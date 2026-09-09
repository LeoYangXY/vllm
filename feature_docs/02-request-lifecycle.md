# 一条请求在 vLLM V1 中的完整生命周期

> 适用版本：vLLM V1（代码位于 `vllm/v1/`，个别入口在 `vllm/entrypoints/`）。

## 0. TL;DR

- vLLM V1 把一条请求的处理切成 **三个进程空间**：前端进程（OpenAI API Server + InputProcessor + Detokenizer + OutputProcessor）、EngineCore 进程（调度器 + Executor）、Worker 进程（真正跑 GPU 的 ModelRunner）。
- 请求从 `/v1/chat/completions` 进来到第一个 token 流出，要跨 **两次 ZMQ 往返**；之后每个 decode step 走一次 `EngineCore.step()` → Worker → `EngineCoreOutput` 回前端。
- 关键设计：输入预处理（tokenize/图像解码）放在**前端**，GPU 计算放在 **Worker**，调度决策放在 **EngineCore**；三者用 `EngineCoreRequest`（msgpack 编码）和 `EngineCoreOutput`（增量 delta）通信。
- 收益：GPU 进程不被 Python 预处理/后处理阻塞，调度循环与 GPU 执行可以真正 overlap；output 用增量传输，省带宽、降低尾延迟。

---

## 1. 场景与痛点

### 1.1 为什么要把进程切开

如果让一个进程既做 HTTP 服务、又做 tokenize、又做调度、又跑 GPU forward，问题很明显：

1. **GIL 单点**：Python 的全局解释器锁让 CPU 密集工作（tokenize、detokenize、结构化输出）和 GPU 工作无法真正并行。decode step 间隙里要做的大量 CPU 工作，会和下一个 step 的 GPU 执行抢同一个线程。
2. **调度延迟不可控**：V0 里调度逻辑跑在和 forward 同一个线程，forward 一卡（CUDA 同步、NCCL 等待），调度就停摆，新的请求进不来。
3. **可扩展性差**：DP（data parallel）多个副本时，每个副本都要独立调度，前端与核心必须解耦。

V1 的解法：把 **EngineCore 单独放进一个进程**，前端（API server）通过 ZMQ 与它通信。这样 EngineCore 的调度循环是一个独立的 busy loop，不受前端 HTTP 协程、tokenizer 的干扰。

### 1.2 一条请求的真实时间线（粗粒度）

```
用户 HTTP  ──>  API Server  ──>  AsyncLLM.generate()  ──>  InputProcessor
   │                                                            │ (CPU: tokenize/mm)
   │                                                            ▼
   │                                                   EngineCore.add_request  (ZMQ)
   │                                                            │
   │                                                   Scheduler.schedule() 每步
   │                                                            ▼
   │                                                   Executor ──> Worker.execute_model()
   │                                                            │  (GPU forward + sample)
   │                                                   EngineCoreOutput (ZMQ 回前端)
   │                                                            ▼
   │                                                   Detokenizer + OutputProcessor
   │                                                            │
   └────────────────────────<── RequestOutput (流式 delta) ─────┘
```

---

## 2. 核心设计

### 2.1 三个进程空间各自负责什么

| 进程 | 关键类 | 职责 | 是否碰 GPU |
| --- | --- | --- | --- |
| 前端 frontend | `AsyncLLM` `InputProcessor` `Detokenizer` `OutputProcessor` | HTTP、tokenize、多模态预处理、detokenize、增量聚合、流式返回 | 否（CPU） |
| 核心 EngineCore | `EngineCore` `Scheduler` `Executor` | 维护请求队列、每步调度、KV cache 分配、把 batch 下发给 workers | 否（CPU，发指令） |
| Worker | `GPUWorker` `GPUModelRunner` | 真正执行 forward、采样、产出 logits/token | 是 |

### 2.2 两条核心数据结构

- **`EngineCoreRequest`**：前端 → EngineCore 的「请求描述」，包含 prompt token ids、SamplingParams、LoRARequest、多模态输入 mm_inputs/mm_hashes、block_size 等。定义在 [vllm/v1/engine/__init__.py:107](../vllm/v1/engine/__init__.py#L107)。
- **`EngineCoreOutput`**：EngineCore → 前端的「一步结果」，包含每个请求的 new token、finished 状态、采样的 logprobs、以及 EngineCore 侧的 metrics。定义在 [vllm/v1/engine/__init__.py:196](../vllm/v1/engine/__init__.py#L196)。多个 `EngineCoreOutput` 打包在 `EngineCoreOutputs` 里（[vllm/v1/engine/__init__.py:253](../vllm/v1/engine/__init__.py#L253)）。

### 2.3 为什么 output 用增量传输

每个 decode step，一个请求只产生 **1 个（或 spec decode 下 k 个）新 token**。如果每次都回传完整序列，带宽会随 sequence 长度线性增长，长会话会爆掉。V1 的 `EngineCoreOutput` 只带 `new_token_ids`（增量）和 `new_logprobs`，前端 `OutputProcessor` 把它 append 到本地维护的 `RequestState` 上（见 [vllm/v1/engine/output_processor.py:622](../vllm/v1/engine/output_processor.py#L622) 的 `process_outputs`）。只有 finish 时才补全 usage。

### 2.4 跨进程通信：ZMQ + msgpack

前端不直接 `import` EngineCore，而是通过 `CoreClient` 抽象发消息：

- `InprocClient`：单进程调试模式（frontend 与 EngineCore 同进程，队列通信）。
- `SyncMPClient` / `AsyncMPClient`：多进程，底层用 **ZMQ 的 IPC/共享内存 socket**。定义在 [vllm/v1/engine/core_client.py:177](../vllm/v1/engine/core_client.py#L177)（`add_request`）。
- `DPAsyncMPClient`：DP 模式下，前端要把请求分发给多个 EngineCore 副本（[vllm/v1/engine/core_client.py:369](../vllm/v1/engine/core_client.py#L369)）。

请求对象跨进程时用 msgpack（`vllm/v1/serial_utils.py`）序列化，避免 pickle 的开销与安全坑。

---

## 3. 代码走读（按时间顺序）

### 3.1 HTTP 入口

OpenAI 兼容的 chat 接口在 `vllm/entrypoints/openai/serving_chat.py` 里把 HTTP body 转成 `Prompt` 与 `SamplingParams`，再调用 `AsyncLLM.generate()`。`AsyncLLM` 是 V1 前端的总入口，它本身是一个异步生成器。

```python
# vllm/v1/engine/async_llm.py:624
async def generate(
    self,
    prompt: Prompt,
    sampling_params: SamplingParams,
    request_id: str,
    ...
) -> AsyncIterator[RequestOutput]:
    # 1) 在协程里做 input processing
    # 2) 把 EngineCoreRequest 通过 core_client 发给 EngineCore
    # 3) 进入循环：从 output queue 取 EngineCoreOutput，转成 RequestOutput yield
```

### 3.2 前端 InputProcessor（CPU 密集，放前端）

`generate()` 内部（或 `LLM.generate` 同步路径）会调用 `InputProcessor.process_inputs`：

```python
# vllm/v1/engine/input_processor.py:281
def process_inputs(self, request_id, prompt, params, ...):
    # - 调 tokenizer 把文本转 token ids
    # - 多模态：解码图像/视频，跑 mm processor，产出 mm_inputs + mm_hashes
    # - 组装 EngineCoreRequest（含 prompt_token_ids / mm_inputs / mm_hashes / ...）
    return engine_core_request
```

**为什么放前端**：tokenize 和图像解码是纯 CPU 工作，且耗时与序列长度相关。如果放 EngineCore 进程，会阻塞调度循环；放前端则可以与 GPU 执行并行（前端协程和 EngineCore busy loop 在两个进程）。

### 3.3 跨进程发请求

`generate()` 把 `EngineCoreRequest` 通过 `core_client.add_request` 发出：

```python
# vllm/v1/engine/core_client.py:177
def add_request(self, request: EngineCoreRequest) -> None:
    # 把请求塞进 ZMQ socket，EngineCore busy loop 的下一次迭代会读到
```

EngineCore 侧在 `EngineCore.step()` 之前消费这个请求，构造 `Request` 对象：

```python
# vllm/v1/engine/core.py:444
def add_request(self, request: Request, request_wave: int = 0):
    # - 用 EngineCoreRequest 构造 vllm.v1.request.Request
    # - 初始化 KV cache prompt 信息（num_cached_tokens 来自 prefix cache 命中）
    # - 进入 waiting 队列，交给 Scheduler
```

### 3.4 请求对象 Request（贯穿全程的状态机）

```python
# vllm/v1/request.py:59
class Request:
    request_id: str
    prompt_token_ids: list[int]
    num_cached_tokens: int        # prefix cache 已经算好的 token 数
    num_computed_tokens: int      # 已经跑过 forward 的 token 数（含 cache）
    num_output_tokens: int        # 已经 decode 出的 token 数
    status: RequestStatus
```

状态机在 [vllm/v1/request.py:364](../vllm/v1/request.py#L364)：

```
WAITING -> RUNNING -> (PREEMPTED -> RUNNING) -> FINISHED_(STOPPED|ABORTED|LENGTH_CAPPED)
```

每产出一个 token 就调 `append_output_token_ids`（[vllm/v1/request.py:265](../vllm/v1/request.py#L265)），更新 `num_output_tokens` 与 `output_token_ids`。

### 3.5 EngineCore 主循环 step()

EngineCore 是一个 busy loop，每个迭代做：消费新请求 → 调度 → 执行 → 收集 output → 回传。

```python
# vllm/v1/engine/core.py:589
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    # 1) 从 input socket 读新请求 / abort
    # 2) scheduler.schedule() -> SchedulerOutput（决定本步跑哪些请求、各跑多少 token）
    # 3) executor.execute_model(scheduler_output) -> List[ModelRunnerOutput]
    # 4) scheduler.update_from_outputs(outputs) 更新状态机
    # 5) 构造 EngineCoreOutput，通过 output socket 发给前端
```

`Scheduler.schedule()` 是另一个话题（见 `05-scheduler-continuous-batching.md`），这里只要知道它产出本步的 batch 计划。

### 3.6 Worker 侧执行

`Executor.execute_model` 把 `SchedulerOutput` 广播给所有 Worker（`collective_rpc`）。`GPUModelRunner.execute_model` 把请求组装成 `InputBatch`（见 [vllm/v1/worker/gpu_input_batch.py](../vllm/v1/worker/gpu_input_batch.py) 的 `InputBatch.add_request` / `CachedRequestState`），构造 forward 张量，跑模型，调 `Sampler`，得到 `ModelRunnerOutput`：

```python
# vllm/v1/outputs.py:320
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict
    sampled_token_ids: list[list[int]]        # 每个请求本步采到的 token（spec decode 时可能 >1）
    logprobs: Optional[...]
    finished_sampling: Optional[set[str]]
```

### 3.7 回前端：Detokenizer + OutputProcessor

`EngineCoreOutput` 回到前端后，先过 `Detokenizer`（把 token id 转文本、检测 stop string，约在 [vllm/v1/engine/detokenizer.py:96](../vllm/v1/engine/detokenizer.py#L96) 与 [vllm/v1/engine/detokenizer.py:310](../vllm/v1/engine/detokenizer.py#L310) 的 stop 检测逻辑），再到 `OutputProcessor`：

```python
# vllm/v1/engine/output_processor.py:558
def add_request(self, request_id, ...):
    # 为请求建 RequestState，准备增量聚合

# vllm/v1/engine/output_processor.py:622
def process_outputs(self, outputs: EngineCoreOutputs, ...):
    # 对每个 EngineCoreOutput：
    #   - 把 new_token_ids 拼到 RequestState.output_token_ids
    #   - 构造 RequestOutput（含 text / token_ids / logprobs）
    #   - 通过 callback / queue 推给 generate() 的调用方
```

> 注意：`OutputProcessor` 在前端进程，所以它持有 **完整的** `RequestState`，而 EngineCore 只发增量。`RequestState` 里缓存的是 `output_token_ids` 累积结果，用于 `usage` 统计与重复 abort 时的回放。

### 3.8 流式返回与收尾

`generate()` 拿到 `RequestOutput` 后 `yield` 给 HTTP handler，handler 再包成 SSE `data: {...}` chunk。finish 时（`RequestStatus.FINISHED_*`）EngineCore 侧会 `KVCacheManager.free` 释放该请求的 KV block，`OutputProcessor` 补上 `usage`（prompt_tokens / completion_tokens）。

### 3.9 abort 路径

用户断连或显式 cancel：前端调 `core_client.abort_request` → EngineCore 在下一个 `step` 把请求状态置 `FINISHED_ABORTED`、释放 KV block、从 batch 移除。abort 也是跨进程消息，保证 EngineCore 不会继续为一个已断开的请求算到结束。

---

## 4. 关键数据结构

| 结构 | 字段要点 | 定义位置 |
| --- | --- | --- |
| `EngineCoreRequest` | prompt_token_ids, mm_inputs, mm_hashes, sampling_params, lora_request, block_size, cache_salt | [vllm/v1/engine/__init__.py:107](../vllm/v1/engine/__init__.py#L107) |
| `EngineCoreOutput` | request_outputs, scheduler_stats, engine_core_metrics, kv_connector_output | [vllm/v1/engine/__init__.py:196](../vllm/v1/engine/__init__.py#L196) |
| `EngineCoreOutputs` | outputs: list[EngineCoreOutput], stop, step_image | [vllm/v1/engine/__init__.py:253](../vllm/v1/engine/__init__.py#L253) |
| `Request` | num_cached_tokens / num_computed_tokens / num_output_tokens / status | [vllm/v1/request.py:59](vllm/v1/request.py#L59) |
| `RequestStatus` | WAITING/RUNNING/PREEMPTED/FINISHED_* | [vllm/v1/request.py:364](vllm/v1/request.py#L364) |
| `ModelRunnerOutput` | sampled_token_ids, logprobs, finished_sampling | [vllm/v1/outputs.py:320](vllm/v1/outputs.py#L320) |
| `RequestOutput` | request_id, prompt, outputs(list[CompletionOutput]), finished, usage | `vllm/outputs.py`（V0/V1 共用定义） |

---

## 5. 收益与代价

**收益**
- GPU 进程不被 tokenizer / detokenizer / 结构化输出阻塞，decode 间隙的 CPU 工作与下一步 GPU 执行 overlap。
- 调度循环独立，新请求延迟稳定，不受长 prefill 的同步阻塞影响。
- output 增量传输，长会话带宽不随长度爆炸。
- DP 多副本天然支持：前端 `DPAsyncMPClient` 把请求散到多个 EngineCore。

**代价 / 注意点**
- 多进程引入 ZMQ 序列化开销（msgpack）+ 一次内核拷贝；小 batch 下这一步不可忽略（这也是 async scheduling 想 overlap 掉的部分，见 `05`）。
- 状态分三处：前端有完整 `RequestState`，EngineCore 有 `Request` 状态机，Worker 有 `InputBatch`（persistent batch）。三处必须靠 `req_id` 对齐；任何一边状态不同步就会出 bug（如重复 token、abort 失效）。
- 调试更难：日志分散在三个进程，需要看 `VLLM_LOGGING_LEVEL` 与各个进程的标准输出。

---

## 6. 面试高频问题

**Q1：vLLM V1 为什么要拆成三个进程？**
A：核心原因一是 Python GIL 让 CPU 密集工作（tokenize/detokenize/guided decoding）与 GPU 工作无法并行；二是让调度循环（EngineCore）独立成 busy loop，不被 forward 的 CUDA 同步阻塞，新请求延迟稳定；三是 DP 多副本需要前端与核心解耦。三者通过 ZMQ + msgpack 通信（`CoreClient` 抽象）。

**Q2：`EngineCoreRequest` 和 `Request` 有什么区别？**
A：`EngineCoreRequest` 是前端 → EngineCore 的「请求描述」（静态、一次性），定义在 [vllm/v1/engine/__init__.py:107](../vllm/v1/engine/__init__.py#L107)；`Request` 是 EngineCore 内部贯穿全程的可变状态机（num_computed_tokens 等会随 step 更新），定义在 [vllm/v1/request.py:59](vllm/v1/request.py#L59)。

**Q3：为什么 output 用增量而不是每次回传完整序列？**
A：每个 decode step 只产 1 个 token，完整回传会让带宽随序列长度线性增长，长会话爆带宽、增尾延迟。V1 回传 `new_token_ids` 增量，前端 `OutputProcessor` 本地聚合（[vllm/v1/engine/output_processor.py:622](../vllm/v1/engine/output_processor.py#L622)）。

**Q4：tokenize 为什么要放在前端进程而不是 EngineCore？**
A：tokenize 是 CPU 密集且与序列长度相关，放 EngineCore 会阻塞调度循环；放前端可以和 GPU 执行并行。多模态的图像解码同理。

**Q5：abort 是怎么跨进程生效的？**
A：前端发 `abort_request` 消息给 EngineCore；EngineCore 在下一个 `step` 把请求置 `FINISHED_ABORTED`、释放 KV block、移出 batch。所以 abort 是「下一 step 生效」，不是立刻停 GPU。

**Q6：DP 模式下请求怎么分发？**
A：前端用 `DPAsyncMPClient`（[vllm/v1/engine/core_client.py:369](../vllm/v1/engine/core_client.py#L369)），把请求散到多个 EngineCore 副本，`DPCoordinator`（[vllm/v1/engine/coordinator.py](../vllm/v1/engine/coordinator.py)）负责负载均衡与结果汇聚。

**Q7：`num_cached_tokens` 和 `num_computed_tokens` 的区别？**
A：`num_cached_tokens` 来自 prefix cache 命中、还没跑 forward 就已可用的 KV；`num_computed_tokens` 是已经过 forward 的 token 总数（含 cache 部分）。调度器据此决定本步从哪开始算（见 `04` / `05`）。

**Q8：Detokenizer 和 OutputProcessor 谁在前、为什么分开？**
A：先 Detokenizer（token→文本+stop 检测）后 OutputProcessor（增量聚合+usage）。Detokenizer 关注「这个 token 是否该停、文本是什么」，OutputProcessor 关注「如何把增量拼成完整的 `RequestOutput` 并推给调用方」，职责不同所以拆开。

**Q9：为什么 `ModelRunnerOutput` 的 `sampled_token_ids` 是 `list[list[int]]` 而不是 `list[int]`？**
A：spec decode 下本步可能一次验证通过多个 draft token（见 `07`），所以每个请求一行可能是多个 token；普通 decode 下每行就是 1 个。

**Q10：单进程调试模式怎么跑？**
A：用 `InprocClient`（frontend 与 EngineCore 同进程，靠队列通信，[vllm/v1/engine/core_client.py](../vllm/v1/engine/core_client.py)），省掉 ZMQ 序列化，便于断点调试。

**Q11：请求从进来到出第一个 token，经过了几次进程间通信？**
A：至少两次：①前端 `add_request` → EngineCore（入队）；②EngineCore 第一个 `step` 的 `EngineCoreOutput` → 前端。之后每个 decode step 一次第②类往返。

**Q12：如果前端挂了但 EngineCore 还在跑会怎样？**
A：EngineCore 不知道前端状态，会继续调度/执行直到请求自然结束或资源用尽；但因为 output 没人消费，KV cache 不会被及时释放，最终会卡在调度（waiting 堆积、cache 打满触发抢占）。生产上一般两者一起由编排系统管理。

---

## 7. 延伸阅读

- 源码：`vllm/v1/engine/async_llm.py`、`core.py`、`core_client.py`、`input_processor.py`、`output_processor.py`、`detokenizer.py`、`coordinator.py`
- 请求状态：`vllm/v1/request.py`、`vllm/v1/engine/__init__.py`（消息体定义）
- 输出体：`vllm/v1/outputs.py`、`vllm/outputs.py`
- 官方：https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- 配套阅读：`01-v1-architecture-overview.md`（整体架构）、`05-scheduler-continuous-batching.md`（调度如何驱动 step）
