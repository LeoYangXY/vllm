# 一条请求在 vLLM V1 中的完整生命周期

> 适用版本：vLLM V1（代码位于 `vllm/v1/`，个别入口在 `vllm/entrypoints/`）。
> 本文只回答一个问题：**一个请求从 HTTP 进来到最后一个 token 出去，经过了哪些地方**。各模块的实现细节见对应篇章（架构见 `01`，调度见 `05`，前缀缓存见 `04`）。

## 0. 部署卡片与 TL;DR

### 0.1 部署卡片

| 项目 | 内容 |
| --- | --- |
| **怎么开** | 没有开关，这是 V1 架构的默认行为。想观察单请求细节加 `--enable-log-requests`；多前端进程用 `--api-server-count N`（见 `01`） |
| **怎么确认生效** | `ps -ef \| grep EngineCore` 能看到独立的核心进程；打开 `--enable-log-requests` 后日志会打印每个请求的到达 / 完成时间点；`/metrics` 里有 `vllm:e2e_request_latency_seconds` |
| **症状 → 调什么** | ① TTFT 高 → 先看 `vllm:num_requests_waiting`（在排队，见 `05`），再看前缀命中率（`04`）；② 首字慢但后续流畅 → prefill 被切成了多个 chunk（`05`）；③ 客户端断开后 GPU 还在算 → abort 要到**下一个 step** 才生效，属预期；④ 多轮对话 TTFT 没改善 → 查 `04` 的命中率 |
| **一条命令试** | `curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"写一首诗"}],"stream":true}'` |

> 延迟的两个来源要分开看：**TTFT ≈ 排队 + prefill**，**TPOT ≈ 每个 decode step 的耗时**。本文把这两段各自穿过哪些进程讲清楚，定位延迟问题时先判断是哪一段。

### 0.2 五句话 TL;DR

1. **是什么**：一条请求要穿过三类进程——前端（HTTP / tokenize / detokenize）、EngineCore（调度）、Worker（跑 GPU）。
2. **解决什么**：这三类工作性质完全不同（CPU 密集 / 决策 / GPU 密集），挤在一个 Python 进程里会互相抢锁、互相阻塞。
3. **怎么做**：前端把 prompt 转成 token id 后发 `EngineCoreRequest` 给 EngineCore；EngineCore 每步决定"这一批算谁、各算几个 token"；Worker 执行完把**增量 token** 回传前端。
4. **收益**：GPU 进程不被 Python 前后处理阻塞，调度循环能独立跑；输出只传增量，长会话的带宽不会随长度线性增长。
5. **代价**：状态分散在三个进程（前端 `RequestState`、EngineCore 的 `Request`、Worker 的 `InputBatch`），靠 `request_id` 对齐，跨进程调试更难。

---

## 1. 场景与痛点

### 1.1 为什么要把进程切开

如果让一个进程既做 HTTP 服务、又做 tokenize、又做调度、又跑 GPU forward，问题很明显：

1. **GIL 单点**：Python 的全局解释器锁让 CPU 密集工作（tokenize、detokenize、结构化输出）和 GPU 工作无法真正并行。decode step 间隙里要做的大量 CPU 工作，会和下一个 step 的 GPU 执行抢同一个线程。
2. **调度延迟不可控**：调度逻辑若和 forward 在同一个线程，forward 一卡（CUDA 同步、NCCL 等待），调度就停摆，新的请求进不来。
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

注意 **EngineCore 自己不碰 GPU**，它只做决策并发指令；真正跑 forward 的是它下面的 Worker 进程。

### 2.2 两条核心数据结构（真实字段）

这两条结构是前后端之间的全部契约，字段以源码为准：

```python
# vllm/v1/engine/__init__.py:107
class EngineCoreRequest(msgspec.Struct, array_like=True, omit_defaults=True):
    request_id: str                                  # :113
    prompt_token_ids: list[int] | None               # :114  已 tokenize，没有原始文本
    mm_features: list[MultiModalFeatureSpec] | None  # :115  多模态特征（不是 mm_inputs）
    sampling_params: SamplingParams | None           # :116
    pooling_params: PoolingParams | None             # :117
    arrival_time: float                              # :118
    lora_request: LoRARequest | None                 # :119
    cache_salt: str | None                           # :120  前缀缓存隔离用
    prompt_embeds: torch.Tensor | None               # :122
    client_index: int = 0                            # :132  多前端时用来把输出送回原前端
    current_wave: int = 0                            # :137  DP 用
    priority: int = 0                                # :138
    external_req_id: str | None = None               # :147  用户原始 request_id
```

```python
# vllm/v1/engine/__init__.py:196
class EngineCoreOutput(msgspec.Struct, array_like=True, omit_defaults=True):
    request_id: str                                  # :202
    new_token_ids: list[int]                         # :203  只有增量！
    new_logprobs: LogprobsLists | None = None        # :205
    new_prompt_logprobs_tensors: LogprobsTensors | None = None  # :206
    pooling_output: torch.Tensor | None = None       # :208
    finish_reason: FinishReason | None = None        # :210
    stop_reason: int | str | None = None             # :211
    events: list[EngineCoreEvent] | None = None      # :212  排队/调度/抢占时间戳
    kv_transfer_params: dict[str, Any] | None = None # :213
    prefill_stats: PrefillStats | None = None        # :218
```

一次 step 的所有输出打包在 `EngineCoreOutputs`（[vllm/v1/engine/__init__.py:253](../vllm/v1/engine/__init__.py#L253)）里，它额外带 `engine_index`（:262，多副本时标识来源）、`outputs`（:265）、`scheduler_stats`（:266）、`finished_requests`（:270）。

> 三个容易记错的点：① 请求侧字段叫 **`mm_features`** 不是 `mm_inputs`；② `EngineCoreRequest` 里**没有** `block_size`（块大小是全局配置，见 `03`）；③ 输出侧**没有** `request_outputs` / `engine_core_metrics` 这类字段。

### 2.3 为什么 output 用增量传输

每个 decode step，一个请求只产生 **1 个（或 spec decode 下 k 个）新 token**。如果每次都回传完整序列，带宽会随 sequence 长度线性增长，长会话会爆掉。V1 的 `EngineCoreOutput` 只带 `new_token_ids`（增量）和 `new_logprobs`，前端 `OutputProcessor` 把它 append 到本地维护的 `RequestState` 上（见 [vllm/v1/engine/output_processor.py:622](../vllm/v1/engine/output_processor.py#L622) 的 `process_outputs`）。只有 finish 时才补全 usage。

### 2.4 跨进程通信：ZMQ + msgpack

前端不直接 `import` EngineCore，而是通过 `EngineCoreClient` 抽象发消息。`add_request` 的抽象定义在 [core_client.py:177](../vllm/v1/engine/core_client.py#L177)（基类里是 `raise NotImplementedError`），各实现的位置是：

| 实现 | 位置 | 场景 |
| --- | --- | --- |
| `InprocClient` | [core_client.py:333](../vllm/v1/engine/core_client.py#L333)（`add_request` 在 [:369](../vllm/v1/engine/core_client.py#L369)） | 单进程调试，直接持有 `EngineCore` 对象 |
| `SyncMPClient` | [core_client.py:869](../vllm/v1/engine/core_client.py#L869)（`add_request` 在 [:976](../vllm/v1/engine/core_client.py#L976)） | 离线 `LLM`，后台线程收包 |
| `AsyncMPClient` | [core_client.py:1046](../vllm/v1/engine/core_client.py#L1046)（`add_request_async` 在 [:1219](../vllm/v1/engine/core_client.py#L1219)） | 在线 `AsyncLLM`，`asyncio.Task` 收包 |
| `DPAsyncMPClient` | [core_client.py:1323](../vllm/v1/engine/core_client.py#L1323) | DP + 外部负载均衡，把请求散发到多个 EngineCore |
| `DPLBAsyncMPClient` | [core_client.py:1507](../vllm/v1/engine/core_client.py#L1507) | DP + 内部负载均衡，按各副本水位挑引擎 |

底层是 ZMQ（同机用 IPC，跨机用 TCP）+ msgpack，序列化细节见 `01` §3.3。

---

## 3. 代码走读（按时间顺序）

### 3.1 HTTP 入口 → `AsyncLLM.generate()`

OpenAI 兼容的 chat 接口在 `vllm/entrypoints/openai/serving_chat.py` 里把 HTTP body 转成 `Prompt` 与 `SamplingParams`，再调用 `AsyncLLM.generate()`。它的签名与文档注释（[vllm/v1/engine/async_llm.py:624](../vllm/v1/engine/async_llm.py#L624)）已经把全流程写清楚了：

```python
624:    async def generate(
625:        self,
626:        prompt: EngineCoreRequest | PromptType | EngineInput
629:            | AsyncGenerator[StreamingInput, None],
630:        sampling_params: SamplingParams,
631:        request_id: str,
...
643:        """
644:        Main function called by the API server to kick off a request
645:            * 1) Making an AsyncStream corresponding to the Request.
646:            * 2) Processing the Input.
647:            * 3) Adding the Request to the Detokenizer.
648:            * 4) Adding the Request to the EngineCore (separate process).
649:
650:        A separate output_handler loop runs in a background AsyncIO task,
651:        pulling outputs from EngineCore and putting them into the
652:        per-request AsyncStream.
...
669:        q: RequestOutputCollector | None = None
670:        try:
671:            q = await self.add_request(
672:                request_id, prompt, sampling_params, ...
684:            )
685:
686:            # The output_handler task pushes items into the queue.
687:            # This task pulls from the queue and yields to caller.
688:            finished = False
689:            while not finished:
692:                out = q.get_nowait() or await q.get()
```

即：`generate()` 本身只做「登记请求 + 从一个队列里取结果并 yield」，真正的输出由后台的 `output_handler` 任务推入队列。

### 3.2 前端 InputProcessor（CPU 密集，放前端）

`add_request` 内部会调 `InputProcessor.process_inputs`（[vllm/v1/engine/input_processor.py:281](../vllm/v1/engine/input_processor.py#L281)）完成 tokenize 与多模态预处理，产出 §2.2 里的 `EngineCoreRequest`。

**为什么放前端**：tokenize 和图像解码是纯 CPU 工作，且耗时与序列长度相关。放 EngineCore 进程会阻塞调度循环；放前端则可以与 GPU 执行并行。多模态场景下单个请求可能耗时几十毫秒，所以 `AsyncLLM` 会把它丢到线程池里异步执行（见 `01` §3.9）。

### 3.3 跨进程发请求

前端把 `EngineCoreRequest` 通过 `core_client.add_request` 发出（各实现见 §2.4）。EngineCore 侧收到后先做 `preprocess_add_request`，构造出内部的 `Request` 对象，再交给调度器：

```python
# vllm/v1/engine/core.py:444
444:    def add_request(self, request: Request, request_wave: int = 0):
445:        """Add request to the scheduler.
446:
447:        `request_wave`: indicate which wave of requests this is expected to
448:        belong to in DP case
449:        """
450:        # Validate the request_id type.
451:        if not isinstance(request.request_id, str):
452:            raise TypeError(
453:                f"request_id must be a string, got {type(request.request_id)}"
454:            )
...
467:        if request.kv_transfer_params is not None and (
468:            not self.scheduler.get_kv_connector()
469:        ):
470:            logger.warning(
471:                "Got kv_transfer_params, but no KVConnector found. "
472:                "Disabling KVTransfer for this request."
473:            )
```

注意这里的入参已经是 `Request`（EngineCore 内部类型），不是 `EngineCoreRequest`——两者由 `preprocess_add_request` 转换。这一步被**故意放在 IO 线程**里做，好让请求初始化（含 block hash 计算）和 GPU forward 重叠（见 `01` §3.4）。

### 3.4 请求对象 Request（贯穿全程的状态机）

`Request`（[vllm/v1/request.py:59](../vllm/v1/request.py#L59)）只存在于 EngineCore 进程，是调度侧的唯一真相：

| 字段 | 含义 |
| --- | --- |
| `request_id` | 内部 request id |
| `status` | 状态机当前状态（初始 `WAITING`） |
| `num_computed_tokens` | **已经跑过 forward 的 token 数**（含前缀缓存命中部分），调度器的核心游标 |
| `num_output_tokens` | 已 decode 出的 token 数（property，不含异步调度的占位符） |
| `num_tokens` | prompt + output 的总长度（property） |

状态机（[vllm/v1/request.py:364](../vllm/v1/request.py#L364)）：

```
WAITING -> RUNNING -> (PREEMPTED -> RUNNING) -> FINISHED_(STOPPED|ABORTED|LENGTH_CAPPED)
```

> 注意：`num_cached_tokens`（本次 prefill 中"没实际算"的 token 数）**不是 `Request` 的字段**，它在 `PrefillStats` 里（[vllm/v1/metrics/stats.py:275](../vllm/v1/metrics/stats.py#L275)），用于统计前缀命中。别和 `num_computed_tokens` 混淆。

### 3.5 EngineCore 主循环 step()

EngineCore 是一个 busy loop，每个迭代做：消费新请求 → 调度 → 执行 → 收集 output → 回传：

```python
# vllm/v1/engine/core.py:589
589:    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
...
598:        if not self.scheduler.has_requests():
599:            return {}, False
600:        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
601:        future = self.model_executor.execute_model(scheduler_output, non_block=True)
602:        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
...
607:            model_output = future.result()
608:            if model_output is None:
609:                model_output = self.model_executor.sample_tokens(grammar_output)
...
613:        self._process_aborts_queue()
614:        engine_core_outputs = self.scheduler.update_from_output(
615:            scheduler_output, model_output
616:        )
```

返回值的 key 是 `client_index`（多前端时用来把输出送回正确的前端）。`Scheduler.schedule()` 的细节是 `05` 篇的主题，这里只需知道它产出本步的 batch 计划。

### 3.6 Worker 侧执行

`Executor.execute_model` 把 `SchedulerOutput` 广播给所有 Worker（`collective_rpc`）。`GPUModelRunner.execute_model` 把请求装进持久化的 `InputBatch`，构造 forward 张量，跑模型，调 `Sampler`，得到 `ModelRunnerOutput`（[vllm/v1/outputs.py:319](../vllm/v1/outputs.py#L319)）：

| 字段 | 含义 |
| --- | --- |
| `req_ids` | 本步参与的请求 id 列表 |
| `req_id_to_index` | id → batch 行号 |
| `sampled_token_ids: list[list[int]]` | **每个请求本步采到的 token**（spec decode 时一行可能有多个） |
| `logprobs` | 采样的 logprobs |
| `finished_sampling` | 本步采样结束的请求 |

### 3.7 回前端：Detokenizer + OutputProcessor

`EngineCoreOutput` 回到前端后，先过 `Detokenizer`（token id → 文本 + stop string 检测，[detokenizer.py:96](../vllm/v1/engine/detokenizer.py#L96) 与 [detokenizer.py:310](../vllm/v1/engine/detokenizer.py#L310)），再到 `OutputProcessor.process_outputs`（[output_processor.py:622](../vllm/v1/engine/output_processor.py#L622)）做增量聚合并产出 `RequestOutput`。

> 注意：`OutputProcessor` 在前端进程，所以它持有 **完整的** `RequestState`，而 EngineCore 只发增量。这也是为什么 stop **string**（需要文本才能判定）只能在前端检测——它甚至没法直接让 engine 停下，只能通过 `finish_requests` 回调，代价是会多生成一个 token 再截断（细节见 `05` §3.11）。

### 3.8 流式返回与收尾

`generate()` 拿到 `RequestOutput` 后 `yield` 给 HTTP handler，handler 再包成 SSE `data: {...}` chunk。finish 时（`RequestStatus.FINISHED_*`）EngineCore 侧会释放该请求的 KV block，`OutputProcessor` 补上 `usage`（prompt_tokens / completion_tokens）。

### 3.9 abort 路径

用户断连或显式 cancel：前端发 abort 消息给 EngineCore；EngineCore 在**下一个 step** 把请求状态置 `FINISHED_ABORTED`、释放 KV block、从 batch 移除。

所以 abort **不是立刻停 GPU**，而是"下一步生效"。这是理解"为什么客户端断开了 GPU 还在算"的关键。

---

## 4. 关键数据结构

| 结构 | 字段要点 | 定义位置 |
| --- | --- | --- |
| `EngineCoreRequest` | `prompt_token_ids` / `mm_features` / `sampling_params` / `lora_request` / `cache_salt` / `client_index` | [vllm/v1/engine/__init__.py:107](../vllm/v1/engine/__init__.py#L107) |
| `EngineCoreOutput` | `new_token_ids` / `new_logprobs` / `finish_reason` / `stop_reason` / `events` / `prefill_stats` | [vllm/v1/engine/__init__.py:196](../vllm/v1/engine/__init__.py#L196) |
| `EngineCoreOutputs` | `engine_index` / `outputs` / `scheduler_stats` / `finished_requests` | [vllm/v1/engine/__init__.py:253](../vllm/v1/engine/__init__.py#L253) |
| `Request` | `num_computed_tokens` / `num_output_tokens` / `num_tokens` / `status` | [vllm/v1/request.py:59](../vllm/v1/request.py#L59) |
| `RequestStatus` | WAITING / RUNNING / PREEMPTED / FINISHED_* | [vllm/v1/request.py:364](../vllm/v1/request.py#L364) |
| `ModelRunnerOutput` | `sampled_token_ids` / `logprobs` / `finished_sampling` | [vllm/v1/outputs.py:319](../vllm/v1/outputs.py#L319) |
| `RequestOutput` | `request_id` / `outputs(list[CompletionOutput])` / `finished` / `usage` | `vllm/outputs.py`（前后端共用定义） |

---

## 5. 收益与代价

**收益**
- GPU 进程不被 tokenizer / detokenizer / 结构化输出阻塞，decode 间隙的 CPU 工作与下一步 GPU 执行 overlap。
- 调度循环独立，新请求延迟稳定，不受长 prefill 的同步阻塞影响。
- output 增量传输，长会话带宽不随长度爆炸。
- DP 多副本天然支持：前端 `DPAsyncMPClient` 把请求散到多个 EngineCore。

**代价 / 注意点**
- 多进程引入 ZMQ 序列化开销（msgpack）+ 一次内核拷贝；小 batch 下这一步不可忽略（这也是异步调度想 overlap 掉的部分，见 `05`）。
- 状态分三处：前端有完整 `RequestState`，EngineCore 有 `Request` 状态机，Worker 有 `InputBatch`（persistent batch）。三处必须靠 `request_id` 对齐；任何一边状态不同步就会出 bug（如重复 token、abort 失效）。
- 调试更难：日志分散在三个进程，需要看 `VLLM_LOGGING_LEVEL` 与各个进程的标准输出。
- stop string 在前端判定 → 命中后 engine 会多产一个 token 才被截断（见 §3.7）。
- abort 下一 step 才生效（见 §3.9）。

---

## 6. 面试高频问题（速查，理解原理可跳过）

> 这一节是面试向的 Q&A 速查，答案大多是正文的压缩版。**如果目标是理解系统，读完 §1~§5 就够了**，不必逐题记忆。

**Q1：vLLM V1 为什么要拆成多个进程？**
A：核心原因一是 Python GIL 让 CPU 密集工作（tokenize/detokenize/guided decoding）与 GPU 工作无法并行；二是让调度循环（EngineCore）独立成 busy loop，不被 forward 的 CUDA 同步阻塞，新请求延迟稳定；三是 DP 多副本需要前端与核心解耦。三者通过 ZMQ + msgpack 通信（`EngineCoreClient` 抽象）。

**Q2：`EngineCoreRequest` 和 `Request` 有什么区别？**
A：`EngineCoreRequest` 是前端 → EngineCore 的「请求描述」（静态、一次性，[vllm/v1/engine/__init__.py:107](../vllm/v1/engine/__init__.py#L107)）；`Request` 是 EngineCore 内部贯穿全程的可变状态机（`num_computed_tokens` 等会随 step 更新，[vllm/v1/request.py:59](../vllm/v1/request.py#L59)）。两者由 `preprocess_add_request` 转换。

**Q3：为什么 output 用增量而不是每次回传完整序列？**
A：每个 decode step 只产 1 个 token，完整回传会让带宽随序列长度线性增长，长会话爆带宽、增尾延迟。V1 回传 `new_token_ids` 增量，前端 `OutputProcessor` 本地聚合（[output_processor.py:622](../vllm/v1/engine/output_processor.py#L622)）。

**Q4：tokenize 为什么要放在前端进程而不是 EngineCore？**
A：tokenize 是 CPU 密集且与序列长度相关，放 EngineCore 会阻塞调度循环；放前端可以和 GPU 执行并行。多模态的图像解码同理，且可能耗时几十毫秒。

**Q5：abort 是怎么跨进程生效的？**
A：前端发 abort 消息给 EngineCore；EngineCore 在**下一个 step** 把请求置 `FINISHED_ABORTED`、释放 KV block、移出 batch。所以 abort 是"下一 step 生效"，不是立刻停 GPU。

**Q6：DP 模式下请求怎么分发？**
A：前端用 `DPAsyncMPClient`（[core_client.py:1323](../vllm/v1/engine/core_client.py#L1323)，外部 LB）或 `DPLBAsyncMPClient`（[core_client.py:1507](../vllm/v1/engine/core_client.py#L1507)，内部 LB，按各副本水位挑引擎），把请求散到多个 EngineCore 副本。

**Q7：`num_computed_tokens` 和"前缀命中数"有什么区别？**
A：`num_computed_tokens` 是已经跑过 forward 的 token 总数（含前缀缓存命中部分），是调度器的核心游标；"本次 prefill 中没实际算的 token 数"是 `PrefillStats.num_cached_tokens`（[stats.py:275](../vllm/v1/metrics/stats.py#L275)），只用于统计。调度器据 `num_computed_tokens` 决定本步从哪开始算（见 `04` / `05`）。

**Q8：Detokenizer 和 OutputProcessor 谁在前、为什么分开？**
A：先 Detokenizer（token→文本+stop 检测）后 OutputProcessor（增量聚合+usage）。Detokenizer 关注"这个 token 是否该停、文本是什么"，OutputProcessor 关注"如何把增量拼成完整的 `RequestOutput` 并推给调用方"，职责不同所以拆开。

**Q9：为什么 `ModelRunnerOutput` 的 `sampled_token_ids` 是 `list[list[int]]` 而不是 `list[int]`？**
A：spec decode 下本步可能一次验证通过多个 draft token（见 `07`），所以每个请求一行可能是多个 token；普通 decode 下每行就是 1 个。

**Q10：单进程调试模式怎么跑？**
A：用 `InprocClient`（[core_client.py:333](../vllm/v1/engine/core_client.py#L333)，frontend 与 EngineCore 同进程，直接持有对象），省掉 ZMQ 序列化，便于断点调试。

**Q11：请求从进来到出第一个 token，经过了几次进程间通信？**
A：至少两次：①前端 `add_request` → EngineCore（入队）；②EngineCore 第一个 `step` 的 `EngineCoreOutput` → 前端。之后每个 decode step 一次第②类往返。

**Q12：如果前端挂了但 EngineCore 还在跑会怎样？**
A：EngineCore 不知道前端状态，会继续调度/执行直到请求自然结束或资源用尽；但因为 output 没人消费，KV cache 不会被及时释放，最终会卡在调度（waiting 堆积、cache 打满触发抢占）。生产上一般两者一起由编排系统管理。

---

## 7. 延伸阅读

- 源码：`vllm/v1/engine/async_llm.py`、`core.py`、`core_client.py`、`input_processor.py`、`output_processor.py`、`detokenizer.py`
- 请求状态：`vllm/v1/request.py`、`vllm/v1/engine/__init__.py`（消息体定义）
- 输出体：`vllm/v1/outputs.py`、`vllm/outputs.py`
- 官方：https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- 配套阅读：`01-v1-architecture-overview.md`（整体架构与进程模型）、`05-scheduler-continuous-batching.md`（调度如何驱动 step）、`04-prefix-caching.md`（TTFT 优化）
