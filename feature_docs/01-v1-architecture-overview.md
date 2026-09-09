# V1 引擎整体架构与进程模型

> 代码基线：本仓库当前 `main` 分支。注意：V0 代码已在本仓库中彻底移除 —— `vllm/engine/llm_engine.py` 现在只是 `vllm.v1.engine.llm_engine.LLMEngine` 的别名（[vllm/engine/llm_engine.py:4](../vllm/engine/llm_engine.py#L4)），`vllm/engine/async_llm_engine.py` 同理（[vllm/engine/async_llm_engine.py:4](../vllm/engine/async_llm_engine.py#L4)）。因此文中凡涉及 V0 的部分一律标注为**「V0 遗留」**且只做架构层面对比，不给出源码行号。

---

## 0. TL;DR

1. **是什么**：vLLM V1 把引擎从「单进程大对象」拆成**前端进程（Frontend）** + **核心进程（EngineCore）**两个角色，中间用 ZMQ + msgpack 通信。
2. **解决什么**：V0 里 tokenize / detokenize / API 序列化这类纯 CPU 的 Python 工作和 GPU 调度循环挤在同一个 Python 解释器里抢 GIL，导致小模型上 CPU 开销成为瓶颈；同时所有 feature（前缀缓存、spec decode、LoRA、结构化输出、KV 传输）都长在同一个 `LLMEngine` 对象上，耦合严重。
3. **怎么做**：`EngineCore`（[vllm/v1/engine/core.py:105](../vllm/v1/engine/core.py#L105)）只负责「调度 + 执行」，跑在自己的进程里跑 busy loop；tokenize / detokenize / 输出聚合 / OpenAI 协议这些留在前端；两者通过 `EngineCoreClient`（[vllm/v1/engine/core_client.py:79](../vllm/v1/engine/core_client.py#L79)）收发 `EngineCoreRequest` / `EngineCoreOutputs`。
4. **怎么做得更快**：EngineCore 内部又用**两个后台 IO 线程 + 两个 queue** 把 ZMQ 收发从 busy loop 里剥离出去（[vllm/v1/engine/core.py:1103](../vllm/v1/engine/core.py#L1103) 的注释写得很直白："These enable us to overlap ZMQ socket IO with GPU since they release the GIL"）。
5. **收益**：GIL 隔离让「CPU 侧调度」和「GPU 侧执行」真正并行；进程边界天然支持多 EngineCore 数据并行（DP）与多 API server 前端横向扩展。

---

## 1. 场景与痛点

### 1.1 V0 遗留：单进程里的一锅粥

V0 时代的 `LLMEngine` 是一个「上帝对象」：它同时持有 tokenizer、Processor（`InputPreprocessor`）、Scheduler、BlockManager、Executor/Worker、Detokenizer、OutputProcessor，以及 `step()` 主循环。`generate()` 的每一次迭代，下面这些 Python 代码都在**同一个 GIL 下**串行执行：

- OpenAI 请求校验 / Pydantic 模型构造 / JSON 序列化
- prompt tokenize、多模态输入处理（图像 resize、pixel_values 打包）
- 调度决策（选哪些 req、分配多少 block）
- 构造 `ModelRunner` 的输入张量（这一步是纯 Python + numpy，V1 里被称为 "Python overhead" 的主要来源）
- detokenize、stop string 判定、输出聚合、统计打点

在 8B 以下的小模型、大 batch、高并发场景下，**单步 wall time 可能只有 5~15 ms，而纯 Python 的调度 + 前后处理开销就能占到一半以上**。此时 GPU 在等 CPU，吞吐被 Python 解释器卡死。

### 1.2 V0 遗留：feature 之间互相污染

V0 里前缀缓存、chunked prefill、speculative decoding、LoRA、多模态、结构化输出这些 feature 共享同一个 `SequenceGroup` / `Scheduler` 抽象，导致：

- 任何 feature 都要理解其它 feature 的存在（`SequenceGroup` 被塞进了几十个字段）；
- 想加一个新的 KV cache 组织形式，要动 block manager + scheduler + worker 三处；
- bug 定位困难：调度逻辑和 IO 逻辑交错，无法单独测试。

### 1.3 V0 遗留：无法做进程级扩展

因为所有状态都在一个 Python 对象图里，V0 无法简单地「起 N 份引擎做数据并行」，也无法「起 M 个 API server 进程共享一份引擎」。而在线服务真实负载恰恰需要这两种扩展方式。

### 1.4 V1 的回答

V1 用一条**进程边界**把上述问题一次性切开：

| 关注点 | V1 的落点 |
| --- | --- |
| 协议适配、tokenize、detokenize、多模态前处理、输出聚合、统计 | 前端进程 |
| 调度决策、KV block 管理、模型执行 | EngineCore 进程（可多份，DP） |
| 模型权重、KV cache 显存、CUDA graph | Worker 进程（TP/PP/EP 可多份） |

---

## 2. 核心设计

### 2.1 三层拓扑

```
┌──────────────────────────── 前端进程 (Frontend, 1..M 个) ────────────────────────────┐
│                                                                                      │
│  FastAPI / OpenAI 协议层                                                             │
│   [vllm/entrypoints/openai/chat_completion/api_router.py:54]                         │
│            │                                                                         │
│            ▼                                                                         │
│  AsyncLLM  (EngineClient)                                                            │
│   [vllm/v1/engine/async_llm.py:77]                                                   │
│   ├── renderer        : chat template / 多模态 processor                              │
│   ├── input_processor : EngineInput --> EngineCoreRequest                             │
│   │     [vllm/v1/engine/input_processor.py:38]      (tokenize / MM 处理)              │
│   ├── output_processor: EngineCoreOutput --> RequestOutput                            │
│   │     [vllm/v1/engine/output_processor.py:448]    (logprobs / 聚合)                 │
│   │       └── IncrementalDetokenizer                                                  │
│   │             [vllm/v1/engine/detokenizer.py:31]  (增量 detok + stop string)        │
│   ├── output_handler  : asyncio task, 拉取 EngineCoreOutputs 并分发                   │
│   │     [vllm/v1/engine/async_llm.py:771]                                             │
│   └── engine_core     : EngineCoreClient  (ZMQ)                                       │
│         [vllm/v1/engine/core_client.py:79]                                            │
└───────────────────────────────┬──────────────────────────────────────────────────────┘
                                │  ZMQ  DEALER/ROUTER (input)  +  PUSH/PULL (output)
                                │  payload = msgpack(EngineCoreRequest / EngineCoreOutputs)
                                ▼
┌────────────────────────── 核心进程 (EngineCore, 1..N 个, DP rank) ────────────────────┐
│  process title = "EngineCore" / "EngineCore_DP{rank}"   [core.py:1298]                │
│                                                                                      │
│  ┌── input_thread ──┐        ┌── main: run_busy_loop ──┐       ┌── output_thread ──┐  │
│  │ process_input_   │        │  [core.py:1403]          │       │ process_output_   │  │
│  │ sockets()        │  put   │  1) _process_input_queue │  put  │ sockets()         │  │
│  │ [core.py:1686]   │──────► │     [core.py:1429]       │──────►│ [core.py:1789]    │  │
│  │ msgpack decode   │        │  2) _process_engine_step │       │ msgpack encode    │  │
│  │ + preprocess_add │        │     [core.py:1460]       │       │ + zero-copy send  │  │
│  └──────────────────┘        │       └─► step_fn()      │       └───────────────────┘  │
│        input_queue           │          [core.py:589]   │          output_queue        │
│                              └────────────┬─────────────┘                              │
│                                           │                                            │
│                             ┌─────────────▼──────────────┐                            │
│                             │ Scheduler (纯 CPU 决策)      │                            │
│                             │ [vllm/v1/core/sched/         │                            │
│                             │      scheduler.py:79]        │                            │
│                             │  .schedule() -> SchedulerOutput                           │
│                             │  .update_from_output() -> ECOs                            │
│                             └─────────────┬──────────────┘                            │
│                                           │  SchedulerOutput                            │
│                             ┌─────────────▼──────────────┐                            │
│                             │ Executor   [executor/abstract.py:38]                      │
│                             │  UniProc / Multiproc / Ray                                │
│                             └─────────────┬──────────────┘                            │
└───────────────────────────────────────────┼────────────────────────────────────────────┘
                                            │  collective_rpc / MessageQueue / Ray DAG
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
            ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
            │ Worker rank 0 │       │ Worker rank 1 │  ...  │ Worker rank N │
            │ [gpu_worker.py:179]                                            │
            │  WorkerBase   [worker_base.py:44]                              │
            │   └── GPUModelRunner [gpu_model_runner.py:503]                 │
            │        ├── InputBatch (persistent) [gpu_input_batch.py:92]     │
            │        ├── KV caches (显存)                                     │
            │        └── Sampler  [vllm/v1/sample/sampler.py:21]            │
            └───────────────┘       └───────────────┘       └───────────────┘
```

### 2.2 为什么 EngineCore 要独立成进程

1. **GIL 隔离（首要动机）**。前端进程里跑的是 asyncio 事件循环、几百个并发 HTTP 连接、tokenizer、detokenizer、JSON 编解码。这些都是「CPU 密集 + 长时间持有 GIL」的活。如果 EngineCore 和它们同进程，一次 detokenize 卡顿就会推迟下一次 `schedule()`，GPU 直接空转。分成两个进程后，**前端的 GIL 抖动完全不会影响 EngineCore 的 busy loop**。
2. **让「CPU 调度」与「GPU 执行」真正并行**。EngineCore 的 `step()`（[vllm/v1/engine/core.py:589](../vllm/v1/engine/core.py#L589)）先 `schedule()`（纯 CPU），然后 `execute_model(..., non_block=True)` 拿到一个 Future，期间去算 grammar bitmask，最后 `future.result()`。再叠加 `step_with_batch_queue`（[vllm/v1/engine/core.py:630](../vllm/v1/engine/core.py#L630)）：调度 N 个 batch 塞进队列，模型异步跑，这样下一步的调度决策可以和上一步的 GPU 执行重叠。
3. **ZMQ IO 与 GPU 的重叠**。EngineCore 内部用两个 daemon 线程 + 两个 `queue.Queue`（[vllm/v1/engine/core.py:1038](../vllm/v1/engine/core.py#L1038)）把 socket 收发挪出主线程。socket 的 `poll()`/`recv()` 会释放 GIL，所以「收新请求 / 发上一批输出」可以和「当前这一步的 forward」同时进行。
4. **可扩展到多进程 DP**。进程边界一旦建立，起 N 个 `EngineCoreProc`（每个持有自己的 Scheduler + KV cache + 一组 Worker）就只是 N 次 fork 的事，见 `launch_core_engines`（[vllm/v1/engine/utils.py:1104](../vllm/v1/engine/utils.py#L1104)）。前端通过 `DPAsyncMPClient` / `DPLBAsyncMPClient` 做路由（[vllm/v1/engine/core_client.py:1323](../vllm/v1/engine/core_client.py#L1323)、[1507](../vllm/v1/engine/core_client.py#L1507)）。
5. **故障隔离 + 多前端**。`CoreEngineProcManager` 会监控子进程存活（[vllm/v1/engine/utils.py:144](../vllm/v1/engine/utils.py#L144)），engine 挂了会通过 `ENGINE_CORE_DEAD` 消息（[vllm/v1/engine/core.py:1022](../vllm/v1/engine/core.py#L1022)）把错误推给前端，而不是把整个 API server 拖死。

### 2.3 设计契约：EngineCore「只管调度 + 执行」

这条契约是理解 V1 的钥匙，具体表现为：

| 职责 | 前端 | EngineCore | Worker |
| --- | --- | --- | --- |
| prompt / 参数校验 | ✅ `InputProcessor._validate_params`（[input_processor.py:84](../vllm/v1/engine/input_processor.py#L84)） | ❌ | ❌ |
| tokenize / 多模态前处理 | ✅ `InputProcessor.process_inputs`（[input_processor.py:281](../vllm/v1/engine/input_processor.py#L281)） | ❌ | ❌ |
| 采样参数默认值填充 | ✅ `process_inputs` 里 `params.clone()` + `update_from_generation_config`（[input_processor.py:356](../vllm/v1/engine/input_processor.py#L356)） | ❌ | ❌ |
| request_id 内部化 | ✅ `assign_request_id`（[input_processor.py:262](../vllm/v1/engine/input_processor.py#L262)） | ❌ | ❌ |
| 调度 / KV block 分配 | ❌ | ✅ `Scheduler.schedule`（[scheduler.py:563](../vllm/v1/core/sched/scheduler.py#L563)） | ❌ |
| 持久 batch / 输入张量 | ❌ | ❌ | ✅ `InputBatch`（[gpu_input_batch.py:92](../vllm/v1/worker/gpu_input_batch.py#L92)） |
| 采样 | ❌ | ❌ | ✅ `GPUModelRunner._sample`（[gpu_model_runner.py:3745](../vllm/v1/worker/gpu_model_runner.py#L3745)） |
| detokenize / stop string | ✅ `BaseIncrementalDetokenizer.update`（[detokenizer.py:96](../vllm/v1/engine/detokenizer.py#L96)） | ❌ | ❌ |
| n>1 并行采样拆分 | ✅ `ParentRequest` + `AsyncLLM.add_request`（[async_llm.py:478](../vllm/v1/engine/async_llm.py#L478)） | ❌（只看到 n 个独立 req） | ❌ |
| 统计 / 日志 / tracing | ✅ `StatLoggerManager` + `OutputProcessor` | 只产出 `SchedulerStats` | ❌ |

一句话概括：**EngineCore 的对外接口只有 `add_request` / `abort_requests` / `step` 三类语义，其余一切都是前端的事**。这也是为什么 `EngineCoreRequest`（[vllm/v1/engine/__init__.py:107](../vllm/v1/engine/__init__.py#L107)）里装的是已经 tokenize 好的 `prompt_token_ids`，而不是原始 prompt 字符串。

---

## 3. 代码走读

### 3.1 前端入口：`EngineClient` 协议与三种实现

`EngineClient` 是抽象前端（[vllm/engine/protocol.py:41](../vllm/engine/protocol.py#L41)），定义了 `generate`（[protocol.py:83](../vllm/engine/protocol.py#L83)）/ `encode`（[protocol.py:106](../vllm/engine/protocol.py#L106)）/ `abort` / `pause_generation` 等。三个实现类：

| 前端 | 场景 | EngineCoreClient |
| --- | --- | --- |
| `AsyncLLM`（[async_llm.py:77](../vllm/v1/engine/async_llm.py#L77)） | 在线服务（OpenAI server） | `AsyncMPClient` / `DPAsyncMPClient` / `DPLBAsyncMPClient` |
| `LLMEngine`（[vllm/v1/engine/llm_engine.py:48](../vllm/v1/engine/llm_engine.py#L48)） | 离线批处理（`vllm.LLM`） | `SyncMPClient`（默认）或 `InprocClient` |
| `InprocClient`（[core_client.py:333](../vllm/v1/engine/core_client.py#L333)） | 单进程调试 / 单测 | 直接持有 `EngineCore` 对象 |

`AsyncLLM.__init__` 里三行就把前端搭好了：

```python
145:        # Convert EngineInput --> EngineCoreRequest.
146:        self.input_processor = InputProcessor(self.vllm_config, renderer)
147:
148:        # Converts EngineCoreOutputs --> RequestOutput.
149:        self.output_processor = OutputProcessor(
150:            renderer.tokenizer,
151:            log_stats=self.log_stats,
152:            stream_interval=self.vllm_config.scheduler_config.stream_interval,
153:            tracing_enabled=tracing_endpoint is not None,
154:        )
155:
156:        # EngineCore (starts the engine in background process).
```

注意 `stream_interval` 是从 `scheduler_config` 读的 —— 这是前端的**输出节流**参数，EngineCore 完全不知道它的存在。

再看 `LLMEngine.__init__`（[vllm/v1/engine/llm_engine.py:109](../vllm/v1/engine/llm_engine.py#L109)）选择 client 的方式：

```python
109:        self.engine_core = EngineCoreClient.make_client(
110:            multiprocess_mode=multiprocess_mode,
111:            asyncio_mode=False,
112:            vllm_config=vllm_config,
113:            executor_class=executor_class,
114:            log_stats=self.log_stats,
115:            renderer=renderer,
116:        )
```

### 3.2 `EngineCoreClient`：工厂与四种实现

工厂在 [vllm/v1/engine/core_client.py:91](../vllm/v1/engine/core_client.py#L91)。它按 `(multiprocess_mode, asyncio_mode)` 二分：

```python
107:        if asyncio_mode and not multiprocess_mode:
108:            raise NotImplementedError(
109:                "Running EngineCore in asyncio without multiprocessing "
110:                "is not currently supported."
111:            )
112:
113:        if multiprocess_mode and asyncio_mode:
114:            return EngineCoreClient.make_async_mp_client(...)
115:
121:        if multiprocess_mode and not asyncio_mode:
122:            return SyncMPClient(...)
129:        return InprocClient(vllm_config, executor_class, log_stats)
```

有意思的是 `asyncio_mode and not multiprocess_mode` 被显式禁止：**EngineCore 的 busy loop 是阻塞式死循环，塞不进 asyncio 事件循环**。

DP 场景下再分两支（[core_client.py:151](../vllm/v1/engine/core_client.py#L151)）：

```python
151:        if parallel_config.data_parallel_size > 1:
152:            if parallel_config.data_parallel_external_lb:
153:                # External load balancer - client per DP rank.
154:                return DPAsyncMPClient(...)
159:            # Internal load balancer - client balances to all DP ranks.
160:            return DPLBAsyncMPClient(...)
163:        return AsyncMPClient(...)
```

#### (a) `InprocClient` —— 契约最小化验证

```python
354:        self.engine_core = EngineCore(
355:            vllm_config,
356:            executor_class,
357:            log_stats,
358:            executor_fail_callback=executor_fail_callback,
359:        )
360:
361:    def get_output(self) -> EngineCoreOutputs:
362:        outputs, model_executed = self.engine_core.step_fn()
363:        self.engine_core.post_step(model_executed=model_executed)
364:        return outputs and outputs.get(0) or EngineCoreOutputs()
```

这里能看到 V1 的一个巧妙设计：**同一个 `EngineCore` 类既能被 busy loop 驱动（多进程），也能被外部同步驱动（in-proc）**。`step_fn` 是构造时绑定的（[core.py:235](../vllm/v1/engine/core.py#L235)）：

```python
235:        self.step_fn = (
236:            self.step if self.batch_queue is None else self.step_with_batch_queue
237:        )
```

#### (b) `MPClient` —— ZMQ 拓扑的建立

`MPClient.__init__`（[core_client.py:558](../vllm/v1/engine/core_client.py#L558)）做四件事：

```python
571:        sync_ctx = zmq.Context(io_threads=2)
572:        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx
```

1）建 ZMQ context（`io_threads=2`，asyncio 模式下包一层 `zmq.asyncio.Context`）。

2）建 socket。前端侧是 **ROUTER（bind）** 收 engine 的握手 + **PULL（bind/connect）** 收输出；EngineCore 侧是 **DEALER（connect）** 发输入 + **PUSH（connect）** 发输出（[core.py:1702](../vllm/v1/engine/core.py#L1702) 与 [core.py:1809](../vllm/v1/engine/core.py#L1809)）：

```python
636:                self.input_socket = self.resources.input_socket = make_zmq_socket(
637:                    self.ctx,
638:                    addresses.inputs[0],
639:                    zmq.ROUTER,
640:                    bind=True,
641:                    router_handover=enable_input_socket_handover,
642:                )
643:                self.resources.output_socket = make_zmq_socket(
644:                    self.ctx, addresses.outputs[0], zmq.PULL
645:                )
```

为什么 input 用 ROUTER/DEALER 而 output 用 PUSH/PULL？因为**一个前端可能要跟多个 EngineCore 通信（DP），需要按 identity 寻址**；而输出方向上，虽然每个 engine 有自己的 PUSH socket，但每个前端 PULL socket 只收属于自己的那路（由 `EngineCoreOutputs.engine_index` + 每个 client 独立的 output 地址区分），不需要回执。

3）地址分配。`get_engine_zmq_addresses`（[utils.py:1039](../vllm/v1/engine/utils.py#L1039)）在「前端与 engine 同机」时用 IPC path，跨节点时用 `tcp://host:0` 占位，bind 后再通过 `getsockopt(zmq.LAST_ENDPOINT)` 拿真实端口回填：

```python
649:                addresses.inputs[0] = self.input_socket.getsockopt(
650:                    zmq.LAST_ENDPOINT
651:                ).decode()
```

4）启动 engine 子进程并等握手：

```python
656:                with launch_core_engines(
657:                    vllm_config, executor_class, log_stats, addresses
658:                ) as engine_launch:
...
709:            # Wait for ready messages from each engine on the input socket.
710:            identities = set(self.core_engines)
711:            sync_input_socket = zmq.Socket.shadow(self.input_socket)
712:            while identities:
713:                if not sync_input_socket.poll(
714:                    timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000
715:                ):
716:                    raise TimeoutError(...)
725:                identity, payload = sync_input_socket.recv_multipart()
726:                identities.remove(identity)
727:                self._apply_ready_response(payload)
```

每个 engine 的 ZMQ identity 就是它的 DP rank 编码成 2 字节小端（[core_client.py:705](../vllm/v1/engine/core_client.py#L705)、[core.py:1045](../vllm/v1/engine/core.py#L1045)）。握手回包 `EngineCoreReadyResponse`（[vllm/v1/engine/__init__.py:72](../vllm/v1/engine/__init__.py#L72)）会把**引擎侧最终生效的配置**（KV cache 自动适配后的 `max_model_len`、`num_gpu_blocks`、`block_size`、dtype、`kv_cache_size_tokens` 等）回传给前端 —— 这是「前端与核心解耦但配置要对齐」的关键机制。

#### (c) `SyncMPClient` —— 给离线 `LLM` 用

它在后台起一个线程专门收输出 socket，塞进 `queue.Queue`（[core_client.py:903](../vllm/v1/engine/core_client.py#L903)）：

```python
919:                    frames = out_socket.recv_multipart(copy=False)
920:                    resources.validate_alive(frames)
921:                    outputs: EngineCoreOutputs = decoder.decode(frames)
922:                    if outputs.utility_output:
923:                        _process_utility_output(outputs.utility_output, utility_results)
924:                    else:
925:                        outputs_queue.put_nowait(outputs)
```

发送侧就是一个同步的 multipart（[core_client.py:956](../vllm/v1/engine/core_client.py#L956)）：

```python
959:        msg = (self.core_engine, request_type.value, *self.encoder.encode(request))
963:        self.input_socket.send_multipart(msg, copy=False)
```

注意第一帧是目标 engine 的 identity，第二帧是 `EngineCoreRequestType`（直接用字节枚举值，省掉一次编码，见 [vllm/v1/engine/__init__.py:284](../vllm/v1/engine/__init__.py#L284)）。

#### (d) `AsyncMPClient` —— 给 `AsyncLLM` 用

输出用 `asyncio.Queue`，收包逻辑跑在 `asyncio.Task` 里（[core_client.py:1111](../vllm/v1/engine/core_client.py#L1111)）：

```python
1114:                    frames = await output_socket.recv_multipart(copy=False)
1115:                    resources.validate_alive(frames)
1116:                    outputs: EngineCoreOutputs = decoder.decode(frames)
...
1156:                    if outputs.outputs or outputs.scheduler_stats:
1157:                        outputs_queue.put_nowait(outputs)
```

`add_request_async`（[core_client.py:1219](../vllm/v1/engine/core_client.py#L1219)）会先把 `client_index` 打进请求，这样多 API server 场景下每个 engine 知道把输出发回哪个前端：

```python
1219:    async def add_request_async(self, request: EngineCoreRequest) -> None:
1220:        request.client_index = self.client_index
1221:        await self._send_input(EngineCoreRequestType.ADD, request)
1222:        self._ensure_output_queue_task()
```

DP 内建负载均衡版本 `DPLBAsyncMPClient`（[core_client.py:1507](../vllm/v1/engine/core_client.py#L1507)）维护 `lb_engines: list[[waiting, running, kv_cache_usage]]`（[core_client.py:1351](../vllm/v1/engine/core_client.py#L1351)），靠 `DPAsyncMPClient._ensure_stats_update_task`（[core_client.py:1369](../vllm/v1/engine/core_client.py#L1369)）订阅 `DPCoordinator` 的 stats 广播来更新，然后覆写 `get_core_engine_for_request`（[core_client.py:1503](../vllm/v1/engine/core_client.py#L1503)）挑引擎。

### 3.3 序列化：msgpack + 零拷贝张量

所有跨进程 payload 都是 `msgspec.Struct` + `array_like=True` + `omit_defaults=True`（例如 [vllm/v1/engine/__init__.py:107](../vllm/v1/engine/__init__.py#L107)、[196](../vllm/v1/engine/__init__.py#L196)）。`array_like=True` 意味着**编码成数组而不是字典** —— 没有字段名，只有位置，体积最小；代价是字段只能追加不能重排（源码注释里多次强调 "Appended last so `array_like` positional serialization stays backward compatible"，见 [vllm/v1/engine/__init__.py:226](../vllm/v1/engine/__init__.py#L226)）。

`MsgpackEncoder`（[vllm/v1/serial_utils.py:136](../vllm/v1/serial_utils.py#L136)）的核心技巧是**把大张量拆成独立的 multipart frame 做零拷贝**：

```python
166:    def encode(self, obj: Any) -> Sequence[bytestr]:
167:        try:
168:            if self.oob_tensor_consumer is not None:
169:                self.oob_tensor_consumer.new_message()
170:            self.aux_buffers = bufs = [b""]
171:            bufs[0] = self.encoder.encode(obj)
...
176:            return bufs
```

`enc_hook`（[serial_utils.py:191](../vllm/v1/serial_utils.py#L191)）遇到 `torch.Tensor` / `np.ndarray` 时，小对象走 `CUSTOM_TYPE_RAW_VIEW` 内联，大对象把 `memoryview` 放进 `aux_buffers`，编码结果里只留一个**下标**（[serial_utils.py:257](../vllm/v1/serial_utils.py#L257)）：

```python
257:    def _encode_tensor(
258:        self, obj: torch.Tensor
259:    ) -> tuple[str, tuple[int, ...], int | dict | memoryview]:
260:        oob_consumer = self.oob_tensor_consumer
261:        # view the tensor as a contiguous 1D array of bytes
262:        if obj.nbytes < self.size_threshold and obj.is_cpu:
263:            # Smaller tensors are encoded inline, just like ndarrays.
264:            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, tensor_data(obj))
265:        elif oob_consumer is not None and (data := oob_consumer(obj)) is not None:
266:            assert isinstance(data, dict)
267:        else:
268:            # Otherwise encode index of backing buffer to avoid copy.
269:            assert self.aux_buffers is not None
270:            data = len(self.aux_buffers)
271:            self.aux_buffers.append(tensor_data(obj))
272:        dtype = str(obj.dtype).removeprefix("torch.")
273:        return dtype, obj.shape, data
```

阈值来自 `VLLM_MSGPACK_ZERO_COPY_THRESHOLD`（[serial_utils.py:155](../vllm/v1/serial_utils.py#L155)）。解码侧 `_decode_tensor`（[serial_utils.py:399](../vllm/v1/serial_utils.py#L399)）用 `torch.frombuffer` 直接建视图，并对 aux buffer 做 `pin_memory()` 以便后续异步 H2D：

```python
416:        arr = torch.frombuffer(buffer, dtype=torch.uint8)
...
420:        if not is_aux:
421:            arr = arr.clone()
422:        elif not self.share_mem:
423:            arr = arr.pin_memory() if self.pin_tensors else arr.clone()
424:        # Convert back to proper shape & type
425:        return arr.view(torch_dtype).view(shape)
```

多模态张量还能走 **out-of-band tensor IPC**（`OOBTensorConsumer` / `OOBTensorProvider`，[serial_utils.py:57](../vllm/v1/serial_utils.py#L57) 与 [75](../vllm/v1/serial_utils.py#L75)），用共享内存队列传大 tensor，避免 msgpack 路径（见 [core_client.py:680](../vllm/v1/engine/core_client.py#L680) 的 `TensorIpcSender` 装配）。

`run_method`（[serial_utils.py:486](../vllm/v1/serial_utils.py#L486)）是 `collective_rpc` 的通用执行器：方法名可以是 str（getattr）、bytes（cloudpickle 反序列化）或 callable。

### 3.4 EngineCore 主循环：一个 busy loop + 两个 IO 线程

进程入口是 `run_engine_core`（[core.py:1283](../vllm/v1/engine/core.py#L1283)），设好进程名、信号处理、DP rank 之后调 `run_busy_loop()`：

```python
1298:                process_title = f"EngineCore_DP{dp_rank}"
1299:            else:
1300:                process_title = "EngineCore"
1301:            set_process_title(process_title)
...
1352:            engine_core.run_busy_loop()
```

busy loop 本体只有 10 行（[core.py:1403](../vllm/v1/engine/core.py#L1403)）：

```python
1403:    def run_busy_loop(self):
1404:        """Core busy loop of the EngineCore."""
1405:        while self._handle_shutdown():
1406:            # 1) Poll the input queue until there is work to do.
1407:            self._process_input_queue()
1408:            # Publish request counts before and after GPU step to ensure freshness.
1409:            self._maybe_publish_request_counts()
1410:            # 2) Step the engine core and return the outputs.
1411:            self._process_engine_step()
1412:            self._maybe_publish_request_counts()
1413:
1414:        raise SystemExit
```

`_process_input_queue`（[core.py:1429](../vllm/v1/engine/core.py#L1429)）在没有工作时**阻塞在 `input_queue.get()`**，有工作时把队列里所有请求一次性抽干：

```python
1433:        while not self.has_work() and self.is_running():
...
1445:                req = self.input_queue.get(block=block)
1446:                self._handle_client_request(*req)
1447:            except queue.Empty:
1448:                break
...
1455:        # Handle any more client requests.
1456:        while not self.input_queue.empty():
1457:            req = self.input_queue.get_nowait()
1458:            self._handle_client_request(*req)
```

`_process_engine_step`（[core.py:1460](../vllm/v1/engine/core.py#L1460)）把 `step_fn()` 产出的 `EngineCoreOutputs` 按 client 分桶塞进 output_queue：

```python
1464:        outputs, model_executed = self.step_fn()
1465:        # Put EngineCoreOutputs into the output queue.
1466:        for output in outputs.items() if outputs else ():
1467:            self.output_queue.put_nowait(output)
1468:        # Post-step hook.
1469:        self.post_step(model_executed)
```

注意 `step()` 返回的是 `dict[int, EngineCoreOutputs]` —— **key 是 `client_index`**，这就是多前端复用同一个 engine 时输出不会串台的原因。

请求预处理被**故意放在 input 线程**里做（[core.py:1756](../vllm/v1/engine/core.py#L1756)）：

```python
1756:                    if request_type == EngineCoreRequestType.ADD:
1757:                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
1758:                        try:
1759:                            request = self.preprocess_add_request(req)
```

`preprocess_add_request`（[core.py:980](../vllm/v1/engine/core.py#L980)）的 docstring 说得很清楚：

> "This function could be directly used in input processing thread to allow request initialization running in parallel with Model forward"

即：`Request` 对象构造（含 block hash 计算、结构化输出 grammar 初始化）也要**和 GPU forward 重叠**。

abort 走双队列（[core.py:1780](../vllm/v1/engine/core.py#L1780)）：既进 `aborts_queue`（让正在跑的 step 结束后立刻处理），也进 `input_queue`（保证顺序、不漏）。源码注释：「aborting in the scheduler is idempotent」。

### 3.5 `EngineCore.step()`：调度 + 执行的核心

```python
589:    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
596:        # Check for any requests remaining in the scheduler - unfinished,
597:        # or finished and not yet removed from the batch.
598:        if not self.scheduler.has_requests():
599:            return {}, False
600:        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
601:        future = self.model_executor.execute_model(scheduler_output, non_block=True)
602:        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
603:        with (
604:            self.capture_iteration_details(scheduler_output) as iteration_details,
605:            self.log_error_detail(scheduler_output),
606:        ):
607:            model_output = future.result()
608:            if model_output is None:
609:                model_output = self.model_executor.sample_tokens(grammar_output)
610:
611:        # Before processing the model output, process any aborts that happened
612:        # during the model execution.
613:        self._process_aborts_queue()
614:        engine_core_outputs = self.scheduler.update_from_output(
615:            scheduler_output, model_output
616:        )
617:        self._attach_iteration_details(engine_core_outputs, iteration_details)
618:
619:        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

这段是全系统的心脏，几个细节值得注意：

- **`non_block=True` + Future**：`execute_model` 立即返回，EngineCore 利用这段 CPU 时间去算 grammar bitmask（结构化输出用），然后才 `future.result()`。
- **`sample_tokens` 是独立的第二次 RPC**（[core.py:609](../vllm/v1/engine/core.py#L609)）。为什么？因为 grammar bitmask 依赖「上一步采样出的 token」，而 async scheduling 下这一步的 logits 还没出来，只能把采样拆成两步：`execute_model` 算出 logits → `sample_tokens(grammar_output)` 应用 bitmask 并采样。这也解释了 `WorkerBase.execute_model` 的返回值语义（[worker_base.py:166](../vllm/v1/worker/worker_base.py#L166)）：返回 `None` 表示「我还没采样，请立刻调 `sample_tokens`」。
- **abort 在 `update_from_output` 之前处理**（[core.py:613](../vllm/v1/engine/core.py#L613)），避免把已经 abort 的请求的输出又算一遍。

`step_with_batch_queue`（[core.py:630](../vllm/v1/engine/core.py#L630)）是 PP 场景消除流水线气泡的版本：维护一个 `deque`，优先把新 batch 排进去（non-blocking），队列满了才阻塞等最老的那个 batch 返回。源码注释里明确写了优先级：「fulfilling the batch queue has a higher priority than getting model outputs」。

### 3.6 Executor 抽象

`Executor`（[vllm/v1/executor/abstract.py:38](../vllm/v1/executor/abstract.py#L38)）是「一台模型副本」的门面。它只有两个真正抽象的方法：`_init_executor` 和 `collective_rpc`（[abstract.py:209](../vllm/v1/executor/abstract.py#L209)），其余（`determine_available_memory`、`get_kv_cache_specs`、`execute_model`、`sample_tokens`、`sleep/wake_up`）全部是 `collective_rpc` 的语法糖：

```python
231:    def execute_model(
232:        self, scheduler_output: SchedulerOutput, non_block: bool = False
233:    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
234:        output = self.collective_rpc(  # type: ignore[call-overload]
235:            "execute_model", args=(scheduler_output,), non_block=non_block
236:        )
237:        return output[0]
```

选择逻辑在 `Executor.get_class`（[abstract.py:48](../vllm/v1/executor/abstract.py#L48)）：

| `distributed_executor_backend` | 实现类 | 文件 |
| --- | --- | --- |
| `"uni"` | `UniProcExecutor` | [uniproc_executor.py:51](../vllm/v1/executor/uniproc_executor.py#L51) |
| `"mp"`（默认） | `MultiprocExecutor` | [multiproc_executor.py:111](../vllm/v1/executor/multiproc_executor.py#L111) |
| `"ray"` | `RayDistributedExecutor` / `RayExecutorV2` | [ray_executor.py:67](../vllm/v1/executor/ray_executor.py#L67) |
| `"external_launcher"` | `ExecutorWithExternalLauncher` | [uniproc_executor.py](../vllm/v1/executor/uniproc_executor.py) |

`UniProcExecutor` 是最简单的参照实现（[uniproc_executor.py:52](../vllm/v1/executor/uniproc_executor.py#L52)）：直接在本进程内 `WorkerWrapperBase(rpc_rank=0)` → `init_worker` → `init_device` → `load_model`，`collective_rpc` 就是本地方法调用（[uniproc_executor.py:90](../vllm/v1/executor/uniproc_executor.py#L90)），非阻塞模式用 `AsyncOutputFuture` 包一层（[uniproc_executor.py:32](../vllm/v1/executor/uniproc_executor.py#L32)）。

`MultiprocExecutor`（[multiproc_executor.py:111](../vllm/v1/executor/multiproc_executor.py#L111)）是生产默认。它为每个 local rank fork 一个 `WorkerProc`（[multiproc_executor.py:195](../vllm/v1/executor/multiproc_executor.py#L195)），用**共享内存 MessageQueue 广播 RPC、per-rank MessageQueue 回收结果**：

```python
420:        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))
...
422:        response_mqs: Sequence[MessageQueue] = self.response_mqs
423:        if output_rank is not None:
424:            response_mqs = (response_mqs[output_rank],)
...
444:        future = FutureWrapper(
445:            self.futures_queue, get_response=get_response, aggregate=aggregate
446:        )
447:
448:        return future if non_block else future.result()
```

`output_rank` 优化非常重要：`execute_model` 只从最后一个 PP rank 取结果（[multiproc_executor.py:340](../vllm/v1/executor/multiproc_executor.py#L340)），其余 rank 的结果直接丢弃，省掉 N-1 次反序列化。

worker 侧的主循环就一行（[multiproc_executor.py:1029](../vllm/v1/executor/multiproc_executor.py#L1029)）：

```python
1029:    def worker_busy_loop(self):
1030:        """Main busy loop for Multiprocessing Workers"""
1031:        assert self.rpc_broadcast_mq is not None
1032:        while True:
1033:            self._execute_worker_rpc(self.rpc_broadcast_mq.dequeue(indefinite=True))
```

一个值得记住的工程细节：**`MultiprocExecutor` 在 fork 完 worker 之后会主动降低自己的 torch 线程数**（[multiproc_executor.py:220](../vllm/v1/executor/multiproc_executor.py#L220)），因为它「只调度，不计算」，多余的 intra-op 并行只会和 worker 抢 CPU：

> "this process only schedules, so it gets no benefit from torch intra-op parallelism, just CPU contention with them."

`RayDistributedExecutor`（[ray_executor.py:67](../vllm/v1/executor/ray_executor.py#L67)）用 Ray actor 承载 worker，并用 **Ray Compiled DAG**（`_execute_dag`，[ray_executor.py:451](../vllm/v1/executor/ray_executor.py#L451)）把一步 forward 的 NCCL 通信固化成静态图以消除调度开销。

### 3.7 Worker 抽象

`WorkerBase`（[vllm/v1/worker/worker_base.py:44](../vllm/v1/worker/worker_base.py#L44)）是「硬件无关」的 worker 接口：`init_device` / `load_model` / `get_kv_cache_spec` / `execute_model` / `sample_tokens` / `compile_or_warm_up_model` 全部 `NotImplementedError`，由 `gpu_worker.Worker`（[gpu_worker.py:179](../vllm/v1/worker/gpu_worker.py#L179)）、`cpu_worker`、`xpu_worker`、`tpu_worker` 实现。

`WorkerWrapperBase`（[worker_base.py:211](../vllm/v1/worker/worker_base.py#L211)）是 worker 的**生命周期包装器**：它先只记住 worker 的类名（延迟 import，避免 CUDA 相关的 import 副作用），真正的构造在子进程的 `init_worker`（[worker_base.py:254](../vllm/v1/worker/worker_base.py#L254)）里发生；`__getattr__`（[worker_base.py:357](../vllm/v1/worker/worker_base.py#L357)）把未定义属性转发给内部 worker。

GPU worker 的 `execute_model` 极薄（[gpu_worker.py:1122](../vllm/v1/worker/gpu_worker.py#L1122)），只处理 PP 的收发，实际工作委派给 `GPUModelRunner`：

```python
1182:        with self.annotate_profile(scheduler_output):
1183:            output = self.model_runner.execute_model(
1184:                scheduler_output, intermediate_tensors
1185:            )
```

`GPUModelRunner`（[gpu_model_runner.py:503](../vllm/v1/worker/gpu_model_runner.py#L503)）持有三样东西：

- **`InputBatch`**（[gpu_input_batch.py:92](../vllm/v1/worker/gpu_input_batch.py#L92)）：**持久 batch**。它把 `max_num_reqs × max_model_len` 的 token 矩阵、block table、sampling metadata 等预分配好，每步只做增量更新（`add_request` / `remove_request` / `condense`），而不是每步重建。这是 V1 相比 V0 最大的单项性能改进之一。
- **`Sampler`**（[gpu_model_runner.py:598](../vllm/v1/worker/gpu_model_runner.py#L598)）与 `RejectionSampler`（[gpu_model_runner.py:706](../vllm/v1/worker/gpu_model_runner.py#L706)，spec decode 用）。
- **KV caches**：由 `initialize_from_config`（[gpu_worker.py:724](../vllm/v1/worker/gpu_worker.py#L724)）按 EngineCore 下发的 `KVCacheConfig` 分配。

### 3.8 为什么 Detokenizer / OutputProcessor 必须在前端

这是最容易被问到的设计问题。三个理由：

1. **detokenize 需要 tokenizer，而 tokenizer 天然属于「协议层」**。前端要输出 `RequestOutput.text`、`logprobs` 的 token 文本、stop string 截断（[detokenizer.py:310](../vllm/v1/engine/detokenizer.py#L310)）—— 这些都是给人类/HTTP 客户端看的。EngineCore 只关心 token id 和 KV block。把 tokenizer 放进 EngineCore 会让 N 个 DP 进程各持一份 tokenizer、各做一遍 detokenize，纯浪费。
2. **`n > 1` 聚合是前端概念**。`SamplingParams(n=4)` 在 EngineCore 眼里就是 4 个独立请求，只有前端的 `ParentRequest`（[async_llm.py:482](../vllm/v1/engine/async_llm.py#L482)）知道它们属于同一个 API 调用，要合并成 `RequestOutput.outputs[4]`。
3. **CPU 开销隔离**。detokenize 是 O(batch × 新 token) 的 Python 循环，logprobs 处理同理。放在前端进程，就不会占用 EngineCore 的调度时间预算。

`OutputProcessor.process_outputs`（[output_processor.py:622](../vllm/v1/engine/output_processor.py#L622)）的 docstring 里有一条对开发者的重要约束：

> "vLLM V1 minimizes the number of python loops over the full batch to ensure system overheads are minimized. This is the only function that should loop over EngineCoreOutputs."

也就是说，**全系统只允许这一处对整批输出做 Python 循环**。这也是为什么 logprobs 计算、tracing、统计都挂在这个循环里（[output_processor.py:660](../vllm/v1/engine/output_processor.py#L660)、[705](../vllm/v1/engine/output_processor.py#L705)、[745](../vllm/v1/engine/output_processor.py#L745)）。

### 3.9 `input_processor` 为什么在前端

`InputProcessor`（[input_processor.py:38](../vllm/v1/engine/input_processor.py#L38)）做三件事：参数校验、prompt → `EngineCoreRequest` 转换、request_id 内部化。它在前端的理由：

- **tokenize 是重 CPU 活**。`process_inputs` 里会调 `renderer.render_cmpl`（[input_processor.py:332](../vllm/v1/engine/input_processor.py#L332)）—— 这一步包含 chat template 渲染 + tokenizer 调用 + 多模态 processor（图像解码/resize）；多模态场景下单个请求可能耗时几十毫秒。绝不能进 EngineCore。
- **必须不阻塞 event loop**。所以类构造时就把它包成异步版本扔到线程池（[input_processor.py:73](../vllm/v1/engine/input_processor.py#L73)）：

```python
70:        # Raw-prompt preprocessing (tokenization and multimodal processing)
71:        # is blocking, so async callers should run it on the renderer's
72:        # thread pool to keep their event loop responsive.
73:        self.process_inputs_async = make_async(
74:            self.process_inputs, executor=self.renderer._executor
75:        )
```

`AsyncLLM.add_request` 据此分流（[async_llm.py:423](../vllm/v1/engine/async_llm.py#L423)）：已经是渲染好的 `EngineInput`（dict 且含 `"type"`）就同步处理；原始 prompt 走 `process_inputs_async`。

- **参数默认值填充也在这里**（[input_processor.py:356](../vllm/v1/engine/input_processor.py#L356)）：

```python
356:            sampling_params = params.clone()
357:            # If unset max tokens, then generate up to the max_model_len.
358:            if sampling_params.max_tokens is None:
359:                seq_len = length_from_prompt_token_ids_or_embeds(
360:                    prompt_token_ids, prompt_embeds
361:                )
362:                sampling_params.max_tokens = self.model_config.max_model_len - seq_len
```

EngineCore 收到的 `sampling_params.max_tokens` 永远是非空的 —— 契约的一部分。

> 注：任务描述里提到的 `input_preprocessor` 在本仓库已不存在（搜索 `vllm/v1/**/input_preprocessor*.py` 无结果）。V1 的对应实现是 `InputProcessor`（[vllm/v1/engine/input_processor.py:38](../vllm/v1/engine/input_processor.py#L38)），配合 `vllm/renderers/` 下的 Renderer 体系完成「协议输入 → EngineInput」的渲染。

---

## 4. 关键数据结构

### 4.1 跨进程传输的三件套

| 结构 | 定义 | 关键字段 | 说明 |
| --- | --- | --- | --- |
| `EngineCoreRequest` | [vllm/v1/engine/__init__.py:107](../vllm/v1/engine/__init__.py#L107) | `request_id: str`(113)、`prompt_token_ids: list[int] \| None`(114)、`mm_features`(115)、`sampling_params`(116)、`pooling_params`(117)、`arrival_time`(118)、`prompt_embeds`(122)、`prompt_is_token_ids`(128)、`client_index: int`(132)、`current_wave: int`(137)、`priority: int`(138)、`external_req_id`(147)、`abort_immediately`(156) | 前端 → EngineCore。已经是 token id，不含原始文本 |
| `EngineCoreOutput` | [vllm/v1/engine/__init__.py:196](../vllm/v1/engine/__init__.py#L196) | `request_id`(202)、`new_token_ids: list[int]`(203)、`new_logprobs`(205)、`finish_reason`(210)、`stop_reason`(211)、`events`(212)、`prefill_stats`(218)、`mm_cache_miss_hashes`(228) | EngineCore → 前端，**只含增量 token** |
| `EngineCoreOutputs` | [vllm/v1/engine/__init__.py:253](../vllm/v1/engine/__init__.py#L253) | `engine_index`(262)、`outputs: list[EngineCoreOutput]`(265)、`scheduler_stats`(266)、`timestamp`(267)、`utility_output`(269)、`finished_requests: set[str]`(270)、`wave_complete`(274) | 一次 step 的一批输出；`__post_init__` 自动打时间戳(279) |

### 4.2 内部状态

| 结构 | 定义 | 关键字段 / 语义 |
| --- | --- | --- |
| `Request` | [vllm/v1/request.py:59](../vllm/v1/request.py#L59) | `status`(98，初始 `WAITING`)、`num_computed_tokens`(182)、`block_hashes`(219)、`num_output_placeholders`(160，async scheduling 用)、`num_preemptions`(210)、`__lt__`(350，priority 调度排序)。**只存在于 EngineCore** |
| `RequestStatus` | [vllm/v1/request.py:364](../vllm/v1/request.py#L364) | `WAITING`(367) / `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`(368) / `WAITING_FOR_REMOTE_KVS`(369) / `WAITING_FOR_STREAMING_REQ`(370) / `RUNNING`(371) / `PREEMPTED`(372) / `FINISHED_*`(375-380)。`is_finished` 的判据是 `status > PREEMPTED`(387) |
| `CachedRequestState` | [vllm/v1/worker/gpu_input_batch.py:35](../vllm/v1/worker/gpu_input_batch.py#L35) | `block_ids: tuple[list[int], ...]`(42)、`num_computed_tokens`(43)、`output_token_ids`(44)、`mrope_positions`(46)、`in_progress_prompt_logprobs_cpu`(54)。**只存在于 Worker**，是 `Request` 在 GPU 侧的镜像 |
| `InputBatch` | [vllm/v1/worker/gpu_input_batch.py:92](../vllm/v1/worker/gpu_input_batch.py#L92) | 持久 batch：`token_ids_cpu_tensor`(134)、`req_id_to_index`(128)。`add_request`(350) / `remove_request`(528) / `condense`(706) 做增量维护 |
| `ModelRunnerOutput` | [vllm/v1/outputs.py:319](../vllm/v1/outputs.py#L319) | `req_ids`(322)、`req_id_to_index`(324)、`sampled_token_ids: list[list[int]]`(330)、`logprobs`(335)、`prompt_logprobs_dict`(341)、`pooler_output`(346)、`kv_connector_output`(348)。注释特别强调 "prefer to use list instead" of tensor(317) |
| `RequestState` | [vllm/v1/engine/output_processor.py:132](../vllm/v1/engine/output_processor.py#L132) | 前端侧的 `Request` 镜像：`detokenizer`(170)、`logprobs_processor`(169)、`is_prefilling`(175)、`sent_tokens_offset`(191)、`queue`(176) |
| `EngineCoreRequestType` | [vllm/v1/engine/__init__.py:284](../vllm/v1/engine/__init__.py#L284) | `ADD=b"\x00"`(290) / `ABORT=b"\x01"`(291) / `START_DP_WAVE=b"\x02"`(292) / `UTILITY=b"\x03"`(293) / `EXECUTOR_FAILED=b"\x04"`(295) / `WAKEUP=b"\x05"`(297) |
| `EngineZmqAddresses` | [vllm/v1/engine/utils.py:86](../vllm/v1/engine/utils.py#L86) | `inputs` / `outputs` / `coordinator_input` / `coordinator_output` / `frontend_stats_publish_address` |

---

## 5. 收益与代价

### 5.1 收益

- **CPU 开销隔离**：前端的 tokenize / detokenize / HTTP 处理彻底不影响 EngineCore 的调度节奏。
- **真正的 CPU/GPU 重叠**：`non_block=True` 的 `execute_model` + `batch_queue` + IO 线程三重重叠。
- **进程级横向扩展**：DP（多 EngineCore）、多 API server 前端（多 client 连同一组 engine）都变成配置问题。
- **故障隔离**：`ENGINE_CORE_DEAD`（[core.py:1022](../vllm/v1/engine/core.py#L1022)）+ `CoreEngineProcManager.monitor_engine_liveness`（[utils.py:256](../vllm/v1/engine/utils.py#L256)）+ `start_engine_core_monitor`（[core_client.py:774](../vllm/v1/engine/core_client.py#L774)）构成完整的死亡检测链。
- **可测试性**：`InprocClient` 让整条链路（除 socket）可以在单进程内跑，单测不需要起子进程。

### 5.2 代价与限制

- **序列化开销**。每步都要 encode/decode 一次 `EngineCoreOutputs`。缓解手段：`array_like=True` 紧凑布局、`omit_defaults=True`、零拷贝 tensor frame、buffer 复用（[core.py:1797](../vllm/v1/engine/core.py#L1797) 的 `reuse_buffers`，上限 `max_reuse_bufs = len(sockets) + 1`）。
- **`execute_model` 输入 `SchedulerOutput` 每步都要序列化**。这是 V1 里「调度 → 执行」方向的主要开销，也是社区里持续在做优化的点（比如 persistent batch 让 `SchedulerOutput` 只传增量）。
- **`asyncio_mode and not multiprocess_mode` 不支持**（[core_client.py:107](../vllm/v1/engine/core_client.py#L107)），即「单进程 + asyncio」组合被禁止。
- **调试变难**：跨进程栈、msgpack 反序列化错误、ZMQ 握手超时（`VLLM_ENGINE_READY_TIMEOUT_S`，[core_client.py:714](../vllm/v1/engine/core_client.py#L714)）都是新的故障模式。
- **多一次内存拷贝的语义风险**：`_decode_tensor` 的 `share_mem=True` 意味着解码出的 ndarray **锁住整个收到的 message buffer**（[serial_utils.py:391](../vllm/v1/serial_utils.py#L391) 注释："We assume the ndarray will not be kept around"）。误持有会导致 buffer 池无法复用。
- **`array_like=True` 的兼容性约束**：字段只能往后追加，不能插入/重排，否则新旧版本 engine/frontend 混跑会静默错位。

---

## 6. 面试高频问题

**Q1：V1 为什么要拆两个进程？拆的边界画在哪里？**
边界画在「协议/文本处理」与「调度/执行」之间。前端（`AsyncLLM` / `LLMEngine`）持有 tokenizer、renderer、detokenizer、OutputProcessor；EngineCore（[core.py:105](../vllm/v1/engine/core.py#L105)）只持有 Scheduler + Executor。核心动机是 GIL 隔离：V0 里 detokenize 这类 Python 重活会推迟下一步调度，导致 GPU 空转。前端输入是 `EngineCoreRequest`（含 token id），输出是 `EngineCoreOutput`（含增量 token id），中间用 ZMQ + msgpack。

**Q2：`EngineCoreClient` 有哪几种实现？分别在什么场景用？**
`InprocClient`（[core_client.py:333](../vllm/v1/engine/core_client.py#L333)，单进程/离线调试，直接持有 `EngineCore`）、`SyncMPClient`（[core_client.py:869](../vllm/v1/engine/core_client.py#L869)，离线 `LLM`，后台线程收包）、`AsyncMPClient`（[core_client.py:1046](../vllm/v1/engine/core_client.py#L1046)，在线 `AsyncLLM`，`asyncio.Task` 收包）、`DPAsyncMPClient`（[core_client.py:1323](../vllm/v1/engine/core_client.py#L1323)，DP + 外部 LB）、`DPLBAsyncMPClient`（[core_client.py:1507](../vllm/v1/engine/core_client.py#L1507)，DP + 内部 LB，按 `lb_engines` 挑引擎）。工厂在 [core_client.py:91](../vllm/v1/engine/core_client.py#L91)，DP 分支在 [core_client.py:151](../vllm/v1/engine/core_client.py#L151)。

**Q3：ZMQ 用了哪些 socket 模式？为什么 input 和 output 不一样？**
前端 input 侧 `ROUTER`（bind），EngineCore input 侧 `DEALER`（connect，带 2 字节 identity）；前端 output 侧 `PULL`，EngineCore output 侧 `PUSH`（linger=4000 保证 `ENGINE_CORE_DEAD` 能发出去，见 [core.py:1809](../vllm/v1/engine/core.py#L1809)）。input 用 ROUTER/DEALER 是因为一个前端要按 identity 寻址多个 DP engine；output 用 PUSH/PULL 是因为输出是单向流、不需要寻址（每个 client 有自己的 output 地址，`EngineCoreOutputs.engine_index` 用于标识来源）。

**Q4：`run_busy_loop` 里到底做了什么？为什么需要两个额外线程？**
[core.py:1403](../vllm/v1/engine/core.py#L1403)：`while self._handle_shutdown(): _process_input_queue() → _maybe_publish_request_counts() → _process_engine_step() → _maybe_publish_request_counts()`。两个线程是 `process_input_sockets`（[core.py:1686](../vllm/v1/engine/core.py#L1686)）和 `process_output_sockets`（[core.py:1789](../vllm/v1/engine/core.py#L1789)），它们把 ZMQ 收发 + msgpack 编解码挪出主线程。因为 socket 的 `poll()`/`recv()` 释放 GIL，所以主线程跑 GPU forward 的同时这两个线程可以继续收新请求 / 发上一批输出。

**Q5：为什么 `add_request` 的预处理（`preprocess_add_request`）放在 input 线程而不是主线程？**
[core.py:980](../vllm/v1/engine/core.py#L980) 的 docstring：「to allow request initialization running in parallel with Model forward」。`Request.from_engine_core_request`（[request.py:237](../vllm/v1/request.py#L237)）要算 block hash（前缀缓存用），结构化输出请求还要 `grammar_init`（[core.py:1001](../vllm/v1/engine/core.py#L1001)），这些都是 CPU 活，放在 IO 线程能和 GPU forward 重叠。线程安全性靠「`mm_receiver_cache` 只在 input 线程访问」「`structured_output_manager` 每个请求独立」两条保证（见 [core.py:986](../vllm/v1/engine/core.py#L986) 与 [996](../vllm/v1/engine/core.py#L996) 的注释）。

**Q6：`step()` 里为什么 `execute_model` 之后还要单独调 `sample_tokens`？**
因为结构化输出的 grammar bitmask 依赖**上一步的采样结果**。`step()`（[core.py:589](../vllm/v1/engine/core.py#L589)）先发 `execute_model(non_block=True)` 拿 Future，利用等待时间算 `get_grammar_bitmask`（[core.py:602](../vllm/v1/engine/core.py#L602)），然后 `future.result()`；如果 worker 返回 `None`（表示它没采样，[worker_base.py:166](../vllm/v1/worker/worker_base.py#L166) 的契约），就再发一次 `sample_tokens(grammar_output)`（[core.py:609](../vllm/v1/engine/core.py#L609)）。

**Q7：`Executor` 的三个实现有什么区别？**
`UniProcExecutor`（[uniproc_executor.py:51](../vllm/v1/executor/uniproc_executor.py#L51)）：本进程内一个 `WorkerWrapperBase`，`collective_rpc` 就是本地调用。`MultiprocExecutor`（[multiproc_executor.py:111](../vllm/v1/executor/multiproc_executor.py#L111)，默认）：fork N 个 `WorkerProc`，共享内存 `MessageQueue` 广播 RPC、`FutureWrapper` 异步收结果，支持 PP（`supports_pp=True`，[multiproc_executor.py:112](../vllm/v1/executor/multiproc_executor.py#L112)）和 async scheduling。`RayDistributedExecutor`（[ray_executor.py:67](../vllm/v1/executor/ray_executor.py#L67)）：Ray actor + Compiled DAG，适合多节点。

**Q8：`MultiprocExecutor.collective_rpc` 的 `unique_reply_rank` / `output_rank` 优化是什么？**
只从产生最终输出的那个 rank 拉结果，其余 rank 的结果丢弃（[multiproc_executor.py:423](../vllm/v1/executor/multiproc_executor.py#L423)）。`execute_model` 传 `unique_reply_rank=self.output_rank`（[multiproc_executor.py:346](../vllm/v1/executor/multiproc_executor.py#L346)），`_get_output_rank`（[multiproc_executor.py:541](../vllm/v1/executor/multiproc_executor.py#L541)）算出最后一个 PP rank。省下 world_size-1 次 msgpack 解码。

**Q9：`Request` 和 `CachedRequestState` 是什么关系？**
`Request`（[request.py:59](../vllm/v1/request.py#L59)）活在 EngineCore（Scheduler 里），管 `status` / `num_computed_tokens` / `block_hashes`。`CachedRequestState`（[gpu_input_batch.py:35](../vllm/v1/worker/gpu_input_batch.py#L35)）活在 Worker，是 `Request` 在 GPU 侧的镜像（`block_ids` / `output_token_ids` / mrope positions / 未完成的 prompt logprobs）。两者通过 `SchedulerOutput` 里的 `new_req_data` / `cached_req_data` 同步：`_update_states`（[gpu_model_runner.py:1204](../vllm/v1/worker/gpu_model_runner.py#L1204)）负责增删，构造点见 [gpu_model_runner.py:1307](../vllm/v1/worker/gpu_model_runner.py#L1307)。

**Q10：V1 怎么支持多 API server 进程共享一份引擎？**
`make_async_mp_client` 接受 `client_addresses` 和 `client_index`（[core_client.py:133](../vllm/v1/engine/core_client.py#L133)）。外部编排的 engine 由传入的 `input_address`/`output_address` 连接（[core_client.py:591](../vllm/v1/engine/core_client.py#L591)），每个 client 把自己的 `client_index` 打进 `EngineCoreRequest`（[core_client.py:1220](../vllm/v1/engine/core_client.py#L1220)），EngineCore 输出时按 `client_index` 分桶（`step()` 返回 `dict[int, EngineCoreOutputs]`），`process_output_sockets` 按 `sockets[client_index]` 发送（[core.py:1849](../vllm/v1/engine/core.py#L1849)）。

**Q11：`EngineCoreReadyResponse` 解决什么问题？**
EngineCore 初始化时会做显存 profiling 并**反向改写配置**（例如 KV cache 不够时 auto-fit 缩小 `max_model_len`，见 [core.py:328](../vllm/v1/engine/core.py#L328)）。前端必须知道最终值才能正确校验请求长度。所以握手时把 `max_model_len` / `num_gpu_blocks` / `block_size` / `dtype` / `kv_cache_size_tokens` / `max_num_seqs` 等回传（[vllm/v1/engine/__init__.py:72](../vllm/v1/engine/__init__.py#L72)），前端在 `_apply_ready_response`（[core_client.py:803](../vllm/v1/engine/core_client.py#L803)）里应用。

**Q12：为什么 `InprocClient.get_output` 调的是 `step_fn` 而不是 `step`？**
`step_fn` 在 `EngineCore.__init__` 末尾绑定（[core.py:235](../vllm/v1/engine/core.py#L235)）：`batch_queue is None` 时用 `step`，否则用 `step_with_batch_queue`。这样 busy loop（多进程）和同步驱动（in-proc）走同一个入口，PP / batch queue 的优化对两者都生效。`InprocClient.get_output`（[core_client.py:361](../vllm/v1/engine/core_client.py#L361)）还会手动调 `post_step`，因为 busy loop 路径下这一步由 `_process_engine_step`（[core.py:1469](../vllm/v1/engine/core.py#L1469)）负责。

---

## 7. 延伸阅读

### 源码清单（按阅读顺序）

| 主题 | 文件 |
| --- | --- |
| 前端协议 | [vllm/engine/protocol.py](../vllm/engine/protocol.py) |
| 在线前端 | [vllm/v1/engine/async_llm.py](../vllm/v1/engine/async_llm.py) |
| 离线前端 | [vllm/v1/engine/llm_engine.py](../vllm/v1/engine/llm_engine.py) |
| 输入处理 | [vllm/v1/engine/input_processor.py](../vllm/v1/engine/input_processor.py) |
| 输出处理 | [vllm/v1/engine/output_processor.py](../vllm/v1/engine/output_processor.py) |
| 增量 detokenize | [vllm/v1/engine/detokenizer.py](../vllm/v1/engine/detokenizer.py) |
| Client 家族 | [vllm/v1/engine/core_client.py](../vllm/v1/engine/core_client.py) |
| EngineCore | [vllm/v1/engine/core.py](../vllm/v1/engine/core.py) |
| 消息定义 | [vllm/v1/engine/__init__.py](../vllm/v1/engine/__init__.py) |
| 序列化 | [vllm/v1/serial_utils.py](../vllm/v1/serial_utils.py) |
| 进程启动 / ZMQ 地址 | [vllm/v1/engine/utils.py](../vllm/v1/engine/utils.py) |
| 张量 IPC | [vllm/v1/engine/tensor_ipc.py](../vllm/v1/engine/tensor_ipc.py) |
| DP 协调器 | [vllm/v1/engine/coordinator.py](../vllm/v1/engine/coordinator.py) |
| 请求状态机 | [vllm/v1/request.py](../vllm/v1/request.py) |
| 输出结构 | [vllm/v1/outputs.py](../vllm/v1/outputs.py) |
| Scheduler | [vllm/v1/core/sched/scheduler.py](../vllm/v1/core/sched/scheduler.py) |
| Executor 抽象 | [vllm/v1/executor/abstract.py](../vllm/v1/executor/abstract.py) |
| 单进程 Executor | [vllm/v1/executor/uniproc_executor.py](../vllm/v1/executor/uniproc_executor.py) |
| 多进程 Executor | [vllm/v1/executor/multiproc_executor.py](../vllm/v1/executor/multiproc_executor.py) |
| Ray Executor | [vllm/v1/executor/ray_executor.py](../vllm/v1/executor/ray_executor.py) |
| Worker 接口 | [vllm/v1/worker/worker_base.py](../vllm/v1/worker/worker_base.py) |
| GPU Worker | [vllm/v1/worker/gpu_worker.py](../vllm/v1/worker/gpu_worker.py) |
| GPU ModelRunner | [vllm/v1/worker/gpu_model_runner.py](../vllm/v1/worker/gpu_model_runner.py) |
| 持久 batch | [vllm/v1/worker/gpu_input_batch.py](../vllm/v1/worker/gpu_input_batch.py) |
| Sampler | [vllm/v1/sample/sampler.py](../vllm/v1/sample/sampler.py) |

### 官方文档

- vLLM V1 设计动机（RFC）：https://github.com/vllm-project/vllm/issues/8779
- vLLM 文档站 · Contributing / Engine 相关章节：https://docs.vllm.ai/en/latest/contributing/
- vLLM 官方博客《vLLM V1: A Major Upgrade to vLLM's Core Architecture》：https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
