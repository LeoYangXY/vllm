# vLLM V1 Feature 深度解析

面向**面试**的 vLLM 源码精读笔记。只讲 **V1 引擎**（`vllm/v1/`），V0 遗留代码仅在对比时提及并明确标注。

## 使用方法

- 文档里的源码链接都是**相对路径 Markdown 链接**，形如
  `[vllm/v1/core/sched/scheduler.py:120](../vllm/v1/core/sched/scheduler.py#L120)`，
  在 IDE 里直接 `Ctrl/Cmd + 点击` 即可跳到对应行。
- 行号对应**你当前仓库 checkout 的版本**，如果代码更新导致行号漂移，按文件名/函数名搜索即可。
- 每篇文档的结构固定：

| 章节 | 作用 |
| --- | --- |
| 0. TL;DR | 5 行记住：是什么 / 解决什么 / 怎么做 / 收益 |
| 1. 场景与痛点 | 为什么必须有这个 feature，量化说明 |
| 2. 核心设计 | 设计思想 + 关键抽象 + ASCII 图 |
| 3. 代码走读 | 按执行顺序逐段讲解，配真实源码链接 + 关键片段 |
| 4. 关键数据结构 | 字段 / 类型 / 含义 / 定义位置 |
| 5. 收益与代价 | 量化收益，以及引入的复杂度、限制、失效场景 |
| 6. 面试高频问题 | Q&A，答案具体到类名 / 函数名 / 行号 |
| 7. 延伸阅读 | 源码文件清单 + 官方链接 |

> 面试复述的黄金顺序：**痛点 → 关键抽象 → 主流程（能说出函数名）→ 权衡 → 指标**。
> 只会背"PagedAttention 省显存"是过不了面试的，能说出 `BlockPool` 的 `ref_cnt` 为什么存在、prefix caching 为什么要求整块命中，才算真懂。

---

## 阅读路线

### 第一条线：先把骨架建起来（必读，按顺序）

1. [01 - V1 引擎整体架构与进程模型](./01-v1-architecture-overview.md)
   前端进程 / EngineCore 进程 / Worker 三段切分，为什么这么切。
2. [02 - 一条请求的完整生命周期](./02-request-lifecycle.md)
   从 `/v1/chat/completions` 到流式 token 返回，全流程串讲。
3. [05 - 调度器：Continuous Batching / Chunked Prefill / 抢占 / 异步调度](./05-scheduler-continuous-batching.md)
   **最重要的一篇**。整个 vLLM 的心脏，面试 80% 的问题落在这里。

### 第二条线：显存与 KV Cache

4. [03 - PagedAttention 与 V1 KV Cache 管理机制](./03-paged-attention-kv-cache.md)
5. [04 - Automatic Prefix Caching](./04-prefix-caching.md)
6. [10 - KV Cache 卸载与 Prefill-Decode 分离（KV Connector）](./10-kv-cache-offload-and-pd-disaggregation.md)

### 第三条线：把延迟和吞吐压到极致

7. [07 - 投机解码 Speculative Decoding](./07-speculative-decoding.md)
8. [08 - CUDA Graph 捕获与 torch.compile](./08-cuda-graph-and-torch-compile.md)
9. [09 - 分布式并行 TP / PP / EP / DP](./09-distributed-parallelism.md)

### 第四条线：功能正确性

10. [06 - 采样流程与 Logits Processor](./06-sampling-logits-processor.md)
11. [11 - 结构化输出 / 约束解码](./11-structured-output.md)
12. [12 - LoRA 与多 LoRA 热插拔服务](./12-lora-and-multilora.md)
13. [13 - 多模态输入处理与 Encoder Cache](./13-multimodal-and-encoder-cache.md)
14. [14 - Metrics 与可观测性](./14-metrics-and-observability.md)

---

## 一张图看 V1 全貌

```
┌──────────────────────── 前端进程 (frontend) ────────────────────────┐
│  OpenAI API Server                                                  │
│      │  AsyncLLM.generate()                                         │
│      ├──> InputProcessor     tokenize / 多模态预处理 (CPU 密集)      │
│      │                                            │                 │
│      │                       ZMQ ── add_request ──┤                 │
│  Detokenizer  <────────────── EngineCoreOutput ───┤                 │
│  OutputProcessor  增量聚合 → RequestOutput         │                 │
└───────────────────────────────────────────────────┼─────────────────┘
                                                    ▼
┌────────────────────── EngineCore 进程 (核心) ───────────────────────┐
│  Scheduler.schedule()  ← KVCacheManager / BlockPool / Prefix Cache  │
│      │  SchedulerOutput (num_scheduled_tokens per req)              │
│      ▼                                                              │
│  Executor → Worker → ModelRunner.execute_model()                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              Model forward → LogitsProcessor → Sampler
              (CUDA Graph / torch.compile / TP-PP-EP 通信)
```

---

## 面试时的"高频追问"速查

| 追问 | 去看 |
| --- | --- |
| 为什么 vLLM 能比 HF 快这么多？ | 03、05 |
| batch size 由什么决定？打满了会怎样？ | 05（token budget / 抢占） |
| 长 prompt 会把系统搞卡吗？怎么解决？ | 05（chunked prefill） |
| 两个请求用了同一个 system prompt，会重复算吗？ | 04 |
| KV cache 占多少显存？怎么算？ | 03 |
| 显存不够了会怎样？ | 05（preemption）、10（offload） |
| 为什么 decode 阶段 GPU 利用率低？怎么优化？ | 07、08 |
| 投机解码真的不改变输出分布吗？ | 07（rejection sampling） |
| CUDA Graph 为什么不能一直用？ | 08 |
| 多卡怎么切？TP 和 PP 怎么选？ | 09 |
| 上百个 LoRA 怎么一起服务？ | 12 |
| 怎么判断系统是不是该扩容了？ | 14 |
