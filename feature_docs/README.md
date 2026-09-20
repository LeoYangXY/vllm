# vLLM V1 Feature 深度解析

面向**理解原理 + 能上手部署**的 vLLM 源码精读笔记。只讲 **V1 引擎**（`vllm/v1/`），V0 遗留代码仅在对比时提及并明确标注。

## 使用方法

- 文档里的源码链接都是**相对路径 Markdown 链接**，形如
  `[vllm/v1/core/sched/scheduler.py:120](../vllm/v1/core/sched/scheduler.py#L120)`，
  在 IDE 里直接 `Ctrl/Cmd + 点击` 即可跳到对应行。
- 行号对应**你当前仓库 checkout 的版本**，如果代码更新导致行号漂移，按文件名/函数名搜索即可。
- `00` 是方法论篇（不对应单个 feature）。**除 `00` 外，每篇功能篇的结构固定**：

| 章节 | 作用 |
| --- | --- |
| 0. 部署卡片与 TL;DR | **先看这里**：怎么开 / 怎么确认生效 / 症状→调什么 / 一条命令；再加 5 句话原理概括 |
| 1. 场景与痛点 | 为什么必须有这个 feature，量化说明 |
| 2. 核心设计 | 设计思想 + 关键抽象 + ASCII 图 |
| 3. 代码走读 | 按执行顺序逐段讲解，配真实源码链接 + 关键片段 |
| 4. 关键数据结构 | 字段 / 类型 / 含义 / 定义位置 |
| 5. 收益与代价 | 量化收益，以及引入的复杂度、限制、失效场景 |
| 6. 面试高频问题 | Q&A 速查（**理解原理可跳过**） |
| 7. 延伸阅读 | 源码文件清单 + 官方链接 |

### 两种读法

- **想建立端到端分析能力**（推荐起点）→ 先读 `00`，按它给的判断树和实验方法跑一遍压测，再带着问题查其余篇章。
- **想搞懂某个机制** → 按上面的顺序读，读到 §5 就够了，§6 是给面试准备的速查。
- **想解决线上问题** → 每篇只看 §0 部署卡片；不确定是哪个环节的问题时，先看
  [14 - Metrics 与可观测性](./14-metrics-and-observability.md) 的「诊断对照表」，再跳到对应篇章。

> 复述一个 feature 的黄金顺序：**痛点 → 关键抽象 → 主流程（能说出函数名）→ 权衡 → 指标**。
> 只会背"PagedAttention 省显存"是不够的，能说出 `BlockPool` 的 `ref_cnt` 为什么存在、prefix caching 为什么要求整块命中，才算真懂。

---

## 术语对照表

同一概念在源码（英文）和本文档（中文）里来回出现，先对齐一下：

| 中文 | 英文 / 代码 | 一句话含义 |
| --- | --- | --- |
| 块 | `block` | KV cache 的最小分配单位，默认 16 token（见 `03`） |
| 块表 | `block table` | 逻辑位置 → 物理块的映射表 |
| 槽位 | `slot` | 一个块内部的 token 位置 |
| 块池 | `BlockPool` | 全局 KV 块资源池，带引用计数与 LRU |
| 连续批处理 | continuous batching | 每个 step 重新组批，请求随到随走（见 `05`） |
| 分块预填充 | chunked prefill | 长 prompt 切成多步逐步算完 |
| 抢占 | preemption | KV 不够时让某个请求从头重算 |
| 牺牲请求 | victim | 被选中抢占的那个请求 |
| 水位 | `watermark` | 给准入预留的空闲块比例，用来降抢占率 |
| 占位符 | `num_output_placeholders` | 异步调度下"预测还没回来的 token 数" |
| 草稿 / 验证 | draft / verify | 投机解码里小模型提议、大模型并行验证（见 `07`） |
| 接受率 | acceptance rate | 草稿 token 被验证通过的比例 |
| 图捕获 | CUDA graph capture | 把一步的 kernel 序列录成图再 replay（见 `08`） |
| 持久批 | persistent batch（`InputBatch`） | worker 侧复用 batch 结构，每步只做增量更新 |
| 前端 / 核心 / 工作者 | frontend / EngineCore / Worker | 三类进程（见 `01`） |
| 适配器 | adapter（LoRA） | 共享 base 权重的小矩阵（见 `12`） |

常见缩写：

| 缩写 | 全称 | 含义 |
| --- | --- | --- |
| TTFT | time to first token | 从请求进来到第一个 token 的耗时（含排队） |
| TPOT | time per output token | 平均每个输出 token 的耗时（**注意 vLLM 的指标是近似口径**，见 `14`） |
| ITL | inter-token latency | 逐 token 间隔，反映"卡不卡" |
| E2E | end-to-end | 端到端总延迟 |
| QPS / RPS | queries per second | 每秒请求数 |
| SLO | service level objective | 服务质量目标（如 P99 TTFT < 1s） |
| ISL / OSL | input / output sequence length | 输入 / 输出序列长度 |
| TP / PP / EP / DP | tensor / pipeline / expert / data parallel | 四种并行方式（见 `09`） |

---

## 阅读路线

### 第零条线：先建立分析方法（强烈建议第一个读）

0. [00 - 端到端优化：分析方法与实践手册](./00-end-to-end-optimization-playbook.md)
   **如果你不确定"该观察什么、能在哪里优化"，先读这篇。** 从算子视角升级到系统视角：五层模型、从症状到层级的归因判断树、每层可做的优化，以及一套可重复的压测实验方法（含命令与记录模板）。

### 第一条线：先把骨架建起来（必读，按顺序）

1. [01 - V1 引擎整体架构与进程模型](./01-v1-architecture-overview.md)
   前端进程 / EngineCore 进程 / Worker 三段切分，为什么这么切。
2. [02 - 一条请求的完整生命周期](./02-request-lifecycle.md)
   从 `/v1/chat/completions` 到流式 token 返回，全流程串讲。
3. [05 - 调度器：Continuous Batching / Chunked Prefill / 抢占 / 异步调度](./05-scheduler-continuous-batching.md)
   **最重要的一篇**。整个 vLLM 的心脏，绝大部分调优参数都在这一篇。

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

## 按症状查文档（部署向）

不知道该看哪篇时，从现象反查：

| 现象 | 先看 | 大概率的原因 |
| --- | --- | --- |
| 吞吐上不去，但显存没用满 | `05` 的 `max_num_seqs`、KV 使用率 | 并发上限没打开 |
| 小模型 / 低并发下 GPU 利用率低 | `05`（异步调度）、`07`、`08` | Python 调度与 launch 开销占比高 |
| 长 prompt 一来，所有请求都变慢 | `05`（chunked prefill） | 单步被长 prefill 独占 |
| 抢占频繁、尾延迟飙升 | `05`（抢占 / watermark）、`03`（KV 容量） | KV cache 打满 |
| 相同 system prompt 没省算力 | `04` | 前缀没逐字一致 / 块被驱逐 |
| 多轮对话没变快 | `04`、`03` | 同上，或 KV 容量不够 |
| 输出重复片段 | `05`（RECOMPUTE 抢占） | 被抢占后重算 |
| 结构化输出慢 / 第一个请求特别慢 | `11` | grammar 编译 |
| 多租户 adapter 请求卡住 | `12` | `max_loras` 打满 |
| 图片请求 OOM 或迟迟不调度 | `13`、`05`（encoder 预算） | 分辨率过大 / encoder 预算占满 |
| 想加机器但不知道 TP 还是 DP | `09` | TP 降延迟、DP 提吞吐 |
| 不确定瓶颈在哪 | `14` 的诊断对照表 | 先用指标归因 |

---

## 面试时的"高频追问"速查

> 下面这些是**面试向**的入口。如果你的目标是理解系统或排查线上问题，用上一节「按症状查文档」更直接。

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
