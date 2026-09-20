# Metrics 与可观测性（问题定位手册）

> 适用版本：vLLM V1。实现在 `vllm/v1/metrics/`（`loggers.py` / `stats.py` / `prometheus.py`）。
> 本文的落地目标是：**给你一套"现象 → 看哪个指标 → 调哪个参数"的对照表**，源码细节只讲必要的部分。

## 0. 部署卡片与 TL;DR

### 0.1 部署卡片

| 项目 | 内容 |
| --- | --- |
| **怎么开** | 周期日志**默认开启**，关掉用 `--disable-log-stats`；Prometheus 指标默认挂在 API server 的 `/metrics`；链路追踪用 `--otlp-traces-endpoint http://...:4318` |
| **怎么确认生效** | 启动后每 30s 会打印一行 `Avg prompt throughput: ..., Running: N reqs, ...`；`curl http://localhost:8000/metrics \| grep vllm:` 能看到全部指标 |
| **症状 → 调什么** | 直接跳到本文 §4「诊断对照表」，按现象查 |
| **一条命令试** | `curl -s http://localhost:8000/metrics \| grep -E 'vllm:(num_requests\|kv_cache_usage\|prefix_cache_hits\|num_preemptions)'` |

> 三个最常见的误读：① **没有直接的"命中率"指标**，要用 `hits / queries` 自己算；② `request_time_per_output_token_seconds` 是「端到端延迟 ÷ 输出 token 数」的近似口径，想看真正的逐 token 间隔要看 `inter_token_latency_seconds`；③ 日志那行里的 `Waiting` 是 `waiting + skipped_waiting` 的和。

### 0.2 五句话 TL;DR

1. **是什么**：EngineCore 每步产出调度/执行统计，随输出回传前端，由日志 logger 和 Prometheus logger 分别落盘与暴露。
2. **解决什么**：线上要回答"该不该扩容""为什么尾延迟高""KV 是不是瓶颈""前缀缓存有没有生效"，没有指标就只能拍脑袋。
3. **怎么做**：`SchedulerStats` 随 `EngineCoreOutputs` 回传 → 前端 `OutputProcessor` 与 logger 聚合 → 日志行或 `/metrics`。
4. **收益**：容量规划、SLO 治理、问题定位都有量化依据，而且 vLLM 自带了**延迟分解**指标（排队 / prefill / decode 分开统计）。
5. **代价**：指标有采样与聚合开销（很小）；多副本/DP 下要按副本聚合，单副本数字不代表全局。

---

## 1. 场景与痛点

生产部署必须可观测，否则只能拍脑袋。典型问题：

| 想回答的问题 | 靠什么指标 |
| --- | --- |
| 该不该扩容了？ | `vllm:num_requests_waiting` 持续高 + `vllm:kv_cache_usage_perc` 接近 1 |
| 为什么尾延迟高？ | `vllm:request_queue_time_seconds`（排队）vs `vllm:request_prefill_time_seconds`（算得慢） |
| KV cache 是不是瓶颈？ | `vllm:kv_cache_usage_perc`、`vllm:num_preemptions` |
| 前缀缓存生效了吗？ | `vllm:prefix_cache_hits / vllm:prefix_cache_queries` |
| GPU 是不是在空转？ | `vllm:num_requests_running` 小 + `vllm:iteration_tokens_total` 低 |

---

## 2. 指标是怎么流出来的

```
Worker/GPU ── ModelRunnerOutput ──> EngineCore.step()
        │
        ├─ SchedulerStats        (调度侧: running/waiting/KV 使用率/前缀命中)
        ├─ EngineCoreMetrics     (核心侧: 迭代 token 数/调度耗时)
        └─ 打包进 EngineCoreOutputs ── ZMQ ──> 前端
                        │
                        ▼
        StatLoggerBase 派生: LoggingStatLogger / PrometheusStatLogger
```

三种 logger（[vllm/v1/metrics/loggers.py](../vllm/v1/metrics/loggers.py)）：

| 类 | 位置 | 作用 |
| --- | --- | --- |
| `StatLoggerBase` | [loggers.py:44](../vllm/v1/metrics/loggers.py#L44) | 抽象基类，定义 `log` / `info` 接口 |
| `LoggingStatLogger` | [loggers.py:99](../vllm/v1/metrics/loggers.py#L99) | 每 30s 打一行汇总日志 |
| `PrometheusStatLogger` | [loggers.py:443](../vllm/v1/metrics/loggers.py#L443) | 注册 Prometheus collector，由 `/metrics` 抓取 |

---

## 3. 先学会看那一行日志

这是**最快的上手方式**——不用配 Prometheus，启动就有。格式（[loggers.py:256-313](../vllm/v1/metrics/loggers.py#L256)）：

```text
Avg prompt throughput: 1234.5 tokens/s, Avg generation throughput: 567.8 tokens/s,
Running: 32 reqs, Waiting: 8 reqs, GPU KV cache usage: 67.3%, Prefix cache hit rate: 43.2%
```

逐项拆解：

| 字段 | 含义 | 怎么用 |
| --- | --- | --- |
| `Avg prompt throughput` | 窗口内 **prefill** token 吞吐 | 高说明在处理大量长 prompt |
| `Avg generation throughput` | 窗口内 **decode** 输出 token 吞吐 | 这是通常说的"系统吞吐" |
| `Running: N reqs` | 当前在跑的请求数 | 远小于 `max_num_seqs` → 并发没打满 |
| `Waiting: M reqs` | 排队请求数（含被推迟的） | 持续 > 0 → 容量不足 |
| `Deferred: D reqs` | 被瞬时约束推迟的请求（仅在 >0 时打印） | 见 `05` 的 `skipped_waiting` |
| `Preemptions: P` | 本窗口累计抢占次数（仅在 >0 时打印） | **一出现就该调优** |
| `GPU KV cache usage` | KV 块占用率 | 接近 100% → 抢占与尾延迟 |
| `Prefix cache hit rate` | 前缀缓存命中率 | 见 `04` |

> 注意最后两项是**窗口内的平均值**，不是瞬时值。看瞬时值要去 `/metrics` 的 `vllm:kv_cache_usage_perc`。

---

## 4. 指标速查表

### 4.1 容量与排队

| 指标 | 含义 |
| --- | --- |
| `vllm:num_requests_running` | 正在被调度的请求数 |
| `vllm:num_requests_waiting` | 排队（waiting + skipped_waiting） |
| `vllm:num_requests_waiting_by_reason` | 按原因拆分：`capacity`（真排队）vs `deferred`（被瞬时约束推迟） |
| `vllm:kv_cache_usage_perc` | KV 块占用率 |
| `vllm:num_preemptions` | 累计抢占次数 |
| `vllm:request_num_preemptions` | 单请求被抢占次数直方图 |

### 4.2 延迟分解（最有诊断价值的一组）

| 指标 | 含义 |
| --- | --- |
| `vllm:time_to_first_token_seconds` | TTFT：进来到第一个 token |
| `vllm:inter_token_latency_seconds` | ITL：逐 token 间隔（**真正的"卡不卡"**） |
| `vllm:request_time_per_output_token_seconds` | 近似 TPOT = 端到端延迟 ÷ 输出 token 数（口径见 §4.5） |
| `vllm:request_queue_time_seconds` | **排队**耗时 |
| `vllm:request_prefill_time_seconds` | **prefill** 耗时 |
| `vllm:request_decode_time_seconds` | **decode** 耗时 |
| `vllm:request_inference_time_seconds` | 推理总耗时（prefill + decode） |
| `vllm:e2e_request_latency_seconds` | 端到端总延迟（含排队） |

有这组指标，就不用再猜"到底是排队慢还是算得慢"了：`queue_time` 大就扩容，`prefill_time` 大就去查前缀命中与切块（`04`/`05`），`decode_time` 大就去查并发与 GPU 空转。

### 4.3 吞吐与 token 计数

| 指标 | 含义 |
| --- | --- |
| `vllm:iteration_tokens_total` | 每步调度的 token 总数（**判断 GPU 空转的关键**） |
| `vllm:prompt_tokens` / `vllm:generation_tokens` | 累计 prefill / decode token 数 |
| `vllm:prompt_tokens_by_source` | prefill token 的来源拆分 |
| `vllm:prompt_tokens_cached` | 命中缓存、没实际计算的 prefill token 数 |
| `vllm:request_prompt_tokens` / `vllm:request_generation_tokens` | 单请求维度直方图 |

`prompt_tokens_cached / prompt_tokens` 就是"前缀缓存实际省了多少计算"，比命中率更直观。

### 4.4 缓存命中

| 指标 | 含义 |
| --- | --- |
| `vllm:prefix_cache_queries` / `vllm:prefix_cache_hits` | 本地前缀缓存查询/命中（**命中率 = hits / queries**） |
| `vllm:external_prefix_cache_queries` / `..._hits` | 外部（KV connector，见 `10`）命中 |
| `vllm:mm_cache_queries` / `vllm:mm_cache_hits` | 多模态 encoder 缓存（见 `13`） |
| `vllm:kv_block_lifetime_seconds` / `..._idle_before_evict_seconds` / `..._reuse_gap_seconds` | KV 块的生命周期/空闲/复用间隔（调 LRU 与块数时有用） |

### 4.5 两个容易读错的口径

1. **`request_time_per_output_token_seconds` ≠ 严格 TPOT**。它是「端到端延迟 ÷ 输出 token 数」的近似，把排队时间也摊进去了。想看生成过程的流畅度，看 `vllm:inter_token_latency_seconds`。
2. **没有现成的命中率指标**。`prefix_cache_hits / prefix_cache_queries` 要自己算（日志里那行已经帮你算好了）。

### 4.6 其他

| 指标 | 含义 |
| --- | --- |
| `vllm:request_success` | 按 finish reason 统计的请求数 |
| `vllm:corrupted_requests` | logits 出现 NaN 的损坏请求（开启 NaN 检测时才有） |
| `vllm:engine_sleep_state` | 引擎 sleep/wake 状态（见 sleep mode） |
| `vllm:lora_requests_info` | LoRA 请求信息（见 `12`） |
| `vllm:spec_decode_num_accepted_tokens_total` / `..._draft_tokens_total` | 投机解码接受情况（见 `07`） |

---

## 5. 诊断对照表（现象 → 看什么 → 调什么）

| 现象 | 先看这几个指标 | 典型结论 | 调什么 |
| --- | --- | --- | --- |
| **吞吐上不去** | `num_requests_running` 远小于 `max_num_seqs`，`kv_cache_usage_perc` 很低 | 并发上限没打开，KV 根本没用满 | 调大 `--max-num-seqs`（`05`） |
| **吞吐上不去** | `num_requests_running` 小、`iteration_tokens_total` 低、GPU 利用率低 | GPU 空转，batch 打不大 | `--async-scheduling`（`05`）、`--max-num-batched-tokens`（`05`） |
| **吞吐上不去，小 batch 场景** | `iteration_tokens_total` 长期很低 | 每步只算很少 token | 投机解码（`07`）、CUDA Graph（`08`） |
| **尾延迟（P99）飙升** | `num_preemptions` 持续上涨、`kv_cache_usage_perc` ≈ 1 | KV 打满触发抢占 | 降 `--max-num-seqs`、加 `--watermark`（`05`） |
| **TTFT 高** | `request_queue_time_seconds` 占大头 | 纯排队，容量不足 | 扩容 / 加 DP 副本（`09`） |
| **TTFT 高** | queue 小但 `request_prefill_time_seconds` 大，且 prefix 命中率低 | 重复算 prefill | 查前缀一致性（`04`） |
| **长 prompt 把系统拖慢** | `inter_token_latency_seconds` 的 P99 抖动 | 长 prefill 抢占了单步时间 | `--long-prefill-token-threshold`（`05`） |
| **前缀缓存没生效** | `prefix_cache_hits / prefix_cache_queries` < 20% | 请求没有真正的公共前缀 | 检查 system prompt 是否逐字一致（`04`） |
| **多模态请求慢** | `mm_cache_hits / mm_cache_queries` 低 | 同一张图反复编码 | 见 `13` |
| **输出出现重复片段** | `num_preemptions` > 0 | 抢占后重算（RECOMPUTE） | 见 `05` §5.2 |
| **多轮对话没变快** | `prompt_tokens_cached` 占 `prompt_tokens` 比例低 | KV 块被 LRU 驱逐了 | 增大 KV 容量（`03`）或降低并发 |

> 判断顺序建议：**先看 `num_requests_running` / `num_requests_waiting`（是不是容量问题）→ 再看 `kv_cache_usage_perc`（是不是显存问题）→ 最后看延迟分解指标（是哪一段慢）**。

---

## 6. 关键数据结构

| 结构 | 字段 | 位置 |
| --- | --- | --- |
| `StatLoggerBase` | 抽象 logger 接口 | [loggers.py:44](../vllm/v1/metrics/loggers.py#L44) |
| `LoggingStatLogger` | 周期汇总日志（格式见 §3） | [loggers.py:99](../vllm/v1/metrics/loggers.py#L99) |
| `PrometheusStatLogger` | Prometheus 指标注册与上报 | [loggers.py:443](../vllm/v1/metrics/loggers.py#L443) |
| `SchedulerStats` | running/waiting/KV 使用率/前缀命中 | [stats.py:186](../vllm/v1/metrics/stats.py#L186) |
| `SpecDecodingStats` | 投机解码的 draft/accept 计数 | [vllm/v1/spec_decode/metrics.py](../vllm/v1/spec_decode/metrics.py) |

指标名的完整清单以 [loggers.py](../vllm/v1/metrics/loggers.py) 中的 `name="vllm:..."` 为准——本文表格是常用子集。

---

## 7. 收益与代价

**收益**
- 量化容量规划：KV 使用率高 → 加显存/降并发；waiting 高 → 加副本。
- SLO 治理：TTFT / ITL 直接对应用户体验，且能按排队/prefill/decode 分解归因。
- 问题定位：命中率低 → 查前缀配置；running 少且 iteration tokens 低 → GPU 空转。

**代价 / 注意点**
- 指标本身有采样与聚合开销（很小，可忽略）。
- 口径易误读：近似 TPOT 与 ITL 不是一回事；命中率要自己相除（§4.5）。
- 多副本 / DP 下要按副本聚合，单个副本的数字不代表全局。
- 部分指标默认不启用（如 `corrupted_requests` 需要开启 NaN 检测，块生命周期类指标需要开启 KV events）。

---

## 8. 面试高频问题（速查，理解原理可跳过）

> 这一节是面试向的 Q&A 速查。**如果目标是能定位线上问题，读完 §1~§5 就够了。**

**Q1：vLLM V1 的指标采集链路是怎样的？**
A：EngineCore 每步把 `SchedulerStats` 和 `EngineCoreMetrics` 塞进 `EngineCoreOutputs`，经 ZMQ 回传前端（见 `02`）；前端 `OutputProcessor` 把数据喂给配置的 logger：`LoggingStatLogger` 打周期日志，`PrometheusStatLogger` 暴露 `/metrics`。

**Q2：日志里那一行 `Waiting` 是什么？和 `Deferred` 有什么区别？**
A：`Waiting` 是 `num_waiting_reqs + num_skipped_waiting_reqs` 的和（[loggers.py:268-271](../vllm/v1/metrics/loggers.py#L268)）。`Deferred` 只在 `num_skipped_waiting_reqs > 0` 时打印，指被**瞬时约束**（grammar 未编译完、远程 KV 未到、streaming 输入未到等）推迟的请求，它们不占容量但也没在跑（见 `05`）。

**Q3：怎么判断系统是"容量不够"还是"GPU 空转"？**
A：容量不够的特征是 `num_requests_waiting` 持续高且 `kv_cache_usage_perc` 接近 1；GPU 空转的特征是 `num_requests_running` 小、`iteration_tokens_total` 低，但 KV 使用率并不高。前者加副本/降并发，后者开异步调度或投机解码。

**Q4：TTFT 高怎么定位？**
A：用延迟分解指标：`request_queue_time_seconds` 大 → 排队，扩容；`request_prefill_time_seconds` 大 → 算得慢，查前缀命中率与 chunked prefill 配置。两者都小但 TTFT 仍高，就要看前端的 tokenize / 多模态预处理（见 `02`）。

**Q5：`request_time_per_output_token_seconds` 能当 TPOT 用吗？**
A：它是「端到端延迟 ÷ 输出 token 数」的近似口径，把排队时间也摊进去了，严格来说不是逐 token 间隔。想看生成过程是否流畅，用 `vllm:inter_token_latency_seconds`。

**Q6：怎么确认前缀缓存真的在省算力？**
A：两个角度：命中率 `prefix_cache_hits / prefix_cache_queries`；以及更直观的 `prompt_tokens_cached / prompt_tokens`（有多少 prefill token 是靠缓存跳过的）。

---

## 9. 延伸阅读

- 源码：`vllm/v1/metrics/loggers.py`（指标定义与日志格式）、`vllm/v1/metrics/stats.py`（统计结构）、`vllm/v1/metrics/prometheus.py`
- 投机解码指标：`vllm/v1/spec_decode/metrics.py`（接受率的 PromQL 写法在源码注释里）
- 官方：https://docs.vllm.ai/en/latest/design/metrics.html
- 配套阅读：`05-scheduler-continuous-batching.md`（running/waiting/抢占的含义）、`03-paged-attention-kv-cache.md`（KV 使用率怎么算）、`04-prefix-caching.md`（命中率）
