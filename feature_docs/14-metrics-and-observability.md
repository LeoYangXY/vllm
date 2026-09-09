# Metrics 与可观测性

> 适用版本：vLLM V1。实现在 `vllm/v1/metrics/`（`loggers.py` / `stats.py` / `prometheus.py`）。

## 0. TL;DR

- **是什么**：V1 在 EngineCore 进程收集调度/执行统计，回前端进程后由 `StatLoggerBase` 派生类落日志或暴露 Prometheus 指标。
- **解决什么**：线上要回答「该不该扩容」「为什么尾延迟高」「KV cache 是不是瓶颈」「prefix 命中率够不够」，没有指标就是盲调。
- **怎么做**：`SchedulerStats` / `EngineCoreMetrics` 随 `EngineCoreOutput` 每步回传 → `OutputProcessor`/`LoggingStatLogger`/`PrometheusStatLogger` 聚合 → 日志 or `/metrics` 端点。
- **收益**：可量化容量规划、SLO 治理、问题定位（排队 / 显存 / 命中率 / GPU 空转）。

---

## 1. 场景与痛点

生产部署必须可观测，否则只能拍脑袋。典型问题：

- **排队**：`num_requests_waiting` 持续高 → 吞吐不够 / 并发上限低。
- **显存瓶颈**：`gpu_cache_usage_perc` 接近 1 → 触发抢占，尾延迟飙升。
- **prefix 失效**：`gpu_prefix_cache_hit_rate` 低 → 重复算 prefill，TTFT 高。
- **GPU 空转**：`iteration_tokens_total` 低、`num_requests_running` 小 → batch 太小，decode 阶段 GPU 利用率低（该上 spec decode / cudagraph）。
- **TTFT / TPOT**：直接反映用户体验的 SLO 指标。

---

## 2. 核心设计

### 2.1 指标数据从哪来

```
Worker/GPU ── ModelRunnerOutput ──> EngineCore.step()
        │
        ├─ SchedulerStats (调度侧: waiting/running/cache 使用/命中率)
        ├─ EngineCoreMetrics (核心侧: 迭代吞吐/调度耗时)
        └─ 打包进 EngineCoreOutput ── ZMQ ──> 前端
                        │
                        ▼
        StatLoggerBase 派生: LoggingStatLogger / PrometheusStatLogger
```

### 2.2 三种 Logger

```python
# vllm/v1/metrics/loggers.py:44
class StatLoggerBase(ABC):           # 抽象基类，定义 log/info 接口

# vllm/v1/metrics/loggers.py:99
class LoggingStatLogger(StatLoggerBase):     # 周期性打印到日志（Avg generation throughput 等）

# vllm/v1/metrics/loggers.py:443
class PrometheusStatLogger(AggregateStatLoggerBase):  # 暴露 Prometheus gauge/counter
```

- `LoggingStatLogger`：每个 `log_interval`（默认 30s）打一行汇总（running/waiting 数、GPU 吞吐、前缀命中率、调度延迟等）。
- `PrometheusStatLogger`：把指标注册成 Prometheus collector，由 `/metrics` 端点抓取。

### 2.3 核心统计结构

```python
# vllm/v1/metrics/stats.py:186
class SchedulerStats:
    # num_running_reqs / num_waiting_reqs
    # gpu_cache_usage_sys / cpu_cache_usage_sys
    # prefix_cache_stats (命中率相关)
    # 各 finish reason 计数
```

`EngineCoreMetrics` 含每步迭代的 token 数、调度耗时、队列长度等，用于计算吞吐与延迟分位。

---

## 3. 代码走读

### 3.1 指标随 output 回传

EngineCore 在 `step` 把 `SchedulerStats` 和 `EngineCoreMetrics` 塞进 `EngineCoreOutput`，经 ZMQ 回前端（见 `02` 的通信链路）。前端 `OutputProcessor` 把数据喂给配置的 logger。

### 3.2 Prometheus 端点

`PrometheusStatLogger` 在 `vllm/entrypoints/metrics.py`（或 API server 启动时）注册 `/metrics` 路由，Prometheus 周期抓取。常见指标名（以实际代码为准）：`vllm:num_requests_running`、`vllm:num_requests_waiting`、`vllm:gpu_cache_usage_perc`、`vllm:gpu_prefix_cache_hit_rate`、`vllm:time_to_first_token_seconds`、`vllm:time_per_output_token_seconds`、`vllm:e2e_request_latency_seconds`、`vllm:request_success_total`、`vllm:iteration_tokens_total`。

### 3.3 吞吐与延迟的计算口径

- **Avg generation throughput (tok/s)** = 窗口内 completion tokens / 窗口时长。
- **TTFT**：从请求进入到第一个 output token 的耗时（含排队）。
- **TPOT（time per output token）**：注意 V1 的 `time_per_output_token_seconds` 实际是「e2e 延迟 / 输出 token 数」的近似口径，并非严格的逐 token 间隔，面试/看板时要明确其口径，避免误读成纯 decode 间隔。
- **prefix cache hit rate**：命中前缀 block 数 / 总前缀 block 数（见 `04`）。

---

## 4. 关键数据结构

| 结构 | 字段 | 位置 |
| --- | --- | --- |
| `StatLoggerBase` | 抽象 logger 接口 | [loggers.py:44](../vllm/v1/metrics/loggers.py#L44) |
| `LoggingStatLogger` | 周期日志 | [loggers.py:99](../vllm/v1/metrics/loggers.py#L99) |
| `PrometheusStatLogger` | Prometheus 指标 | [loggers.py:443](../vllm/v1/metrics/loggers.py#L443) |
| `SchedulerStats` | waiting/running/cache/命中率 | [stats.py:186](../vllm/v1/metrics/stats.py#L186) |
| `EngineCoreMetrics` | 每步吞吐/调度耗时 | [vllm/v1/engine/__init__.py](../vllm/v1/engine/__init__.py) |

---

## 5. 收益与代价

**收益**
- 量化容量规划：cache usage 高 → 加显存/降并发；waiting 高 → 加副本。
- SLO 治理：TTFT/TPOT 直接对应用户体验。
- 问题定位：命中率低 → 查 prefix 配置；running 少 + iteration tokens 低 → GPU 空转，该上 spec decode/cudagraph。

**代价 / 注意点**
- 指标本身有采样与聚合开销（小，可忽略）。
- `time_per_output_token_seconds` 口径易误读，需结合文档理解。
- Prometheus 端点需独立暴露端口并配 scrape，否则指标采集不到。
- 多副本/多 DP 下指标需按副本聚合，单副本数字不代表全局。

---

## 6. 面试高频问题

**Q1：vLLM V1 的指标数据采集链路？**
A：Worker 产出 `ModelRunnerOutput` → EngineCore 汇总成 `SchedulerStats`/`EngineCoreMetrics` → 随 `EngineCoreOutput` 经 ZMQ 回前端 → `StatLoggerBase` 派生 logger 落日志或 Prometheus。

**Q2：num_requests_waiting 高说明什么？怎么处理？**
A：说明请求在排队，吞吐或并发上限不足。处理：提高 `max_num_seqs`/`max_num_batched_tokens`、加 DP 副本、或降低单请求显存占用（如降 prefix 缓存保留）。

**Q3：gpu_cache_usage_perc 接近 1 会怎样？**
A：KV cache 打满，新请求分配不到 block，触发抢占（RECOMPUTE，见 `05`），已 running 请求被踢重算，尾延迟飙升。

**Q4：gpu_prefix_cache_hit_rate 低但预期很高，可能原因？**
A：prompt 含随机因素（随机 few-shot 顺序）导致前缀不一致；block_size 过大浪费；或 LoRA 不同 adapter 前缀被隔离。查 prompt 构造与 `enable_prefix_caching` 开关。

**Q5：time_per_output_token_seconds 的真实口径？**
A：它是「e2e 延迟 / 输出 token 数」的近似，并非严格逐 token 间隔，看板时要按文档理解，别当纯 decode 间隔。

**Q6：怎么判断该上投机解码？**
A：看 `num_requests_running` 小、`iteration_tokens_total` 低（batch 小、GPU 带宽受限、decode 慢），且输出长、draft 与 target 分布接近 → spec decode 收益大（见 `07`）。

**Q7：Prometheus 指标在哪暴露？**
A：`PrometheusStatLogger` 配合 API server 的 `/metrics` 路由（[vllm/entrypoints/metrics.py](../vllm/entrypoints/metrics.py)），由 Prometheus 抓取。

**Q8：LoggingStatLogger 打哪些关键数？**
A：running/waiting 数、GPU 生成吞吐（tok/s）、前缀命中率、调度延迟、cache 使用率（[vllm/v1/metrics/loggers.py:99](../vllm/v1/metrics/loggers.py#L99)）。

**Q9：缓存命中率类指标对容量规划有什么用？**
A：命中率低意味着大量重复 prefill，可通过加大 prefix 缓存、统一 system prompt、合并相同文档来提升，直接降 TTFT 与 GPU 负载。

**Q10：spec decode 相关的可观测指标？**
A：草稿接受率（`num_accepted_tokens / num_draft_tokens`）、接受长度分布，来自 `vllm/v1/spec_decode/metrics.py`，用于判断是否该调 `num_speculative_tokens` 或关掉（见 `07`）。

---

## 7. 延伸阅读

- 实现：`vllm/v1/metrics/loggers.py`、`stats.py`、`prometheus.py`、`vllm/entrypoints/metrics.py`
- 配合：`04-prefix-caching.md`、`05-scheduler-continuous-batching.md`、`07-speculative-decoding.md`
- 官方：https://docs.vllm.ai/en/latest/serving/metrics.html
