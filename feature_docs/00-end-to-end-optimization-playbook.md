# 端到端推理优化：分析方法与实践手册

> **建议第一个读这篇**。其余 14 篇讲"某个机制怎么实现"，这篇讲"面对一个真实的推理服务，我该观察什么、怎么归因、能改什么"。
> 假设你已经了解常见的推理优化技术（KV cache、continuous batching、量化、投机解码等），但缺少把它们串成分析链条的经验。

## 0. 先换心智模型

### 0.1 从算子视角到系统视角的四个转变

算子视角问的是"这个 kernel 多快"，系统视角问的是"GPU 的时间里装了多少有用工作"。四个必须完成的转变：

| # | 算子视角 | 系统视角 |
| --- | --- | --- |
| 1 | 关注单次 kernel 耗时 | 关注**每个 step 处理多少有用 token**（`vllm:iteration_tokens_total`）。batch=1 时 kernel 再快，一步也只算 1 个 token |
| 2 | 关注平均延迟 | 关注**延迟的组成与分位数**。TTFT = 排队 + prefill，P99 才是用户感受；先分解再优化 |
| 3 | 觉得哪里都能优化 | **只优化瓶颈**。非瓶颈层的优化收益为 0，甚至为负（GPU 已饱和时开投机解码就是负收益） |
| 4 | 追求最大吞吐 | 追求 **SLO 下的最大吞吐（goodput）**。吞吐和延迟是权衡关系，不存在脱离 SLO 的"最优配置" |

### 0.2 分层模型

把服务拆成 5 层，每一层都有自己的"瓶颈信号"和"可调旋钮"。**分析时从上往下走，不要跳层**：

```
L0  目标层     SLO / goodput / 单卡成本
     ↑  "够不够快？"
L1  负载层     QPS、ISL/OSL 分布、并发数、到达模式
     ↑  "要处理多少活？"
L2  调度层     running/waiting、batch 大小、token 预算、KV 占用、抢占
     ↑  "活怎么被安排？"
L3  执行层     单步耗时、CPU/GPU 是否重叠、CUDA Graph、kernel 效率
     ↑  "一步跑得多快？"
L4  硬件层     GPU 利用率、显存带宽、KV 容量、TP/PP 通信开销
```

一个关键原则：**L0~L1 决定你要什么，L2 决定你能不能达到，L3~L4 决定上限在哪**。绝大多数线上问题出在 L2（活没安排好），而不是 L4（硬件不够快）。

### 0.3 五个必须记住的公式

分析时反复用到，先记住：

1. **系统吞吐** ≈ 每步处理的 token 总数 ÷ 单步耗时
   （对应指标：`vllm:iteration_tokens_total` ÷ step time）
2. **单步耗时** = 有重叠时 `max(CPU 调度, GPU 执行)`；无重叠时两者**相加**
   → 这就是为什么小模型上要开异步调度（`05`）：把相加变成取 max
3. **并发上限** = `min(max_num_seqs, KV 总 token 容量 ÷ 平均序列长度)`
   → 两个上限都要看，只调一个会误判（`05` + `03`）
4. **TTFT** = 排队时间 + prefill 时间
   **E2E** ≈ TTFT + 输出长度 × TPOT
   → vLLM 自带分解指标：`request_queue_time_seconds` / `request_prefill_time_seconds` / `request_decode_time_seconds`（`14`）
5. **KV 容量换算**：可用 token 数 ≈（显存 × `gpu_memory_utilization` − 权重 − activation）÷ 单 token KV 字节数
   → 启动日志里的 `GPU KV cache size: N tokens` 就是这个值

---

## 1. 观察清单：每一层看什么

### L0 / L1：外部观测（压测侧，最容易拿到）

| 指标 | 怎么拿 | 说明 |
| --- | --- | --- |
| P99 TTFT / TPOT / ITL / E2EL | `vllm bench serve --percentile-metrics ...` | **你的优化目标**，先看这个 |
| goodput（满足 SLO 的请求率） | `vllm bench serve --goodput ttft:500 tpot:20` | **最诚实的性能指标**：不满足 SLO 的吞吐不算数 |
| 输出 token 吞吐（tok/s） | bench 输出 / `Avg generation throughput` | 系统能力 |
| 请求吞吐（req/s） | bench 输出 | 受输出长度影响，跨负载不可比 |

> **goodput 是被严重低估的指标**。两个配置 A 吞吐高但 P99 超标、B 吞吐略低但全满足 SLO，真实业务里 B 更好。`--goodput` 直接帮你算这个数（单位是毫秒）。

### L2~L4：内部观测（引擎侧）

| 层 | 指标 | 看到什么就说明什么 |
| --- | --- | --- |
| L2 | `vllm:num_requests_running` | 远小于 `max_num_seqs` → 并发没打满，先别动 L3/L4 |
| L2 | `vllm:num_requests_waiting` | 持续 > 0 → 容量不足或准入被卡 |
| L2 | `vllm:kv_cache_usage_perc` | 接近 1 → KV 是瓶颈；很低 → KV 根本不是瓶颈，别加显存 |
| L2 | `vllm:num_preemptions` | > 0 → 已在重算，尾延迟必炸 |
| L2 | `vllm:iteration_tokens_total` | 每步实际算的 token 数，**判断 GPU 有没有吃饱的第一指标** |
| L2 | `prefix_cache_hits / prefix_cache_queries` | 低 → 大量重复 prefill |
| L3 | GPU 利用率（`nvidia-smi dmon`） | 低 + running 少 → CPU/GPU 没重叠或 batch 太小 |
| L3 | 单步耗时 vs 理论 FLOPs | 差得远 → kernel / 编译 / 图捕获有问题 |
| L4 | 显存占用 | KV 容量不足的根源 |
| L4 | TP/PP 通信耗时 | 多卡扩展效率低的原因 |

---

## 2. 归因：从症状到层级的判断树

拿到一个"不够快"的系统，按这个顺序走，每步都能排除掉一批可能性：

```
① 你的 SLO 是什么？（P99 TTFT < X ms，P99 TPOT < Y ms）
   没有 SLO 就无法判断"够不够快"，先定这个
              │
              ▼
② 在固定并发下压测，P99 满足吗？
   ├─ 满足 → 逐步加大 request-rate，直到某个 rate 开始超标
   │        → 这个临界点就是当前配置的容量。换配置重测，比谁更靠右
   └─ 不满足 → 用延迟分解指标定位是哪一段
              │
              ▼
③ 看 request_queue_time_seconds 占 E2E 的比例
   ├─ 占大头（>50%）→ 纯排队，是"容量/准入"问题（L1/L2）
   │     ├─ kv_cache_usage_perc 高（>0.9）→ KV 不够 → 看 ⑤
   │     ├─ kv 低但 running 已达 max_num_seqs → 并发配置太小 → 调 --max-num-seqs
   │     └─ kv 低且 running 也低 → 请求根本没进来，查前端（tokenize/多模态，`02`）
   │
   ├─ prefill_time 占大头 → L2 的 prefill 问题
   │     ├─ prefix 命中率低 → 前缀没对齐 / 块被驱逐（`04`）
   │     ├─ 单个超长 prompt 拖慢全部 → chunked prefill / long_prefill_token_threshold（`05`）
   │     └─ 多模态 encoder 占时间 → encoder 预算（`13`）
   │
   └─ decode（TPOT）占大头 → L3/L4
         ├─ iteration_tokens_total 低 → batch 太小，GPU 没喂饱 → L2 的 batch 配置
         ├─ GPU 利用率低但 running 高 → CPU/GPU 没重叠 → 异步调度 / CUDA Graph（`05`/`08`）
         └─ GPU 利用率已高 → 真的到算力/带宽上限 → 量化、投机解码、加卡（`07`/`09`）
              │
              ▼
④ 有没有抢占？（num_preemptions > 0）
   有 → 尾延迟的主要来源，优先解决：降 max_num_seqs、加 watermark、减 KV 占用
              │
              ▼
⑤ KV 不够怎么办？
   降并发（换延迟） / 加显存或 gpu_memory_utilization / 缩短 max_model_len /
   开前缀缓存减少实际占用 / 上 offload 或 PD 分离（`10`）
```

**最重要的一句话**：先确认 `kv_cache_usage_perc` 和 `num_requests_running` 这两个数，再决定动哪一层。它们能排除掉一半的误判。

---

## 3. 每一层能做什么：观察点 ↔ 手段对照表

### L1 负载层

| 观察 | 结论 | 能做的 |
| --- | --- | --- |
| ISL/OSL 分布 | 决定你是"prefill 型"还是"decode 型"负载 | 负载本身通常改不了，但**必须知道它是哪种**，否则后面全错 |
| 到达模式（均匀/突发） | 突发会让排队指标失真 | bench 时用 `--burstiness` 模拟 |
| 输出长度长尾 | 长尾请求独占 slot | 长度分组路由、优先级调度（`--scheduling-policy priority`） |

### L2 调度层（**收益最大的一层**）

| 观察 | 结论 | 能做的 |
| --- | --- | --- |
| running << max_num_seqs，kv 使用率低 | 并发没打开 | 调大 `--max-num-seqs` |
| iteration_tokens_total 低 | 单步 token 太少 | 调大 `--max-num-batched-tokens` |
| 单步都被长 prompt 独占、TPOT 抖动 | 长 prefill 抢预算 | 设 `--long-prefill-token-threshold` |
| num_preemptions > 0 | KV 打满在重算 | 降 `--max-num-seqs` / 加 `--watermark` |
| prefix 命中率低 | 重复计算 | 对齐前缀、检查 `--block-size`（`04`） |
| waiting 高但 kv 不满 | 准入被并发卡住 | `--max-num-seqs`（不是显存问题！） |

### L3 执行层

| 观察 | 结论 | 能做的 |
| --- | --- | --- |
| 小 batch 下单步延迟高 | launch 开销占比大 | 开 CUDA Graph（`-O3`，`08`） |
| 小模型 GPU 利用率低 | CPU 调度占时间 | `--async-scheduling`（`05`） |
| 单步时间远大于理论值 | 编译/图没生效 | 检查是否被 `--enforce-eager` 或不支持的 feature 降级（`08`） |
| decode 慢、batch 小 | 访存受限、算力闲置 | 投机解码（`07`） |

### L4 硬件层

| 观察 | 结论 | 能做的 |
| --- | --- | --- |
| 单卡放得下却用了 TP | 白付通信开销 | 改 DP（`09`） |
| 高 QPS、延迟已达标 | 要的是吞吐 | 加 DP 副本（`09`） |
| 显存吃紧 | KV 容量是瓶颈 | `--gpu-memory-utilization`、量化、缩短 `max_model_len` |
| MoE 吞吐上不去 | 专家并行没开 | `--enable-expert-parallel`（`09`） |

---

## 4. 四类典型负载的优化优先级

先判断自己的负载属于哪一类，再按下表的**顺序**尝试（顺序 = 性价比）：

| 负载 | 例子 | 主要瓶颈 | 优先级顺序 |
| --- | --- | --- | --- |
| **短 ISL + 短 OSL** | 分类、短问答、embedding 后的 rerank | 调度与 launch 开销 | ① batch 打满 ② CUDA Graph ③ 异步调度 |
| **长 ISL + 短 OSL** | RAG、摘要、信息抽取 | prefill 算力、重复计算 | ① 前缀缓存 ② chunked prefill 配置 ③ PD 分离 |
| **短 ISL + 长 OSL** | 创作、代码生成、Agent 输出 | decode 访存带宽 | ① 大 batch ② 投机解码 ③ 量化 |
| **长 ISL + 长 OSL** | 长文生成、复杂 Agent | 显存 + 两者叠加 | ① KV 容量 ② PD 分离 ③ offload（`10`） |

> 一个常见错误：拿着"长 ISL 短 OSL"的负载去调投机解码。投机解码只加速 decode，对 prefill 型负载几乎无收益，还会抢算力。

---

## 5. 实践：一套可重复的实验方法

### 5.1 关键原则

1. **固定 workload**：ISL / OSL / 并发三者固定，否则数字不可比。
2. **扫 request-rate，而不是只跑"无限速率"**。无限速率下系统必然饱和，只能测出吞吐上限，测不出延迟。
3. **一次只改一个变量**，否则不知道是谁的功劳。
4. **比较的是曲线，不是单点**：把每个配置在不同 rate 下的 (吞吐, P99 延迟) 画出来，整体靠右上的配置更好。

### 5.2 标准操作流程

```bash
# ① 起服务（一次改一个参数）
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256

# ② 另开终端，扫不同的请求速率
for rate in 2 4 8 16 32 64; do
  vllm bench serve \
    --backend openai \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 256 \
    --num-prompts 200 \
    --request-rate $rate \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 99 \
    --goodput ttft:2000 tpot:50
done
```

参数说明（都以 `vllm/benchmarks/serve.py` 为准）：

| 参数 | 作用 |
| --- | --- |
| `--random-input-len` / `--random-output-len` | 固定 ISL / OSL，做**受控**实验 |
| `--request-rate` | 请求到达速率，默认 `inf`（全速）。**扫这个才有延迟曲线** |
| `--max-concurrency` | 另一种模式：固定并发数而不是速率 |
| `--percentile-metrics` | 可选 `ttft` / `tpot` / `itl` / `e2el` / `client_queue_time` |
| `--metric-percentiles` | 默认 `99`，可设 `25,50,75,99` 看分布 |
| `--goodput` | SLO 定义，格式 `--goodput ttft:2000 tpot:50`（**空格分隔，单位毫秒**） |
| `--burstiness` | 模拟突发到达 |

### 5.3 服务端指标同步采集

压测的同时抓引擎内部指标，才能把"外部现象"和"内部原因"对上：

```bash
curl -s http://localhost:8000/metrics | grep -E \
  'vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions|iteration_tokens_total|prefix_cache_(hits|queries))'
```

同时看日志那一行（`14` §3 有完整拆解）：

```text
Avg prompt throughput: ... , Avg generation throughput: ... ,
Running: 32 reqs, Waiting: 8 reqs, GPU KV cache usage: 67.3%, Prefix cache hit rate: 43.2%
```

### 5.4 实验记录模板

每次实验填一行，**只改一个变量**：

| # | 改了什么 | rate | 输出 tok/s | P99 TTFT | P99 TPOT | goodput | running | kv usage | preemptions | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | baseline | 16 | | | | | | | | |
| 2 | `max_num_seqs` 128→256 | 16 | | | | | | | | |
| 3 | `max_num_batched_tokens` 2048→8192 | 16 | | | | | | | | |
| 4 | 开 `--async-scheduling` | 16 | | | | | | | | |

最后三列（running / kv usage / preemptions）是**归因用的**：如果改了参数但吞吐没变，看这三列就知道是卡在哪。

### 5.5 一个完整的诊断练习

建议按这个顺序自己做一遍，每一步都能对应到具体篇章：

1. 起默认配置，扫 rate，画出 baseline 曲线 → 得到容量上限
2. 看 `kv_cache_usage_perc` 和 `num_requests_running` → 判断是 KV 瓶颈还是并发配置瓶颈（`03`/`05`）
3. 故意把 `max_num_seqs` 调到很小 → 观察 waiting 堆积但 KV 用不满 → 记住这个"症状图案"
4. 故意关掉前缀缓存并发重复前缀的请求 → 观察 TTFT 与 `prompt_tokens_cached` 的变化（`04`）
5. 用超长 prompt（如 16K）压测 → 观察 TPOT 抖动，再调 `long_prefill_token_threshold` 看改善（`05`）
6. 小 batch 下开关 `-O3` → 感受 CUDA Graph 对单步延迟的影响（`08`）

做完这 6 步，你会对每个"症状图案"对应的根因形成条件反射——这就是部署经验。

---

## 6. 六种常见误判

| 误判 | 为什么错 | 正确做法 |
| --- | --- | --- |
| "GPU 利用率低所以要优化 kernel" | 利用率低通常是因为**没喂饱**（batch 小 / CPU 阻塞），不是 kernel 慢 | 先看 `iteration_tokens_total` 和 running 数 |
| "显存没用满就是没优化好" | KV 用不满往往是 `max_num_seqs` 太小 | 看 waiting 与 running 的关系（`05` §5.2） |
| "吞吐越高越好" | 超 SLO 的吞吐是无效吞吐 | 用 goodput 衡量 |
| "上 TP 一定更快" | 单卡放得下时 TP 只增加通信 | TP 是为了放不下 / 降延迟，DP 才是提吞吐（`09`） |
| "投机解码总是划算" | 高 QPS 下抢算力，反而变慢 | 看 running 是否长期打满（`07`） |
| "平均延迟达标就行" | 用户感受的是 P99 | 始终看分位数 |

---

## 7. 与其余篇章的对应关系

| 想解决 | 读 |
| --- | --- |
| 并发 / 批处理 / 切块 / 抢占 | `05`（最重要） |
| 显存与 KV 容量 | `03` |
| 重复前缀、TTFT | `04` |
| decode 太慢 | `07`、`08` |
| 多卡怎么切 | `09` |
| 请求经过哪些环节、延迟怎么分解 | `02`、`14` |
| 不确定瓶颈在哪 | `14` 的诊断对照表 → 再按 §2 的判断树走 |

**推荐路径**：先读本篇 → 再读 `05` → 然后按 §5.5 做一遍实验 → 遇到具体环节再查对应篇章。带着实验中的疑问去读源码，比从头通读有效得多。
