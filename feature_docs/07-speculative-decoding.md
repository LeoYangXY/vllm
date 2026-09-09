# 投机解码 Speculative Decoding

> 适用版本：vLLM V1。核心在 `vllm/v1/spec_decode/`（`metadata.py` / `eagle.py` / `medusa.py` / `ngram_proposer*.py` / `suffix_decoding.py` / `rejection_sampler.py` / `metrics.py`），调度协同在 `vllm/v1/core/sched/scheduler.py`，执行在 `vllm/v1/worker/gpu_model_runner.py`。

## 0. TL;DR

- **是什么**：用一个小而快的 draft 模型一次「提议」k 个 token，再让 target 模型**一次 forward 并行验证**这 k 个 token；接受最长匹配前缀，拒绝处重采样。
- **解决什么**：自回归 decode 是 memory-bandwidth-bound，batch 小时 GPU 算力闲置；串行逐 token 解码把延迟卡在「每 token 一次完整 forward」。
- **怎么做**：proposer 产出 draft token + 其概率 → `rejection_sampler` 用接受概率 `min(1, p_target/p_draft)` 决定接受几个 → 接受的分布**严格等于** target 模型分布（rejection sampling 保证）。
- **收益**：batch 小 + 输出长 + draft 与 target 分布接近时，吞吐接近线性提升（理想 k 倍）；且**不改变输出质量/分布**。

---

## 1. 场景与痛点

### 1.1 decode 为什么慢

自回归生成每步只产 1 个 token，要跑一次完整模型 forward。当 batch 小（低 QPS / 长输出），GPU 的算力被显存带宽限制（memory-bound）：每步只搬一个 token 的 KV 读 + 一次 matmul，大量 FLOPs 闲置。投机解码把「k 次串行 forward」压成「1 次 draft forward + 1 次 target verify forward」，让 GPU 每个 step 处理 k 个 token。

### 1.2 什么负载才划算

- **低 QPS / 小 batch**：GPU 带宽受限，verify 一次能验证多个 token。
- **长输出**：token 数多，累计收益大。
- **draft 与 target 分布接近**：接受率高。

高 QPS 下 GPU 已饱和，draft 反而抢算力、verify 的额外开销不划算，应关闭（用 `disable_by_batch_size`）。

---

## 2. 核心设计

### 2.1 数学保证（面试必考）

设 target 分布为 `p`，draft 分布为 `q`（q 易采样，如小模型/ngram）。对每个位置独立做：

- 以概率 `min(1, p(x)/q(x))` **接受** draft token `x`；
- 若拒绝（概率 `1 - min(1, p/q)`），从修正分布 `p'(x) = normalize(max(0, p(x) - q(x)))` 重采样一个 token。

可证明：接受或重采样的 token 边际分布 = `p`。即 **输出分布与只用 target 模型贪婪/采样完全一致**，仅延迟降低。

贪婪（greedy target）时：接受 draft 与 target 一致的最长前缀，再采 1 个 bonus token（rejection sampler 的 `bonus_token`，保证分布正确性的「额外一抽」）。

### 2.2 数据结构

```python
# vllm/v1/spec_decode/metadata.py:10
class SpecDecodeMetadata:
    draft_token_ids: list[list[int]]     # 每个请求的本步 draft token（树或序列）
    cu_num_draft_tokens: Tensor           # 每段 draft 数量的累积计数
    cu_num_sampled_tokens: Tensor
    max_num_draft_tokens: int
    target_logits_indices: Tensor         # 验证时哪些位置需要 target logits
    # ... bonus token / spec_decode_metadata 相关字段
```

### 2.3 Proposer 家族

| Proposer | 思路 | 位置 |
| --- | --- | --- |
| `EagleProposer` | EAGLE/EAGLE3：用 target 的隐状态 + 上一步 token 预测下一 token，树形草稿 | [vllm/v1/spec_decode/eagle.py:10](../vllm/v1/spec_decode/eagle.py#L10) |
| `MedusaProposer` | Medusa 多头并行预测多位置 | [vllm/v1/spec_decode/medusa.py:40](../vllm/v1/spec_decode/medusa.py#L40) |
| `NgramProposer` | 纯 CPU/GPU 的 n-gram 匹配（无需 draft 模型） | [vllm/v1/spec_decode/ngram_proposer.py:135](../vllm/v1/spec_decode/ngram_proposer.py#L135)、`ngram_proposer_gpu.py:316` |
| `SuffixDecodingProposer` | 后缀匹配 | [vllm/v1/spec_decode/suffix_decoding.py:35](../vllm/v1/spec_decode/suffix_decoding.py#L35) |
| `DraftModelProposer` / `LLMBaseProposer` | 用独立小模型做 draft | [vllm/v1/spec_decode/llm_base_proposer.py:71](../vllm/v1/spec_decode/llm_base_proposer.py#L71) |
| `Step3p5Proposer` / `DFlashProposer` / `Gemma4Proposer` | 厂商特定结构 | 各自文件 |

所有 proposer 继承 `SpecDecodeBaseProposer`（[vllm/v1/spec_decode/llm_base_proposer.py:71](../vllm/v1/spec_decode/llm_base_proposer.py#L71)），实现 `propose`（[vllm/v1/spec_decode/llm_base_proposer.py:510](../vllm/v1/spec_decode/llm_base_proposer.py#L510)）。

### 2.4 验证核心：Rejection Sampler

```python
# vllm/v1/sample/rejection_sampler.py:92
def forward(self, ...):
    # 输入: target logits (验证位置) + draft token ids + draft probs
    # 输出: 每个请求接受的 token 序列 + 接受长度 + bonus token
    # greedy / 随机两条路径; parse_output 解析树形 attention 结果
```

EAGLE 用**树形注意力**（tree attention）：draft 是一棵树而非一条链，验证时把树展平一次性算 target logits，再按树结构回溯最长接受路径。attention backend（如 flash-attn）需支持 tree/cascade attention。

### 2.5 调度与执行协同

- 调度器在 `schedule` 时为 spec 请求打标记（[vllm/v1/core/sched/scheduler.py:1993](../vllm/v1/core/sched/scheduler.py#L1993) 附近的 spec 分支）。
- `GPUModelRunner.execute_model` 分两段：先 `proposer.propose` 产出 draft，再 target forward 验证（draft token 拼进 batch，`gpu_input_batch.py` 存 draft token）。
- spec decode 下 draft token 还要过 structured output 校验（[vllm/v1/core/sched/scheduler.py:2425](../vllm/v1/core/sched/scheduler.py#L2425)）。
- profiling / dummy_run 时会造假的 spec 输入，确保图能捕获。

### 2.6 配置与指标

`SpeculativeConfig`（`vllm/config/speculative.py`）：`method`、`num_speculative_tokens`、`draft_model_config`、`disable_by_batch_size`、`disable_logprobs` 等。

指标（[vllm/v1/spec_decode/metrics.py](../vllm/v1/spec_decode/metrics.py)）：草稿接受率 = `num_accepted_tokens / num_draft_tokens`、接受长度分布、接受率随时间变化。

---

## 3. 代码走读

### 3.1 一个 step 的流程

```
1) propose: draft_model/EAGLE 产出 k 个 draft token（树）
2) 把 draft token 拼到 batch，target 一次 forward 算所有验证位置的 logits
3) rejection_sampler.forward: 逐请求算接受/拒绝，得接受序列 + bonus token
4) 接受长度 = n -> 本步实际产出 n 个（或 n+1 含 bonus）token
5) 更新 Request.num_output_tokens，调 grammar.accept_tokens（若有约束解码）
```

### 3.2 拒绝采样保证分布

核心在 `rejection_sampler.py:92` 的 `forward`：对每个验证位置，比较 `p_target` 与 `p_draft`，接受概率 `min(1, p_t/p_d)`；拒绝处按 `max(0, p_t - p_d)` 归一化重采。数学上保证边际分布 = target。bonus token 是「即使全接受也要多抽一个」以补齐分布（贪婪下等价于多走一步）。

### 3.3 EAGLE 树形注意力

EAGLE 的 draft 是树：不同分支代表不同可能的后续 token。验证时 TreeAttention 把整棵树作为一次 attention 的输入（用 tree 结构的 `cu_num_draft_tokens` / parent 指针），一次性拿到所有节点的 target logits，再按接受规则回溯最长一致路径。这让「一个 step 验证多个分支」成为可能，比纯链式 draft 接受率更高。

---

## 4. 关键数据结构

| 结构 | 字段 | 位置 |
| --- | --- | --- |
| `SpecDecodeMetadata` | draft_token_ids / cu_num_draft_tokens / max_num_draft_tokens | [vllm/v1/spec_decode/metadata.py:10](../vllm/v1/spec_decode/metadata.py#L10) |
| `SpecDecodeBaseProposer` | propose 基类 | [vllm/v1/spec_decode/llm_base_proposer.py:71](../vllm/v1/spec_decode/llm_base_proposer.py#L71) |
| `EagleProposer` | EAGLE 实现 | [vllm/v1/spec_decode/eagle.py:10](../vllm/v1/spec_decode/eagle.py#L10) |
| `RejectionSampler` | forward | [vllm/v1/sample/rejection_sampler.py:92](../vllm/v1/sample/rejection_sampler.py#L92) |
| `SpeculativeConfig` | 配置 | [vllm/config/speculative.py](../vllm/config/speculative.py) |

---

## 5. 收益与代价

**收益**
- 不改变输出分布（rejection sampling 保证），仅降低延迟、提升小 batch 吞吐。
- draft 模型小，propose 成本远低于 target verify；理想接受率下吞吐近 k 倍。
- ngram proposer 无需额外模型，几乎零成本，对重复文本/模板收益明显。

**代价 / 限制 / 失效场景**
- **高 QPS 下反而更慢**：GPU 已饱和，draft + verify 的额外计算抢资源。用 `disable_by_batch_size` 自动关。
- **与 async scheduling 互斥**：async 调度靠「预测状态提前调度」，spec decode 的 draft/verify 两阶段状态难预测，V1 中 spec decode 通常关闭 async scheduling。
- **与 chunked prefill / prefix caching**：spec decode 主要针对 decode 段；长 prefill 阶段仍走普通 prefill。prefix caching 不影响 spec 正确性。
- **显存 / 图**：draft 模型占额外显存；tree attention 需要 attention backend 支持，且影响 CUDA Graph 捕获（常走 piecewise 或不捕获 attention）。
- **接受率依赖 draft 质量**：draft 与 target 分布差 → 接受率低 → 收益归零甚至负。
- **bonus token 与 logprobs**：开 spec decode 时 logprobs 计算需特别处理（`disable_logprobs` 可选）。

---

## 6. 面试高频问题

**Q1：投机解码为什么不改变输出分布？**
A：rejection sampling 保证每个位置的边际分布 = target 分布。接受概率 `min(1, p_t/p_d)`，拒绝处按 `max(0, p_t - p_d)` 归一化重采（[vllm/v1/sample/rejection_sampler.py:92](../vllm/v1/sample/rejection_sampler.py#L92)）。

**Q2：bonus token 是什么、为什么需要？**
A：即使 draft 全部接受，也要多抽一个 token 来「补齐」分布（贪婪下等价于再走一步），保证采样语义与 target 完全一致。它是 rejection sampler 正确性的一部分。

**Q3：EAGLE 的树形注意力相比链式 draft 好在哪？**
A：draft 是一棵树，验证时 TreeAttention 一次算所有分支的 target logits，再回溯最长接受路径，比纯链式 draft 接受率更高（[vllm/v1/spec_decode/eagle.py:10](../vllm/v1/spec_decode/eagle.py#L10)）。

**Q4：什么时候该关投机解码？**
A：高 QPS（GPU 已饱和）、draft 与 target 分布差（接受率低）、或开了 async scheduling 时。用 `disable_by_batch_size` 按 batch 大小自动切换。

**Q5：ngram proposer 为什么几乎零成本？**
A：它不做模型前向，只用已生成 token 的 n-gram 匹配历史来猜后续，纯查表/匹配，适合重复文本、模板、few-shot（[vllm/v1/spec_decode/ngram_proposer.py:135](../vllm/v1/spec_decode/ngram_proposer.py#L135)）。

**Q6：投机解码和 CUDA Graph 冲突吗？**
A：部分冲突。tree attention 的动态形状让 attention 段难以进 graph，通常 spec decode 下 attention 走 eager/piecewise（见 `08`）。

**Q7：SpecDecodeMetadata 里 draft_token_ids 为什么是二维 list？**
A：每个请求一行，每行是该请求的 draft token 序列（或树展开）；配合 `cu_num_draft_tokens` 定位每段长度，喂给 target 一次 verify forward。

**Q8：spec decode 下 step 实际产出几个 token？**
A：接受长度 n，则产出 n 个（greedy 下含 1 个 bonus 共 n+1 个，见 `02` 中 `ModelRunnerOutput.sampled_token_ids` 是二维的原因）。

**Q9：如何衡量 spec decode 是否值得？**
A：看接受率指标（`num_accepted_tokens / num_draft_tokens`，[vllm/v1/spec_decode/metrics.py](../vllm/v1/spec_decode/metrics.py)）；接受率接近 `num_speculative_tokens` 才划算。

**Q10：spec decode 和约束解码怎么协同？**
A：draft token 先过 `grammar.validate_tokens` 过滤不合法项（[vllm/v1/core/sched/scheduler.py:2425](../vllm/v1/core/sched/scheduler.py#L2425)），再进 rejection sampler，保证输出仍合法。

**Q11：draft 模型如何与 target 对齐词表？**
A：draft 与 target 共享词表（或经 `vocab_mapping.py` 映射），proposer 把 draft 概率对齐到 target 词表维度供 rejection sampler 使用。

**Q12：为什么 draft 模型可以比 target 小很多还有效？**
A：因为验证是「并行的、一次 forward」，draft 只需近似 target 的下一步分布；即使 draft 不精确，rejection sampling 也会纠正，正确性不依赖 draft 质量，只影响接受率。

---

## 7. 延伸阅读

- 实现：`vllm/v1/spec_decode/`（metadata / eagle / medusa / ngram / suffix / rejection_sampler / metrics）
- 调度协同：`vllm/v1/core/sched/scheduler.py`（spec 分支）、`vllm/v1/worker/gpu_model_runner.py`
- 配套：`08-cuda-graph-and-torch-compile.md`、`06-sampling-logits-processor.md`
- 官方：https://docs.vllm.ai/en/latest/serving/spec_decode.html
