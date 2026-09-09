# 06 · 采样流程与 Logits Processor 深度解析

## 0. TL;DR

- **是什么**：vLLM V1 把「从 hidden states 到下一个 token」的整条链路拆成三段——`model.compute_logits`（lm_head 投影）、`Sampler.apply_logits_processors`（约束/偏置/惩罚）、`Sampler.sample`（温度 + top-k/top-p/min-p + 随机采样）。
- **解决什么**：让 min_tokens / logit_bias / bad_words / 惩罚 / grammar bitmask 这些**逐请求异构**的约束，能在**一个 batched `[B, V]` 张量**上完成，而不是退化成 per-request Python 循环。
- **怎么做**：所有 per-request 参数都被预先摊成 pinned CPU 侧数组 + 按批异步 H2D 的设备张量（`SamplingMetadata`），processor 只维护「行索引 → 稀疏索引张量」，用 `index_put_` / `masked_fill_` 原地改 logits；整个过程零 CPU-GPU 同步。
- **关键收益**：logits `[B, V]` 永远不离开 GPU（B=256、V=128k、fp32 就是 131 MB）；绝大多数请求走 fast-path（无惩罚 / 无 top-p / 无 logprobs 时直接跳过整段算子）；top-k/top-p 有 FlashInfer / Triton pivot 两条免排序实现。
- **代价**：processor 的状态维护是「persistent batch + BatchUpdate(added/removed/moved)」的心智负担；per-request seed 会让采样退化成逐请求循环，并强制从 FlashInfer 回落到 native 路径。

---

## 1. 场景与痛点

### 1.1 一次 decode step 里，采样处在什么位置

decode step 的尾端是这样的（以 GPU model runner 为例）：

```
模型 forward
  → hidden_states [num_tokens, H]
  → sample_hidden_states = hidden_states[logits_indices]     # 只取需要采样的行
  → logits = model.compute_logits(sample_hidden_states)      # [num_sampled, V]
  → apply_grammar_bitmask(...)                               # 结构化输出（见 11 篇）
  → Sampler.forward(logits, sampling_metadata)               # ← 本篇主角
  → SamplerOutput(sampled_token_ids, logprobs_tensors)
  → bookkeeping / 回传 scheduler
```

注意第一步：「只取需要采样的行」。prefill 时 `num_tokens` 可以是几万，但真正需要出 logits 的只有每个 request 的最后 1 个位置（+ 投机解码的 draft 位置）。这一步在 [vllm/v1/worker/gpu_model_runner.py:4584](../vllm/v1/worker/gpu_model_runner.py#L4584)：

```4584:4604:vllm/v1/worker/gpu_model_runner.py
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_tokens_padded
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    logits = self.model.compute_logits(sample_hidden_states)
```

> **面试点**：vLLM 里**没有**一个叫 `LogitsProcessor` 的类负责 lm_head。`LogitsProcessor`（[vllm/v1/sample/logits_processor/interface.py:60](../vllm/v1/sample/logits_processor/interface.py#L60)）是「对已算出的 logits 做变换」的插件抽象；lm_head 投影是模型自己的 `compute_logits`（如 [vllm/model_executor/models/llama.py:420](../vllm/model_executor/models/llama.py#L420) 附近的 `self.logits_processor(self.lm_head, hidden_states)`，那里的 `self.logits_processor` 是 `vllm/model_executor/layers/logits_processor.py` 的模块，跟 V1 的 logits processor 体系**不是一回事**，别混。

随后进入采样（[vllm/v1/worker/gpu_model_runner.py:3745](../vllm/v1/worker/gpu_model_runner.py#L3745)）：

```3745:3774:vllm/v1/worker/gpu_model_runner.py
    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        # Update output token ids with tokens sampled in last step
        # if async scheduling and required by current sampling params.
        self.input_batch.update_async_output_token_ids()
        if spec_decode_metadata is None:
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        # Update spec_token_ids with real draft tokens from pre step only when
        # output_token_ids is needed (penalties or bad_words are in use).
        if self.use_async_scheduling and self._draft_token_req_ids is not None:
            draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
            self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)

        draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            draft_probs,
            logits,
            sampling_metadata,
        )
        return sampler_output
```

### 1.2 痛点清单

1. **Vocab 维度的代价**：`[B, V]` 中 V 通常是 32k~200k。任何一次全 vocab 的 sort / softmax 都是主要成本；任何一次 `[B, V]` 的 D2H 拷贝都是几百 MB/s 量级的浪费。所以「采样必须在 GPU 上做」。
2. **请求级参数高度异构**：同一个 batch 里可能同时有 greedy、temperature=0.7+top_p、带 min_tokens、带 logit_bias、带 frequency_penalty 的请求。朴素做法是 for 循环逐请求处理，O(B) 次 kernel launch。
3. **同步是杀手**：async scheduling 下，任何一次 GPU→CPU 同步（比如 `.item()`、`.max()` 拿标量）都会打断流水线。所以 vLLM 里到处能看到 `async_tensor_h2d` / `non_blocking=True` / `pin_memory`。
4. **状态要跟着 persistent batch 走**：V1 的 `InputBatch` 是持久化的，请求在 batch 内的 index 会因为 remove/swap 而变化。logits processor 的 per-request 状态必须能跟着「added / removed / moved」增量更新。

---

## 2. 核心设计

### 2.1 分层架构

```
                     ┌──────────────────────────────────────────────────┐
                     │ InputBatch (vllm/v1/worker/gpu_input_batch.py)   │
                     │  · temperature/top_p/top_k/min_p 的 pinned CPU   │
                     │    数组 + 设备张量                                │
                     │  · BatchUpdateBuilder: added/removed/moved       │
                     │  · generators: dict[int, torch.Generator]        │
                     └───────────────┬──────────────────────────────────┘
                                     │ refresh_metadata()
                                     ▼
                          ┌────────────────────────┐
                          │  SamplingMetadata      │  ← 纯 dataclass，每步重建
                          │  (v1/sample/metadata.py)│
                          └──────────┬─────────────┘
                                     │
        ┌────────────────────────────┼─────────────────────────────┐
        ▼                            ▼                             ▼
┌───────────────────┐   ┌────────────────────────┐   ┌────────────────────┐
│ apply_logits_     │   │ LogitsProcessors       │   │ TopKTopPSampler    │
│   processors()    │──▶│  · non_argmax_invariant│   │  · temperature     │
│  allowed / bad /  │   │    (min_tokens, bias)  │   │  · argmax_invariant│
│  penalties /      │   │  · argmax_invariant    │   │    (min_p)         │
│  thinking budget  │   │    (min_p)             │   │  · top-k / top-p   │
└───────────────────┘   └────────────────────────┘   │  · random_sample   │
                                                     └────────────────────┘
        │                                                     │
        └──────────────────────┬──────────────────────────────┘
                               ▼
                    SamplerOutput(sampled_token_ids [B,1] int32,
                                  logprobs_tensors: LogprobsTensors)
```

### 2.2 三个核心抽象

| 抽象 | 位置 | 职责 |
|---|---|---|
| `SamplingMetadata` | [vllm/v1/sample/metadata.py:14](../vllm/v1/sample/metadata.py#L14) | 一次 step 的采样输入快照，含所有设备张量 + fast-path 标志 |
| `LogitsProcessor` (ABC) | [vllm/v1/sample/logits_processor/interface.py:60](../vllm/v1/sample/logits_processor/interface.py#L60) | 插件式 logits 变换：`apply` / `update_state` / `is_argmax_invariant` |
| `LogitsProcessors` | [vllm/v1/sample/logits_processor/state.py:148](../vllm/v1/sample/logits_processor/state.py#L148) | 把 processor 按 `argmax_invariant` 分成两条链，greedy 时只跑一条 |
| `TopKTopPSampler` | [vllm/v1/sample/ops/topk_topp_sampler.py:77](../vllm/v1/sample/ops/topk_topp_sampler.py#L77) | 温度无关部分之后的截断 + 随机采样，按平台分派实现 |
| `RejectionSampler` | [vllm/v1/sample/rejection_sampler.py:38](../vllm/v1/sample/rejection_sampler.py#L38) | 投机解码的验证；复用 `Sampler` 出 bonus token |

### 2.3 为什么分 argmax-invariant 和 non-argmax-invariant

这是一处很见功力的设计。`is_argmax_invariant()` 回答的是：**这个 processor 会不会改变 `argmax(logits)` 的结果？**

- `min_p`：把低于 `max_prob * min_p` 的位置置为 `-inf`。由于被 mask 的一定不是最大值，**不影响 argmax** → `True`。
- `logit_bias`：给特定 token 加常数，可能把它抬到最大 → `False`。
- `min_tokens`：把 stop token 置 `-inf`，可能改变 argmax → `False`。

后果在 [vllm/v1/sample/sampler.py:283](../vllm/v1/sample/sampler.py#L283)：**只有非 greedy 采样才会跑 argmax_invariant 链**。也就是说 min_p 对全 greedy 的 batch 完全不执行：

```274:303:vllm/v1/sample/sampler.py
        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )

        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Apply top_k and/or top_p.
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, processed_logprobs
```

注意最后 `torch.where(..., out=greedy_sampled)`：**用一个 `torch.where` 把 greedy 行和 random 行合并**，而不是分支执行。这让「混合 greedy/random batch」只付一次随机采样的代价。

---

## 3. 代码走读

### 3.1 Step 0：构造 `SamplingMetadata`

`InputBatch.refresh_metadata()` 每步调用（[vllm/v1/worker/gpu_input_batch.py:838](../vllm/v1/worker/gpu_input_batch.py#L838)）。它先把增量变更交给所有 logits processor，再决定是否重建 metadata：

```838:861:vllm/v1/worker/gpu_input_batch.py
    def refresh_metadata(self):
        """Apply any batch updates to sampling metadata."""

        if self.is_pooling_model:
            batch_changed = self.batch_update_builder.reset()
            if batch_changed:
                self.sampling_metadata = self._make_sampling_metadata()
            return

        # For non-pooling models - generate and apply logitsprocs update;
        # reset batch update tracking.
        # Update sampling metadata if batch state is changed.
        batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
        if self.thinking_budget_state_holder is not None and batch_update:
            self.thinking_budget_state_holder.sync_batch(batch_update)
        for logit_proc in self.logitsprocs.all:
            logit_proc.update_state(batch_update)
        if batch_update:
            self.sampling_metadata = self._make_sampling_metadata()

        num_reqs = self.num_reqs
```

`_make_sampling_metadata`（[vllm/v1/worker/gpu_input_batch.py:858](../vllm/v1/worker/gpu_input_batch.py#L858)）就是「fast-path 判定」的落地处：**任何一个维度上「整个 batch 都没人用」就把对应张量设为 `None` 或直接不拷贝**：

```858:920:vllm/v1/worker/gpu_input_batch.py
    def _make_sampling_metadata(self) -> SamplingMetadata:
        num_reqs = self.num_reqs
        if not self.all_greedy:
            temperature = copy_slice(
                self.temperature_cpu_tensor, self.temperature, num_reqs
            )
        else:
            temperature = None
        if not self.no_top_p:
            copy_slice(self.top_p_cpu_tensor, self.top_p, num_reqs)
        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)

        if not self.no_penalties:
            # Since syncing these tensors is expensive only copy them
            # if necessary i.e. if there are requests which require
            # penalties to be applied during sampling.
            copy_slice(
                self.frequency_penalties_cpu_tensor, self.frequency_penalties, num_reqs
            )
            copy_slice(
                self.presence_penalties_cpu_tensor, self.presence_penalties, num_reqs
            )
            copy_slice(
                self.repetition_penalties_cpu_tensor,
                self.repetition_penalties,
                num_reqs,
            )

        needs_prompt_token_ids = (
            not self.no_penalties
            or self.logits_processing_needs_token_ids[:num_reqs].any()
        )
        # The device prompt tokens are used only for applying penalties or
        # pooling methods that explicitly request GPU token IDs.
        # Hence copy these tensors only when there are requests which
        # need penalties/step_pooler to be applied.
        prompt_token_ids_cpu = (
            self._make_prompt_token_ids_cpu_tensor() if needs_prompt_token_ids else None
        )
        prompt_token_ids = (
            prompt_token_ids_cpu.to(device=self.device, non_blocking=True)
            if prompt_token_ids_cpu is not None
            else None
        )

        # Only set output_token_ids if required by the current requests'
        # sampling parameters.
        holder = self.thinking_budget_state_holder
        thinking_budget_tracks_reqs = (
            holder is not None and holder.has_tracked_requests()
        )
        needs_output_token_ids = (
            not self.no_penalties
            or bool(self.bad_words_token_ids)
            or self.logitsprocs_need_output_token_ids
            or thinking_budget_tracks_reqs
        )
        output_token_ids = (
            cast(list[list[int]], self.req_output_token_ids)
            if needs_output_token_ids
            else []
        )
```

这些 `no_*` 全部是「计数器为 0」的 property（[vllm/v1/worker/gpu_input_batch.py:1116](../vllm/v1/worker/gpu_input_batch.py#L1116)）：

```1116:1153:vllm/v1/worker/gpu_input_batch.py
    @property
    def all_greedy(self) -> bool:
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        return len(self.greedy_reqs) == 0

    @property
    def no_top_p(self) -> bool:
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        return len(self.top_k_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        return (
            len(self.presence_penalties_reqs) == 0
            and len(self.frequency_penalties_reqs) == 0
            and len(self.repetition_penalties_reqs) == 0
        )

    @property
    def no_thinking_budget(self) -> bool:
        return (
            self.thinking_budget_state_holder is None
            or len(self.thinking_token_budget_reqs) == 0
        )

    @property
    def max_num_logprobs(self) -> int | None:
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    @property
    def no_allowed_token_ids(self) -> bool:
        return len(self.has_allowed_token_ids) == 0
```

> `max_num_logprobs` 是**取 batch 内最大值**（不是 sum），所以一个请求要 20 个 logprobs，整个 batch 都会按 20 来 topk —— 这是 logprobs 会拖慢整个 batch 的原因。

`SamplingMetadata` 本体（[vllm/v1/sample/metadata.py:14](../vllm/v1/sample/metadata.py#L14)）：

```14:55:vllm/v1/sample/metadata.py
@dataclass
class SamplingMetadata:
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool

    top_p: torch.Tensor | None
    top_k: torch.Tensor | None

    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: int | None

    no_penalties: bool
    prompt_token_ids: torch.Tensor | None
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: list[list[int]]

    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    allowed_token_ids_mask: torch.Tensor | None

    # req_index -> bad_words_token_ids
    bad_words_token_ids: dict[int, list[list[int]]]

    # Loaded logits processors
    logitsprocs: LogitsProcessors

    # Specific token IDs to compute logprobs for (more efficient than full vocab)
    # When set, logprobs are computed only for these token IDs using gather
    # req_index -> list of token IDs to get logprobs for
    logprob_token_ids: dict[int, list[int]] | None = None

    # Speculative token ids
    spec_token_ids: list[list[int]] | None = None
    # When non-None, use ``holder.has_tracked_requests()`` to see if this batch applies
    # thinking-token-budget logits (holder may exist with an empty tracking set).
    thinking_budget_state_holder: ThinkingBudgetStateHolder | None = None
```

### 3.2 Step 1：`Sampler.forward` 主流程

`Sampler` 的 docstring 本身就是一份权威的执行顺序说明（[vllm/v1/sample/sampler.py:21](../vllm/v1/sample/sampler.py#L21)）：

```21:60:vllm/v1/sample/sampler.py
class Sampler(nn.Module):
    """
    A layer that samples the next tokens from the model's outputs
    with the following steps in order:

    1. If logprobs are requested:
        a) If `logprobs_mode` is `raw_logprobs`, compute logprobs
           as the final logprobs to return.
        b) If `logprobs_mode` is `raw_logits`, clone the logits
           as the final logprobs to return.
    2. Convert logits to float32.
    3. Apply allowed token ids whitelist.
    4. Apply bad words exclusion.
    5. Apply logit processors which are not argmax-invariant,
       i.e. that can impact greedy sampling.
        a) Min tokens processor
        b) Logit bias processor
    6. Apply penalties
        a) Repetition penalty
        b) Frequency penalty
        c) Presence penalty
    7. Sample the next tokens. `sample` method performs the following steps:
        a) If not `all_random`, perform greedy sampling. If `all_greedy`,
           return the greedily sampled tokens and final logprobs if requested.
        b) Apply temperature.
        c) Apply logit processors which are argmax-invariant, by default
           the min_p processor.
        d) Apply top_k and/or top_p.
        e) Sample the next tokens with the probability distribution.
        f) If `all_random` or temperature >= epsilon (1e-5), return the
           randomly sampled tokens and final logprobs if requested. Else,
           return the greedily sampled tokens and logprobs if requested.
    8. Gather the logprobs of the top `max_num_logprobs` and sampled token
       (if requested). Note that if the sampled token is within the top
       `max_num_logprobs`, the logprob will be eventually merged in
       `LogprobsProcessor` during output processing. Therefore, the
       final output may contain either `max_num_logprobs + 1` or
       `max_num_logprobs` logprobs.
    9. Return the final `SamplerOutput`.
    """
```

`forward` 的实现（[vllm/v1/sample/sampler.py:73](../vllm/v1/sample/sampler.py#L73)），**关键点：logprobs 用的是「未经任何 penalty/温度的原始 logits」**，这是 V1 与 V0 的语义差异：

```73:101:vllm/v1/sample/sampler.py
    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        num_logprobs = sampling_metadata.max_num_logprobs
        raw_logprobs: torch.Tensor | None = None
        if num_logprobs is not None or sampling_metadata.logprob_token_ids:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                if logits.dtype == torch.float32:
                    raw_logprobs = logits.clone()
                else:
                    raw_logprobs = logits.to(torch.float32)

        # Use float32 for the logits.
        logits = logits.to(torch.float32)

        logits = self.apply_logits_processors(
            logits, sampling_metadata, predict_bonus_token
        )
```

> **面试高频**：「vLLM 返回的 logprob 是加了 repetition penalty 之后的吗？」——不是。V1 显式用**原始 logits** 的 `log_softmax`（见上面 `NOTE(woosuk)`），而 V0 用的是采样时的 logits。这是破坏性变更，也是面试里能拉开差距的细节。

### 3.3 Step 2：`apply_logits_processors` —— 约束/惩罚落地

[vllm/v1/sample/sampler.py:373](../vllm/v1/sample/sampler.py#L373)：

```373:419:vllm/v1/sample/sampler.py
    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        bad_words_token_ids = sampling_metadata.bad_words_token_ids
        any_penalties_or_bad_words = (
            bool(bad_words_token_ids) or not sampling_metadata.no_penalties
        )
        output_token_ids = sampling_metadata.output_token_ids
        if predict_bonus_token and any_penalties_or_bad_words:
            # Combine base outputs with spec tokens when speculative decoding
            # is enabled.
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Apply allowed token ids.
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids:
            apply_bad_words(logits, bad_words_token_ids, output_token_ids)

        # Apply logits processors which can impact greedy sampling.
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        # Apply penalties (e.g., freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
        holder = sampling_metadata.thinking_budget_state_holder
        if holder is not None and holder.has_tracked_requests():
            # Committed outputs only; spec drafts live in ``spec_token_ids``.
            holder.update_state(
                sampling_metadata.output_token_ids,
                sampling_metadata.spec_token_ids,
                repeat_indices=None,
            )
            logits = holder.apply_to_logits(
                logits,
                predict_bonus_token,
                sampling_metadata.spec_token_ids,
            )
        return logits
```

顺序值得背：`allowed_token_ids → bad_words → non_argmax_invariant processors → penalties → thinking_budget`。

**为什么 thinking budget 放最后？** 因为它用「把 end-of-thinking token 的 logit 加成 `1e9`」的方式**强制**输出结束思考的 token（[vllm/v1/sample/thinking_budget_state.py:579](../vllm/v1/sample/thinking_budget_state.py#L579)），放在最后才能保证不被前面的 mask 或 penalty 抵消：

```553:581:vllm/v1/sample/thinking_budget_state.py
        if active_indices_cpu:
            device = logits.device
            if current_platform.is_rocm() and logits.is_contiguous():
                # Flattened index_fill avoids ROCm faults seen with 2-D
                # advanced-indexing writes on the thinking-budget path.
                vocab_size = logits.shape[1]
                flat_indices_cpu = [
                    row * vocab_size + token
                    for row, token in zip(active_indices_cpu, force_tokens_cpu)
                ]
                flat_indices = async_tensor_h2d(
                    flat_indices_cpu, dtype=torch.long, device=device
                )
                logits.view(-1).index_fill_(0, flat_indices, 1e9)
            elif current_platform.is_rocm():
                fill = logits.new_tensor(1e9)
                for row, token in zip(active_indices_cpu, force_tokens_cpu):
                    logits[row, token] = fill
            else:
                active_indices = async_tensor_h2d(
                    active_indices_cpu, dtype=torch.long, device=device
                )
                force_tokens = async_tensor_h2d(
                    force_tokens_cpu, dtype=torch.long, device=device
                )
                # Avoid CPU->GPU sync.
                fill = logits.new_full((len(active_indices_cpu),), 1e9)
                logits.index_put_((active_indices, force_tokens), fill)
```

注意这三段：**同样的语义，因平台 bug 走了三条不同实现**。ROCm 上 2-D advanced indexing 写会 fault，所以退化成 flatten 后的一维 `index_fill_`。这类「平台特化」是 vLLM 代码里非常常见的形态。

另外注意一个细节：强制值用 `1e9` 而不是 `float("inf")`。用 inf 会让后续 softmax 出现 `inf - inf = nan`；`1e9` 在 fp32 里是有限的，softmax 后概率≈1 但不会产生 NaN。

### 3.4 Step 3：penalties 与为什么走 CUDA kernel

[vllm/v1/sample/sampler.py:421](../vllm/v1/sample/sampler.py#L421) → [vllm/v1/sample/ops/penalties.py:10](../vllm/v1/sample/ops/penalties.py#L10)：

```10:38:vllm/v1/sample/ops/penalties.py
def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """
    Applies presence, frequency and repetition penalties to the logits.
    """
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)

    # In the async scheduling case, rows that won't have penalties applied may contain
    # -1 placeholder token ids. We must replace these with valid token ids so that the
    # scatter done in apply_penalties is valid.
    # NOTE(nick): The penalties implementation is currently quite inefficient and
    # will be reworked anyhow.
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

    return apply_penalties(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )
```

真正的算子在 [vllm/model_executor/layers/utils.py:43](../vllm/model_executor/layers/utils.py#L43)：

```43:81:vllm/model_executor/layers/utils.py
def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """
    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts
        are padded to the maximum prompt length within the batch using
        `vocab_size` as the padding value. The value `vocab_size` is used
        for padding because it does not correspond to any valid token ID
        in the vocabulary.
    ...
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_tokens_tensor, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs
    )

    # Apply repetition penalties as a custom op
    from vllm._custom_ops import apply_repetition_penalties

    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits
```

**为什么 repetition penalty 一定要自定义 CUDA kernel？**

repetition penalty 的定义是「若 logit > 0 则除以 r，否则乘以 r」，是一个**依赖元素符号的条件分支**。纯 PyTorch 写法需要 `torch.where(logits > 0, logits / r, logits * r)`，这会物化 `[B, V]` 的 bool mask 加两个 `[B, V]` 中间张量；再叠加 prompt_mask / output_mask 两个 `[B, V]` 掩码，峰值显存是 4~5 份 `[B, V]`。对 B=256、V=128k 的 fp32，一份是 131 MB，5 份就是 650 MB —— 直接吃掉 KV cache。融合成一个 kernel（读一次、写一次）把峰值压回 1 份。

而 frequency / presence 是**线性**的（`logits -= c * counts`），没有数据依赖分支，用 broadcasting 的 elementwise 就够，PyTorch 自己会 fuse，所以没做 kernel。这个「按算术性质决定要不要写 kernel」的取舍很值得讲。

`prompt_tokens_tensor` 用 `vocab_size` 作为 padding 值也很妙：`vocab_size` 不是合法 token id，所以 `get_token_bin_counts_and_mask` 里 scatter 到 index=vocab_size 的位置是安全的……实际上 bin count 张量宽度是 `vocab_size + 1`，多出的那一格专门接 padding 的计数，天然被丢弃。

### 3.5 Step 4：内置 Logits Processor

#### 3.5.1 接口

[vllm/v1/sample/logits_processor/interface.py:60](../vllm/v1/sample/logits_processor/interface.py#L60)：

```60:108:vllm/v1/sample/logits_processor/interface.py
class LogitsProcessor(ABC):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        """Validate sampling params for this logits processor.

        Raise ``VLLMValidationError`` (preferred) / ``ValueError`` (backward compatible)
        for invalid params. Bare ``ValueError`` is converted to ``VLLMValidationError``
        at the engine boundary so online serving returns HTTP 400.
        """
        return None

    @abstractmethod
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply LogitsProcessor to batch logits tensor.

        The updated tensor must be returned but may be
        modified in-place.
        """
        raise NotImplementedError

    @abstractmethod
    def is_argmax_invariant(self) -> bool:
        """True if logits processor has no impact on the
        argmax computation in greedy sampling.
        ...
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(
        self,
        batch_update: "BatchUpdate | None",
    ) -> None:
        """Called when there are new output tokens, prior
        to each forward pass.
        ...
        """
        raise NotImplementedError
```

`BatchUpdate`（[vllm/v1/sample/logits_processor/interface.py:36](../vllm/v1/sample/logits_processor/interface.py#L36)）是 persistent batch 的增量描述：

```36:57:vllm/v1/sample/logits_processor/interface.py
@dataclass(frozen=True)
class BatchUpdate:
    """Persistent batch state change info for logitsprocs"""

    batch_size: int  # Current num reqs in batch

    # Metadata for requests added to, removed from, and moved
    # within the persistent batch.
    #
    # Key assumption: the `output_tok_ids` list (which is an element of each
    # tuple in `added`) is a reference to the request's running output tokens
    # list; via this reference, the logits processors always see the latest
    # list of generated output tokens.
    #
    # NOTE:
    # * Added or moved requests may replace existing requests with the same
    #   index.
    # * Operations should be processed in the following order:
    #   - removed, added, moved
    removed: Sequence[RemovedRequest]
    added: Sequence[AddedRequest]
    moved: Sequence[MovedRequest]
```

那个「`output_tok_ids` 是引用而非拷贝」的假设非常重要：`MinTokensLogitsProcessor` 靠这个引用，每步 `apply` 时看到的都是**最新**的 output token 数，因此不需要每步重新登记状态。

#### 3.5.2 `MinTokensLogitsProcessor`：稀疏索引的教科书

[vllm/v1/sample/logits_processor/builtin.py:165](../vllm/v1/sample/logits_processor/builtin.py#L165)。它把「(req_idx, stop_token_id)」摊平成一维索引对，用 `index_put_` 一次性写入 `-inf`：

```194:249:vllm/v1/sample/logits_processor/builtin.py
    @staticmethod
    def add_request(
        params: SamplingParams, _: list[int] | None, output_tok_ids: list[int]
    ) -> tuple[int, Sequence[int], set[int], bool] | None:
        min_tokens = params.min_tokens
        if not min_tokens or len(output_tok_ids) >= min_tokens:
            return None
        return (
            min_tokens,
            output_tok_ids,
            params.all_stop_token_ids,
            params.structured_outputs is not None,
        )

    def update_state(self, batch_update: BatchUpdate | None):
        needs_update = process_dict_updates(
            self.min_toks, batch_update, self.add_request
        )
        if self.min_toks:
            # Check for any requests that have attained their min tokens.
            to_remove = tuple(
                index
                for index, (min_toks, out_tok_ids, _, _) in self.min_toks.items()
                if len(out_tok_ids) >= min_toks
            )
            if to_remove:
                needs_update = True
                for index in to_remove:
                    del self.min_toks[index]

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            restore_reqs: list[int] = []
            restore_tok_ids: list[int] = []
            for req, (
                _,
                _,
                stop_tok_ids,
                uses_structured_output,
            ) in self.min_toks.items():
                reqs.extend([req] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)
                if uses_structured_output:
                    restore_reqs.extend([req] * len(stop_tok_ids))
                    restore_tok_ids.extend(stop_tok_ids)
```

它还有一个和结构化输出耦合的坑：如果 `min_tokens` 把 stop token 全 mask 了，而 grammar bitmask 又把其它 token 全 mask 了，整行就会变成全 `-inf` → softmax 出 NaN。所以有 `restore_logits_slice` 机制（[vllm/v1/sample/logits_processor/builtin.py:254](../vllm/v1/sample/logits_processor/builtin.py#L254)）：

```254:288:vllm/v1/sample/logits_processor/builtin.py
    def _mask_stop_token_logits(
        self,
        logits: torch.Tensor,
        logits_slice: tuple[torch.Tensor, torch.Tensor],
        restore_logits_slice: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        restore_rows, restore_toks = restore_logits_slice
        # Advanced indexing already returns a copy, so this survives the
        # in-place masking below.
        stop_logits = logits[restore_logits_slice]
        logits.index_put_(logits_slice, self.neg_inf_tensor)

        # Stop-token entries for the same row are emitted together, so this
        # checks each row once without sorting the index tensor.
        unique_rows, inverse_indices = torch.unique_consecutive(
            restore_rows, return_inverse=True
        )
        row_needs_restore = torch.isneginf(logits[unique_rows]).all(dim=-1)
        restore_mask = row_needs_restore[inverse_indices] & torch.isfinite(stop_logits)
        restore_slice = (
            restore_rows[restore_mask],
            restore_toks[restore_mask],
        )
        logits.index_put_(restore_slice, stop_logits[restore_mask])

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit stop tokens for requests which have not reached min length.
            if self.restore_logits_slice[0].numel() == 0:
                logits.index_put_(self.logits_slice, self.neg_inf_tensor)
            else:
                self._mask_stop_token_logits(
                    logits, self.logits_slice, self.restore_logits_slice
                )
        return logits
```

「如果整行都成了 `-inf`，就把 stop token 的 logit 恢复回来」——即**宁可违反 min_tokens，也绝不产出 NaN**。这个「降级优先于崩溃」的思路在 vLLM 里反复出现。

#### 3.5.3 `MinPLogitsProcessor` 与「计数驱动的懒惰更新」

[vllm/v1/sample/logits_processor/builtin.py:23](../vllm/v1/sample/logits_processor/builtin.py#L23)。核心技巧是 `min_p_count`：只要 batch 里**没有任何一个请求**用了 min_p，就完全不构建张量、不 H2D：

```54:116:vllm/v1/sample/logits_processor/builtin.py
    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update:
            return

        needs_update = False
        # Process added requests.
        for index, params, _, _ in batch_update.added:
            min_p = params.min_p
            min_p_before = self.min_p_cpu[index]
            if min_p_before != min_p:
                needs_update = True
                self.min_p_cpu[index] = min_p
                if min_p and not min_p_before:
                    self.min_p_count += 1
                elif not min_p and min_p_before:
                    self.min_p_count -= 1

        if self.min_p_count:
            # Process removed requests.
            if batch_update.removed:
                needs_update = True
                for index in batch_update.removed:
                    if self.min_p_cpu[index]:
                        self.min_p_cpu[index] = 0
                        self.min_p_count -= 1
        ...
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits

        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
        # Adjust min_p
        adjusted_min_p = max_probabilities.mul_(self.min_p)
        # Identify valid tokens using threshold comparison
        invalid_token_mask = probability_values < adjusted_min_p
        # Apply mask using boolean indexing
        logits.masked_fill_(invalid_token_mask, -float("inf"))
        return logits
```

`apply` 里注意：先 `softmax` 再 `amax`，然后用 `masked_fill_` 写回 logits。这里有个微妙点——它 masked 的是 **logits**（写 `-inf`），但判据算的是 **概率**，语义才正确。

#### 3.5.4 `LogitBiasLogitsProcessor`：最简的一行

[vllm/v1/sample/logits_processor/builtin.py:159](../vllm/v1/sample/logits_processor/builtin.py#L159)：

```159:162:vllm/v1/sample/logits_processor/builtin.py
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits
```

`logits_slice = (req_indices[int32], tok_indices[int32])`，`bias_tensor` 是 float32。一次 advanced-index add。总共就 4 行，却是整个体系的最佳例证：**把 per-request 的稀疏参数摊成索引对，剩下的交给 PyTorch 的 gather/scatter**。

#### 3.5.5 插件与自定义 processor

- 内置列表：[vllm/v1/sample/logits_processor/__init__.py:50](../vllm/v1/sample/logits_processor/__init__.py#L50)
- 装配入口 `build_logitsprocs`：[vllm/v1/sample/logits_processor/__init__.py:185](../vllm/v1/sample/logits_processor/__init__.py#L185)，注意两个短路：pooling 模型直接不装；**开了投机解码时只装 `MinTokensLogitsProcessor`**，并 warn「min_p 和 logit_bias 不生效」。
- 想包装旧的 per-request processor（V0 风格 `vllm/logits_process.LogitsProcessor`），用 `AdapterLogitsProcessor`（[vllm/v1/sample/logits_processor/__init__.py:240](../vllm/v1/sample/logits_processor/__init__.py#L240)）。它的 `apply` 是**逐请求 Python 循环**：

```337:347:vllm/v1/sample/logits_processor/__init__.py
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.req_info:
            # Apply per-request logits processors to corresponding rows of
            # logits tensor
            for req_idx, req_lp in self.req_info.items():
                req_logits = logits[req_idx]
                new_logits = req_lp(req_logits)
                if new_logits is not req_logits:
                    # Modify logits tensor row in-place if necessary
                    logits[req_idx] = new_logits
        return logits
```

> 这就是「自定义 logits processor 很慢」的根源：O(B) 次 Python 调用 + 可能的逐行 kernel。生产上必须自己实现 batched 的 `LogitsProcessor` 子类。

### 3.6 Step 5：温度、top-k、top-p、min-p

#### 3.6.1 温度与 greedy 的合并

[vllm/v1/sample/sampler.py:228](../vllm/v1/sample/sampler.py#L228)：

```228:242:vllm/v1/sample/sampler.py
    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        # Avoid division by zero if there are greedy requests.
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)
```

`temp < 1e-5` 的行被替换成 1.0 再除，避免除零；随后 `torch.where(temp < _SAMPLING_EPS, greedy, random)` 把这些行换回 greedy 结果。

`_SAMPLING_EPS = 1e-5`（[vllm/v1/sample/sampler.py:18](../vllm/v1/sample/sampler.py#L18)）——「temperature 是否为 0」的判定阈值，不是严格的 0。

#### 3.6.2 `TopKTopPSampler`：按平台分派 forward

[vllm/v1/sample/ops/topk_topp_sampler.py:85](../vllm/v1/sample/ops/topk_topp_sampler.py#L85) 在 `__init__` 里就把 `self.forward` 绑到具体实现上（避免每步判分支）：

```85:129:vllm/v1/sample/ops/topk_topp_sampler.py
    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
    ) -> None:
        super().__init__()
        self.logprobs_mode = logprobs_mode
        self.use_fp64_gumbel = use_fp64_gumbel
        if current_platform.is_cuda():
            # FlashInfer doesn't expose post-top-k/top-p logits/logprobs,
            # so it can't be used when the configured mode requires them.
            can_use_flashinfer = (
                logprobs_mode not in PROCESSED_LOGPROBS_MODES
                and flashinfer_sampler_supported()
            )
            self.forward = (
                self.forward_cuda if can_use_flashinfer else self.forward_native
            )
        elif current_platform.is_cpu():
            arch = current_platform.get_cpu_architecture()
            # Fall back to native implementation for POWERPC and RISCV.
            # On PowerPC argmax produces incorrect output with torch.compile.
            # PR: https://github.com/vllm-project/vllm/pull/26987
            if arch in (CpuArchEnum.RISCV, CpuArchEnum.POWERPC):
                self.forward = self.forward_native
            else:
                self.forward = self.forward_cpu
        elif current_platform.is_xpu():
            if envs.VLLM_XPU_USE_SAMPLER_KERNEL:
                self.forward = self.forward_xpu
            else:
                self.forward = self.forward_native
```

CUDA 路径有**两个回退条件**（[vllm/v1/sample/ops/topk_topp_sampler.py:155](../vllm/v1/sample/ops/topk_topp_sampler.py#L155)）：

```155:182:vllm/v1/sample/ops/topk_topp_sampler.py
    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """More optimized implementation for top-k and top-p sampling."""
        # Fall back to the PyTorch-native path when FlashInfer has nothing
        # to do (no top-k / top-p filter) or when per-request generators
        # are present (unsupported by FlashInfer 0.2.3+).
        if (k is None and p is None) or generators:
            if generators:
                logger.debug_once(
                    "FlashInfer 0.2.3+ does not support "
                    "per-request generators. Falling back to "
                    "PyTorch-native implementation."
                )
            return self.forward_native(logits, generators, k, p)
        if self.use_fp64_gumbel:
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in PROCESSED_LOGPROBS_MODES, (
            "FlashInfer does not support returning logits/logprobs"
        )
        # flashinfer sampling functions expect contiguous logits.
        # In flex_attn/triton_attn fp32 inference, logits can be non-contiguous
        # because of slicing operation in logits_processor.
        return flashinfer_sample(logits.contiguous(), k, p, generators), None
```

**这是「seed 会拖慢采样」的直接代码证据**：只要 batch 里有任何一个请求带 `seed`（`generators` 非空），整个 batch 从 FlashInfer 回落到 PyTorch native 路径。

#### 3.6.3 top-p 的 sort 实现 vs 免排序实现

**PyTorch fallback（有 sort）** —— [vllm/v1/sample/ops/topk_topp_sampler.py:362](../vllm/v1/sample/ops/topk_topp_sampler.py#L362)：

```362:403:vllm/v1/sample/ops/topk_topp_sampler.py
def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    allow_cpu_sync: bool = False,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits

        if allow_cpu_sync:
            # Avoid sorting vocab for top-k only case.
            return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)
```

代价分析（这一段值得背）：
- `sort` 对 `[B, V]` 是 O(V log V)，是**整条采样链路上最贵的算子**。
- top-p 的实现用的是「**升序 cumsum**，把 `cumsum <= 1 - p` 的部分 mask 掉」——等价于保留累积概率 `> 1-p` 的尾部。`top_p_mask[:, -1] = False` 保证**至少保留一个 token**（最大值永远不被 mask）。
- 最后 `scatter_` 把排序后的值写回原位置。

**top-k only 的免排序实现** —— [vllm/v1/sample/ops/topk_topp_sampler.py:406](../vllm/v1/sample/ops/topk_topp_sampler.py#L406)：

```406:426:vllm/v1/sample/ops/topk_topp_sampler.py
def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.
    Note however that it involves a GPU->CPU sync which can be detrimental for
    async scheduling performance.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    return logits.masked_fill_(logits < top_k_mask, -float("inf"))
```

用 `topk(max_top_k)` 拿到每行的第 k 大值作为阈值，避开全 vocab sort。但 `k.max()` 会产生 **GPU→CPU 同步**（`max_top_k` 要作为 Python int 传给 `topk`），所以只在 `allow_cpu_sync=True`（即 CPU 平台）时用。这是「同步 vs 排序」的典型权衡。

**默认路径：Triton pivot 算法** —— [vllm/v1/sample/ops/topk_topp_sampler.py:349](../vllm/v1/sample/ops/topk_topp_sampler.py#L349)：

```349:359:vllm/v1/sample/ops/topk_topp_sampler.py
def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    if HAS_TRITON:
        return apply_top_k_top_p_triton(logits, k, p)

    is_cpu = current_platform.is_cpu()
    return apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=is_cpu)
```

Triton 实现在 [vllm/v1/sample/ops/topk_topp_triton.py:71](../vllm/v1/sample/ops/topk_topp_triton.py#L71)（`_topk_topp_kernel`），核心思想写在文件头：

```
Based on the paper "Qrita: High-performance Top-k and Top-p Algorithm for GPUs
using Pivot-based Truncation and Selection" By Park et al.
```

即：**不做全排序，而是用统计（均值/方差 + 查表得到 sigma）猜一个 pivot，把 outlier 收集进 buffer，再在 buffer 上做三分/二分搜索找真正的阈值**。算法细节：

1. 采样一个 block 算 `avg_logit` / `std_logit`；
2. 用 `percentile = k / V * 200` 查 `_PERCENTILE_TO_STD_TABLE`，得到 `outlier_pivot = avg + std * sigma`（[topk_topp_triton.py:155](../vllm/v1/sample/ops/topk_topp_triton.py#L155)）；
3. 一遍扫描收集所有 `> outlier_pivot` 的值到 `BUFFER`；
4. 在 buffer 上做最多 18 次三分搜索找 `k_pivot`（top-k 阈值）；
5. 若启用 top-p，再在 k 截断后的集合上做概率的二分搜索找 `p_pivot`；
6. 最后一遍扫描写 mask。

两个值得注意的工程细节：
- **`-inf` 处理**：grammar bitmask 会产生大量 `-inf`，必须排除出统计量否则 pivot 变 NaN。[vllm/v1/sample/ops/topk_topp_triton.py:137](../vllm/v1/sample/ops/topk_topp_triton.py#L137) 的注释写得很清楚：
  ```137:141:vllm/v1/sample/ops/topk_topp_triton.py
                  # Exclude -inf values (e.g. from grammar bitmasks) from
                  # statistics to avoid NaN in pivot computation.
                  finite_mask = (logits_blk0 > -float("inf")) & mask_n
                  num_finite = tl.sum(finite_mask)
  ```
- **小 batch 的 split-row pipeline**：batch ≤ 64 且只开 top-p 时，monolithic kernel（每行一个 program）会让大部分 SM 空转，于是 `_apply_topp_split` 把一行切成最多 32 片并行做部分归约（[vllm/v1/sample/ops/topk_topp_triton.py:1316](../vllm/v1/sample/ops/topk_topp_triton.py#L1316)）。

**FlashInfer 路径**（[vllm/v1/sample/ops/topk_topp_sampler.py:470](../vllm/v1/sample/ops/topk_topp_sampler.py#L470)）则用 rejection sampling 避开排序：

```470:507:vllm/v1/sample/ops/topk_topp_sampler.py
def flashinfer_sample(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    generators: dict[int, torch.Generator] = {},  # noqa
) -> torch.Tensor:
    """Sample from the logits using FlashInfer.

    Statistically, this function is equivalent to the `random_sample` function.
    However, this function is faster because it avoids sorting the logits tensor
    via rejection sampling.

    NOTE: The outputs of this function do not necessarily match the outputs of
    the `random_sample` function. It only guarantees that the outputs are
    statistically equivalent.
    """
    import flashinfer

    assert not (k is None and p is None)
    if k is None:
        # Top-p only.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, deterministic=True
        )
    elif p is None:
        # Top-k only.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, deterministic=True
        )
    else:
        # Both top-k and top-p.
        next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, deterministic=True
        )

    return next_token_ids.view(-1)
```

**注意那句 NOTE**：FlashInfer 与 native 路径的输出**不保证逐位相同**，只保证统计等价。`deterministic=True` 指的是「同一输入多次调用结果一致」，不是「与 PyTorch 一致」。

#### 3.6.4 随机采样：为什么不用 `torch.multinomial`

[vllm/v1/sample/ops/topk_topp_sampler.py:445](../vllm/v1/sample/ops/topk_topp_sampler.py#L445)：

```445:467:vllm/v1/sample/ops/topk_topp_sampler.py
def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = empty_exponential_noise_like(probs, use_fp64_gumbel)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return sample_with_exponential_noise(probs, q)
```

这是 **Gumbel-max trick**：`argmax(probs / Exp(1) 噪声)` 的分布等价于按 probs 采样。因为 `argmax` 不产生同步，而 `multinomial` 会。

`sample_with_exponential_noise`（[vllm/v1/sample/ops/topk_topp_sampler.py:436](../vllm/v1/sample/ops/topk_topp_sampler.py#L436)）就是 `probs / q` 再 argmax。`use_fp64_gumbel` 时 `q` 用 float64，降低并列概率（提升可复现性），代价是双倍显存和更慢的指数采样。

### 3.7 Step 6：logprobs

三条产出路径（[vllm/v1/sample/sampler.py:121](../vllm/v1/sample/sampler.py#L121)）：

- `num_logprobs is None` → 不产出（除非有 `logprob_token_ids`）
- `num_logprobs == -1` → 返回**全 vocab 未排序**的 logprobs
- 否则 → `gather_logprobs`：top-k + 采样 token

```309:358:vllm/v1/sample/sampler.py
    @staticmethod
    def gather_logprobs(
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.
        ...
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        # Avoid 0/1 specialization recompile on the batch dimension
        # of the compiled batched_count_greater_than. mark_unbacked makes
        # the size fully symbolic so dynamo doesn't specialize when
        # batch_size transitions from 1 to >=2.
        with gpu_sync_allowed(first_only=True):
            torch._dynamo.decorators.mark_unbacked(logprobs, 0)
            torch._dynamo.decorators.mark_unbacked(token_logprobs, 0)
            token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)
```

`batched_count_greater_than` 是 rank 的计算（[vllm/v1/sample/ops/logprobs.py:10](../vllm/v1/sample/ops/logprobs.py#L10)）：

```10:27:vllm/v1/sample/ops/logprobs.py
@torch.compile(backend=current_platform.simple_compile_backend)
def batched_count_greater_than(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Counts elements in each row of x that are greater than the corresponding
    value in values.  Use torch.compile to generate an optimized kernel for
    this function. otherwise, it will create additional copies of the input
    tensors and cause memory issues.

    Args:
        x (torch.Tensor): A 2D tensor of shape (batch_size, n_elements).
        values (torch.Tensor): A 2D tensor of shape (batch_size, 1).

    Returns:
        torch.Tensor: A 1D tensor of shape (batch_size,) with the counts.
    """
    torch._check(x.shape[0] >= 1)
    torch._check(x.shape[0] == values.shape[0])
    return (x >= values).sum(-1)
```

`(x >= values).sum(-1)` 朴素写法会物化一个 `[B, V]` 的 bool 张量；用 `torch.compile` 让它融合成一个 reduction kernel。这是「用编译器代替手写 kernel」的例子。

`LogprobsTensors` 定义（[vllm/v1/outputs.py:80](../vllm/v1/outputs.py#L80)）：三个张量 `logprob_token_ids` / `logprobs` / `selected_token_ranks`，shape 都是 `[B, num_logprobs + 1]`（采样 token 放在第 0 列）。

前端侧的归并、detokenize、UTF-8 修正在 `LogprobsProcessor`（[vllm/v1/engine/logprobs.py:29](../vllm/v1/engine/logprobs.py#L29)）。它有个很有意思的细节：byte-fallback tokenizer 会把一个多字节 UTF-8 字符切成多个 token，逐个 decode 会得到 `U+FFFD`；`_correct_decoded_token`（[vllm/v1/engine/logprobs.py:249](../vllm/v1/engine/logprobs.py#L249)）用前 4 个 token 作为上下文重新 decode 来还原正确字符串。

### 3.8 Step 7：seed 与可复现性

per-request generator 的创建在 **worker 侧**（[vllm/v1/worker/gpu_model_runner.py:1289](../vllm/v1/worker/gpu_model_runner.py#L1289)），注意是 `device=self.device` 的 CUDA generator：

```1289:1296:vllm/v1/worker/gpu_model_runner.py
            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None
```

`SamplingType.RANDOM_SEED` 的判定（[vllm/sampling_params.py:779](../vllm/sampling_params.py#L779)）：

```779:783:vllm/sampling_params.py
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM
```

generator 被存进 `InputBatch.generators`（[vllm/v1/worker/gpu_input_batch.py:430](../vllm/v1/worker/gpu_input_batch.py#L430)）：

```430:433:vllm/v1/worker/gpu_input_batch.py
            # NOTE(woosuk): self.generators should not include the requests that
            # do not have their own generator.
            if request.generator is not None:
                self.generators[req_index] = request.generator
```

**per-request generator 的三重代价**（面试必考）：

1. `random_sample` 里对每个带 seed 的请求**单独**调用 `q[i].exponential_(generator=generator)`（[topk_topp_sampler.py:463](../vllm/v1/sample/ops/topk_topp_sampler.py#L463)），源码注释直言 "This can be slow because we handle each request one by one"。
2. `forward_cuda` 检测到 `generators` 非空 → **整个 batch 从 FlashInfer 回落到 native**（[topk_topp_sampler.py:166](../vllm/v1/sample/ops/topk_topp_sampler.py#L166)）。
3. 投机解码路径同样受影响：`generate_uniform_probs` 里对带 generator 的请求逐段重采样（[rejection_sampler.py:650](../vllm/v1/sample/rejection_sampler.py#L650)）。

另外，即使不设 per-request seed，vLLM 也**不保证跨版本/跨 batch 组成的可复现性**：batch 大小变了，`q.exponential_()` 消耗的随机数流就变了。真正的「可复现」只能通过 per-request seed + 固定 batch 组成实现，而不是全局 seed。

### 3.9 Step 8：`RejectionSampler` —— 复用 `Sampler` 做投机解码验证

`RejectionSampler`（[vllm/v1/sample/rejection_sampler.py:38](../vllm/v1/sample/rejection_sampler.py#L38)）持有 `self.sampler`，用三种方式复用它：

1. **bonus token**：直接调 `self.sampler(...)`，但 `logprobs_mode_override` 强制成 `raw_logits` / `processed_logits`（因为后面要用 logits 算 accepted token 的 logprob）：

```125:152:vllm/v1/sample/rejection_sampler.py
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            # Override the logprobs mode to return logits because they are
            # needed later to compute the accepted token logprobs.
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        raw_target_logits = logits[target_logits_indices]
        # Use float32 for the target_logits.
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = raw_target_logits
```

2. **processor 复用**：`apply_logits_processors`（[vllm/v1/sample/rejection_sampler.py:289](../vllm/v1/sample/rejection_sampler.py#L289)）是 `Sampler` 版本的一份「按 draft token 展开」的变体。核心是 `repeat_indices`：penalty / allowed_token_ids 是 `[B]` 的，而 target logits 是 `[num_tokens]` 的，所以要按每个请求的 draft 数展开：

```306:332:vllm/v1/sample/rejection_sampler.py
        # Calculate indices of target logits.
        repeat_indices: torch.Tensor | None = None
        need_repeat_indices = (
            sampling_metadata.allowed_token_ids_mask is not None or has_penalties
        )
        if need_repeat_indices:
            num_requests = len(metadata.num_draft_tokens)
            num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
            original_indices = torch.arange(num_requests, device="cpu")
            repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
            repeat_indices = repeat_indices_cpu.to(
                device=logits.device, non_blocking=True
            )
            logits = self.apply_penalties(
                logits, sampling_metadata, metadata, repeat_indices, output_token_ids
            )

            # Apply allowed token ids.
            if sampling_metadata.allowed_token_ids_mask is not None:
                token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
                logits.masked_fill_(token_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
            apply_bad_words_with_drafts(
                logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
            )

        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            if isinstance(processor, MinTokensLogitsProcessor):
                logits = processor.apply_with_spec_decode(
                    logits, metadata.num_draft_tokens
                )
```

注意：spec 路径**只跑 `MinTokensLogitsProcessor`**，且用的是专门的 `apply_with_spec_decode`（[builtin.py:290](../vllm/v1/sample/logits_processor/builtin.py#L290)），因为它知道「前 n_mask 个 draft 位置还要 mask，之后的不用」：

```335:351:vllm/v1/sample/logits_processor/builtin.py
        for req_idx, min_tok, current_len, stop_toks, uses_structured_output in entries:
            remaining = min_tok - current_len
            # How many leading draft positions still need stop-token masking.
            n_mask = int(min(max(remaining, 0), num_draft_arr[req_idx]))

            if n_mask > 0:
                offset = cumsum[req_idx]
                row_indices = np.arange(offset, offset + n_mask, dtype=np.int64)
                n_stop = len(stop_toks)
                rows = np.repeat(row_indices, n_stop)
                toks = np.tile(stop_toks, n_mask)
                all_rows.append(rows)
                all_toks.append(toks)
                if uses_structured_output:
                    restore_rows.append(rows)
                    restore_toks.append(toks)
```

3. **温度/top-k/top-p 复用**：`apply_sampling_constraints`（[rejection_sampler.py:510](../vllm/v1/sample/rejection_sampler.py#L510)）—— 但注意它**只做温度和 top-k/top-p，不做 min_p 等 argmax_invariant processor**。原因：投机解码的验证必须严格用目标模型的分布，而 V1 在开启 spec decode 时 `build_logitsprocs` 只装 `MinTokensLogitsProcessor`（见 3.5.5），所以没有其它 processor 需要跑。

```533:565:vllm/v1/sample/rejection_sampler.py
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))
    ...
    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    return apply_top_k_top_p(logits, top_k, top_p)
```

4. **输出解析**：被拒绝的位置填 `PLACEHOLDER_TOKEN_ID = -1`（[rejection_sampler.py:31](../vllm/v1/sample/rejection_sampler.py#L31)），在 `parse_output`（[rejection_sampler.py:252](../vllm/v1/sample/rejection_sampler.py#L252)）里过滤掉。这样避免 CPU-GPU 同步去拿「每个请求接受了多少个」。

### 3.10 Step 9：`thinking_budget_state.py`

`ThinkingBudgetStateHolder`（[vllm/v1/sample/thinking_budget_state.py:34](../vllm/v1/sample/thinking_budget_state.py#L34)）实现「思考 token 预算」：一旦思考超过预算，强制输出 `thinking_end_token_ids`。

设计要点：
- **使能开关就是 `reasoning_config is not None`**（[line 53](../vllm/v1/sample/thinking_budget_state.py#L53)），没有单独的 flag。
- `has_tracked_requests()` 区分「holder 存在」和「batch 里真的有带 budget 的请求」（[line 74](../vllm/v1/sample/thinking_budget_state.py#L74)）—— 因为 reasoning parser 打开了但没人传 `thinking_token_budget` 是常态。
- state 是 per-request 的 dict，跟着 `BatchUpdate` 的 added/removed/moved 走（[line 83](../vllm/v1/sample/thinking_budget_state.py#L83)）。
- 与投机解码交互复杂：`force_index` 记录「在哪一个 spec 位置开始强制」，因为 spec 的多个 draft 位置可能跨过预算边界（[line 422](../vllm/v1/sample/thinking_budget_state.py#L422)）。
- 有个反直觉的回退：如果 `in_end` 状态下 rejection sampler 把 end token 拒了，要退回 think 模式重新等（[line 353](../vllm/v1/sample/thinking_budget_state.py#L353) 的注释）。

---

## 4. 关键数据结构

### 4.1 `SamplingMetadata`（[vllm/v1/sample/metadata.py:14](../vllm/v1/sample/metadata.py#L14)）

| 字段 | 类型 | 含义 | 何时为 None/空（fast-path） |
|---|---|---|---|
| `temperature` | `Tensor \| None` | `[B]` 温度 | `all_greedy` 时为 `None` |
| `all_greedy` | `bool` | batch 内无随机请求 | — |
| `all_random` | `bool` | batch 内无 greedy 请求 | — |
| `top_p` | `Tensor \| None` | `[B]` | `no_top_p` 时为 `None` |
| `top_k` | `Tensor \| None` | `[B]` int32 | `no_top_k` 时为 `None` |
| `generators` | `dict[int, Generator]` | 带 seed 的请求 | 无 seed 请求时为空 |
| `max_num_logprobs` | `int \| None` | batch 内 logprobs 最大值 | 无人要 logprobs 时为 `None` |
| `no_penalties` | `bool` | 三种惩罚全为 0/1 | — |
| `prompt_token_ids` | `Tensor \| None` | `[B, max_prompt_len]` | 无 penalty 且 processor 不需要时 `None` |
| `frequency_penalties` / `presence_penalties` / `repetition_penalties` | `Tensor` | `[B]` | 总是存在，但不拷贝 |
| `output_token_ids` | `list[list[int]]` | 逐请求已生成 token，**引用** | 无人需要时为空 list |
| `allowed_token_ids_mask` | `Tensor \| None` | `[max_B, V]` bool | `no_allowed_token_ids` 时 `None` |
| `bad_words_token_ids` | `dict[int, list[list[int]]]` | req_idx → 词 id 序列 | 空 dict |
| `logitsprocs` | `LogitsProcessors` | 两条 processor 链 | — |
| `logprob_token_ids` | `dict[int, list[int]] \| None` | 只要特定 token 的 logprob（scoring API） | — |
| `spec_token_ids` | `list[list[int]] \| None` | 投机 draft | 非 spec 时 `None` |
| `thinking_budget_state_holder` | `ThinkingBudgetStateHolder \| None` | 思考预算 | — |

### 4.2 `LogitsProcessor` 接口（[vllm/v1/sample/logits_processor/interface.py:60](../vllm/v1/sample/logits_processor/interface.py#L60)）

| 方法 | 语义 |
|---|---|
| `__init__(vllm_config, device, is_pin_memory)` | 统一构造签名（便于插件按 FQCN 加载） |
| `apply(logits) -> logits` | 原地或返回新张量 |
| `is_argmax_invariant() -> bool` | 决定是否进 greedy 跳过的那条链 |
| `update_state(batch_update)` | 每步 forward 前调用，维护 per-request 状态 |
| `validate_params(sampling_params)` | 类方法，请求入口校验，抛 `VLLMValidationError` → HTTP 400 |

### 4.3 内置 processor 对照

| Processor | 位置 | argmax-invariant | 实现手法 |
|---|---|---|---|
| `MinTokensLogitsProcessor` | [builtin.py:165](../vllm/v1/sample/logits_processor/builtin.py#L165) | `False` | `index_put_` 写 `-inf`；带 restore 兜底 |
| `LogitBiasLogitsProcessor` | [builtin.py:119](../vllm/v1/sample/logits_processor/builtin.py#L119) | `False` | advanced-index `+=` |
| `MinPLogitsProcessor` | [builtin.py:23](../vllm/v1/sample/logits_processor/builtin.py#L23) | `True` | softmax → amax → 阈值 → `masked_fill_` |
| `AdapterLogitsProcessor` | [\_\_init\_\_.py:240](../vllm/v1/sample/logits_processor/__init__.py#L240) | 子类决定 | 逐请求 Python 循环（慢） |

### 4.4 `SamplingParams` 关键字段（[vllm/sampling_params.py](../vllm/sampling_params.py)）

| 字段 | 行 | 默认 | 说明 |
|---|---|---|---|
| `temperature` | [252](../vllm/sampling_params.py#L252) | 1.0 | `< 1e-5` 视为 greedy |
| `top_p` | [256](../vllm/sampling_params.py#L256) | 1.0 | 1.0 = 不过滤 |
| `top_k` | [259](../vllm/sampling_params.py#L259) | 0 | 0/-1 = 不过滤 |
| `min_p` | [262](../vllm/sampling_params.py#L262) | 0.0 | 相对最大概率的阈值 |
| `seed` | [266](../vllm/sampling_params.py#L266) | `None` | 触发 `RANDOM_SEED`，创建 CUDA generator |
| `min_tokens` | [280](../vllm/sampling_params.py#L280) | 0 | 抑制 stop token |
| `logprobs` | [283](../vllm/sampling_params.py#L283) | `None` | `-1` 表示全 vocab |
| `logprob_token_ids` | [~297](../vllm/sampling_params.py#L297) | `None` | scoring 场景，只 gather 指定 token |
| `logit_bias` | [339](../vllm/sampling_params.py#L339) | `None` | dict[token_id, bias] |
| `allowed_token_ids` | [342](../vllm/sampling_params.py#L342) | `None` | 白名单 |
| `bad_words` | [350](../vllm/sampling_params.py#L350) | `None` | 字符串列表，预转成 `_bad_words_token_ids` |
| `thinking_token_budget` | [357](../vllm/sampling_params.py#L357) | `None` | 思考预算 |

---

## 5. 收益与代价

### 5.1 量化收益

| 优化 | 收益 |
|---|---|
| 采样全在 GPU | 避免 `[B, V]` fp32 D2H。B=256、V=128k → 131 MB/step；60 step/s 就是 7.9 GB/s 的 PCIe 流量，直接打满 PCIe 4.0 x16 的一半 |
| 只算需要采样的行的 logits | prefill 时把 `[num_tokens, V]` 降到 `[num_reqs, V]`，长 prompt 场景省掉一到两个数量级的 lm_head FLOPs 和显存 |
| `torch.where` 合并 greedy/random | 混合 batch 只付一次随机采样；`all_greedy` 时 `argmax` 后直接返回（[sampler.py:262](../vllm/v1/sample/sampler.py#L262)） |
| argmax_invariant 分流 | 全 greedy batch 完全不跑 min_p 的 softmax（省 2 次 `[B, V]` 遍历） |
| `no_*` 门控 | 无 penalty 时跳过 prompt_token_ids 的 `[B, max_len]` 构造与 H2D；无 top-k/top-p 时跳过整个 `apply_top_k_top_p` |
| Gumbel-max 替代 `multinomial` | 消除 CPU-GPU 同步 |
| Triton pivot 替代 sort | top-k 从 O(V log V) 降到约 O(V) 的若干遍扫描（常数约 4~6 遍） |
| FlashInfer rejection sampling | 完全避开排序，但只保证统计等价 |

### 5.2 代价与限制

1. **per-request seed 是性能陷阱**：三重回退（逐请求循环 / 失去 FlashInfer / spec 路径逐段重采样）。
2. **自定义 logits processor 是性能陷阱**：`AdapterLogitsProcessor.apply` 是 Python 逐行循环；且 `build_logitsprocs` 在 spec decode 开启时**直接拒绝**自定义 processor（[\_\_init\_\_.py:201](../vllm/v1/sample/logits_processor/__init__.py#L201)）。
3. **spec decode 下 min_p / logit_bias 静默失效**：只在初始化时打一条 warning（[\_\_init\_\_.py:205](../vllm/v1/sample/logits_processor/__init__.py#L205)）。生产上很容易踩。
4. **logprobs 的 batch 放大效应**：`max_num_logprobs` 取最大值，一个请求要 20 个，全 batch 都按 20 做 topk；`num_logprobs=-1` 会返回**全 vocab** 的 logprobs 张量，显存和带宽开销极大。
5. **状态机复杂度**：`added/removed/moved` 的顺序语义（removed → added → moved）必须严格遵守，否则 index 会错乱。`BatchUpdateBuilder` 甚至专门维护「removed 是否读过」的标志来防止误用（[state.py:76](../vllm/v1/sample/logits_processor/state.py#L76)）。
6. **`apply_top_k_only` 的同步**：免排序但带 GPU→CPU sync，只在 CPU 平台启用。
7. **penalties 实现被作者自己标记为「quite inefficient」**（[ops/penalties.py:27](../vllm/v1/sample/ops/penalties.py#L27)）：`_convert_to_tensors` 每步都要把 `list[list[int]]` 做成 padded 张量并 H2D，是 Python + 拷贝开销。

### 5.3 失效场景

- **全 `-inf` 行**：grammar bitmask + min_tokens 叠加可能把一整行 mask 光 → NaN。vLLM 的做法是「宁可违反约束也要恢复有限值」（见 3.5.2 的 restore 机制）。
- **投机解码 + 复杂 sampling**：验证阶段不支持 min_p/logit_bias；`build_logitsprocs` 直接把它们摘掉。
- **TPU 平台**：`_load_custom_logitsprocs` 直接返回空（[\_\_init\_\_.py:177](../vllm/v1/sample/logits_processor/__init__.py#L177)），TPU 不支持自定义 logits processor。

---

## 6. 面试高频问题

**Q1：vLLM V1 里 `compute_logits` 和 `LogitsProcessor` 分别是什么？**
`model.compute_logits(hidden_states)` 是模型侧的 lm_head 投影（如 [vllm/model_executor/models/llama.py:420](../vllm/model_executor/models/llama.py#L420) 里 `self.logits_processor(self.lm_head, hidden_states)`，那个 `logits_processor` 是 `vllm/model_executor/layers/logits_processor.py` 的层，做 TP 相关的 gather/logits 后处理）。而 V1 的 `LogitsProcessor`（[vllm/v1/sample/logits_processor/interface.py:60](../vllm/v1/sample/logits_processor/interface.py#L60)）是**对已算出的 `[B, V]` logits 做逐请求约束**的插件抽象，由 `Sampler.apply_logits_processors` 调用。两者名字撞车但完全无关。

**Q2：`Sampler.forward` 的执行顺序是什么？**
背 [sampler.py:21-59](../vllm/v1/sample/sampler.py#L21) 的 docstring：(1) 用**原始 logits** 算 logprobs → (2) 转 fp32 → (3) allowed_token_ids → (4) bad_words → (5) non_argmax_invariant processors（min_tokens、logit_bias）→ (6) penalties（repetition/frequency/presence）→ (7) sample：greedy / 温度 / argmax_invariant processors（min_p）/ top-k+top-p / 随机采样 / `torch.where` 合并 → (8) gather logprobs。thinking budget 在 (6) 之后。

**Q3：为什么 logprobs 用原始 logits 而不是采样用的 logits？**
[sampler.py:81](../vllm/v1/sample/sampler.py#L81) 的 `NOTE(woosuk)` 明确说明：V1 用未经 penalty 和温度缩放的原始 logits 的 `log_softmax`；V0 用的是采样时的 logits。所以开了 repetition penalty 后，V1 返回的 logprob 反映的是**模型原始分布**，这是与 V0 的破坏性行为差异。

**Q4：`is_argmax_invariant` 有什么用？举例说明。**
决定 processor 是否能在「全 greedy」场景下被跳过。`min_p` 返回 `True`（[builtin.py:47](../vllm/v1/sample/logits_processor/builtin.py#L47)），因为它只 mask 掉概率远低于最大值的 token，不改变 argmax；`logit_bias` 返回 `False`（[builtin.py:130](../vllm/v1/sample/logits_processor/builtin.py#L130)），`min_tokens` 返回 `False`（[builtin.py:189](../vllm/v1/sample/logits_processor/builtin.py#L189)）。分流在 [state.py:151](../vllm/v1/sample/logits_processor/state.py#L151) 的 `LogitsProcessors.__init__`，消费在 [sampler.py:283](../vllm/v1/sample/sampler.py#L283)（只给 random 采样跑）和 [sampler.py:401](../vllm/v1/sample/sampler.py#L401)（greedy 也要跑）。

**Q5：为什么 repetition penalty 用自定义 CUDA kernel，而 frequency/presence 不用？**
repetition penalty 是「logit > 0 则除以 r，否则乘以 r」的**符号依赖条件运算**（[vllm/model_executor/layers/utils.py:75](../vllm/model_executor/layers/utils.py#L75) 的 `apply_repetition_penalties`）。纯 PyTorch 需要物化多个 `[B, V]` 中间张量（bool mask + 两个分支结果），峰值显存 4~5 份 `[B, V]`；B=256、V=128k fp32 时是 650 MB。融合 kernel 读一次写一次，峰值回落到 1 份。而 frequency/presence 是纯线性的 `logits -= c * counts`（[utils.py:79-80](../vllm/model_executor/layers/utils.py#L79)），PyTorch 的 broadcasting elementwise 会自动融合，没必要写 kernel。

**Q6：top-p 有哪几种实现？各自代价？**
(1) **PyTorch sort 版** [topk_topp_sampler.py:362](../vllm/v1/sample/ops/topk_topp_sampler.py#L362)：升序 sort + cumsum，`cumsum <= 1-p` 的 mask 掉，`top_p_mask[:, -1] = False` 保证至少留一个；O(V log V)，还要一次 `scatter_` 写回。(2) **Triton pivot 版** [topk_topp_triton.py:71](../vllm/v1/sample/ops/topk_topp_triton.py#L71)：用均值/方差+查表猜 pivot，收集 outlier 后做三分/二分搜索，避免全排序；默认路径。(3) **FlashInfer** [topk_topp_sampler.py:470](../vllm/v1/sample/ops/topk_topp_sampler.py#L470)：rejection sampling，完全不排序，但**只保证统计等价，不保证与 PyTorch 逐位一致**，且不支持 per-request generator、不支持返回处理后的 logits/logprobs。(4) **top-k only 免排序版** [topk_topp_sampler.py:406](../vllm/v1/sample/ops/topk_topp_sampler.py#L406)：`topk(max_top_k)` 取阈值，但 `k.max()` 有 GPU→CPU 同步，只在 CPU 平台用。

**Q7：为什么不用 `torch.multinomial`？**
[topk_topp_sampler.py:450](../vllm/v1/sample/ops/topk_topp_sampler.py#L450) 的 docstring 直接写了：`torch.multinomial` 会引发 CPU-GPU 同步。改用 Gumbel-max trick：`argmax(probs / Exponential(1))`（[topk_topp_sampler.py:436](../vllm/v1/sample/ops/topk_topp_sampler.py#L436)）。`argmax` 不产生同步。`use_fp64_gumbel` 时用 float64 噪声提升精度/降低并列。

**Q8：per-request seed 的代价是什么？**
三重：(a) `random_sample` 里对带 generator 的请求**逐个**调用 `q[i].exponential_(generator=...)`（[topk_topp_sampler.py:463](../vllm/v1/sample/ops/topk_topp_sampler.py#L463)，源码注释 "This can be slow"）；(b) `forward_cuda` 发现 `generators` 非空就从 FlashInfer 回落到 native（[topk_topp_sampler.py:166](../vllm/v1/sample/ops/topk_topp_sampler.py#L166)），**整个 batch 一起降级**；(c) spec decode 的 `generate_uniform_probs` 也要逐段重采样（[rejection_sampler.py:650](../vllm/v1/sample/rejection_sampler.py#L650)）。

**Q9：`RejectionSampler` 怎么复用 `Sampler`？**
它持有 `self.sampler`（[rejection_sampler.py:68](../vllm/v1/sample/rejection_sampler.py#L68)），三处复用：(a) **bonus token**：`self.sampler(bonus_logits, ..., predict_bonus_token=True, logprobs_mode_override=...)`（[line 134](../vllm/v1/sample/rejection_sampler.py#L134)）；(b) **logits processors**：自己实现 `apply_logits_processors`，用 `repeat_indices = arange(B).repeat_interleave(num_draft_tokens)` 把 `[B]` 的 penalty/allowed mask 展到 `[num_tokens]`（[line 315](../vllm/v1/sample/rejection_sampler.py#L315)），并用 `MinTokensLogitsProcessor.apply_with_spec_decode` 的 spec 版本；(c) **温度/top-k/top-p**：`apply_sampling_constraints`（[line 510](../vllm/v1/sample/rejection_sampler.py#L510)）。被拒的位置填 `PLACEHOLDER_TOKEN_ID = -1`，在 `parse_output`（[line 252](../vllm/v1/sample/rejection_sampler.py#L252)）过滤，全程无 CPU-GPU 同步。

**Q10：`logits_indices` 是什么？为什么需要它？**
`hidden_states` 是 `[num_tokens, H]`，但只有部分位置需要采样（每请求最后 1 个 prefill 位置，或 spec decode 的所有 draft + bonus 位置）。`logits_indices = query_start_loc[1:] - 1` 之类的计算（[gpu_model_runner.py:2276](../vllm/v1/worker/gpu_model_runner.py#L2276)）给出这些位置，然后 `hidden_states[logits_indices]` 再送 lm_head。这一步避免了对整个 prefill 序列做 `[num_tokens, V]` 的投影——对 8k token 的 prompt 是 8000/1 ≈ 三个数量级的节省。

**Q11：`thinking_budget` 是怎么强制结束思考的？为什么用 `1e9` 而不是 `inf`？**
在 `apply_to_logits` → `_apply_forcing_to_logits`（[thinking_budget_state.py:480](../vllm/v1/sample/thinking_budget_state.py#L480)）里，用 `index_put_` 把 end-of-thinking token 的 logit 设成 `1e9`。用 `1e9` 而非 `float("inf")` 是因为 inf 会让后续 softmax 产生 `inf - inf = NaN`；`1e9` 在 fp32 下有限，softmax 后概率≈1 但不会 NaN。且它被安排在 `apply_logits_processors` 的**最后**（[sampler.py:406](../vllm/v1/sample/sampler.py#L406)），保证不被前面的 mask/penalty 抵消。

**Q12：开投机解码后哪些 sampling 参数会失效？为什么？**
`build_logitsprocs`（[logits_processor/\_\_init\_\_.py:201](../vllm/v1/sample/logits_processor/__init__.py#L201)）在 `speculative_config` 存在时，只装 `MinTokensLogitsProcessor`，并 warn "min_p and logit_bias parameters won't work with speculative decoding"；同时自定义 logits processor 直接抛 `ValueError`。根因：验证阶段用的是 `apply_sampling_constraints`（只做温度 + top-k/top-p），不带 processor 链；因为 rejection sampling 要求 target 分布是「模型真实分布」，而 processor 会破坏这个前提。

**Q13：为什么说「采样必须在 GPU 上做」？量化一下。**
`[B, V]` fp32：B=256、V=128256 → 256 × 128256 × 4 B ≈ 131 MB。若每 step 都 D2H 一次，60 step/s 就是 ~7.9 GB/s，还要加上 CPU 侧做 softmax/sort 的延迟（几十 ms 级），完全不可接受。而且 penalties、bad_words、grammar bitmask 都要在 `[B, V]` 上做元素级操作，放 CPU 上就是纯内存带宽瓶颈。所以整条链路从头到尾不离开 GPU，只有最终的 `sampled_token_ids`（`[B, 1]` int32）和 logprobs 张量回传。

**Q14：`apply_top_k_top_p` 里 `-inf` 为什么要特殊处理？**
grammar bitmask（见 11 篇）会把大量非法 token 的 logit 设成 `-inf`。Triton kernel 在做统计（均值/方差）和二分搜索时必须排除 `-inf`，否则：均值被拉到 `-inf` → `std` 变 NaN → `outlier_pivot` 变 NaN → 整行结果崩坏。代码里有两处显式处理（[topk_topp_triton.py:137](../vllm/v1/sample/ops/topk_topp_triton.py#L137) 和 [line 172](../vllm/v1/sample/ops/topk_topp_triton.py#L172)），注释直接点名 "e.g. from grammar bitmasks"。

---

## 7. 延伸阅读

**源码（本文全部链接的相对根）**

- 采样主循环：[vllm/v1/sample/sampler.py](../vllm/v1/sample/sampler.py)
- 元数据：[vllm/v1/sample/metadata.py](../vllm/v1/sample/metadata.py)
- Processor 接口：[vllm/v1/sample/logits_processor/interface.py](../vllm/v1/sample/logits_processor/interface.py)
- 内置 processor：[vllm/v1/sample/logits_processor/builtin.py](../vllm/v1/sample/logits_processor/builtin.py)
- 状态管理：[vllm/v1/sample/logits_processor/state.py](../vllm/v1/sample/logits_processor/state.py)
- top-k/top-p：[vllm/v1/sample/ops/topk_topp_sampler.py](../vllm/v1/sample/ops/topk_topp_sampler.py)、[vllm/v1/sample/ops/topk_topp_triton.py](../vllm/v1/sample/ops/topk_topp_triton.py)
- 惩罚算子：[vllm/model_executor/layers/utils.py:43](../vllm/model_executor/layers/utils.py#L43)
- 投机验证：[vllm/v1/sample/rejection_sampler.py](../vllm/v1/sample/rejection_sampler.py)
- 思考预算：[vllm/v1/sample/thinking_budget_state.py](../vllm/v1/sample/thinking_budget_state.py)
- 前端 logprobs：[vllm/v1/engine/logprobs.py](../vllm/v1/engine/logprobs.py)
- 批处理状态：[vllm/v1/worker/gpu_input_batch.py](../vllm/v1/worker/gpu_input_batch.py)

**关联文档**

- [11-structured-output.md](./11-structured-output.md) —— grammar bitmask 如何把 `-inf` 写进 logits，以及它和 min_tokens 的冲突
- [14-metrics-and-observability.md](./14-metrics-and-observability.md) —— 采样阶段的耗时如何体现到 TTFT / ITL 指标上

**论文 / 外部**

- Qrita（pivot-based top-k/top-p）：https://arxiv.org/abs/2602.01518 —— Triton kernel 的算法来源
- Accelerating Large Language Model Decoding with Speculative Sampling：https://arxiv.org/abs/2211.17192 —— `RejectionSampler` 遵循的算法
- Gumbel-max trick：https://en.wikipedia.org/wiki/Gumbel_max_trick
- FlashInfer sampling：https://docs.flashinfer.ai/api/sampling.html
- xgrammar token bitmask：https://xgrammar.mlc.ai/docs/api/python/index.html
