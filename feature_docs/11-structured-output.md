# 结构化输出 / 约束解码（Structured Output）

> 适用版本：vLLM V1。实现在 `vllm/v1/structured_output/`、`vllm/v1/core/sched/scheduler.py`、`vllm/v1/worker/gpu_model_runner.py`。

## 0. TL;DR

- **是什么**：在采样阶段用「语法/正则/JSON schema」生成的 token bitmask 屏蔽不合法 token，保证每个输出 token 都满足约束，最终结果必然符合格式。
- **解决什么**：仅靠 prompt 让模型「输出 JSON」不可靠（模型会漏字段、加多余文字、格式错）；function calling / tool use / 表单抽取要求 **100% 可解析**。
- **怎么做**：后端把 grammar 编译成「每步合法 token 集合」，产出 `vocab_size` 长度的 bitmask；ModelRunner 在采样前把 bitmask 应用到 logits（置 -inf），只从合法 token 采样。每步根据已生成 token `accept_tokens` 推进 grammar 状态。
- **收益**：输出零解析失败、无需重试；比「生成后校验再重采样」省一轮往返。默认后端 xgrammar 编译快、支持增量接受。

---

## 1. 场景与痛点

### 1.1 哪些场景必须用约束解码

- **Function calling / Tool use**：模型必须输出可被 `json.loads` 解析、且字段类型匹配 tool schema 的内容，否则调用失败。
- **JSON schema 抽取**：从非结构化文本抽取结构化字段（人名、日期、金额）。
- **分类标签**：输出必须是候选标签集合之一。
- **表单 / 模板填充**：输出必须是固定 schema。

### 1.2 为什么不能靠 prompt

- 模型会在 JSON 前后加解释性文字、漏字段、用单引号、转义出错。
- 长输出越往后越容易「跑偏」。
- 后处理校验 + 重采样延迟高、成本翻倍，且仍可能再失败。

约束解码把「格式正确性」从「模型自觉」变成「采样时刻的硬约束」——只要 bitmask 正确，输出 100% 合法。

---

## 2. 核心设计

### 2.1 后端抽象

`StructuredOutputBackend`（抽象基类，[vllm/v1/structured_output/backend_types.py:99](../vllm/v1/structured_output/backend_types.py#L99)）定义统一接口：

```python
# vllm/v1/structured_output/backend_types.py:107
def compile_grammar(self, grammar_spec, refs, tokenizer_info) -> StructuredOutputGrammar:
    # 把 schema / regex / EBNF 编译成可增量推进的 grammar 对象

# vllm/v1/structured_output/backend_types.py:129
def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
    # 预分配 [max_num_seqs, vocab_size] 的 bitmask 张量（每请求 vocab_size/8 字节）
```

内置后端：`xgrammar`（`backend_xgrammar.py`，默认）、`guidance`、`outlines`、`lm_format_enforcer`（见各自文件）。

`StructuredOutputManager`（[vllm/v1/structured_output/__init__.py:37](../vllm/v1/structured_output/__init__.py#L37)）是调度器侧的统一入口，负责持有各请求的 grammar、编译、以及产出 bitmask。

### 2.2 bitmask 如何作用到采样

核心思路：vocab 中每个 token 用一个 bit 表示是否合法。`apply_grammar_bitmask` 把非法 token 的 logit 置 `-inf`，采样只在合法 token 上做。

```python
# vllm/v1/worker/gpu_model_runner.py:4683 （示意）
if grammar_output is not None:
    apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)
```

`apply_grammar_bitmask` 在 [vllm/v1/structured_output/utils.py](../vllm/v1/structured_output/utils.py)（由 `gpu_model_runner.py:215` 导入）。它在采样前于 GPU 上修改 logits，开销是 vocab_size 量级的位运算，远小于一次 forward。

### 2.3 调度器侧协同

调度器每步为带 grammar 的请求构造 `GrammarOutput`（bitmask）：

```python
# vllm/v1/core/sched/scheduler.py:1845
def get_grammar_bitmask(self, scheduler_output) -> GrammarOutput | None:
    if not scheduler_output.has_structured_output_requests:
        return None
    structured_output_request_ids = [r for r in ... if r.use_structured_output and not r.is_prefill_chunk]
    bitmask = self.structured_output_manager.grammar_bitmask(...)
    return GrammarOutput(structured_output_request_ids, bitmask)
```

注意：**prefill chunk 阶段不做 grammar**（`not request.is_prefill_chunk`），只对 decode 步做 bitmask 约束。

### 2.4 grammar 状态推进与 spec decode 的协同

每产出一个 token，调度器让 grammar 接受并推进状态：

```python
# vllm/v1/core/sched/scheduler.py:2043
if advance_token_ids and self.structured_output_manager.should_advance(request):
    grammar = request.structured_output_request.grammar
    if not grammar.accept_tokens(...):   # 理论上不应被拒（采样已被 bitmask 约束）
        ...
```

spec decode 下，草稿 token 也必须过 grammar 校验，否则会被过滤：

```python
# vllm/v1/core/sched/scheduler.py:2425 / 2453
if self.structured_output_manager.should_advance(request):
    spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # 过滤不合法草稿
```

### 2.5 编译失败的处理

grammar 编译（尤其是复杂 schema）可能失败。V1 不阻塞整个 batch：把该请求标记为 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`，编译失败则加入 `grammar_compile_error_reqs`，最终以 per-request error 结束（[vllm/v1/core/sched/scheduler.py:2981](../vllm/v1/core/sched/scheduler.py#L2981)）。

---

## 3. 代码走读

### 3.1 请求携带 grammar

`SamplingParams`（`vllm/sampling_params.py:215`）中的 `structured_outputs` 字段描述约束。EngineCore 构造 `Request` 时建立 `structured_output_request`（类型 `StructuredOutputGrammar`）。调度器在 `add_request` 触发 `structured_output_manager` 的编译（线程池，避免阻塞调度循环）。

### 3.2 每步构造 bitmask

```python
# vllm/v1/core/sched/scheduler.py:1862 （示意）
bitmask = self.structured_output_manager.grammar_bitmask(
    scheduler_output, structured_output_request_ids, self.input_batch)  # 调各后端
```

`grammar_bitmask` 对处于 decode 步的每个 grammar 请求，基于其当前 grammar 状态算出合法 token 的 bitmask，写进统一分配的 `[max_num_seqs, vocab_size]` 张量。

### 3.3 应用到 logits

```python
# vllm/v1/worker/gpu_model_runner.py:4684
apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)
```

在 `Sampler` 之前执行：把 `logits[非法 token] = -inf`，合法 token 保持不变。随后 `Sampler` 正常采样/贪婪，结果必然合法。

### 3.4 xgrammar 为何是默认后端

- 编译快：xgrammar 把上下文无关文法编译成高效的 token 匹配机，支持 JSON schema 与 EBNF。
- 增量 `accept_tokens`：每步只推进已生成 token，O(1) 级别。
- 用线程池编译（`backend_xgrammar.py` 中的 `ThreadPoolExecutor`），不阻塞调度。
- 支持 `trim_reasoning_for_advance`：对带 thinking 的模型，可把 reasoning 内容剔除再喂给 grammar（见 [vllm/v1/core/sched/scheduler.py:2054](../vllm/v1/core/sched/scheduler.py#L2054)）。

---

## 4. 关键数据结构

| 结构 | 作用 | 位置 |
| --- | --- | --- |
| `StructuredOutputBackend` | 后端抽象（compile_grammar / allocate_token_bitmask） | [backend_types.py:99](../vllm/v1/structured_output/backend_types.py#L99) |
| `StructuredOutputManager` | 调度器侧统一入口 | [__init__.py:37](../vllm/v1/structured_output/__init__.py#L37) |
| `GrammarOutput` | 本步 bitmask + 请求 id 列表 | [vllm/v1/core/sched/output.py](../vllm/v1/core/sched/output.py) |
| `StructuredOutputGrammar` | 单请求的 grammar 状态机（accept_tokens） | `vllm/v1/structured_output/` |
| `SamplingParams.structured_outputs` | 用户侧的约束描述 | [vllm/sampling_params.py:215](vllm/sampling_params.py#L215) |

---

## 5. 收益与代价

**收益**
- 输出 100% 可解析，function calling / 抽取零失败，无需「生成-校验-重采」重试。
- bitmask 应用开销很小（vocab 量级位运算），远小于 forward。
- 与 spec decode 协同：草稿 token 先过 grammar 校验再验证。

**代价 / 限制**
- **显存/内存**：每个请求常驻 `vocab_size/8` 字节的 bitmask（~0.5MB@128k vocab），以及 per-request grammar 状态（xgrammar AST）。并发高时显存占用可观。
- **与 prefix caching 冲突**：约束解码下每个请求的合法 token 受 grammar 状态影响，但 KV 只依赖前缀 token（与 grammar 无关），前缀 KV 仍可跨请求复用——不过「命中后还要从对应 grammar 状态继续」是正确的。需注意：同一前缀 + 不同 grammar 不能共享 grammar 状态。
- **编译开销**：复杂 schema 编译慢，V1 用线程池异步编译并把请求挂 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`，避免阻塞 batch。
- **性能回退**：bitmask 会把合法 token 集合收窄，理论上不改变「在合法集合上的分布」，但实现上若强行 mask 后 renormalize 可能轻微改变采样；需配合 `logit_bias` 一起用时要小心顺序。
- **与 CUDA Graph**：grammar bitmask 应用在采样前，若采样走 graph 需确保 bitmask 作为 graph 输入（piecewise / 不在 graph 内的情况需特殊处理）。

---

## 6. 面试高频问题

**Q1：约束解码为什么比「生成后校验」好？**
A：约束解码在采样时刻就屏蔽非法 token，输出必然合法，零重试；后校验失败要整段重生成，延迟和成本翻倍且仍可能再失败。

**Q2：bitmask 是怎么作用到采样的？**
A：`apply_grammar_bitmask`（[vllm/v1/worker/gpu_model_runner.py:4683](../vllm/v1/worker/gpu_model_runner.py#L4683)）在 `Sampler` 前把非法 token 的 logit 置 `-inf`，采样只在合法集合上进行。

**Q3：为什么 prefill chunk 阶段不做 grammar？**
A：grammar 约束针对「逐 token 生成」的 decode 步；prefill chunk 是一次性算完已有 prompt 的 KV，不做 token 采样，无需 bitmask（[vllm/v1/core/sched/scheduler.py:1857](../vllm/v1/core/sched/scheduler.py#L1857) 的 `not request.is_prefill_chunk`）。

**Q4：grammar 编译失败会阻塞整个 batch 吗？**
A：不会。编译失败请求被加入 `grammar_compile_error_reqs`，以 per-request error 结束（[vllm/v1/core/sched/scheduler.py:2981](../vllm/v1/core/sched/scheduler.py#L2981)），其余请求照常。

**Q5：约束解码会改变输出分布吗？**
A：只把非法 token 概率置 0，在合法集合上保持原分布（等价于条件分布），语义上不改变「合法输出上的相对偏好」。

**Q6：spec decode 下约束解码怎么做？**
A：草稿 token 先过 `grammar.validate_tokens` 过滤不合法项（[vllm/v1/core/sched/scheduler.py:2425](../vllm/v1/core/sched/scheduler.py#L2425)），再进 rejection sampler 验证。

**Q7：为什么默认用 xgrammar？**
A：编译快、支持增量 `accept_tokens`、用线程池异步编译不阻塞调度、支持 JSON schema 与 EBNF，还能 `trim_reasoning_for_advance` 处理带 thinking 的模型。

**Q8：约束解码和 prefix caching 能共存吗？**
A：能。KV 只依赖前缀 token，与 grammar 无关，前缀 KV 可跨请求复用；只是命中后 grammar 状态要从对应位置继续推进。

**Q9：bitmask 的显存代价？**
A：每请求 `vocab_size/8` 字节（128k vocab 约 0.5MB），加上 per-request grammar 状态机。并发高时需要注意。

**Q10：StructuredOutputManager 和 StructuredOutputBackend 的分工？**
A：`Backend` 是具体引擎（xgrammar/guidance/outlines）的封装，负责编译与 bitmask；`Manager`（[vllm/v1/structured_output/__init__.py:37](../vllm/v1/structured_output/__init__.py#L37)）是调度器侧的协调者，持有各请求 grammar、触发编译、产出 `GrammarOutput`。

---

## 7. 延伸阅读

- 实现：`vllm/v1/structured_output/`（backend_types / backend_xgrammar / backend_guidance / backend_outlines / manager / utils）
- 调度侧：`vllm/v1/core/sched/scheduler.py:1845`（get_grammar_bitmask）、`vllm/v1/core/sched/output.py`（GrammarOutput）
- 采样侧：`vllm/v1/worker/gpu_model_runner.py:4683`（apply_grammar_bitmask）、`06-sampling-logits-processor.md`
- 官方：https://docs.vllm.ai/en/latest/serving/structured_outputs.html
