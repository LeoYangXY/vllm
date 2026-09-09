# 多模态输入处理与 Encoder Cache

> 适用版本：vLLM V1。多模态处理在 `vllm/multimodal/`，前端预处理在 `vllm/v1/engine/input_processor.py`，encoder 执行在 `vllm/v1/worker/mm_encoder_model_runner.py`，encoder 缓存管理在 `vllm/v1/core/encoder_cache_manager.py`。

## 0. TL;DR

- **是什么**：VLM 的图像/视频先经轻量 encoder（ViT 类）得到 embedding，再拼进 LLM 做 prefill。V1 把「encoder 计算」与「LLM 解码」解耦，并用 `EncoderCacheManager` 按 mm_hash 缓存 encoder 输出。
- **解决什么**：同图在多轮/多次请求中重复出现，重复跑 ViT 是纯浪费；大图 prefill 长时间独占 GPU。
- **怎么做**：前端 `InputProcessor` 做图像解码/预处理（CPU，不阻塞 EngineCore），带 `mm_inputs`/`mm_hashes` 进 EngineCore；调度器用 `EncoderCacheManager` 管 encoder 预算与缓存；`MMEncoderModelRunner` 跑 encoder，输出按 hash 缓存、跨请求共享。
- **收益**：同图零重复计算、显存可控、与 prefix caching 两层缓存叠加。代价是 encoder 预算与显存需 profile。

---

## 1. 场景与痛点

### 1.1 多模态请求的额外成本

VLM 把图像过 ViT 得到视觉 embedding，再作为「特殊 token 序列」拼进 LLM prompt。问题：

- **同图重复**：多轮对话里同一张图每次都重跑 ViT；同一文档的多个子问题重复编码。
- **大图占 GPU**：高分辨率图 token 数巨大，prefill 长时间独占 GPU，阻塞其他请求。
- **encoder 计算贵**：ViT 虽比 LLM 小，但批量大图仍可观。

### 1.2 为什么预处理放前端

图像解码/resize/tile 是 CPU 密集且与序列长度相关，放 EngineCore 会阻塞调度循环。V1 在前端 `InputProcessor` 完成（与 `02` 的 tokenize 同理），只把 `mm_inputs`（已处理好的张量/元数据）和 `mm_hashes`（内容 hash）传给 EngineCore。

---

## 2. 核心设计

### 2.1 流水线总览

```
前端 InputProcessor
  - 图像/视频解码、预处理、tile -> mm_inputs
  - 计算 mm_hashes（内容 hash）
  - 组装 EngineCoreRequest(mm_inputs, mm_hashes) -> EngineCore

EngineCore / Scheduler
  - EncoderCacheManager.check_and_update_cache: 该 mm_hash 是否已缓存？
  - can_allocate: encoder 预算够吗？（encoder_compute_budget）
  - 决定本步哪些请求的 encoder 要跑 / 可复用缓存

Worker / MMEncoderModelRunner.execute_model
  - 跑 ViT 得 embedding
  - 缓存到 EncoderCacheManager（按 mm_hash）
  - embedding 拼进 LLM prefill
```

### 2.2 EncoderCacheManager（按 embedding 粒度缓存）

```python
# vllm/v1/core/encoder_cache_manager.py:19
class EncoderCacheManager:
    # cached: mm_hash -> set[request_id]  (哪些请求引用该缓存)
    # num_free_slots / num_freeable_slots: 缓存容量（按 encoder embedding 数计）
    # freeable: 无请求引用的可驱逐项（LRU）
    # freed: 本次被驱逐的 mm_hash 列表
```

关键方法（真实定义行号）：

```python
# vllm/v1/core/encoder_cache_manager.py:102
def check_and_update_cache(self, request, input_id) -> bool:
    # 该 mm_hash 是否已缓存；是则把 request 加入引用集，返回 True

# vllm/v1/core/encoder_cache_manager.py:131
def can_allocate(self, num_encoder_embeds) -> bool:
    # encoder 预算（encoder_compute_budget）是否够放新 embedding

# vllm/v1/core/encoder_cache_manager.py:192
def allocate(self, request, input_id) -> None:
    # 分配缓存槽，登记 mm_hash -> request

# vllm/v1/core/encoder_cache_manager.py:220
def get_cached_input_ids(self, request) -> set[int]:
    # 返回该请求已缓存的 input_id 集合（避免重复跑 encoder）

# vllm/v1/core/encoder_cache_manager.py:224 / 262
def free_encoder_input(self, request, input_id)  /  free(self, request):
    # 释放单个 / 整个请求的 encoder 缓存引用
```

注意（源码 docstring 明确）：缓存粒度是 **encoder embedding**，不是 encoder token；文本间隔 token 不计入缓存大小。驱逐在分配时发生（无空闲槽才驱逐 `freeable` 中无引用的 LRU 项）。

### 2.3 MMEncoderModelRunner

```python
# vllm/v1/worker/mm_encoder_model_runner.py:32
class MMEncoderModelRunner(GPUModelRunner):
    # get_kv_cache_spec (line 67) / capture_model (line 70) / _dummy_run (line 73)
    # execute_model (line 91): 跑 encoder forward，产出视觉 embedding
```

encoder 是**独立 ModelRunner**，与 LLM 的 `GPUModelRunner` 分离执行（V1 在 Worker 内分别驱动），使 encoder 计算可与 LLM 的某些阶段 overlap，并按需只跑未缓存的部分。

### 2.4 mm hash 与 prefix caching 的两层缓存

- **EncoderCacheManager** 缓存的是「视觉 embedding」（按 mm_hash）：同图跨请求/多轮零重复编码。
- **Prefix Caching**（`04`）缓存的是「LLM 的 KV」：同前缀 prompt（含视觉 token 占位）跨请求复用 KV。

两者叠加：同图 + 同前缀 → 既不用重跑 encoder，也不用重算 prefill KV。mm_hash 也参与 block hash（`04` 提到），保证「文本同图不同」不误命中。

### 2.5 chunked prefill 下的多模态

大图产生大量 embedding token，prefill 很长。`Scheduler` 在调度时受 `encoder_compute_budget` / `max_num_encoder_input_tokens` 约束，把多图/大图请求切成多个 step，避免一次占满 GPU（见 `05` chunked prefill）。

---

## 3. 代码走读

### 3.1 前端预处理

```python
# vllm/v1/engine/input_processor.py:281 （process_inputs，示意）
def process_inputs(self, ...):
    # 调多模态 processor：图像解码/tile -> mm_inputs
    # 算 mm_hashes
    # 组装 EngineCoreRequest(mm_inputs=..., mm_hashes=...)
```

### 3.2 调度时查 encoder 缓存

```python
# vllm/v1/core/encoder_cache_manager.py:102 （示意）
if manager.check_and_update_cache(request, input_id):
    # 命中：跳过 encoder 计算，直接用缓存 embedding
else:
    if manager.can_allocate(num_embeds):
        manager.allocate(request, input_id)
        # 标记该 input 需要跑 encoder
```

### 3.3 encoder 执行与拼入 LLM

```python
# vllm/v1/worker/mm_encoder_model_runner.py:91 （execute_model，示意）
def execute_model(self, scheduler_output):
    # 对需要跑 encoder 的 input 跑 ViT -> embedding
    # embedding 写回，供 LLM prefill 拼入
```

---

## 4. 关键数据结构

| 结构 | 作用 | 位置 |
| --- | --- | --- |
| `EncoderCacheManager` | mm embedding 缓存与预算 | [encoder_cache_manager.py:19](../vllm/v1/core/encoder_cache_manager.py#L19) |
| `check_and_update_cache` | 查/登记命中 | [encoder_cache_manager.py:102](../vllm/v1/core/encoder_cache_manager.py#L102) |
| `can_allocate` | encoder 预算判断 | [encoder_cache_manager.py:131](../vllm/v1/core/encoder_cache_manager.py#L131) |
| `MMEncoderModelRunner` | 跑 ViT encoder | [mm_encoder_model_runner.py:32](../vllm/v1/worker/mm_encoder_model_runner.py#L32) |
| `mm_hashes` | 多模态内容 hash | `EngineCoreRequest`（`vllm/v1/engine/__init__.py:107`） |

---

## 5. 收益与代价

**收益**
- 同图跨请求/多轮零重复 encoder 计算，省算力与延迟。
- encoder 预算（`encoder_compute_budget`）约束显存与 GPU 占用，避免大图独占。
- 与 prefix caching 两层缓存叠加，长多模态对话收益大。

**代价 / 限制**
- **显存**：encoder 输出 embedding 也占显存，`cache_size` 需调。
- **profile**：启动期需 `MMProfiler` 估算 encoder 峰值显存（[vllm/multimodal/profiling.py](../vllm/multimodal/profiling.py)），否则可能 OOM。
- **与 CUDA Graph / LoRA / prefix caching**：encoder 动态、与 graph 配合需 piecewise；LoRA 不影响 encoder；prefix caching 的 mm_hash 隔离已处理。
- **驱逐**：缓存满时按 LRU 驱逐无引用项，热点图频繁重编码会有抖动。

---

## 6. 面试高频问题

**Q1：EncoderCacheManager 缓存的是什么粒度？**
A：encoder **embedding**（不是 encoder token）；文本间隔 token 不计入缓存大小（[encoder_cache_manager.py:19](../vllm/v1/core/encoder_cache_manager.py#L19) docstring 明确）。

**Q2：为什么多模态预处理放前端进程？**
A：图像解码/resize/tile 是 CPU 密集，放 EngineCore 会阻塞调度循环；放前端可与 GPU 计算并行（同 `02` 的 tokenize 思路）。

**Q3：EncoderCacheManager 和 Prefix Caching 什么关系？**
A：两层缓存——EncoderCache 缓存视觉 embedding（按 mm_hash），Prefix Caching 缓存 LLM KV（按 block hash）。同图+同前缀两者都命中，收益叠加（见 `04`）。

**Q4：encoder_compute_budget 是什么？**
A：调度器允许的 encoder 计算预算（按 embedding 数），约束每步能跑多少 encoder 输入，避免大图占满 GPU（[encoder_cache_manager.py:131](../vllm/v1/core/encoder_cache_manager.py#L131) 的 `can_allocate`）。

**Q5：同图在多轮对话里怎么零重复计算？**
A：第一轮跑 encoder 后按 mm_hash 缓存；后续轮 `check_and_update_cache` 命中（[encoder_cache_manager.py:102](../vllm/v1/core/encoder_cache_manager.py#L102)），直接用缓存 embedding。

**Q6：大图/多图请求怎么不被一步占满 GPU？**
A：调度受 `encoder_compute_budget` / `max_num_encoder_input_tokens` 约束，配合 chunked prefill 把多图请求切成多步（见 `05`）。

**Q7：MMEncoderModelRunner 和普通 GPUModelRunner 的区别？**
A：前者只跑 ViT 类 encoder 得 embedding（[mm_encoder_model_runner.py:32](../vllm/v1/worker/mm_encoder_model_runner.py#L32)），后者跑 LLM 自回归；两者在 Worker 内分离执行、可 overlap。

**Q8：mm_hash 为什么参与 KV block hash？**
A：保证「文本相同但图像不同」不会误命中 prefix cache 的 KV（见 `04`）。

**Q9：缓存满时怎么驱逐？**
A：分配时发现无空闲槽，驱逐 `freeable`（无请求引用、LRU）的 embedding（[encoder_cache_manager.py:19](../vllm/v1/core/encoder_cache_manager.py#L19) 的 `freeable`/`freed`）。

**Q10：encoder 缓存和 KV cache 争显存吗？**
A：都占显存，需通过 `cache_size` 与 `num_gpu_blocks` 一起规划；启动期 profiling 估算两者峰值（[vllm/multimodal/profiling.py](../vllm/multimodal/profiling.py)）。

---

## 7. 延伸阅读

- 实现：`vllm/multimodal/`（registry/processing/inputs/cache/profiling）、`vllm/v1/engine/input_processor.py`、`vllm/v1/core/encoder_cache_manager.py`、`vllm/v1/worker/mm_encoder_model_runner.py`
- 配套：`04-prefix-caching.md`、`05-scheduler-continuous-batching.md`、`02-request-lifecycle.md`
- 官方：https://docs.vllm.ai/en/latest/models/mm_inputs.html
