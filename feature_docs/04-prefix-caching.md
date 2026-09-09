# Automatic Prefix Caching（自动前缀缓存）

> 适用版本：vLLM V1。核心实现在 `vllm/v1/core/kv_cache_utils.py`、`block_pool.py`、`single_type_kv_cache_manager.py`、`kv_cache_manager.py`。

## 0. TL;DR

- **是什么**：把「相同的前缀 token 序列」算出来的 KV cache 按 block 复用，新请求命中后这段 prefill 直接跳过。
- **解决什么**：system prompt、多轮对话历史、RAG 同一文档、few-shot、并行/自洽采样等场景下，大量请求前缀完全相同，重复算 prefill 纯浪费算力、拉高 TTFT。
- **怎么做**：每个填满的 KV block 用「父 block hash + 本 block token」算出内容 hash，存进 `BlockPool` 的 hash→block 表；新请求按 block 顺序查 longest common prefix，命中的 block 直接复用、只算剩余 token。
- **收益**：命中前缀比例越高，prefill 计算量越接近「增量」，TTFT 大幅下降，GPU 吞吐上升；对长 system prompt / 多轮对话收益尤其明显。

---

## 1. 场景与痛点

### 1.1 哪些真实负载前缀高度重合

| 场景 | 重合部分 | 收益 |
| --- | --- | --- |
| 固定 system prompt + 用户问题 | system prompt 整段 | 几乎每个请求都命中 |
| 多轮对话 | 历史对话 | 第 2+ 轮只算新问题 |
| RAG / 长文档 QA | 同一篇被检索文档 | 同一文档的多个子问题全命中 |
| few-shot / 固定模板 | 示例与模板 | 命中示例段 |
| 并行采样 / 自洽采样（best-of-n） | 整个 prompt | n 个请求前缀 100% 一致 |
| 批量摘要同一长文 | 原文 | 命中原文段 |

### 1.2 为什么必须按 block 而不是整序列

KV cache 是**分页**的（见 `03-paged-attention-kv-cache.md`），物理上就是一块块 block。前缀要能复用，最自然的单位就是 block：

- 只有**填满的整块**（block_size 个 token）才值得缓存——半块无法被后续请求完整复用，也没法做内容 hash 对齐。
- 命中判断沿 block 链做「最长公共前缀」，命中几个 block 就复用几块。

### 1.3 量化收益直觉

设 system prompt 长 1000 token，block_size=16，则约 62 个 block。若 100 个并发请求共享该系统 prompt，命中后每个请求 prefill 从 1000 token 降到 ~几十 token，prefill FLOPs 降到 1/20 以下，TTFT 同比例下降，且省下的 KV 块让并发上限更高。

---

## 2. 核心设计

### 2.1 三个硬约束（必须记住，面试高频）

1. **只有填满的整块才缓存**。半块不进缓存表。
2. **hash 是「内容哈希」，靠父 hash 链防冲突**。第 i 块的 hash = H(第 i-1 块的 hash, 第 i 块的 token)。这样任意位置 token 变了，后续所有块 hash 都变，天然隔离。
3. **block 内 token 必须完全一致才命中**（即要求生成是 deterministic 的、且同一前缀的 token 序列确定）。不同采样结果不会让同一前缀产生不同 KV（KV 只依赖前向，不依赖采样），所以前缀 KV 可安全复用。

### 2.2 hash 计算链路

- `init_none_hash`：初始化「空前缀」的 hash 种子（[vllm/v1/core/kv_cache_utils.py:148](../vllm/v1/core/kv_cache_utils.py#L148)）。
- `hash_block_tokens`：给定父 hash + 本 block token，算本 block 的 hash（[vllm/v1/core/kv_cache_utils.py:622](../vllm/v1/core/kv_cache_utils.py#L622)）。
- `get_request_block_hasher`：从句子的 block 序列构造一个 hasher，按块顺序产出每块的 `BlockHashType`（[vllm/v1/core/kv_cache_utils.py:796](../vllm/v1/core/kv_cache_utils.py#L796)）。
- 多模态时，block hash 还会混入 `mm_hash`（图像/视频的 hash），保证「文本相同但图像不同」不会误命中。

### 2.3 BlockPool：hash → block 的注册表 + 引用计数

`BlockPool`（`vllm/v1/core/block_pool.py`）持有：

- `block_hash_to_block_id`：内容 hash → 物理 block id 表（缓存命中查询）。
- `free_block_queue`：空闲 block 队列，按 LRU 语义回收。
- `ref_cnt`：每个 block 的引用计数（被多少请求占用）。正在被占用的 block 不能被驱逐。

关键方法：

```python
# vllm/v1/core/block_pool.py:198
def get_cached_block(self, block_hash: BlockHashType) -> Optional[int]:
    # 查 hash 表，命中返回 block_id，未命中返回 None

# vllm/v1/core/block_pool.py:225
def cache_full_blocks(self, request_id, blocks, num_cached_blocks, ...
                      extra_hash: BlockHashType | None):
    # 把「填满且尚未缓存」的 block 登记进 hash 表，ref_cnt 归请求所有
```

### 2.4 命中查询：find_longest_cache_hit

每个 KV cache group 由 `SingleTypeKVCacheManager` 管理（full attention / sliding window / mamba 不同实现）。命中查询：

```python
# vllm/v1/core/single_type_kv_cache_manager.py:578
def find_longest_cache_hit(self, request, num_computed_tokens, ...) -> tuple[int, ...]:
    # 从 num_computed_tokens 起，沿 block 顺序查 BlockPool：
    #   父 hash + 本 block token -> hash -> get_cached_block
    # 命中就继续，不命中就停。返回命中的 block 数（公共前缀长度）
```

同文件还有 `get_num_common_prefix_blocks`（[vllm/v1/core/single_type_kv_cache_manager.py:561](../vllm/v1/core/single_type_kv_cache_manager.py#L561)），用于已 RUNNING 请求与已有 block 串的公共前缀计算（配合 prefix cache 的「运行中请求也能被新请求复用其前缀」）。

### 2.5 与调度器协同

调度器在 `add_request` / `schedule` 时调用这些查询，把命中的前缀块数填进 `Request.num_cached_tokens` / `num_external_computed_tokens`：

- 命中 k 个 block → 该请求 prefill 只需算「k×block_size 之后的 token」。
- 与 chunked prefill 叠加：即使前缀没全命中，长剩余部分也能被切成 chunk（见 `05`）。
- prefix cache 命中率指标在 `vllm/v1/core/kv_cache_metrics.py` 与 `gpu_model_runner` 中统计，暴露为 `gpu_prefix_cache_hit_rate`（见 `14`）。

---

## 3. 代码走读

### 3.1 请求进来：算 hash 链

```python
# vllm/v1/core/kv_cache_utils.py:622 （示意）
def hash_block_tokens(
    hash_fn,
    parent_hash: BlockHashType,
    curr_block_tokens: list[int],          # 本 block 的 token
    extra_hash: BlockHashType | None = None,  # 多模态 hash
) -> BlockHashType:
    # H = hash_fn(父hash的字节 + 本block token字节 + extra_hash字节)
    return hash_fn(parent_hash + curr_block_tokens + extra_hash)
```

哈希输入包含 **父 hash**，所以前缀任意处改动都会向后传播，天然防冲突。

### 3.2 调度时查命中

```python
# vllm/v1/core/single_type_kv_cache_manager.py:578 （示意）
def find_longest_cache_hit(self, request, num_computed_tokens, ...):
    block_hashes = request.block_hashes
    num_matching_blocks = 0
    for i in range(num_computed_blocks, total_blocks):
        block_hash = block_hashes[i]
        if self.block_pool.get_cached_block(block_hash) is not None:
            num_matching_blocks += 1
        else:
            break
    return num_matching_blocks  # 这就是本次可复用的前缀块数
```

### 3.3 计算 num_cached_tokens

调度器拿到 `num_matching_blocks` 后：

```
num_cached_tokens = num_matching_blocks * block_size
num_external_computed_tokens = num_cached_tokens
```

之后 `KVCacheManager.allocate_slots` 只为「剩余未命中 token」分配新 block（见 `03`）。

### 3.4 算完登记缓存

请求跑完它的 prefill（包含未命中的部分）后，把填满的 block 登记：

```python
# vllm/v1/core/block_pool.py:225 （示意）
def cache_full_blocks(self, request_id, blocks, num_cached_blocks, ...):
    for block in blocks[num_cached_blocks:]:
        if block.full and not block.is_cached:
            block_pool.block_hash_to_block_id[block.hash] = block.block_id
            block.ref_cnt += 1
```

### 3.5 驱逐

当 KV cache 不够（`allocate_slots` 拿不到空闲块）时，调度器触发释放：从 `free_block_queue` 的队首（LRU）取 block，若其 `ref_cnt == 0` 则回收并从 hash 表删除；`ref_cnt > 0` 表示正被某请求占用，不能回收。这保证了「正在用的前缀不会被误删」。

---

## 4. 关键数据结构

| 结构 | 字段 | 含义 / 位置 |
| --- | --- | --- |
| `BlockHashType` | 哈希值类型 | 内容 hash，[vllm/v1/core/kv_cache_utils.py](../vllm/v1/core/kv_cache_utils.py) |
| `BlockPool` | `block_hash_to_block_id` | hash → 物理 block id，[vllm/v1/core/block_pool.py](vllm/v1/core/block_pool.py) |
| `BlockPool` | `free_block_queue` | 空闲块 LRU 队列 |
| `BlockPool` | `ref_cnt` | 引用计数，保护占用中 block |
| `SingleTypeKVCacheManager` | `find_longest_cache_hit` | 查最长公共前缀，[vllm/v1/core/single_type_kv_cache_manager.py:578](../vllm/v1/core/single_type_kv_cache_manager.py#L578) |
| `Request` | `block_hashes` | 每块的 hash 链 |
| `Request` | `num_cached_tokens` / `num_external_computed_tokens` | 命中长度 |

---

## 5. 收益与代价

**收益**
- 命中前缀直接跳过 prefill，TTFT 与 prefill FLOPs 大幅下降。
- 省下的 KV 块提高并发上限。
- 对长 system prompt、多轮、RAG、并行采样收益巨大。

**代价 / 限制 / 失效场景**
- **必须 deterministic**：同一前缀必须产生相同 token 序列才能对齐 hash；若 prompt 含随机因素（如随机 few-shot 顺序）则命中率掉。
- **整块才缓存**：block_size 越大，命中粒度越粗，浪费越多（半块不缓存）；block_size 越小命中率越高，但 block 元数据和 hash 表开销上升。
- **sliding window**：滑动窗口注意力下，很早的 block 会被窗口踢出，长前缀的远端 block 无法复用，命中率受限。
- **mamba / 递归状态**：非注意力类 KV（如 Mamba 的 conv_state/ssm_state）通常不支持前缀缓存（KV 是状态而非按 block 的 KV）。
- **LoRA 场景**：不同 adapter 的 KV 不能跨请求复用——代码通过把 lora 相关信息纳入 hash 或禁止跨 lora 命中来隔离（具体以本仓库实现为准）。
- **显存代价**：缓存表本身占 CPU 内存（block_hash 字典）；GPU KV block 被缓存占用，会减少可服务的新请求数，因此需要 LRU 驱逐做权衡。
- **与 CUDA Graph / spec decode**：前缀命中只影响 prefill 长度，不影响 graph 捕获；spec decode 下前缀命中的请求照常可被 draft（无冲突）。

---

## 6. 面试高频问题

**Q1：prefix caching 和 PagedAttention 的关系？**
A：PagedAttention 提供「分页 KV cache + block 引用」的基础设施（见 `03`）；prefix caching 是建在这之上的「按内容 hash 复用 block」的策略。`ref_cnt` 来自 PagedAttention 的 block 机制，prefix caching 复用它实现共享前缀。

**Q2：为什么 hash 要包含父 block 的 hash？**
A：用父 hash 链保证前缀任意位置的改动都向后传播，避免不同前缀产生相同 hash 造成误命中（冲突）。这是内容寻址缓存的标准做法。

**Q3：为什么只缓存填满的整块？**
A：半块无法被后续请求完整复用，也没法跟 block 边界对齐做确定性 hash；整块才能安全地跨请求共享。

**Q4：正在被请求占用的前缀 block 会被驱逐吗？**
A：不会。`ref_cnt > 0` 的 block 受引用计数保护，驱逐时跳过。只有 `ref_cnt == 0` 的空闲块（来自 `free_block_queue` 队首，LRU）才会被回收。

**Q5：并行采样（best-of-n）为什么天然高命中？**
A：n 个请求的 prompt 完全相同，前缀 hash 链一模一样，第一个算完后续 n-1 个直接全命中，prefill 几乎免费。

**Q6：prefix caching 对 TTFT 的影响取决于什么？**
A：取决于「命中前缀 token 数 / 总 prompt token 数」。命中比例越高，prefill 越接近只算剩余 token，TTFT 越低。长 system prompt 场景收益最大。

**Q7：怎么看命中率？指标叫什么？**
A：`gpu_prefix_cache_hit_rate`，由 `vllm/v1/core/kv_cache_metrics.py` 与 `gpu_model_runner` 统计，经 `vllm/v1/metrics/` 暴露（见 `14`）。

**Q8：和多模态怎么结合？**
A：block hash 会混入 `mm_hash`（图像/视频的 hash），所以「文本相同但图不同」不会误命中；同一个图的多次请求则命中，避免重复跑 ViT（与 `13` 的 encoder cache 是两层缓存）。

**Q9：prefix caching 和 chunked prefill 冲突吗？**
A：不冲突，反而叠加。命中只减少要算的 token 数，没命中的剩余部分仍可被 chunked prefill 切成小块（见 `05`）。

**Q10：block_size 对 prefix caching 的影响？**
A：block_size 越小，命中粒度越细、浪费越少、命中率越高，但 hash 表和 block 元数据开销上升，且单次 alloc 的块数变多。需要在两者间权衡。

**Q11：sliding window 下 prefix caching 还有效吗？**
A：有效但受限——窗口外的早期 block 会被注意力机制踢出，长前缀的远端 block 无法参与复用，命中长度受窗口大小约束。

**Q12：如果生成是 sampling（非 greedy），前缀 KV 还能复用吗？**
A：能。KV cache 只依赖前向计算的输入 token，与采样策略无关；同一前缀必然产生相同 KV，所以即使后面采样不同，前缀 KV 也安全复用。

---

## 7. 延伸阅读

- 实现：`vllm/v1/core/kv_cache_utils.py`、`block_pool.py`、`single_type_kv_cache_manager.py`、`kv_cache_manager.py`、`kv_cache_metrics.py`
- 配套：`03-paged-attention-kv-cache.md`、`05-scheduler-continuous-batching.md`、`13-multimodal-and-encoder-cache.md`、`14-metrics-and-observability.md`
- 官方：https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html
