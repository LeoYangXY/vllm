# PagedAttention 与 vLLM V1 的 KV Cache 管理机制

> 本文只讨论 V1 代码（`vllm/v1/` 下）。所有链接=真实源码行号，可跳转核对。

---

## 0. TL;DR

1. **是什么**：把 KV cache 切成固定大小（`block_size`，默认 16 token）的 **block**，请求按需从全局 `BlockPool` 申请 block，逻辑上连续、物理上离散，用一张 **block table** 记录 `token 位置 → 物理 slot` 的映射。
2. **解决什么**：消除「按 `max_model_len` 预留连续显存」带来的内部碎片、外部碎片，并让**相同前缀的 block 可以被多个请求共享**（引用计数 + hash），这是 prefix caching 与 KV 复用的物理基础。
3. **怎么做**：`KVCacheSpec`（每层怎么存）→ `KVCacheGroupSpec`（哪些层共享 block table）→ `KVCacheConfig` + `KVCacheTensor`（一个扁平字节池的 stride 描述）→ `BlockPool` / `SingleTypeKVCacheManager` / `KVCacheCoordinator`（调度侧分配）→ `BlockTable` + `slot_mapping`（worker 侧索引）→ attention kernel。
4. **收益**：显存几乎零碎片（`BlockPool.get_usage()` 即真实占用率），系统吞吐随空闲 block 数线性增长；`num_blocks = available_memory // bytes_per_block`，`bytes_per_block` 由 spec 精确算出，不留余量。
5. **代价**：kernel 必须通过 block table 间接寻址（slot mapping 多一次 gather）；block 粒度带来尾部内部碎片（平均 `block_size/2` token/请求）；SWA / Mamba 这类「非全量保留」的注意力需要额外的 block 回收与 null block 填充逻辑。

---

## 1. 场景与痛点

### 1.1 朴素方案：为每个请求预留一段连续显存

Transformer 解码时，第 `t` 个 token 的 attention 需要 `[0, t)` 全部历史的 K/V。最直觉的实现是：为每条请求分配一块**连续的** `[max_model_len, num_layers, 2, num_kv_heads, head_size]` 显存，di 一个 token 就往第 `t` 行写。

这个方案有三重浪费：

| 浪费类型 | 成因 | 量化 |
|---|---|---|
| **预留浪费（reservation）** | 每条请求都按 `max_model_len` 预留，但绝大多数请求远没跑满 | 平均输出/输入长度 `s`，浪费率 `1 - s/M`（`M = max_model_len`） |
| **内部碎片（internal frag.）** | 最后一个 block / 预分配粒度内未使用的部分，无法被别人用 | 分页后每请求平均 `block_size/2` token，相对占比 `(block_size/2)/s`；`block_size=16`、`s=1024` 时仅 **0.78%** |
| **外部碎片（external frag.）** | 不同长度的请求释放后留下大小不一的空洞，后来的请求长度对不上就用不了 | 取决于分配器；无法解析给出，通常远大于内部碎片 |

vLLM SOSP'23 论文（*Efficient Memory Management for Large Language Model Serving with PagedAttention*）给出的实测结论是：现有系统（FasterTransformer / Orca 风格）的 KV cache 显存**有 60%~80% 被浪费**，主要来源是预留 + 外部碎片，而不是内部碎片。（此为论文数据，非本仓库代码，仅作量级参考。）

### 1.2 更致命的问题：无法共享

连续分配下，两个请求即使有**完全相同的 system prompt**（比如同一个应用的 2k token 指令模板），也只能各存一份——因为它们各自占据不同的连续地址区间，物理上没法让第二个请求的 block table 指向第一个请求的地址。

这直接杀死了几类高价值负载的复用可能：多轮对话、RAG 同文档多问、few-shot、beam search / best-of-n、self-consistency。详见《04-prefix-caching.md》。

### 1.3 vLLM 的答案

> 借操作系统虚拟内存的思路：**分页**。

- 显存被切成 `num_blocks` 个等大的 block（`block_size` token）；
- 请求维护一张 **block table**（`[max_num_blocks_per_req]` 的 int32 数组），第 `i` 项是第 `i` 个逻辑 block 的物理 block id；
- token `pos` 的物理 slot 由 `slot = block_table[pos // block_size] * block_size + pos % block_size` 算出，这个映射由 **slot mapping** 张量显式物化（Triton kernel 生成，见 §3.10）；
- 一个 block 可以被多个请求的 block table 同时指向，用 **引用计数 `ref_cnt`** 保护，归零才回到 free queue。

---

## 2. 核心设计

### 2.1 五层抽象

```
┌────────────────────────────────────────────────────────────────────────────┐
│ L1  Spec 层              vllm/v1/kv_cache_interface.py                      │
│     KVCacheSpec / FullAttentionSpec / SlidingWindowSpec / MambaSpec / ...   │
│     描述「一层怎么存」：block_size、num_kv_heads、head_size、dtype、page 大小 │
├────────────────────────────────────────────────────────────────────────────┤
│ L2  Group 层             KVCacheGroupSpec                                    │
│     若干层共享同一张 block table（同 spec 的层被合并成 1 个 group）            │
├────────────────────────────────────────────────────────────────────────────┤
│ L3  Config 层            KVCacheConfig + KVCacheTensor                       │
│     num_blocks + 每个 (layers, layer_stride, block_stride, offset) 的字节布局 │
├────────────────────────────────────────────────────────────────────────────┤
│ L4  调度侧管理            BlockPool / SingleTypeKVCacheManager                │
│                          / KVCacheCoordinator / KVCacheManager               │
│     分配、free、hash 缓存、LRU 驱逐、SWA/Mamba 的 block 回收                   │
├────────────────────────────────────────────────────────────────────────────┤
│ L5  Worker 侧            MultiGroupBlockTable / BlockTable / slot_mapping     │
│     CPU 侧 numpy 拼表 → H2D → Triton 生成 slot_mapping → attention kernel     │
└────────────────────────────────────────────────────────────────────────────┘
```

这套分层的关键价值是：**调度器（engine core 进程）只操作 L1~L4 的纯 Python 元数据，完全不碰显存**；只有 worker 在 L5 把 block id 变成真实的 GPU 张量。

### 2.2 Spec 类型体系

`KVCacheSpec` 是所有 spec 的基类，定义在 [vllm/v1/kv_cache_interface.py:151](../vllm/v1/kv_cache_interface.py#L151)，核心字段只有一个 `block_size`，其余都是抽象方法：

| 属性/方法 | 语义 | 定义处 |
|---|---|---|
| `block_size` | 一个 block 装多少 token | [:157](../vllm/v1/kv_cache_interface.py#L157) |
| `num_heads` | page 逻辑 shape `[B,H,N,C]` 的 H | [:409](../vllm/v1/kv_cache_interface.py#L409)（`AttentionSpec`）|
| `tokens_per_state` | 一个 state 覆盖几个 token（>1 压缩，<1 多 state/token）| [:400](../vllm/v1/kv_cache_interface.py#L400) |
| `state_content_size_bytes` | 一个 `(head slot, state)` 单元的字节数 C | [:416](../vllm/v1/kv_cache_interface.py#L416) |
| `num_states` | `block_size // tokens_per_state` | [:186](../vllm/v1/kv_cache_interface.py#L186) |
| `page_size_bytes` | 一个 block 的字节数（含 padding） | [:427](../vllm/v1/kv_cache_interface.py#L427) |
| `max_memory_usage_bytes` | 单请求最长序列要多少字节 | [:195](../vllm/v1/kv_cache_interface.py#L195) |
| `max_num_blocks_per_req` | worker 侧 block table 的行宽 | [:204](../vllm/v1/kv_cache_interface.py#L204) |
| `prefix_cacheable` | 本组是否参与 prefix caching | [:159](../vllm/v1/kv_cache_interface.py#L159) |

几个关键派生 spec：

- **`FullAttentionSpec`**（[:447](../vllm/v1/kv_cache_interface.py#L447)）：最常见的全注意力。注意它的 `max_memory_usage_bytes` 是 `cdiv(max_model_len, block_size) * page_size_bytes`，即**请求全程保留所有 block**。
- **`SlidingWindowSpec`**（[:707](../vllm/v1/kv_cache_interface.py#L707)）：带 `sliding_window`。它的内存上界是 `max_admission_blocks_per_request(...)` 算出来的 **窗口 + in-flight tokens**，而不是 `max_model_len`（[:716](../vllm/v1/kv_cache_interface.py#L716)）；`+1` 是因为窗口起点可能不在 block 边界上（注释里的 `[XXCD][EF]` 例子）。
- **`ChunkedLocalAttentionSpec`**（[:667](../vllm/v1/kv_cache_interface.py#L667)）：chunked local attention，上界是 `attention_chunk_size + max_in_flight_tokens`（[:670](../vllm/v1/kv_cache_interface.py#L670)）。
- **`MambaSpec`**（[:885](../vllm/v1/kv_cache_interface.py#L885)）：**完全不分页**。`tokens_per_state = -1`，`num_heads = 1`；一个 block 装的是一段**递归状态**而不是 token 序列，`page_size_bytes = sum(prod(shape) * dtype_size)`（[:912](../vllm/v1/kv_cache_interface.py#L912)）。它的内存上界依赖 `mamba_cache_mode`（[:922](../vllm/v1/kv_cache_interface.py#L922)）：`"all"` 是 `max_model_len/block_size`，`"align"` 只留 2 + speculative + checkpoint 个 block，否则只要 1 个。
- **`UniformTypeKVCacheSpecs`**（[:1072](../vllm/v1/kv_cache_interface.py#L1072)）：把「类型相同但 page size 不同」的若干层打包成一个 group，`page_size_bytes` 是各层之和（[:1093](../vllm/v1/kv_cache_interface.py#L1093)），`max_memory_usage_bytes` 取各层需要的 page 数的最大值（[:1096](../vllm/v1/kv_cache_interface.py#L1096)）——因为整个 group 只能按同一个 `num_blocks` 分配。

spec → manager 的映射不是 if-else，而是一张**注册表** `KVCacheSpecRegistry`（[vllm/v1/kv_cache_spec_registry.py:39](../vllm/v1/kv_cache_spec_registry.py#L39)），见 §3.1.1。

### 2.3 物理布局：`KVCacheLayout`

[vllm/v1/kv_cache_layout.py:15](../vllm/v1/kv_cache_layout.py#L15) 的 `KVCacheLayout` 枚举描述「逻辑 5 维 `[L, B, H, N, C]` 如何映射到物理内存序」。成员就是一个 stride 置换：

```
LBHNC = (0,1,2,3,4)   [L, B, H, N, C]  identity —— 层优先，块内层间不连续
LBNHC = (0,1,3,2,4)   [L, B, N, H, C]
LHBNC = (0,2,1,3,4)   [L, H, B, N, C]
BLHNC = (1,0,2,3,4)   [B, L, H, N, C]  块优先
BLNHC = (1,0,3,2,4)   [B, L, N, H, C]
BHLNC = (1,2,0,3,4)   [B, H, L, N, C]
```

三个判据（[:39](../vllm/v1/kv_cache_layout.py#L39)–[:57](../vllm/v1/kv_cache_layout.py#L57)）：

- `is_layer_compact`：L 是最外层 → 每层占一段连续显存，H2D / 跨层拷贝友好；
- `is_block_compact`：L 和 B 都在最外两维 → 每个 page 的 `[H,N,C]` 是一段连续字节，混合 page size 时可以「一个 block 内塞多个层」；
- `is_block_outermost`：B 是最外层 → 一个 block 包含**所有层**的 page，适合 P/D 传输时按 block 整块搬。

### 2.4 Block 布局 ASCII 图

```
KVCacheConfig 描述的是「一个扁平 int8 字节池」+ 若干「视图描述」：

  buf = torch.zeros(bytes_per_block * num_blocks, dtype=int8)     # 一次分配，见 worker/utils.py:429
  ├───────────────────────────────────────────────────────────────────────────┤
  │                        backing allocation (bytes)                          │
  └───────────────────────────────────────────────────────────────────────────┘

  KVCacheTensor: size / layers[] / layer_stride / block_stride / offset
  （kv_cache_interface.py:1241）
  layer l 的 block b 起点 = offset + l * layer_stride + b * block_stride

  ── 层优先（L 最外，如 LBNHC）──────────────────────────────────────────────
   offset=0
   ├─ L0: [b0][b1][b2] ... [bN-1] ─┤  layer_stride = page * num_blocks
   ├─ L1: [b0][b1][b2] ... [bN-1] ─┤  block_stride = page
   ├─ ...                          │
   └─ LL-1:[b0][b1][b2] ... [bN-1]─┘

  ── 块优先（B 最外，如 BLHNC）──────────────────────────────────────────────
   ├─ b0: [L0 page][L1 page] ... [LL-1 page] ─┤  layer_stride = page
   ├─ b1: [L0 page][L1 page] ... [LL-1 page] ─┤  block_stride = bytes_per_block
   ├─ ...                                      │
   └─ bN-1: ...                                ┘

  单 page 内部（FullAttentionSpec，逻辑 [B, H, N, C]）：
   ┌─── block b ────────────────────────────────────────────────────────┐
   │ H0: [ s0 ][ s1 ][ s2 ] ... [ s_{block_size-1} ]                    │
   │ H1: [ s0 ][ s1 ][ s2 ] ... [ s_{block_size-1} ]                    │   s_i = 第 i 个 token 的
   │ ...                                                                │         K/V 拼接内容
   │ H_{num_kv_heads-1}: [ ... ]                                        │
   └────────────────────────────────────────────────────────────────────┘
     N = num_states = block_size // tokens_per_state
     C = state_content_size_bytes = (head_size + head_size_v) * dtype_size

  多 group 时，各 group 的 KVCacheTensor **从字节 0 开始互相 alias**（见
  kv_cache_utils.py:1714 的注释）：block id 同一时刻只被一个 group 拥有，
  所以重叠是安全的——这正是 Mamba + FullAttention 混合模型能省显存的原因。
```

### 2.5 请求 → block table → slot mapping

```
 请求 token 序列 (长度 37, block_size=16)
 pos:  0 ....... 15 | 16 ...... 31 | 32 .. 36
       ├── blk 0 ──┤├── blk 1 ──┤├── blk 2 ─┤ (半满)

 KVCacheManager (调度侧) 分配物理 block:
   req_to_blocks = [KVCacheBlock(7), KVCacheBlock(1024), KVCacheBlock(3)]
                        │                  │                  │
                        ▼                  ▼                  ▼
 worker 侧 BlockTable 行 (int32[max_num_blocks_per_req]):
   block_table[req_row] = [7, 1024, 3, 0, 0, ...]
                            ▲  num_blocks_per_row = 3（append_row 时累加，block_table.py:157）

 Triton kernel ComputeSlotMappingKernel 生成 slot_mapping（block_table.py:413）:
   对 token pos:
     block_idx  = pos // block_size
     slot_off   = pos %  block_size
     slot       = block_table[req_row, block_idx] * block_size + slot_off
     slot_mapping[token_idx] = slot

   pos=0  → 7*16+0  = 112
   pos=16 → 1024*16+0 = 16384
   pos=36 → 3*16+4  = 52

 reshape_and_cache → 把 K/V 写到 kv_cache[layer].view(...)[slot]  （_custom_ops.py:2600）
 attention kernel  → 按 block_table 直接做 gather-KV 的分块 attention
```

---

## 3. 代码走读

### 3.1 起点：模型声明「我需要什么样的 KV cache」

#### 3.1.1 `get_kv_cache_spec()`

[vllm/v1/worker/gpu_model_runner.py:7633](../vllm/v1/worker/gpu_model_runner.py#L7633)：

```python
def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
    if has_ec_transfer() and not get_ec_transfer().is_consumer:
        return {}
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    layer_type = cast(type[Any], AttentionLayerBase)
    attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
    for layer_name, attn_module in attn_layers.items():
        if isinstance(attn_module, Attention) and (
            kv_tgt_layer := attn_module.kv_sharing_target_layer_name
        ):
            # 该层复用目标层的 KV cache，不生成自己的 spec → 不分配显存
            self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
            continue
        if spec := attn_module.get_kv_cache_spec(self.vllm_config):
            if isinstance(spec, AttentionSpec):
                spec = attn_module.get_attn_backend().customize_spec(spec)
            kv_cache_spec[layer_name] = spec
    return kv_cache_spec
```

三个要点：

1. **spec 是「声明式」的**：每个 attention 模块自己报 `FullAttentionSpec` / `SlidingWindowSpec` / `MambaSpec`，框架不关心模型结构。
2. **跨层 KV 共享**（YOCO / You-Only-Cache-Once，`kv_sharing_target_layer_name`）通过「不给该层生成 spec」实现，一层不占 block → 显存直接省掉（[:7660](../vllm/v1/worker/gpu_model_runner.py#L7660) 的 `continue`）。
3. **`customize_spec` 是 backend 改写 layout 的钩子**：例如 [triton_attn.py:276](../vllm/v1/attention/backends/triton_attn.py#L276) 对 per-token-head 量化模式，把 fp32 scale 内联进 page：

```python
def customize_spec(cls, spec: "AttentionSpec") -> "AttentionSpec":
    """Per-token-head modes pack inline fp32 scales after each head's
    data, so the content is (data + one scale) per K/V side."""
    mode = spec.kv_quant_mode
    if spec.state_content_bytes is not None or not mode.is_per_token_head:
        return spec
    hs_k, hs_v = spec.head_size, spec.head_size_v
    if mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        hs_k, hs_v = hs_k // 2, hs_v // 2
    scale_bytes = get_dtype_size(torch.float32)
    content = (hs_k + hs_v) * get_dtype_size(spec.dtype) + 2 * scale_bytes
    return replace(spec, state_content_bytes=content)
```

#### 3.1.2 spec → manager 的注册表

[vllm/v1/kv_cache_spec_registry.py:105](../vllm/v1/kv_cache_spec_registry.py#L105) 的 `get_manager_class` 沿 MRO 向上找注册项：

```python
@classmethod
def get_manager_class(cls, kvcache_spec: "KVCacheSpec") -> ...:
    cls._ensure_registered()
    kvcache_spec_cls = type(kvcache_spec)
    # Walk up the MRO to find a registered base class
    for base in kvcache_spec_cls.__mro__:
        if base in _REGISTRY_KVCACHESPEC_LIST:
            return _REGISTRY_KVCACHESPEC_LIST[base].manager_class
    return None
```

注册动作集中在 [single_type_kv_cache_manager.py:2198](../vllm/v1/core/single_type_kv_cache_manager.py#L2198) 的 `register_all_kvcache_specs`：

```python
KVCacheSpecRegistry.register(FullAttentionSpec, FullAttentionManager,
                             uniform_type_base_spec=FullAttentionSpec)
KVCacheSpecRegistry.register(SlidingWindowSpec, SlidingWindowManager,
                             uniform_type_base_spec=SlidingWindowSpec)
KVCacheSpecRegistry.register(MambaSpec, MambaManager,
                             uniform_type_base_spec=MambaSpec)
# FullAttentionSpec 的子类统一归到 FullAttentionSpec 组：
KVCacheSpecRegistry.register(MLAAttentionSpec, FullAttentionManager,
                             uniform_type_base_spec=FullAttentionSpec)
```

`uniform_type_base_spec` 决定**哪些 spec 可以合并成一个 group**（见 [:128](../vllm/v1/kv_cache_spec_registry.py#L128)），这是「out-of-tree backend 不用改 vLLM 核心」的关键扩展点（装饰器 `@register_kv_cache_spec` 在 [:176](../vllm/v1/kv_cache_spec_registry.py#L176)）。

### 3.2 `determine_available_memory()`：先算「还剩多少显存给 KV cache」

[vllm/v1/worker/gpu_worker.py:525](../vllm/v1/worker/gpu_worker.py#L525)：

```python
@torch.inference_mode()
def determine_available_memory(self) -> int:
    maybe_apply_startup_plan(self)
    if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
        # 用户手动指定：仍要跑一次 profile_run 把模型编译出来
        self.model_runner.profile_run()
        ...
        return reserve_mm_ipc_gpu_memory(kv_cache_memory_bytes, ...)

    # 用 dummy 输入跑一次 forward，profile 峰值显存
    with memory_profiling(self.init_snapshot,
                          weights_memory=int(self.model_runner.model_memory_usage),
                          ) as profile_result:
        self.model_runner.profile_run()
    ...
    self.available_kv_cache_memory_bytes = (
        self.requested_memory
        - profile_result.non_kv_cache_memory
        - cudagraph_memory_estimate_applied
    )
    return reserve_mm_ipc_gpu_memory(
        int(self.available_kv_cache_memory_bytes), ...)
```

要点：

- **先 profile 再分配**：`requested_memory = total * gpu_memory_utilization`，扣掉权重 + activation 峰值 +（可选）CUDA graph 内存，剩下的**全部**给 KV cache。这也是为什么 vLLM 几乎不会因为你设了 0.9 就 OOM——它按实测峰值算的。
- `kv_cache_memory_bytes` 是逃生舱：手工指定大小，跳过 profiling（但仍要 `profile_run()` 做编译）。
- CUDA graph 内存是否计入由 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 控制（[:585](../vllm/v1/worker/gpu_worker.py#L585)），默认自 v0.21.0 起计入。

调用方在 engine core：[vllm/v1/engine/core.py:308](../vllm/v1/engine/core.py#L308) `available_gpu_memory = self.model_executor.determine_available_memory()`。

### 3.3 从 spec + 可用显存 → `KVCacheConfig`

[vllm/v1/core/kv_cache_utils.py:1614](../vllm/v1/core/kv_cache_utils.py#L1614) `get_kv_cache_config_from_groups`：

```python
layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
validate_kv_cache_layout(layout, kv_cache_groups)
bytes_per_block = _get_kv_cache_bytes_per_block(kv_cache_groups)
interleaved_block_stride = bytes_per_block if layout.is_block_outermost else None

num_blocks = available_memory // bytes_per_block
num_blocks = may_override_num_blocks(vllm_config, num_blocks)
size = bytes_per_block * num_blocks

kv_cache_tensors = []
for group in kv_cache_groups:
    ...
    for spec, layer_names in layers_by_spec.items():
        layer_stride, block_stride, _, _, _ = compute_layout_strides(
            spec, num_blocks, len(layer_names), layout,
            fixed_strides=(None, interleaved_block_stride, None, None, None),
        )
        kv_cache_tensors.append(KVCacheTensor(
            size=size, layers=layer_names,
            layer_stride=layer_stride, block_stride=block_stride, offset=offset))
        byte_offset += len(layer_names) * spec.page_size_bytes
```

这就是「分页」的**量化核心**：

```
num_blocks = available_memory // bytes_per_block
bytes_per_block = Σ_layers page_size_bytes(layer)
page_size_bytes = num_kv_heads × block_size × (head_size + head_size_v) × dtype_size
```

**举例**（Llama-3-8B，32 层、8 KV head、head_dim 128、bf16、`block_size=16`）：

```
每个 (head, state) 单元: (128+128) × 2 B           = 512 B
单 page (一层一个 block): 8 heads × 16 states × 512 = 65,536 B = 64 KiB
整个 pool 一个 block:    32 层 × 64 KiB            = 2 MiB
20 GiB 可用 → num_blocks = 20×1024/2 = 10,240 blocks
             = 163,840 token 的 KV 容量
```

注意所有 group 的 `KVCacheTensor` 共用 `size=bytes_per_block*num_blocks`、且 `offset` 从 0 开始——**多 group 在字节层面互相 alias**（[:1714](../vllm/v1/core/kv_cache_utils.py#L1714) 的注释解释了为什么安全：同一时刻一个 block id 只属于一个 group）。

### 3.4 worker 侧真正分配显存

`initialize_kv_cache` → `initialize_kv_cache_tensors` → `allocate_kv_cache`：

[gpu_model_runner.py:7491](../vllm/v1/worker/gpu_model_runner.py#L7491)：

```python
kv_cache_config = deepcopy(kv_cache_config)
self.kv_cache_config = kv_cache_config
self.may_add_encoder_only_layers_to_kv_cache_config()
self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
self.initialize_attn_backend(kv_cache_config, is_profiling=is_profiling)
initialize_mamba_ssu_backend(...)
# kernel block size：manager 用 256，但 backend 只支持 64 时会拆成 4 个虚拟 block
kernel_block_sizes = prepare_kernel_block_sizes(kv_cache_config, self.attn_groups)
self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)
self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
kv_caches = self.initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes, ...)
```

**kernel block size vs manager block size** 是 V1 一个重要的解耦：调度侧按 256 分配（减少 Python 侧管理开销），attention kernel 只支持 64 时，worker 把每个 manager block 拆成 4 个虚拟 block（[block_table.py:238](../vllm/v1/worker/block_table.py#L238) 的 `map_to_kernel_blocks`）。

[worker/utils.py:397](../vllm/v1/worker/utils.py#L397) `allocate_kv_cache`：

```python
sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
assert len(sizes) == 1, "KV cache tensors must share one backing allocation."
raw_size = sizes.pop()
...
buf = torch.zeros(buf_size, dtype=torch.int8, device=device)     # ← 一次分配

for tensor in kv_cache_config.kv_cache_tensors:
    ...
    views = create_kv_cache_views(buf, spec, num_blocks, layout, tensor,
                                  kernel_block_size=kernel_block_size)
    kv_caches.update(zip(tensor.layers, views))
return kv_caches
```

`create_kv_cache_views`（[kv_cache_interface.py:319](../vllm/v1/kv_cache_interface.py#L319)）用 `torch.as_strided` 把扁平 int8 buffer 变成每层的 4D `[B,H,N,C]` 视图，**零拷贝、零 padding 分配**：

```python
view_5d = torch.as_strided(
    raw, size=logical_shape, stride=strides,
    storage_offset=raw.storage_offset() + kv_cache_tensor.offset)
views = []
for layer_idx in range(num_layers):
    cache_logical = view_5d[layer_idx]
    if dtype is not None:
        cache_logical = cache_logical.view(dtype)   # int8 字节视图 → 真实 dtype
    views.append(cache_logical)
```

注意 `compute_layer_kv_cache_shape_bytes`（[:261](../vllm/v1/kv_cache_interface.py#L261)）把 C 维**按字节算**，所以 int8 buffer 才能被任意 dtype 重新 view。

最后 `bind_kv_cache`（[gpu_model_runner.py:7454](../vllm/v1/worker/gpu_model_runner.py#L7454)）把张量挂到 `static_forward_context` 的每个 attention 层上。

### 3.5 `BlockPool`：block 的唯一物理所有者

[vllm/v1/core/block_pool.py:143](../vllm/v1/core/block_pool.py#L143)：

```python
def __init__(self, num_gpu_blocks, enable_caching, hash_block_size,
             enable_kv_cache_events=False, metrics_collector=None):
    self.num_gpu_blocks = num_gpu_blocks
    self.enable_caching = enable_caching
    self.hash_block_size = hash_block_size
    # All kv-cache blocks.
    self.blocks: list[KVCacheBlock] = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
    # Free block queue：双向链表，按驱逐顺序组织
    self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
    self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
    self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}
    # To represent a placeholder block with block_id=0.
    # The ref_cnt of null_block is not maintained, needs special care to avoid freeing it.
    self.null_block = self.free_block_queue.popleft()
    self.null_block.is_null = True
```

**`null_block` 是一个设计得很有意思的技巧**：SWA / Mamba 会「逻辑上跳过」某些 token，但 worker 的 block table 必须是一张**矩形**张量，不能留空洞。于是用 `null_block`（block_id=0、永不缓存、引用计数不维护）占位，其 KV 内容为全 0，配合 kernel 侧的 mask 跳过即可。见 [block_pool.py:190](../vllm/v1/core/block_pool.py#L190)。

分配（[:658](../vllm/v1/core/block_pool.py#L658)）：

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    if num_blocks > self.get_num_free_blocks():
        raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
    ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
    if self.enable_caching:
        for block in ret:
            self._maybe_evict_cached_block(block)   # 从 prefix cache 里摘掉
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
    ...
```

注意 `_maybe_evict_cached_block` 在 `ref_cnt` 归零的 block 上才可能被调用——**正在被别的请求引用的 block 根本不在 free queue 里**，所以不会被驱逐（详见 doc 04）。

回收（[:734](../vllm/v1/core/block_pool.py#L734)）**区分有无 hash**：

```python
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    blocks_to_evict_last = []
    blocks_to_evict_first = []
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            if block.block_hash is None or not self.enable_caching:
                # LIFO reuse of non-cached blocks for better GPU locality.
                blocks_to_evict_first.append(block)
            else:
                # FIFO reuse of cached blocks for LRU eviction behavior.
                blocks_to_evict_last.append(block)
    self.free_block_queue.prepend_n(blocks_to_evict_first)
    self.free_block_queue.append_n(blocks_to_evict_last)
```

- 无 hash（不可能被 prefix 命中）→ `prepend_n` 到队头，**LIFO 复用**，刚释放的 block 还在 cache 里热着；
- 有 hash（是 prefix cache 的候选）→ `append_n` 到队尾，**FIFO 复用**，等价于 LRU（最早进入 free queue 的最先被 `popleft` 拿走）。

### 3.6 `FreeKVCacheBlockQueue`：O(1) 中间删除的双向链表

[vllm/v1/core/kv_cache_utils.py:230](../vllm/v1/core/kv_cache_utils.py#L230)。为什么不用 `collections.deque`？因为 `BlockPool.touch()` 需要把**队中间的**某个 block 摘出来（`ref_cnt` 从 0 变 1，说明它被新请求命中了）——deque 做不到 O(1)。

```python
class FreeKVCacheBlockQueue:
    """... We implement this class instead of using Python builtin deque to support
    removing a block in the middle of the queue in O(1) time. To close the performance
    gap to the builtin deque which is implemented in C++, this class does not allocate
    any Python objects when manipulating the linked list. Instead, this class manipulates
    the prev_free_block and next_free_block attributes of the given blocks.
    ...
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the same sequence),
       the one with more hash tokens (the tail of a block chain) is at the front.
    """
```

链表节点就是 `KVCacheBlock` 自身的两个字段（[:180](../vllm/v1/core/kv_cache_utils.py#L180)–[:181](../vllm/v1/core/kv_cache_utils.py#L181)），另外造了两个 `block_id=-1` 的 **fake head / fake tail** 哨兵（[:268](../vllm/v1/core/kv_cache_utils.py#L268)）来消除分支判断。`init` 时按 block id 顺序串起来，所以**初始队列顺序 = block id 顺序**。

### 3.7 `SingleTypeKVCacheManager`：单类型 group 的分配逻辑

[vllm/v1/core/single_type_kv_cache_manager.py:43](../vllm/v1/core/single_type_kv_cache_manager.py#L43)。它维护：

| 字段 | 语义 | 行号 |
|---|---|---|
| `block_size` | 本 manager 的分配粒度（`= spec.block_size * dcp_world_size`，Mamba 除外）| [:85](../vllm/v1/core/single_type_kv_cache_manager.py#L85) |
| `req_to_blocks` | `req_id → list[KVCacheBlock]` | [:106](../vllm/v1/core/single_type_kv_cache_manager.py#L106) |
| `num_cached_block` | `req_id → 已缓存 block 数` | [:112](../vllm/v1/core/single_type_kv_cache_manager.py#L112) |
| `new_block_ids` | 本次分配的新 block（供 worker 侧 zeroing）| [:101](../vllm/v1/core/single_type_kv_cache_manager.py#L101) |
| `_null_block` | 占位 block | [:115](../vllm/v1/core/single_type_kv_cache_manager.py#L115) |

**`get_num_blocks_to_allocate`**（[:156](../vllm/v1/core/single_type_kv_cache_manager.py#L156)）是「还需要几个 block」的核心计算：

```python
num_required_blocks = cdiv(num_tokens, self.block_size)
if apply_admission_cap and self._max_admission_blocks_per_request is not None:
    # SWA / chunked-local 会回收 block，这里用 spec 算出的上界夹紧，
    # 保证「启动时按此上界定 pool 大小」与「运行时准入判断」用同一个数
    num_required_blocks = min(num_required_blocks, self._max_admission_blocks_per_request)
num_req_blocks = len(self.req_to_blocks.get(request_id, ()))
if request_id in self.num_cached_block:
    # Fast-path: 运行中的请求不会有新的 prefix 命中
    assert len(new_computed_blocks) == 0
    return max(num_required_blocks - num_req_blocks, 0)
num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
num_skipped_blocks = num_skipped_tokens // self.block_size
num_new_blocks = max(
    num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks), 0)
```

对 SWA 来说 `num_skipped_blocks` 是**滑窗之外已经被丢弃的 block**——它们物理上已经 free 但位置上还占着（已被替换成 null block），所以不算进"已持有"。

**`remove_skipped_blocks`**（[:658](../vllm/v1/core/single_type_kv_cache_manager.py#L658)）是 block 复用的关键，配合 `get_num_skipped_tokens`：

```python
# SlidingWindowManager.get_num_skipped_tokens   (:1123)
return max(0, num_computed_tokens - self.sliding_window + 1 - self.extra_retained_tokens)

# MambaManager.get_num_skipped_tokens           (:1929)
return num_computed_tokens - 1          # 只保留最后一个 token 的递归状态

# FullAttentionManager 沿用基类                (:697)
return 0                                 # 永不跳过 → 永不复用
```

`_remove_blocks_in_range`（[:631](../vllm/v1/core/single_type_kv_cache_manager.py#L631)）**从后往前**遍历，把要丢弃的 block free 掉并替换成 `_null_block`，保证 block table 的矩形性：

```python
for i in range(last_block - 1, first_block - 1, -1):
    if blocks[i] == self._null_block:
        break
    freed.append(blocks[i])
    blocks[i] = self._null_block
if freed:
    self.block_pool.free_blocks(freed)
```

### 3.8 `KVCacheCoordinator`：多 group 的仲裁者

[vllm/v1/core/kv_cache_coordinator.py:66](../vllm/v1/core/kv_cache_coordinator.py#L66)。工厂函数 `get_kv_cache_coordinator`（[:978](../vllm/v1/core/kv_cache_coordinator.py#L978)）三选一：

| 策略 | 条件 | 类 | 行号 |
|---|---|---|---|
| 无 prefix cache | `not enable_caching` | `KVCacheCoordinatorNoPrefixCache` | [:451](../vllm/v1/core/kv_cache_coordinator.py#L451) |
| 单 group | `enable_caching and len(groups)==1` | `UnitaryKVCacheCoordinator` | [:503](../vllm/v1/core/kv_cache_coordinator.py#L503) |
| 多 group | 其余 | `HybridKVCacheCoordinator` | [:589](../vllm/v1/core/kv_cache_coordinator.py#L589) |

基类构造时就把 `BlockPool` 和每个 group 的 manager 建好（[:98](../vllm/v1/core/kv_cache_coordinator.py#L98)、[:136](../vllm/v1/core/kv_cache_coordinator.py#L136)），DCP 的 block size 缩放也在这一层完成（[:144](../vllm/v1/core/kv_cache_coordinator.py#L144)）。

`HybridKVCacheCoordinator.verify_and_split_kv_cache_groups`（[:706](../vllm/v1/core/kv_cache_coordinator.py#L706)）把**相同 spec 的 group 合并成一个 `SpecGroup`**，一次查表服务多个 group；并显式把 full attention 排到第一个：

```python
# Put full attention first: its efficient left-to-right scan provides
# a tighter initial bound, reducing work for subsequent groups.
self.attention_groups.sort(key=lambda g: not isinstance(g.spec, FullAttentionSpec))
```

### 3.9 `KVCacheManager`：调度器唯一面对的门面

[vllm/v1/core/kv_cache_manager.py:119](../vllm/v1/core/kv_cache_manager.py#L119)。`KVCacheBlocks`（[:35](../vllm/v1/core/kv_cache_manager.py#L35)）是它与 scheduler 之间的数据契约，外层维度是 **group** 而不是 token block（注释解释了原因：未来不同 group 可能 block 数不同）。

`allocate_slots`（[:360](../vllm/v1/core/kv_cache_manager.py#L360)）的 docstring 里有一张很好的布局图：

```
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                                          |   < to be computed >     |
----------------------------------------------------------------------
                          |            < to be allocated >           |
```

三段式：

```python
# 1) 先回收滑窗外 / Mamba 旧状态的 block（即使后续分配失败也要做，能减少碎片）
self.coordinator.remove_skipped_blocks(
    request.request_id,
    max(0, total_computed_tokens - request.num_in_flight_tokens),
    num_prompt_tokens=request.num_prompt_tokens)

# 2) 询问需要多少 block，不够就返回 None（调度器据此停止准入）
num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(...)
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks:
    return None

# 3) 先挂上 prefix 命中的 block，再分配新 block，最后 cache 完成的 block
self.coordinator.allocate_new_computed_blocks(...)
new_blocks = self.coordinator.allocate_new_blocks(...)
if not self.enable_caching or delay_cache_blocks:
    return self.create_kv_cache_blocks(new_blocks)
num_tokens_to_cache = min(total_computed_tokens + num_new_tokens, request.num_tokens)
self.coordinator.cache_blocks(request, num_tokens_to_cache)
```

注意 `num_tokens_to_cache` 被 `request.num_tokens` 截断——**投机解码中被拒绝的 draft token 不会被缓存**（[:570](../vllm/v1/core/kv_cache_manager.py#L570) 的 `NOTE(woosuk)`）。

### 3.10 worker 侧：`MultiGroupBlockTable` 与 slot mapping

[vllm/v1/worker/block_table.py:288](../vllm/v1/worker/block_table.py#L288) 的 `MultiGroupBlockTable` 就是一组 `BlockTable`（每个 group 一张）。`BlockTable` 的关键设计：

- **CPU/GPU 双缓冲**：`self.block_table = CpuGpuBuffer(max_num_reqs, max_num_blocks_per_req, dtype=torch.int32)`（[:114](../vllm/v1/worker/block_table.py#L114)）。调度侧写 numpy（`append_row` [:157](../vllm/v1/worker/block_table.py#L157)），`commit_block_table` 一次性 H2D（[:231](../vllm/v1/worker/block_table.py#L231)）。
- **虚拟 block 拆分**：`kernel_block_size != block_size` 时，`map_to_kernel_blocks`（[:238](../vllm/v1/worker/block_table.py#L238)）把 manager block id 展开成多个 kernel block id。
- **`SlotMappingMode.NONE`**（[:52](../vllm/v1/worker/block_table.py#L52)）：Mamba 类 group 的 block table 是**状态索引**而不是 token 索引，不需要 slot mapping。

slot mapping 由 Triton kernel 生成（[:413](../vllm/v1/worker/block_table.py#L413)）：

```python
pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
virtual_block_indices  = pos // virtual_block_size
virtual_block_offsets  = pos - virtual_block_indices * virtual_block_size
is_local = (virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
local_block_offsets = ...   # DCP/PCP 的本地偏移重映射
block_indices = virtual_block_indices * BLOCKS_PER_KV_BLOCK + local_block_offsets // block_size
block_numbers = tl.load(block_table_ptr + row_offset + block_indices,
                        mask=mask & is_local, other=0).to(tl.int64)
slot_offsets = local_block_offsets % block_size
slot_ids = block_numbers * block_size + slot_offsets
slot_ids = tl.where(is_local, slot_ids, PAD_ID)
tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)
```

`PAD_ID = PAD_SLOT_ID = -1`（[backends/utils.py:45](../vllm/v1/attention/backends/utils.py#L45)），kernel 侧遇到 -1 就跳过写。最后一个 program 还负责把尾部 pad 成 -1，保证 CUDA graph 兼容（[:433](../vllm/v1/worker/block_table.py#L433)）。

### 3.11 attention backend 如何消费 block table

以 FlashAttention 为例，[flash_attn.py:545](../vllm/v1/attention/backends/flash_attn.py#L545) 的 `build()` 直接把 `CommonAttentionMetadata` 里的现成张量塞进 metadata：

```python
block_table_tensor = common_attn_metadata.block_table_tensor   # [num_reqs, max_blocks]
slot_mapping       = common_attn_metadata.slot_mapping         # [num_tokens]
...
attn_metadata = FlashAttentionMetadata(
    num_actual_tokens=num_actual_tokens,
    max_query_len=max_query_len,
    query_start_loc=query_start_loc,
    max_seq_len=max_seq_len,
    seq_lens=seq_lens,
    block_table=block_table_tensor,      # ← 直接传给 FA kernel
    slot_mapping=slot_mapping,           # ← 写 KV 用
    ...)
```

也就是说，**现代 backend 已经不显式「查 block table 再 gather」了**：`block_table` 作为一个 `[num_reqs, max_num_blocks_per_req]` 的 int32 张量整体传给 kernel，由 kernel 内部按 `block_table[req, pos // block_size]` 做 paged 访问。`slot_mapping` 只在**写** KV 时用一次。

`use_cascade_attention` / `cascade_attention`（[:1666](../vllm/v1/attention/backends/flash_attn.py#L1666)、[:1744](../vllm/v1/attention/backends/flash_attn.py#L1744)）利用了共享前缀：`common_prefix_len > 0` 时，把 attention 拆成「公共前缀一次 attention（batch=1）+ 各请求后缀 attention」，公共前缀的 KV 只读一遍。这依赖 `KVCacheManager.get_num_common_prefix_blocks`（[kv_cache_manager.py:649](../vllm/v1/core/kv_cache_manager.py#L649)），其实现就是数 `ref_cnt == len(req_to_blocks)` 的前缀 block（[single_type_kv_cache_manager.py:864](../vllm/v1/core/single_type_kv_cache_manager.py#L864)）—— 注意 SWA 和 Mamba 都返回 0（[:1160](../vllm/v1/core/single_type_kv_cache_manager.py#L1160)、[:1625](../vllm/v1/core/single_type_kv_cache_manager.py#L1625)），因为它们的前缀是 null block。

### 3.12 PagedAttention kernel：读与写两条路径

**写路径**（把新算出的 K/V 落进 paged cache）：

[vllm/_custom_ops.py:2600](../vllm/_custom_ops.py#L2600) → `torch.ops._C_cache_ops.reshape_and_cache` → [csrc/libtorch_stable/cache_kernels.cu:759](../csrc/libtorch_stable/cache_kernels.cu#L759)：

```cpp
void reshape_and_cache(
    torch::stable::Tensor& key,     // [num_tokens, num_heads, head_size]
    torch::stable::Tensor& value,
    torch::stable::Tensor& key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::stable::Tensor& value_cache,  // [num_blocks, num_heads, head_size, block_size]
    torch::stable::Tensor& slot_mapping, // [num_tokens]
    const std::string& kv_cache_dtype, ...)
```

注意 `key_cache` 的 shape 是 `[num_blocks, num_heads, head_size/x, block_size, x]`（`x = 16 / element_size()`，即 16 字节向量化宽度）——这就是 PagedAttention 原论文的 **K cache 布局**：把 `head_size` 拆成 `(head_size/x, x)` 两段，让同一 token 的连续 `x` 个元素在内存里相邻，一次 16B 向量访存。

`reshape_and_cache_flash`（[:805](../csrc/libtorch_stable/cache_kernels.cu#L805)）是给 FA 用的 `[num_blocks, block_size, num_heads, head_size]` 布局，支持任意 stride（`block_stride / page_stride / head_stride`）。

Python 侧封装在 [vllm/v1/attention/ops/paged_attn.py:32](../vllm/v1/attention/ops/paged_attn.py#L32) `PagedAttention.write_to_paged_cache`。

> **重要事实核对**：本仓库（写作时）的 `csrc/attention/` 下**已经没有** `paged_attention_v1_kernel.cu` / `paged_attention_v2_kernel.cu`，目录只剩 `attention_dtypes.h`、`attention_generic.cuh` 和几个 dtype 头。**读路径的 paged attention 已由 FlashAttention / FlashInfer / Triton 等外部或自研 backend 承担**，vLLM 自己保留的是「paged 显存管理 + 写 KV（`reshape_and_cache`）+ block table 传递」这套机制。`PagedAttention` 类（[paged_attn.py:15](../vllm/v1/attention/ops/paged_attn.py#L15)）现在只剩 `split_kv_cache` 和 `write_to_paged_cache` 两个静态方法。面试时不要再说「推理走 paged_attention_v1/v2 kernel」，那是 V0 时代的实现。

### 3.13 Mamba / 混合模型：非分页的 KV cache

`MambaSpec`（[kv_cache_interface.py:885](../vllm/v1/kv_cache_interface.py#L885)）是一个「分页机制的特例」：

```python
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.MAMBA2
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0
    num_prefill_checkpoint_blocks: int = 0
    num_heads: int = 1
    tokens_per_state: int = -1          # ← 不是 token 序列，是一段状态
    tp_replicated: bool = False
```

- 一个 block 装的是**一段递归状态**（conv state + SSM state），不是 `block_size` 个 token 的 K/V；
- `page_size_bytes = Σ prod(shape) * dtype_size`（[:912](../vllm/v1/kv_cache_interface.py#L912)）；
- `get_num_skipped_tokens = num_computed_tokens - 1`（[single_type_kv_cache_manager.py:1929](../vllm/v1/core/single_type_kv_cache_manager.py#L1929)）——**只保留最后一个 token 的状态**，前面的全部可以 free，所以理论上占用是 O(1) 个 block；
- `mamba_cache_mode == "align"` 时 `MambaManager` 用 `last_state_block_idx` 追踪「上上步分配的 block」并在下一步 free（[:1606](../vllm/v1/core/single_type_kv_cache_manager.py#L1606)），配合 null block 维持 block table 的矩形性；
- 因为状态是**原地覆盖**的（下一步直接写同一个 block），Mamba 的 prefix caching 必须做 copy-on-write，见 doc 04。

### 3.14 KV cache 量化（FP8）对 block 的影响

先明确一点：**FP8 不改变 `block_size`**（还是 16 个 token 一个 block），它改变的是 **每个 block 的字节数**，从而改变 `num_blocks`。

链路是：`kv_cache_dtype` 字符串 → `KVQuantMode`（[kv_cache_interface.py:39](../vllm/v1/kv_cache_interface.py#L39)，映射函数 `get_kv_quant_mode` 在 [:84](../vllm/v1/kv_cache_interface.py#L84)）→ `spec.dtype` → `state_content_size_bytes`：

```python
@property
def state_content_size_bytes(self) -> int:
    """Bytes per (head slot, stored state) cell of the page."""
    if self.state_content_bytes is not None:
        return self.state_content_bytes
    return (self.head_size + self.head_size_v) * get_dtype_size(self.dtype)
```

再回到 §3.3 的公式：

```
bytes_per_block = Σ_layers num_kv_heads × block_size × C
C_bf16 = (128 + 128) × 2 = 512 B
C_fp8  = (128 + 128) × 1 = 256 B      →  bytes_per_block 减半 → num_blocks 翻倍
```

用 §3.3 的例子：bf16 时 2 MiB/block、20 GiB → 10240 blocks（163,840 token）；**fp8(e4m3) per-tensor 时 1 MiB/block → 20480 blocks（327,680 token），KV 容量翻倍**，并发上限（`get_max_concurrency_for_kv_cache_config` [kv_cache_utils.py:1051](../vllm/v1/core/kv_cache_utils.py#L1051)）同样翻倍。

但有几处**抵消**要注意：

1. **per-token-head 量化模式**（`INT8_PER_TOKEN_HEAD` / `FP8_PER_TOKEN_HEAD` / `INT4_PER_TOKEN_HEAD`）会在 page 内联 fp32 scale。以 Triton backend 为例（[triton_attn.py:286](../vllm/v1/attention/backends/triton_attn.py#L286)）：`C = (hs_k + hs_v) * 1 + 2 * 4`，对 `head_size=128` 是 `256 + 8 = 264 B`，比理想 256 B 多 3%。
2. **page padding**：MLA 类 spec 的 `page_size_padded` 会向上取整到 `alignment`（[kv_cache_interface.py:544](../vllm/v1/kv_cache_interface.py#L544) 的 `_apply_alignment_padding`），`page_size_bytes` 取 padded 值（[:427](../vllm/v1/kv_cache_interface.py#L427)）——量化后 page 变小，padding 的**相对**开销反而变大了。
3. **混合精度**：`KVCacheConfig.needs_kv_cache_zeroing`（[:1361](../vllm/v1/kv_cache_interface.py#L1361)）为 True 时，worker 必须对新分配的 block 清零，因为「同一 block 被不同 group 复用时，按另一种精度解释旧字节会得到 NaN/Inf」。这是一笔真实的前向开销。
4. **kernel block size 约束**：量化 dtype 往往只有部分 backend 支持（`FlashAttentionBackend.supports_kv_cache_dtype` [flash_attn.py:178](../vllm/v1/attention/backends/flash_attn.py#L178)），`prepare_kernel_block_sizes`([worker/utils.py:462](../vllm/v1/worker/utils.py#L462)) 会据此选择 kernel block size，间接影响虚拟 block 拆分比例。

---

## 4. 关键数据结构

### 4.1 `KVCacheBlock`（[kv_cache_utils.py:164](../vllm/v1/core/kv_cache_utils.py#L164)）

| 字段 | 类型 | 含义 |
|---|---|---|
| `block_id` | `int` | 物理 block 编号，也是 GPU 侧 block table 里存的数 |
| `ref_cnt` | `int` | 引用计数；为 0 表示在 free queue 里（驱逐候选） |
| `_block_hash` | `BlockHashWithGroupId \| None` | 该 block 满块且被缓存时的 hash key（含 group id） |
| `_block_hash_num_tokens` | `int \| None` | 该 hash 覆盖的前缀 token 数（partial entry 可 < block 边界） |
| `prev_free_block` / `next_free_block` | `KVCacheBlock \| None` | 只由 `FreeKVCacheBlockQueue` 操作的链表指针 |
| `is_null` | `bool` | 是否为 null block（永不缓存、ref_cnt 不维护） |

### 4.2 `KVCacheTensor`（[kv_cache_interface.py:1241](../vllm/v1/kv_cache_interface.py#L1241)）

| 字段 | 类型 | 含义 |
|---|---|---|
| `size` | `int` | 整个 backing allocation 的字节数（所有 tensor 相同） |
| `layers` | `list[str]` | 按 L 维排列的层名 |
| `layer_stride` | `int` | 相邻层的字节 stride |
| `block_stride` | `int` | 相邻 block 的字节 stride |
| `offset` | `int` | `layers[0]` 的 block 0 的字节偏移 |

### 4.3 `KVCacheConfig`（[kv_cache_interface.py:1282](../vllm/v1/kv_cache_interface.py#L1282)）

| 字段 | 类型 | 含义 |
|---|---|---|
| `num_blocks` | `int` | block pool 的 block 总数 |
| `kv_cache_tensors` | `list[KVCacheTensor]` | 每个「同 shape 层集合」的字节布局 |
| `kv_cache_groups` | `list[KVCacheGroupSpec]` | 逻辑分组，决定有几张 block table |
| `prefix_cache_retention_interval` | `int \| None` | 稀疏 checkpoint 保留策略（SWA/Mamba） |
| `kv_cache_layout` | `str \| None` | engine core 解析出的物理布局名 |

### 4.4 `SingleTypeKVCacheManager` 家族（[single_type_kv_cache_manager.py](../vllm/v1/core/single_type_kv_cache_manager.py)）

| 类 | 对应 spec | `get_num_skipped_tokens` | `supports_fine_grained_hash_lookup` |
|---|---|---|---|
| `FullAttentionManager` [:714](../vllm/v1/core/single_type_kv_cache_manager.py#L714) | `FullAttentionSpec` / MLA | 0（永不复用）| `True` |
| `SlidingWindowManager` [:921](../vllm/v1/core/single_type_kv_cache_manager.py#L921) | `SlidingWindowSpec` | `n - w + 1 - extra` | `False` |
| `ChunkedLocalAttentionManager` [:1263](../vllm/v1/core/single_type_kv_cache_manager.py#L1263) | `ChunkedLocalAttentionSpec` | `n // chunk * chunk` | `False` |
| `MambaManager` [:1421](../vllm/v1/core/single_type_kv_cache_manager.py#L1421) | `MambaSpec` | `n - 1` | `True` |
| `CircularBufferManager` [:1170](../vllm/v1/core/single_type_kv_cache_manager.py#L1170) | `CircularBufferSpec` | 0 | `False` |
| `CrossAttentionManager` [:2064](../vllm/v1/core/single_type_kv_cache_manager.py#L2064) | `CrossAttentionSpec` | 0 | — |

### 4.5 `BlockTable`（[block_table.py:57](../vllm/v1/worker/block_table.py#L57)）

| 字段 | 类型 | 含义 |
|---|---|---|
| `block_table` | `CpuGpuBuffer[int32]` | `[max_num_reqs, max_num_blocks_per_req]` 的 CPU/GPU 双缓冲 |
| `num_blocks_per_row` | `np.ndarray[int32]` | 每行有效长度 |
| `slot_mapping` | `CpuGpuBuffer[int64]` | `[max_num_batched_tokens]`，token → 物理 slot |
| `blocks_per_kv_block` | `int` | manager block 拆成几个 kernel block |
| `use_hybrid_blocks` | `bool` | 是否发生虚拟 block 拆分 |

---

## 5. 收益与代价

### 5.1 收益

| 收益 | 量化 |
|---|---|
| 消除预留浪费 | 显存按实际 token 数按需分配，`BlockPool.get_usage()`（[block_pool.py:823](../vllm/v1/core/block_pool.py#L823)）= `1 - free/(num_blocks-1)` 就是真实占用率 |
| 消除外部碎片 | block 等大，任意 block 可被任意请求复用，无空洞问题 |
| 内部碎片可控 | 每请求平均浪费 `block_size/2` token；`block_size=16`、平均 1k token 时 ≈ 0.78% |
| 前缀共享 | 相同前缀只存一份，N 个请求共享 P token 前缀 → 省 `(N-1)×P` token 的 KV（详见 doc 04） |
| 精确容量规划 | `num_blocks = available_memory // bytes_per_block`，`max_memory_usage_bytes` 由 spec 精确推导（`estimate_max_model_len` [kv_cache_utils.py:905](../vllm/v1/core/kv_cache_utils.py#L905) 二分求最大可服务长度） |
| 支持复杂注意力 | 同一套 block 抽象通过 `get_num_skipped_tokens` 覆盖 full / sliding window / chunked local / Mamba |
| 支持 P/D 分离与 CPU offload | block 是离散的，天然可按 block 粒度搬运（`KVCacheConfig.transfer_group_ids` [kv_cache_interface.py:1305](../vllm/v1/kv_cache_interface.py#L1305)） |

### 5.2 代价与限制

1. **间接寻址开销**：kernel 每次访存都要过 block table。现代 backend 把整张 block table 传进 kernel，开销被隐藏，但相比连续内存仍多一次索引 load。
2. **尾部内部碎片**：`block_size` 越大碎片越大（平均 `block_size/2`），但 `block_size` 越小 Python 侧管理开销与 block table 宽度越大（`max_num_blocks_per_req = cdiv(max_model_len, block_size)`，直接决定 GPU 显存占用与 kernel 的循环次数）。默认 16 是折中。
3. **null block 的隐性成本**：SWA / Mamba 的 block table 里塞了大量 null block，它们的 KV 内容被读进 kernel 但被 mask 掉——**浪费带宽**。这也是 `reachable_block_mask`（[single_type_kv_cache_manager.py:504](../vllm/v1/core/single_type_kv_cache_manager.py#L504)）存在的原因（只在可能命中的位置缓存，减少无效状态）。
4. **混合模型的 block size 耦合**：多 group 时 `scheduler_block_size = lcm(所有 group block_size)`（[kv_cache_utils.py:730](../vllm/v1/core/kv_cache_utils.py#L730)），一个 group 用 16、另一个用 64 → 调度粒度被放大到 64，短请求的对齐浪费变大。
5. **kernel block size 约束**：`block_size` 必须是 backend 支持的整数倍，否则要虚拟拆分（`map_to_kernel_blocks`），这会放大 block table 宽度。
6. **不适用场景**：`CrossAttentionSpec`（encoder-decoder 的 cross attention）直接禁用 prefix caching（[single_type_kv_cache_manager.py:2126](../vllm/v1/core/single_type_kv_cache_manager.py#L2126)）；`CircularBufferSpec` / `KpoolTailSpec` 标记 `prefix_cacheable = False`（[:793](../vllm/v1/kv_cache_interface.py#L793)、[:880](../vllm/v1/kv_cache_interface.py#L880)）。

---

## 6. 面试高频问题

**Q1：为什么需要分页 KV cache？连续分配到底浪费在哪？**
答：三重浪费 —— (a) **预留**：按 `max_model_len` 预留，实际远用不满；(b) **内部碎片**：最后一个分配单元没用满；(c) **外部碎片**：长短请求释放后留下大小不一的空洞。vLLM SOSP'23 论文实测现有系统浪费 60%~80% 显存。分页后 (b) 降到平均 `block_size/2` token/请求，(a)(c) 彻底消失。代码侧对应：`num_blocks = available_memory // bytes_per_block`（[kv_cache_utils.py:1710](../vllm/v1/core/kv_cache_utils.py#L1710)），没有任何预留系数。

**Q2：`block_size` 怎么影响系统？默认多少？**
答：默认 16。影响三处：内部碎片（∝ `block_size/2`）、block table 宽度 `cdiv(max_model_len, block_size)`（GPU 显存 + kernel 循环）、prefix caching 的粒度（不满块不缓存，见 doc 04）。混合模型还会触发 `scheduler_block_size = lcm(...)`（[kv_cache_utils.py:730](../vllm/v1/core/kv_cache_utils.py#L730)）。

**Q3：`null_block` 是什么？为什么需要它？**
答：[block_pool.py:190](../vllm/v1/core/block_pool.py#L190) 从 free queue 队头取出的 block，`is_null=True`。SWA 滑窗外、Mamba 旧状态这些「逻辑上已丢弃」的位置，block table 里必须有个值（张量是矩形的），于是填 null block；其 KV 全 0，配合 kernel mask 跳过。`free_blocks` 里显式跳过 `is_null`（[:747](../vllm/v1/core/block_pool.py#L747)），`touch` 里也跳过（[:724](../vllm/v1/core/block_pool.py#L724)）。

**Q4：`ref_cnt` 怎么保证正在使用的 block 不被驱逐？**
答：引用计数 > 0 的 block **不在 free queue 里**。分配时 `get_new_blocks` 用 `popleft_n` 从队头取（[block_pool.py:672](../vllm/v1/core/block_pool.py#L672)）；prefix 命中时 `touch` 把 `ref_cnt == 0` 的 block 从队中 `remove` 掉（[:713](../vllm/v1/core/block_pool.py#L713)）。所以驱逐只可能发生在 free queue 内的 block 上，天然安全。

**Q5：`free_blocks` 为什么对「有 hash / 无 hash」的 block 用不同的入队方式？**
答：[block_pool.py:734](../vllm/v1/core/block_pool.py#L734)。无 hash 的 block 永远不可能被 prefix 命中，用 `prepend_n` 放到队头做 **LIFO** 复用，刚释放的 block 还在各级 cache 里热着，GPU 局部性好；有 hash 的 block 是 prefix cache 的候选，用 `append_n` 放到队尾做 **FIFO**，等价于 LRU——最早进队的（最久未用的）最先被 `popleft` 分配出去。

**Q6：V1 的 spec / group / coordinator 三层是怎么分工的？**
答：`KVCacheSpec` 描述**一层**怎么存（[kv_cache_interface.py:151](../vllm/v1/kv_cache_interface.py#L151)）；同 spec 的层合成 `KVCacheGroupSpec`（[:1265](../vllm/v1/kv_cache_interface.py#L1265)）共享一张 block table；`KVCacheCoordinator`（[kv_cache_coordinator.py:66](../vllm/v1/core/kv_cache_coordinator.py#L66)）仲裁多 group 的一致行为（尤其 prefix hit 长度必须各 group 一致）。每个 group 一个 `SingleTypeKVCacheManager`，由 `KVCacheSpecRegistry` 按 MRO 查表得到（[kv_cache_spec_registry.py:105](../vllm/v1/kv_cache_spec_registry.py#L105)）。

**Q7：SWA 的 block 是怎么复用的？**
答：`SlidingWindowManager.get_num_skipped_tokens(n) = max(0, n - w + 1 - extra)`（[single_type_kv_cache_manager.py:1155](../vllm/v1/core/single_type_kv_cache_manager.py#L1155)），`remove_skipped_blocks` → `_remove_blocks_in_range`（[:631](../vllm/v1/core/single_type_kv_cache_manager.py#L631)）**从后往前**把滑窗外的 block free 掉并替换为 null block。准入时用 `max_admission_blocks_per_request`（[:716](../vllm/v1/kv_cache_interface.py#L716)）夹紧，注释指出它必须与启动时的 pool sizing 用同一个公式，否则会重现 issue #39734 的死锁。

**Q8：Mamba 的 KV cache 和 attention 的有什么本质区别？**
答：`MambaSpec.tokens_per_state = -1`、`num_heads = 1`（[kv_cache_interface.py:894](../vllm/v1/kv_cache_interface.py#L894)），一个 block 装的是**递归状态**而不是 token 序列，`page_size_bytes = Σ prod(shape)*dtype_size`（[:912](../vllm/v1/kv_cache_interface.py#L912)）。`get_num_skipped_tokens(n) = n - 1`（[single_type_kv_cache_manager.py:1929](../vllm/v1/core/single_type_kv_cache_manager.py#L1929)）——只需保留最后一个 token 的状态。代价是状态被原地覆盖，prefix caching 必须 copy-on-write（doc 04）。

**Q9：slot mapping 是怎么生成的？为什么需要它？**
答：[block_table.py:413](../vllm/v1/worker/block_table.py#L413) 的 Triton kernel：每个 program 处理一个请求，按 `pos // block_size` 查 block table、按 `pos % block_size` 算块内偏移，`slot = block_id * block_size + offset`，非本地（DCP/PCP）的位置填 `PAD_SLOT_ID = -1`。写 KV 时传给 `reshape_and_cache`，kernel 按它 scatter。

**Q10：manager block size 和 kernel block size 不一样怎么办？**
答：`prepare_kernel_block_sizes`（[worker/utils.py:462](../vllm/v1/worker/utils.py#L462)）按 backend 支持列表选一个能整除 manager block size 的值；`BlockTable.map_to_kernel_blocks`（[block_table.py:238](../vllm/v1/worker/block_table.py#L238)）把 manager block id `b` 展开成 `[b*k, b*k+1, ..., b*k+k-1]`。这样调度侧可以按大 block 降低 Python 开销，kernel 侧仍用小 block。

**Q11：FP8 KV cache 到底省在哪？对 block size 有影响吗？**
答：不影响 `block_size`（还是 16 token/block），影响 `page_size_bytes`：`C = (head_size + head_size_v) * get_dtype_size(dtype)`（[kv_cache_interface.py:420](../vllm/v1/kv_cache_interface.py#L420)），bf16 → fp8 让 C 从 512 B 降到 256 B → `bytes_per_block` 减半 → `num_blocks` 翻倍 → 并发上限翻倍。但 per-token-head 模式会内联 fp32 scale（[triton_attn.py:286](../vllm/v1/attention/backends/triton_attn.py#L286)，+8 B/单元），MLA 的 `page_size_padded` 对齐也会吃掉一部分收益；混合精度还会打开 `needs_kv_cache_zeroing`（[kv_cache_interface.py:1361](../vllm/v1/kv_cache_interface.py#L1361)）带来清零开销。

**Q12：本仓库还有 `paged_attention_v1_kernel.cu` 吗？推理走的是哪个 kernel？**
答：**没有**。`csrc/attention/` 现在只有 `attention_dtypes.h` / `attention_generic.cuh` / 几个 dtype 头。V1 的读路径交给 FlashAttention / FlashInfer / Triton 等 backend，它们整体接收 `block_table` 张量（[flash_attn.py:729](../vllm/v1/attention/backends/flash_attn.py#L729)）在 kernel 内部做 paged 访问；vLLM 自己保留的是 paged 显存管理 + `reshape_and_cache` 写路径（`_custom_ops.py:2600` → `csrc/libtorch_stable/cache_kernels.cu:759`）。

**Q13：`available_memory` 是怎么算出来的？为什么设了 `gpu_memory_utilization=0.9` 还是不 OOM？**
答：[gpu_worker.py:525](../vllm/v1/worker/gpu_worker.py#L525)：先跑一次 `profile_run()` 实测峰值（`memory_profiling` 上下文），`available = requested_memory - non_kv_cache_memory - cudagraph_estimate`。`requested_memory = total * gpu_memory_utilization`。因为减的是**实测**模型+激活峰值而非估值，所以不会 OOM。`kv_cache_memory_bytes` 可以手工指定跳过 profiling。

**Q14：多 group 时各 group 的 KV cache 在显存里是怎么排的？**
答：所有 `KVCacheTensor` 共享同一个 `size = bytes_per_block * num_blocks` 的 backing buffer，且 `offset` 从 0 开始——**互相 alias**（[kv_cache_utils.py:1714](../vllm/v1/core/kv_cache_utils.py#L1714) 注释）。安全性来自「同一时刻一个 block id 只属于一个 group」。具体 stride 由 `KVCacheLayout`（[kv_cache_layout.py:15](../vllm/v1/kv_cache_layout.py#L15)）决定：层优先给每层一段连续显存，块优先让一个 block 包含所有层的 page（后者对 P/D 传输友好）。

---

## 7. 延伸阅读

### 源码文件清单

| 文件 | 职责 |
|---|---|
| [vllm/v1/kv_cache_interface.py](../vllm/v1/kv_cache_interface.py) | spec / group / config / tensor 全部数据类 + 布局计算 |
| [vllm/v1/kv_cache_layout.py](../vllm/v1/kv_cache_layout.py) | 物理布局枚举（stride 置换） |
| [vllm/v1/kv_cache_spec_registry.py](../vllm/v1/kv_cache_spec_registry.py) | spec → manager 注册表，out-of-tree 扩展点 |
| [vllm/v1/core/kv_cache_utils.py](../vllm/v1/core/kv_cache_utils.py) | `KVCacheBlock`、`FreeKVCacheBlockQueue`、hash、config 生成 |
| [vllm/v1/core/block_pool.py](../vllm/v1/core/block_pool.py) | block 池、hash→block 映射、LRU 驱逐 |
| [vllm/v1/core/single_type_kv_cache_manager.py](../vllm/v1/core/single_type_kv_cache_manager.py) | 各注意力类型的分配/回收逻辑 |
| [vllm/v1/core/kv_cache_coordinator.py](../vllm/v1/core/kv_cache_coordinator.py) | 多 group 仲裁 |
| [vllm/v1/core/kv_cache_manager.py](../vllm/v1/core/kv_cache_manager.py) | 调度器门面 |
| [vllm/v1/core/kv_cache_metrics.py](../vllm/v1/core/kv_cache_metrics.py) | block 驻留时间采样 |
| [vllm/v1/worker/block_table.py](../vllm/v1/worker/block_table.py) | GPU block table + slot mapping Triton kernel |
| [vllm/v1/worker/utils.py](../vllm/v1/worker/utils.py) | `allocate_kv_cache` / `prepare_kernel_block_sizes` |
| [vllm/v1/worker/gpu_model_runner.py](../vllm/v1/worker/gpu_model_runner.py) | `get_kv_cache_spec` / `initialize_kv_cache` |
| [vllm/v1/worker/gpu_worker.py](../vllm/v1/worker/gpu_worker.py) | `determine_available_memory` |
| [vllm/v1/attention/backends/flash_attn.py](../vllm/v1/attention/backends/flash_attn.py) | backend 消费 block table |
| [vllm/v1/attention/ops/paged_attn.py](../vllm/v1/attention/ops/paged_attn.py) | `PagedAttention` 封装（只剩写路径）|
| [vllm/_custom_ops.py](../vllm/_custom_ops.py) | `reshape_and_cache` Python 入口（:2600）|
| [csrc/libtorch_stable/cache_kernels.cu](../csrc/libtorch_stable/cache_kernels.cu) | `reshape_and_cache` CUDA 实现（:759）|

### 官方与论文

- 论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention*（Kwon et al., SOSP 2023）
- 仓库内设计文档：[docs/design/paged_attention.md](../docs/design/paged_attention.md)
- 官方文档：https://docs.vllm.ai/en/latest/design/
- 配套文档：[04-prefix-caching.md](./04-prefix-caching.md)
