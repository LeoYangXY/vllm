# 12. LoRA 与多 LoRA 热插拔服务

## 0. TL;DR

- **是什么**：vLLM 在一个进程里同时持有上百个 LoRA adapter，让它们共享同一份 base 权重，只把 `A/B` 低秩矩阵按需换入 GPU。
- **解决什么**：多租户微调服务里"每个租户一份模型"的显存与成本爆炸（8B bf16 底座 16GB，r=16 的 adapter 只有几十 MB，量级差 1000 倍）。
- **怎么做**：静态预分配 `max_loras` 个 adapter slot（`lora_a_stacked` / `lora_b_stacked`），用 LRU 在 CPU/GPU 两级缓存间换入换出；每个 token 带一个 lora id，Punica 把「一 batch 里 N 个不同 adapter 的 GEMM」合成**一次** kernel launch。
- **额外收益**：请求级 adapter 选择（`LoRARequest` 随请求走），无需重启引擎即可 `add_lora` / `remove_lora` / `pin_lora`。
- **代价**：slot 数量是硬上限（调度器会卡住），prefix caching 按 adapter 名隔离，CUDA Graph 需要为 LoRA 多抓一份（或专门化多份）。

---

## 1. 场景与痛点

### 1.1 多租户 SaaS：base weight 必须共享

一个 LoRA adapter 的参数量是 `2 * r * d_in * d_out`（一层），一个 7B 模型如果对所有 linear 都加 LoRA，r=16 时总参数量大约 **1%~3%** 的 base，也就是几十 MB 到一两百 MB。而 base 权重本身 bf16 就是 14GB。

如果按"一个 adapter 一份模型"部署：

| 部署方式 | 100 个 adapter 的显存 |
| --- | --- |
| 每 adapter 一份完整模型（H100 80G） | 100 × 16GB ≈ 1.6TB → 20+ 卡 |
| 共享 base + 常驻 adapter | 16GB + 100 × 0.1GB ≈ 26GB → 1 卡 |

所以工程上唯一可行的路线是：**一份 base 权重 + 一堆随时换入换出的小矩阵**。

### 1.2 朴素做法为什么不行

最直觉的"多 LoRA"实现是逐请求做 GEMM：

```text
for req in batch:
    y[req] += x[req] @ A[req.lora].T @ B[req.lora].T * scale
```

这条路线上有三个致命问题：

1. **kernel launch 数量 = batch_size × num_layers × 2**。batch=256、32 层模型，一次 forward 就是 16K 次 launch，launch 开销（~5us）远超计算本身（r=16 的 GEMM 只有几十 us 但 size 极小）。
2. **batch 内 adapter 混杂时，每个请求的 token 数很小**，GEMM 退化成 memory-bound 的 skinny GEMM，tensor core 利用率个位数。
3. **权重搬运**：每 step 都把 adapter 从 host 拷到 device，PCIe 带宽被吃光。

vLLM/Punica 的解法是把「逐请求 GEMM」变成「按 adapter 分组的 grouped GEMM」——见 §3.5。

### 1.3 热插拔的工程约束

多租户场景下 adapter 集合是**动态**的：新租户上线要加载、老租户下线要卸载、热点 adapter 要钉住不被驱逐。这要求：

- 加载不能阻塞太久（不能为每个 adapter 重新 `torch.compile`）；
- 显存上界必须可预测（否则 OOM 掉整个服务）；
- 换入换出必须和调度器协同（不能一个 batch 需要 5 个 adapter 而 GPU 上只有 3 个 slot）。

---

## 2. 核心设计

### 2.1 三个核心抽象

```text
┌──────────────────────────────────────────────────────────────────────┐
│                        LoRAConfig (静态, 启动时)                       │
│   max_loras=8   max_lora_rank=16   max_cpu_loras=8   lora_dtype       │
└──────────────────────────────────────────────────────────────────────┘
                                  │
      ┌───────────────────────────┼───────────────────────────┐
      ▼                           ▼                           ▼
┌───────────────┐        ┌──────────────────┐        ┌────────────────┐
│ LoRAModelManager│       │  LoRA layers     │        │ PunicaWrapper  │
│ (slot 分配/LRU) │◄─────►│ (被注入到 model) │◄──────►│ (kernel 元数据) │
└───────────────┘        └──────────────────┘        └────────────────┘
   lora_index_to_id          lora_a_stacked              token_lora_mapping
   _registered_adapters      [max_loras, 1, r, in]       active_lora_ids
   _active_adapters          lora_b_stacked              num_tokens_per_lora
                             [max_loras, 1, out, r]      lora_token_start_loc
```

- **`LoRAModelManager`** 管"哪些 adapter 在 CPU 上（`capacity = max_cpu_loras`）、哪些在 GPU slot 上（`lora_slots = max_loras`）"。[model_manager.py:71](../vllm/lora/model_manager.py#L71)
- **`BaseLayerWithLoRA`** 是包在原始 linear 外面的 wrapper，持有**预分配**的 `lora_a_stacked` / `lora_b_stacked`，形状第一维就是 `max_loras`——这是 slot 概念的物化。[base_linear.py:129](../vllm/lora/layers/base_linear.py#L129)
- **`PunicaWrapper`** 每 step 根据 batch 计算一次"token → adapter"的索引元数据，供所有 LoRA 层共用。[punica_gpu.py:34](../vllm/lora/punica_wrapper/punica_gpu.py#L34)

### 2.2 一 step 的完整数据流

```text
Scheduler                ModelRunner                    LoRA Layers
   │                          │                              │
   │ scheduled_reqs (带 lora) │                              │
   ├─────────────────────────►│                              │
   │                          │ set_active_loras()           │
   │                          │  ① make_lora_inputs() 生成   │
   │                          │     token_lora_mapping        │
   │                          │  ② lora_manager               │
   │                          │     .set_active_adapters()    │
   │                          │      → 换入缺失 adapter       │
   │                          │      → 分配 slot index        │
   │                          │  ③ punica.update_metadata()  │
   │                          │      准备 kernel 元数据       │
   │                          ├────────────────────────────►│
   │                          │                              │ forward:
   │                          │                              │  base GEMM
   │                          │                              │  + add_shrink
   │                          │                              │  + add_expand
   │                          │◄─────────────────────────────┤
```

关键点：**adapter 的权重拷贝（②）和 kernel 元数据准备（③）每 step 只做一次**，与层数无关；层里只做 `add_lora_linear(output, x, ...)` 这样的一次调用。

### 2.3 slot 分配示意

```text
lora_index_to_id = [103, None, 7, None, ...]     # 长度 = max_loras
                     │            │
                     │            └── slot 2 → adapter 7
                     └────────────── slot 0 → adapter 103

token_lora_mapping = [-1, -1, 0, 0, 2, 2, 2, ...]
                                 │  │  └────── slot 2
                                 └──┴───────── slot 0
                     -1 = 该 token 不用 LoRA
```

注意 `token_lora_mapping` 里放的是 **slot index**（0..max_loras-1），不是 `lora_int_id`。这个转换在 `convert_mapping` 里做，用反向查表代替 `list.index()` 线性扫描。[punica_wrapper/utils.py:54](../vllm/lora/punica_wrapper/utils.py#L54)

---

## 3. 代码走读

### 3.1 请求入口：`LoRARequest` 怎么跟请求走

`LoRARequest` 是一个 `msgspec.Struct`，`array_like=True`，所以它可被高效序列化跨进程（API server → EngineCore）传输。[request.py:8](../vllm/lora/request.py#L8)

```python
# vllm/lora/request.py:8
class LoRARequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    lora_name: str
    lora_int_id: int
    lora_path: str = ""
    base_model_name: str | None = msgspec.field(default=None)
    tensorizer_config_dict: dict | None = None
    load_inplace: bool = False
    is_3d_lora_weight: bool = False
```

两个容易踩坑的细节：

- `__eq__` / `__hash__` **只看 `lora_name`**（[request.py:58](../vllm/lora/request.py#L58)）。这意味着 `set[LoRARequest]` 里"同一个 adapter"的判等不依赖 `lora_int_id`；但缓存的 key 用的是 `lora_int_id`（见 `list_adapters()` 返回 `set[int]`）。**实践中必须保证 name ↔ int_id 一一对应**，否则会出现"名字相同但 id 不同"导致缓存穿透。
- `load_inplace=True` 会**强制重载**同名 adapter，即使缓存里已经有——用于热更新 adapter 权重。[worker_manager.py:294](../vllm/lora/worker_manager.py#L294)

请求侧：[vllm/v1/request.py:71](../vllm/v1/request.py#L71) 里 `Request.__init__` 接 `lora_request`，[vllm/v1/request.py:87](../vllm/v1/request.py#L87) 存成 `self.lora_request`。

调度时 `InputBatch` 把它摊平成一条 int 数组：

```python
# vllm/v1/worker/gpu_input_batch.py:1003
def make_lora_inputs(
    self, num_scheduled_tokens: np.ndarray, num_sampled_tokens: np.ndarray
) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
    req_lora_mapping = self.request_lora_mapping[: self.num_reqs]
    prompt_lora_mapping = tuple(req_lora_mapping.repeat(num_sampled_tokens))
    token_lora_mapping = tuple(req_lora_mapping.repeat(num_scheduled_tokens))

    active_lora_requests: set[LoRARequest] = set(
        self.lora_id_to_lora_request.values()
    )
    return prompt_lora_mapping, token_lora_mapping, active_lora_requests
```

`np.repeat` 把"每请求一个 lora id"扩成"每 token 一个 lora id"，同时 `prompt_lora_mapping` 按**采样 token** 扩（logits LoRA 只需要最后几个位置）。`lora_id_to_lora_request` 在请求加入/移除 batch 时维护（[gpu_input_batch.py:262](../vllm/v1/worker/gpu_input_batch.py#L262)、[:487](../vllm/v1/worker/gpu_input_batch.py#L487)、[:555](../vllm/v1/worker/gpu_input_batch.py#L555)）。

### 3.2 启动期：层替换与显存预分配

`LoRAModelRunnerMixin.load_lora_model` 在模型加载后调用，先建 manager，再用 manager 改写模型：[lora_model_runner_mixin.py:47](../vllm/v1/worker/lora_model_runner_mixin.py#L47)

```python
# vllm/v1/worker/lora_model_runner_mixin.py:47
def load_lora_model(self, model, vllm_config, device) -> nn.Module:
    if not supports_lora(model):
        raise ValueError(f"{model.__class__.__name__} does not support LoRA yet.")
    self.lora_manager = LRUCacheWorkerLoRAManager(
        vllm_config, device, model.embedding_modules,
    )
    return self.lora_manager.create_lora_manager(model, vllm_config)
```

真正的替换逻辑在 `LoRAModelManager._create_lora_modules()`（[model_manager.py:407](../vllm/lora/model_manager.py#L407)），它对每个 submodule 调 `from_layer()`：

```python
# vllm/lora/utils.py:107
def from_layer(layer, max_loras, lora_config, packed_modules_list, model_config=None):
    for lora_cls in _all_lora_classes:
        if lora_cls.can_replace_layer(
            source_layer=layer, lora_config=lora_config,
            packed_modules_list=packed_modules_list, model_config=model_config,
        ):
            instance_layer = lora_cls(layer)
            instance_layer.create_lora_weights(max_loras, lora_config, model_config)
            return instance_layer
    return layer
```

`_all_lora_classes` 是一个**有序** tuple（[utils.py:79](../vllm/lora/utils.py#L79)），注释明确说"更具体的 wrapper 必须排在通用 wrapper 前面"——因为 `MergedQKVParallelLinearWithLoRA`（3 个 packed module）和 `QKVParallelLinearWithLoRA`（1 个）都可能匹配 `QKVParallelLinear`，靠 `packed_modules_list` 长度区分，顺序错了会选错类。

权重缓冲在 `create_lora_weights` 里一次性 `torch.zeros` 出来：

```python
# vllm/lora/layers/base_linear.py:100
def create_lora_weights(self, max_loras, lora_config, model_config=None) -> None:
    self.lora_config = lora_config
    if isinstance(self.base_layer, ReplicatedLinear):
        lora_a_out_size = lora_config.max_lora_rank
        lora_b_out_size = self.output_size
    elif isinstance(self.base_layer, ColumnParallelLinear):
        lora_a_out_size = (
            lora_config.max_lora_rank
            if not lora_config.fully_sharded_loras
            else divide(lora_config.max_lora_rank, self.tp_size)
        )
        lora_b_out_size = self.output_size
    elif isinstance(self.base_layer, RowParallelLinear):
        lora_a_out_size = lora_config.max_lora_rank
        lora_b_out_size = (
            self.output_size
            if not lora_config.fully_sharded_loras
            else divide(self.output_size, self.tp_size)
        )
    ...
    self.lora_a_stacked = tuple(
        torch.zeros(max_loras, 1, lora_a_out_size, self.input_size,
                    dtype=lora_config.lora_dtype, device=self.device)
        for _ in range(self.n_slices)
    )
    self.lora_b_stacked = tuple(
        torch.zeros(max_loras, 1, lora_b_out_size, lora_config.max_lora_rank,
                    dtype=lora_config.lora_dtype, device=self.device)
        for _ in range(self.n_slices)
    )
```

注意：

- 第一维 `max_loras` 是**静态**的，所以 LoRA 显存占用完全可预测：`num_lora_layers × 2 × max_loras × r × d`。这也是 [config/scheduler.py:249](../vllm/config/scheduler.py#L249) 注释说的"LoRA 基于 `max_num_batched_tokens` 创建静态 buffer，tensor 的 shape/stride 会被 torch.compile 显式捕获"的原因。
- 第 2 维恒为 1，是为了统一 `n_slices` 语义（把多个 slice 放在 tuple 第 0 维）。
- `fully_sharded_loras` 改变的是 **A/B 谁被切分**，见 §3.6。

### 3.3 调度器的 `max_loras` 硬约束

调度器在调度 running 请求后先统计已占用的 adapter 集合：[scheduler.py:837](../vllm/v1/core/sched/scheduler.py#L837)

```python
# vllm/v1/core/sched/scheduler.py:837
scheduled_loras: set[int] = set()
if self.lora_config:
    scheduled_loras = set(
        req.lora_request.lora_int_id
        for req in scheduled_running_reqs
        if req.lora_request and req.lora_request.lora_int_id > 0
    )
    assert len(scheduled_loras) <= self.lora_config.max_loras
```

然后从 waiting 队列取请求时，如果"再加一个新 adapter 就超了"，**直接跳过该请求**（而不是中断调度）：[scheduler.py:890](../vllm/v1/core/sched/scheduler.py#L890)

```python
# vllm/v1/core/sched/scheduler.py:890
if (
    self.lora_config
    and request.lora_request
    and (
        len(scheduled_loras) == self.lora_config.max_loras
        and request.lora_request.lora_int_id not in scheduled_loras
    )
):
    # Scheduling would exceed max_loras, skip.
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

被跳过的请求被放进 `step_skipped_waiting`，本 step 不再考虑。成功调度的请求在 [scheduler.py:1249](../vllm/v1/core/sched/scheduler.py#L1249) 把自己的 adapter 加入集合：

```python
if self.lora_config and request.lora_request:
    scheduled_loras.add(request.lora_request.lora_int_id)
```

**面试要点**：这个约束是"**并发 adapter 数 ≤ max_loras**"，不是"并发请求数"。同一个 adapter 上的 100 个请求只占 1 个 slot。这也是为什么多租户场景要把 `max_loras` 调到 32/64 这种量级，而显存代价只有线性增长。

另一个隐含点：抢占路径**不会**因为 LoRA 约束被触发——上一步的 `assert` 保证了 running 集合永远不超。也就是说 LoRA 不会导致"为了腾 slot 而抢占"。

### 3.4 运行期：`set_active_loras` → slot 分配

model runner 每 step 在准备输入时调用：[gpu_model_runner.py:2307](../vllm/v1/worker/gpu_model_runner.py#L2307)

```python
# vllm/v1/worker/gpu_model_runner.py:2306
# Hot-Swap lora model
if self.lora_config:
    assert (
        np.sum(num_sampled_tokens)
        <= self.vllm_config.scheduler_config.max_num_batched_tokens
    )
    self.set_active_loras(
        self.input_batch, num_scheduled_tokens, num_sampled_tokens
    )
```

mixin 里构造 `LoRAMapping`（注意 `is_prefill=True` 是**故意**的，注释说"让非 CUDA 平台也走 SGMV kernel"）：

```python
# vllm/v1/worker/lora_model_runner_mixin.py:64
def _set_active_loras(self, prompt_lora_mapping, token_lora_mapping,
                      lora_requests, mapping_type=LoRAMappingType.LANGUAGE) -> None:
    self._ensure_lora_enabled()
    # Set is_prefill to True, so we always use the SGMV kernels on
    # non-cuda platforms.
    lora_mapping = LoRAMapping(
        token_lora_mapping, prompt_lora_mapping,
        is_prefill=True, type=mapping_type,
    )
    self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
```

`LoRAMapping` 本身就是一个 dataclass：[layers/utils.py:33](../vllm/lora/layers/utils.py#L33)

```python
@dataclass
class LoRAMapping:
    index_mapping: tuple[int, ...]
    prompt_mapping: tuple[int, ...]
    is_prefill: bool = False
    type: LoRAMappingType = LoRAMappingType.LANGUAGE
```

`LoRAMappingType` 有 `LANGUAGE / TOWER / CONNECTOR` 三种（[layers/utils.py:27](../vllm/lora/layers/utils.py#L27)）——多模态模型可以对 vision tower 和 connector 用不同的 adapter 映射，见 §3.6 末。

`WorkerLoRAManager.set_active_adapters` 分两步：先 `_apply_adapters`（换入换出），再 `set_adapter_mapping`（准备 kernel 元数据）：[worker_manager.py:193](../vllm/lora/worker_manager.py#L193)

LRU 版本只换入、不主动卸载不在本 batch 的 adapter（交给"容量满时淘汰最旧"）：[worker_manager.py:272](../vllm/lora/worker_manager.py#L272)

```python
# vllm/lora/worker_manager.py:272
def _apply_adapters(self, lora_requests: set[LoRARequest]) -> None:
    loras_map = {
        lora_request.lora_int_id: lora_request
        for lora_request in lora_requests if lora_request
    }
    if len(loras_map) > self._adapter_manager.lora_slots:
        raise RuntimeError(
            f"Number of requested LoRAs ({len(loras_map)}) is greater "
            "than the number of GPU LoRA slots "
            f"({self._adapter_manager.lora_slots})."
        )
    for lora in loras_map.values():
        self.add_adapter(lora)
```

换入的核心逻辑（含"先加载成功再淘汰"的顺序保证）：[worker_manager.py:287](../vllm/lora/worker_manager.py#L287)

```python
# vllm/lora/worker_manager.py:287
def add_adapter(self, lora_request: LoRARequest) -> bool:
    with gpu_sync_allowed():
        if (lora_request.lora_int_id not in self.list_adapters()
                or lora_request.load_inplace):
            # Load the new adapter first to ensure it is actually valid, before
            # evicting any existing adapters.
            lora = self._load_adapter(lora_request)
            self._adapter_manager.remove_adapter(lora.id)
            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                self._adapter_manager.remove_oldest_adapter()
            loaded = self._adapter_manager.add_adapter(lora)
        else:
            loaded = (self._adapter_manager.get_adapter(lora_request.lora_int_id)
                      is not None)
        self._adapter_manager.activate_adapter(lora_request.lora_int_id)
    return loaded
```

这里有个很值得说的细节：**先 `_load_adapter`（可能失败）再 `remove_oldest_adapter`**。如果先淘汰再加载，加载失败就会白白丢掉一个已经热的 adapter。代价是"已加载数量可能瞬时超过 `max_cpu_loras`"——注释里明确承认了这一点。

slot 真正落位在 `activate_adapter`：[model_manager.py:315](../vllm/lora/model_manager.py#L315)

```python
# vllm/lora/model_manager.py:315
def activate_adapter(self, lora_id: int) -> bool:
    """Move LoRA into a GPU buffer to be used in the forward pass."""
    if lora_id in self._active_adapters:
        return False
    first_free_slot = next(
        ((i, lora_id) for i, lora_id in enumerate(self.lora_index_to_id)
         if lora_id is None), None,
    )
    if first_free_slot is None:
        raise ValueError("No free lora slots")
    index, _ = first_free_slot
    self._active_adapters[lora_id] = None
    lora_model = self._registered_adapters[lora_id]
    self.lora_index_to_id[index] = lora_model.id
    for module_name, module in self.modules.items():
        module_lora = self._get_lora_layer_weights(lora_model, module_name)
        if not module_lora:
            module.reset_lora(index)
            continue
        module.set_lora(index, module_lora.lora_a, module_lora.lora_b)
    return True
```

`_active_adapters` 也是个 LRU（容量 `lora_slots`），满了就淘汰最旧的（[model_manager.py:1213](../vllm/lora/model_manager.py#L1213)）：

```python
# vllm/lora/model_manager.py:1213
def activate_adapter(self, lora_id: int) -> bool:
    if (lora_id not in self._active_adapters
            and len(self._active_adapters) >= self.lora_slots):
        self._active_adapters.remove_oldest()
    result = super().activate_adapter(lora_id)
    self._active_adapters.touch(lora_id)
    return result
```

**面试要点**：换入一个 adapter 的成本是"遍历所有 LoRA 层 × 每层一次 `copy_`"。对一个 32 层模型来说就是几百次小 H2D copy（虽然 `non_blocking=True`，但次数摆在那）。这就是为什么 `max_loras` 开得太大 + adapter 频繁换入会明显掉吞吐，以及为什么需要 `pin_lora`。

### 3.5 Punica：为什么 grouped kernel 比逐请求 GEMM 快

Punica 的核心思想是：**把 batch 里属于同一个 adapter 的 token 聚到一起，让每个 adapter 只发一个 GEMM**，grid 的第三个维度就是 adapter 数。

元数据准备（每 step 一次）：[lora_kernel_metadata.py:109](../vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L109)

```python
# vllm/lora/ops/triton_ops/lora_kernel_metadata.py:109
def prepare_tensors(self, token_lora_mapping: torch.Tensor) -> None:
    self._reset()
    no_lora = torch.all(token_lora_mapping == -1)
    self.no_lora_flag_cpu[0] = no_lora
    if no_lora:
        # Early exit. LoRA kernels will not be run.
        return
    num_tokens = token_lora_mapping.size(0)
    self.token_lora_mapping[:num_tokens].copy_(token_lora_mapping, non_blocking=True)
    _, token_indices_sorted_by_lora_ids = torch.sort(token_lora_mapping, stable=True)
    self.token_indices_sorted_by_lora_ids[:num_tokens].copy_(
        token_indices_sorted_by_lora_ids, non_blocking=True)
    lora_ids, num_tokens_per_lora = torch.unique(
        token_lora_mapping, sorted=True, return_counts=True)
    self.active_lora_ids[:lora_ids.size(0)].copy_(lora_ids, non_blocking=True)
    self.num_tokens_per_lora[:num_tokens_per_lora.size(0)].copy_(
        num_tokens_per_lora, non_blocking=True)
    num_active_loras = lora_ids.size(0)
    ...
    lora_token_start_loc = torch.cumsum(num_tokens_per_lora, dim=0)
    self.lora_token_start_loc[1:1 + lora_token_start_loc.size(0)].copy_(
        lora_token_start_loc, non_blocking=True)
```

四个关键张量（`LoRAKernelMeta` dataclass，[lora_kernel_metadata.py:13](../vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L13)）：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `token_lora_mapping` | `[max_num_tokens]` | 每个 token 属于哪个 slot（-1 = 无 LoRA） |
| `token_indices_sorted_by_lora_ids` | `[max_num_tokens]` | 按 slot 稳定排序后的**原始 token 下标** |
| `active_lora_ids` | `[max_loras+1]` | 本 step 活跃的 slot 列表 |
| `num_tokens_per_lora` | `[max_loras+1]` | 每个 slot 有多少 token |
| `lora_token_start_loc` | `[max_loras+2]` | 每个 slot 在排序后序列里的起始位置（前缀和） |

kernel 侧怎么用：[lora_shrink_op.py:24](../vllm/lora/ops/triton_ops/lora_shrink_op.py#L24)

```python
# vllm/lora/ops/triton_ops/lora_shrink_op.py:70
slice_id = tl.program_id(axis=1)
lora_idx = tl.program_id(axis=2)

lora_id = tl.load(lora_ids + lora_idx)
if lora_id == -1:
    # Early exit for the no-lora case.
    return

lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

cta_m_offset = pid_m * BLOCK_M
if cta_m_offset >= lora_m_size:
    # Early exit CTA.
    return

cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
cta_lora_seq_indices = (
    token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
)
offset_m = tl.arange(0, BLOCK_M) % cta_m_len
ram = tl.load(cta_lora_seq_indices + offset_m)
```

launch 的 grid：[lora_shrink_op.py:223](../vllm/lora/ops/triton_ops/lora_shrink_op.py#L223)

```python
grid = (
    SPLIT_K * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
    NUM_SLICES,
    num_active_loras.item(),
)
```

**为什么这样更快**：

1. **launch 次数从 `O(batch_size × layers × 2)` 降到 `O(layers × 2)`**。每个 LoRA 层只 launch 一次 shrink + 一次 expand，不管 batch 里有 1 个还是 100 个 adapter。
2. **每个 CTA 处理的 M 维是"该 adapter 的 token 数"，而不是 1**。adapter 数远小于请求数时（典型 8 vs 256），每个 adapter 平均 32 个 token，`BLOCK_M=64` 能填满，避免 skinny GEMM。
3. **`ram`（row gather）让访存连续**：`token_indices_sorted_by_lora_ids` 是 stable sort 的结果，同一 adapter 的 token 在原始 `x` 里可能不连续，但至少每个 CTA 内部只 gather `BLOCK_M` 行，且这些行在 L2 里局部性尚可。
4. **空 adapter / 空 CTA 早退**：`lora_id == -1` 和 `cta_m_offset >= lora_m_size` 两处 early exit，让"batch 里只有 1 个 adapter"时只跑 1/8 的 CTA。

对应的 bus 级别对比：

```text
逐请求 GEMM:   launch × (B × L × 2),  每次 M=1~4,  N=r=16      → launch bound
Punica SGMV:   launch × (L × 2),      每次 M=N_tokens/L_adapters → compute/DRAM bound
```

**关于 bgmv / sgmv**：仓库里仍然保留了这两个算子封装（[ops/torch_ops/lora_ops.py:7](../vllm/lora/ops/torch_ops/lora_ops.py#L7)），`sgmv_expand` 内部就是"按 adapter 循环调 `bgmv_expand`"。`bgmv` 是 **B**atched **G**ather-**M**atrix-**V**ector（单个 adapter 的一次 gather + GEMV），`sgmv` 是 **S**egmented GEMV（多个 adapter 拼成一段，用 `b_seq_start_loc` 分段）。`compute_meta()`（[punica_wrapper/utils.py:15](../vllm/lora/punica_wrapper/utils.py#L15)）还在给这条路径准备 `b_seq_start_loc / seq_lengths / lora_indices_per_batch`，它用 `torch.unique_consecutive` 把**相邻且同 adapter** 的请求合并，进一步减少段数。CUDA 平台默认走 Triton 的 shrink/expand，CPU/XPU 走 sgmv/bgmv。

### 3.6 各层实现：TP 下 A/B 怎么切

所有 linear 类 LoRA 的公共基类是 `BaseLinearLayerWithLoRA`。默认（非 `fully_sharded`）策略是**只切 B**（column parallel）或**只切 A**（row parallel），这样一层内 LoRA 部分**不需要额外通信**，只在 base layer 原本的 all-gather / all-reduce 里顺带完成。

**ColumnParallelLinearWithLoRA**（[column_parallel_linear.py:85](../vllm/lora/layers/column_parallel_linear.py#L85)）：

```python
# vllm/lora/layers/column_parallel_linear.py:107
def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
    if self.is_merged_col_linear:
        shard_size = self.output_size // 2
        offset = lora_b.shape[0] // 2
        left_weight = lora_b[self.tp_rank * shard_size:(self.tp_rank + 1) * shard_size, :]
        right_weight = lora_b[offset + self.tp_rank * shard_size:
                              offset + (self.tp_rank + 1) * shard_size, :]
        lora_b = torch.cat([left_weight, right_weight], dim=0)
    else:
        shard_size = self.output_size
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        lora_b = lora_b[start_idx:end_idx, :]
    return lora_b
```

`gate_up_proj` 这种 merged 层必须**左右两半各切一段再拼回来**，因为 `MergedColumnParallelLinear` 的输出排布是 `[gate_shard | up_shard]` 而不是 `[gate | up]` 整体再切。

**QKVParallelLinearWithLoRA**（[column_parallel_linear.py:418](../vllm/lora/layers/column_parallel_linear.py#L418)）要考虑 GQA：`kv_shard_id = tp_rank // num_kv_head_replicas`，因为 K/V head 数可能少于 TP 度，多个 rank 共享同一份 KV。

**RowParallelLinearWithLoRA**（[row_parallel_linear.py:32](../vllm/lora/layers/row_parallel_linear.py#L32)）反过来切 A 的输入维：

```python
# vllm/lora/layers/row_parallel_linear.py:32
def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
    shard_size = self.input_size
    start_idx = self.tp_rank * shard_size
    end_idx = (self.tp_rank + 1) * shard_size
    lora_a = lora_a[:, start_idx:end_idx]
    return lora_a
```

**`fully_sharded_loras=True`（S-LoRA 策略）** 则两层都切，代价是 LoRA 路径上要多一次通信：

- Column parallel：`add_shrink` 后对 rank 维做 **all-gather**（[column_parallel_linear.py:64](../vllm/lora/layers/column_parallel_linear.py#L64)），再 `add_expand`。
- Row parallel：`add_shrink` 后对 buffer 做 **all-reduce**（[row_parallel_linear.py:134](../vllm/lora/layers/row_parallel_linear.py#L134)），再用 `offset_start = tp_rank * shard_size` 写到输出的分片上，与 base 的 partial sum 相加后只做一次常规 all-reduce。

这是经典的"用一次额外通信换掉一半的冗余计算"：r 很大 / TP 很大 / seq 很长时，每个 rank 只算 `r/tp` 的 rank 维能省下可观的 FLOPs。

**QKV fused / merged**：`MergedQKVParallelLinearWithLoRA`（[column_parallel_linear.py:456](../vllm/lora/layers/column_parallel_linear.py#L456)）把 q/k/v 当 3 个 slice 放在 `lora_a_stacked` / `lora_b_stacked` 的 tuple 第 0 维，`output_slices = (q_shard, kv_shard, kv_shard)`，一次 `add_expand` 用 `offset` 累加写到输出的三段。这样 qkv_proj 只 launch 一次而不是三次。

**LogitsProcessorWithLoRA**（[logits_processor.py:20](../vllm/lora/layers/logits_processor.py#L20)）：lm_head 上的 LoRA 会改变 vocab 分布，所以必须**先 gather 完整 logits 再加 delta**：

```python
# vllm/lora/layers/logits_processor.py:153
def _get_logits(self, hidden_states, lm_head, embedding_bias=None, skip_gather=False):
    # The LoRA delta is accumulated into the full gathered logits, so the
    # TP gather cannot be skipped here.
    if skip_gather:
        raise NotImplementedError(
            "Skipping the logits TP gather is not supported with an lm_head LoRA."
        )
    ...
    logits = self.base_layer._gather_logits(logits)
    if logits is None:
        return None
    if self.sharded_to_full_mapping_gpu is not None:
        # 每个 TP shard 会 pad 到 num_embeddings_per_partition，gather 之后
        # padding 和真实 vocab 交错，需要按映射重排
        logits = logits[:, self.sharded_to_full_mapping_gpu]
    lora_output = self.punica_wrapper.add_lora_logits(
        logits, hidden_states, self.lora_a_stacked, self.lora_b_stacked, 1.0)
```

并且有硬限制：`vocab_size > 258048` 直接报错（[logits_processor.py:91](../vllm/lora/layers/logits_processor.py#L91)），因为 `lora_b_stacked` 形状是 `[max_loras, 1, vocab_size, r]`。

实际调用链：[punica_gpu.py:206](../vllm/lora/punica_wrapper/punica_gpu.py#L206) `add_lora_linear` 就是"建一个 fp32 buffer → `add_shrink` → `add_expand`"：

```python
# vllm/lora/punica_wrapper/punica_gpu.py:245
r = lora_b_stacked[0].size(-1)
# We set the buffer to be float32 by default, refer to:
# https://github.com/triton-lang/triton/issues/1387
buffer = torch.empty(
    (len(output_slices), x.size(0), r), dtype=torch.float32, device=x.device
)
add_inputs = kwargs.pop("add_inputs", True)
self.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
self.add_expand(y, buffer, lora_b_stacked, output_slices,
                add_inputs=add_inputs, **kwargs)
```

**多模态 tower / connector LoRA**：`LoRAConfig.enable_tower_connector_lora=True` 时，`_set_adapter_mapping` 会根据 `mapping.type` 选不同的 punica wrapper：[model_manager.py:374](../vllm/lora/model_manager.py#L374)。model runner 里为 encoder 单独构造映射（[gpu_model_runner.py:3122](../vllm/v1/worker/gpu_model_runner.py#L3122)），因为 encoder 的 batch 结构和主 batch 不同。

**MoE LoRA**：`FusedMoEWithLoRA` / `FusedMoE3DWithLoRA` 走 `add_lora_w13` / `add_lora_w2`，额外需要 `moe_lora_align_block_size`（[punica_gpu.py:330](../vllm/lora/punica_wrapper/punica_gpu.py#L330)）把 (token, expert) 按 `max_loras` 展开分桶，`sorted_ids` 形状是 `(max_loras * max_num_tokens_padded,)`。

### 3.7 热插拔：`add_lora` / `remove_lora` / `pin_lora`

对外 API 链路：`AsyncLLM.add_lora` → `EngineCore.add_lora` → `Executor.add_lora` → `Worker.add_lora` → `LoRAModelRunnerMixin.add_lora`（[lora_model_runner_mixin.py:290](../vllm/v1/worker/lora_model_runner_mixin.py#L290)）。

```python
# vllm/v1/worker/lora_model_runner_mixin.py:290
def add_lora(self, lora_request: LoRARequest) -> bool:
    self._ensure_lora_enabled()
    return self.lora_manager.add_adapter(lora_request)

def remove_lora(self, lora_id: int) -> bool:
    self._ensure_lora_enabled()
    return self.lora_manager.remove_adapter(lora_id)

def pin_lora(self, lora_id: int) -> bool:
    self._ensure_lora_enabled()
    return self.lora_manager.pin_adapter(lora_id)
```

`pin_adapter` 只在 LRU 版本里实现（基类直接 `raise NotImplementedError`，[model_manager.py:367](../vllm/lora/model_manager.py#L367)）：

```python
# vllm/lora/model_manager.py:1233
def pin_adapter(self, lora_id: int) -> bool:
    """Pin a LoRAModel in the manager cache."""
    self._pin_lora_in_cpu_cache(lora_id)
    self._pin_lora_in_gpu_cache(lora_id)
    return True

def _pin_lora_in_gpu_cache(self, lora_id: int):
    if lora_id not in self._active_adapters:
        # move lora to gpu if not already active
        self.activate_adapter(lora_id)
    self._active_adapters.pin(lora_id)
```

**pin 的语义**：把条目从 LRU 的淘汰队列里摘出来（pin 之后 `remove_oldest` 不会选中它），同时保证它已经 `activate_adapter` 到 GPU slot。适用场景：

- 有一个"永远在线"的高频 adapter，不能因为一次冷启动批量加载被挤掉；
- 做 A/B 测试时需要对比的两个 adapter 同时驻留。

**注意坑**：pin 会**永久占用**一个 GPU slot。若 pin 的数量达到 `max_loras`，`activate_adapter` 会抛 `ValueError("No free lora slots")`。

还有 `reset_lora_state()`（[lora_model_runner_mixin.py:35](../vllm/v1/worker/lora_model_runner_mixin.py#L35)）在 base 权重被替换后调用：它会 `remove_all_adapters()` 并重建 `LogitsProcessorWithLoRA` 的 `sharded_to_full_mapping`——因为权重 reload 可能复用/覆盖 `lora_b_stacked` 的显存。

### 3.8 LoRA 与 CUDA Graph

LoRA 和 CUDA Graph 的根本冲突是：**LoRA 的 kernel grid 依赖 `num_active_loras`，而 CUDA Graph 要求 grid 固定**。此外 `set_lora()` 里的 H2D copy 本身是 graph-unsafe 的（不能在 replay 里做）。

vLLM 的解法分三层：

**(1) 元数据张量常驻，只改内容**。`LoRAKernelMeta` 的所有张量都是预分配的固定 buffer，`prepare_tensors` 只是往里 `copy_`（[lora_kernel_metadata.py:131](../vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L131)）。这样 graph 里捕获的指针永远有效。

**(2) `no_lora_flag` / `num_active_loras` 伪装成 CPU tensor**。因为"torch.compile 会把 python 标量烘焙成常量，且不能处理动态控制流，但 `torch.ops` 内部不被 trace"，所以把这两个标志存成 **CPU tensor**，让 kernel 内部做 early exit（[lora_kernel_metadata.py:21](../vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L21) 的注释写得很清楚）。CPU tensor 的值变化不需要重新 capture graph。

**(3) 按 active LoRA 数量抓多份 graph**。开关是 `compilation_config.cudagraph_specialize_lora`（[config/compilation.py:660](../vllm/config/compilation.py#L660)，默认 `True`）和 `lora_config.specialize_active_lora`（[config/lora.py:68](../vllm/config/lora.py#L68)，默认 `False`）。

捕获哪些 case 由**单一事实来源**决定（[lora/utils.py:50](../vllm/lora/utils.py#L50)）：

```python
# vllm/lora/utils.py:50
def get_captured_lora_counts(max_loras: int, specialize: bool) -> list[int]:
    if not specialize:
        return [max_loras + 1]
    return [
        n for n in range(1, max_loras + 2) if (n & (n - 1)) == 0 or n == max_loras + 1
    ]
```

- `specialize=False`：只抓 `max_loras + 1` 一种（最坏情况），运行时 grid 恒定为最大值。**代价**：即使 batch 里 0 个 LoRA，也会跑满 grid（靠 `lora_id == -1` early exit 兜底，但 CTA 已经发出去了）。
- `specialize=True`：抓 1, 2, 4, 8, ..., max_loras, max_loras+1（2 的幂 + 上界）。运行时把实际 `num_active_loras` **向上取整**到最近的捕获值（[lora_kernel_metadata.py:157](../vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L157)）。

dispatcher 侧同样调用这个函数（[cudagraph_dispatcher.py:111](../vllm/v1/cudagraph_dispatcher.py#L111)）：

```python
# vllm/v1/cudagraph_dispatcher.py:111
def _get_lora_cases(self) -> list[int]:
    lora_config = self.vllm_config.lora_config
    if lora_config is None:
        return [0]
    if self.compilation_config.cudagraph_specialize_lora:
        captured_counts = get_captured_lora_counts(
            lora_config.max_loras, self.specialize_lora_count)
        return [0] + captured_counts
    return [lora_config.max_loras + 1]
```

dispatch 时向上取整到最近的捕获 key（[cudagraph_dispatcher.py:283](../vllm/v1/cudagraph_dispatcher.py#L283)）：

```python
effective_num_active_loras = num_active_loras
if has_lora and num_active_loras > 0:
    if self.specialize_lora_count:
        idx = bisect.bisect_left(self.captured_lora_counts, num_active_loras)
        if idx < len(self.captured_lora_counts):
            effective_num_active_loras = self.captured_lora_counts[idx]
    else:
        effective_num_active_loras = self.vllm_config.lora_config.max_loras + 1
```

model runner 在 real run 里从 batch 读 `num_active_loras` 传给 dispatcher：[gpu_model_runner.py:4076](../vllm/v1/worker/gpu_model_runner.py#L4076)

```python
num_active_loras = (
    force_num_active_loras
    if force_num_active_loras is not None
    else len(self.input_batch.lora_id_to_lora_request)
)
has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora
```

capture 时用 dummy LoRA 造出对应的 mapping（[lora_model_runner_mixin.py:148](../vllm/v1/worker/lora_model_runner_mixin.py#L148) `maybe_select_dummy_loras`），并在 capture 结束清掉（[gpu_model_runner.py:6884](../vllm/v1/worker/gpu_model_runner.py#L6884) `maybe_remove_all_loras`）。

**所以"LoRA 必须走 eager"这个说法在当前版本已经不准确**：CUDA 平台走 piecewise cudagraph + 专门的 LoRA graph；`num_active_loras` 作为 BatchDescriptor 的一部分参与 dispatch key。真正退化到 eager 的是别的原因（如 `use_inductor_graph_partition`、`enable_sp` 等）。

### 3.9 LoRA 与 prefix caching

不同 adapter 的 KV 语义不同，**不能共享**。vLLM 的做法是把 adapter 名字塞进 block hash 的 extra keys：[kv_cache_utils.py:543](../vllm/v1/core/kv_cache_utils.py#L543)

```python
# vllm/v1/core/kv_cache_utils.py:543
def _gen_lora_extra_hash_keys(request: Request) -> list[str]:
    """Generate extra keys related to LoRA for block hash computation."""
    if not request.lora_request:
        return []
    return [request.lora_request.lora_name]
```

并在 `generate_block_hash_extra_keys` 里拼在最前面：[kv_cache_utils.py:604](../vllm/v1/core/kv_cache_utils.py#L604)

```python
lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)
...
extra_keys: list[Any] = (
    lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
)
```

结论：**hash 里包含的是 `lora_name`（字符串），不是 `lora_int_id`**。这与 `LoRARequest.__hash__` 只认 name 是一致的。副作用：

- 同名 adapter 即使权重更新了（`load_inplace`），prefix cache 仍会命中旧 KV——需要配合 `cache_salt` 或重启。
- 无 LoRA 的请求和有 LoRA 的请求，第一个 block 的 extra keys 不同，因此天然隔离。

---

## 4. 关键数据结构

| 字段 / 类 | 类型 | 含义 | 定义处 |
| --- | --- | --- | --- |
| `LoRARequest.lora_int_id` | `int`（>0） | adapter 的全局整型 id，缓存 key | [request.py:26](../vllm/lora/request.py#L26) |
| `LoRARequest.lora_name` | `str` | 判等/哈希依据，**也进 block hash** | [request.py:25](../vllm/lora/request.py#L25) |
| `LoRARequest.load_inplace` | `bool` | 强制重载同名 adapter | [request.py:30](../vllm/lora/request.py#L30) |
| `LoRARequest.is_3d_lora_weight` | `bool` | MoE adapter 是否 3D fused 布局 | [request.py:31](../vllm/lora/request.py#L31) |
| `LoRAConfig.max_loras` | `int` | 单 batch 活跃 adapter 上限 = GPU slot 数 | [config/lora.py:37](../vllm/config/lora.py#L37) |
| `LoRAConfig.max_lora_rank` | 枚举 1..512 | 决定 A/B buffer 的 rank 维 | [config/lora.py:35](../vllm/config/lora.py#L35) |
| `LoRAConfig.max_cpu_loras` | `int \| None` | CPU 侧 LRU 容量，默认 = `max_loras` | [config/lora.py:44](../vllm/config/lora.py#L44) |
| `LoRAConfig.fully_sharded_loras` | `bool` | 开启 S-LoRA 全切分（多一次通信） | [config/lora.py:39](../vllm/config/lora.py#L39) |
| `LoRAConfig.specialize_active_lora` | `bool` | 按 adapter 数量专门化多份 cudagraph | [config/lora.py:68](../vllm/config/lora.py#L68) |
| `LoRAConfig.target_modules` | `list[str] \| None` | 部署时限定的模块后缀白名单 | [config/lora.py:49](../vllm/config/lora.py#L49) |
| `LoRAMapping.index_mapping` | `tuple[int, ...]` | 每 token 的 `lora_int_id`（0 = 无） | [layers/utils.py:35](../vllm/lora/layers/utils.py#L35) |
| `LoRAMapping.prompt_mapping` | `tuple[int, ...]` | 每请求（采样位置）的 `lora_int_id` | [layers/utils.py:36](../vllm/lora/layers/utils.py#L36) |
| `LoRAMapping.type` | `LoRAMappingType` | LANGUAGE / TOWER / CONNECTOR | [layers/utils.py:38](../vllm/lora/layers/utils.py#L38) |
| `LoRAModelManager.lora_index_to_id` | `list[int \| None]` | slot → adapter id，长度 = `max_loras` | [model_manager.py:113](../vllm/lora/model_manager.py#L113) |
| `LoRAModelManager._registered_adapters` | `AdapterLRUCache` | CPU 侧缓存，容量 = `max_cpu_loras` | [model_manager.py:106](../vllm/lora/model_manager.py#L106) |
| `LoRAModelManager._active_adapters` | `AdapterLRUCache` | GPU slot 占用，容量 = `max_loras` | [model_manager.py:109](../vllm/lora/model_manager.py#L109) |
| `lora_a_stacked` | tuple of `[max_loras,1,r,in]` | 每层预分配的 A 矩阵 | [base_linear.py:129](../vllm/lora/layers/base_linear.py#L129) |
| `lora_b_stacked` | tuple of `[max_loras,1,out,r]` | 每层预分配的 B 矩阵 | [base_linear.py:140](../vllm/lora/layers/base_linear.py#L140) |
| `LoRAKernelMeta.active_lora_ids` | `[max_loras+1] int32` | 本 step 活跃 slot | [lora_kernel_metadata.py:17](../vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L17) |
| `LoRAKernelMeta.num_tokens_per_lora` | `[max_loras+1] int32` | 每 slot 的 token 数 | [lora_kernel_metadata.py:18](../vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L18) |
| `LoRAKernelMeta.lora_token_start_loc` | `[max_loras+2] int32` | 排序后每 slot 起始位置 | [lora_kernel_metadata.py:19](../vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L19) |
| `LoRAKernelMeta.no_lora_flag_cpu` | CPU bool tensor | 让 kernel 内部 early exit，避免 re-capture | [lora_kernel_metadata.py:30](../vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L30) |

---

## 5. 收益与代价

### 5.1 收益

- **显存**：从 `N × base` 降到 `base + max_loras × adapter`。1000 个 r=16 adapter 的 7B 模型，常驻部分约 16GB + 8 × 0.1GB ≈ 16.8GB。
- **换入延迟**：`add_lora` 只做文件读取 + `copy_`，不做编译、不重建 CUDA Graph。实测（典型 7B、32 层）单次换入在 10ms 量级。
- **吞吐**：相比逐请求 GEMM，Punica 把 LoRA 部分的 launch 数降低 1~2 个数量级；`max_loras=8` 的混合 batch 相对单 adapter batch 的吞吐衰减通常在 10% 以内（取决于 r 和 adapter 分散度）。
- **热插拔**：`add_lora` / `pin_lora` 是在引擎运行中生效的，不需要重启。

### 5.2 代价与限制

- **slot 是硬上限**：`max_loras` 之外的 adapter 请求会被**静默跳过**（进 `skipped_waiting`），表现为尾部延迟升高而不是报错。生产上要监控。
- **`max_lora_rank` 决定显存**：buffer 按 `max_lora_rank` 分配而非实际 rank。一个 r=8 的 adapter 在 `max_lora_rank=64` 的引擎里也占 64 的空间。
- **adapter 换入是同步开销**：`activate_adapter` 遍历所有层做 H2D copy，且包在 `gpu_sync_allowed()` 里。高频换入会直接打在 P99 上 → 用 `pin_lora` 或提高 `max_loras`。
- **`fully_sharded_loras` 增加通信**：TP 下每层 LoRA 多一次 all-gather/all-reduce，小 batch 时反而更慢；只在 r 大 / TP 大 / seq 长时开启。
- **CUDA Graph 数量膨胀**：`specialize_active_lora=True` 时 graph 数 ×（2 的幂个数），启动时间和显存都上升。
- **与 adaptive verification 不兼容**：[config/vllm.py:2687](../vllm/config/vllm.py#L2687) 明确拒绝 LoRA + adaptive verification 的组合。
- **lm_head LoRA 有 vocab 上限 258048**（[logits_processor.py:91](../vllm/lora/layers/logits_processor.py#L91)）。
- **prefix cache 按 adapter 名隔离**：切换 adapter 会完全 miss。

### 5.3 失效场景

| 场景 | 表现 | 缓解 |
| --- | --- | --- |
| adapter 数 >> `max_loras` 且访问分散 | 每 step 都在换入换出，吞吐塌方 | 提高 `max_loras`；热点 `pin_lora` |
| 单个请求很大、adapter 独占 | batch 里只有 1 个 adapter，Punica 收益消失 | 无解，这是 LoRA 固有开销 |
| r 很大（≥128） | `add_shrink` 的 N 维变大，接近 base GEMM 成本 | 考虑合并到 base 权重 |
| 量化 + LoRA | `slice_lora_a/b` 要走 `_get_lora_device` 的分支（qweight/w2_qweight 等），部分量化方法不支持 | [layers/utils.py:45](../vllm/lora/layers/utils.py#L45) |

---

## 6. 面试高频问题

**Q1：`max_loras`、`max_lora_rank`、`max_cpu_loras` 分别控制什么？**

`max_loras` = 单 batch 内**不同 adapter 的最大数量**，同时就是 GPU slot 数（`LoRAModelManager.lora_slots`，[model_manager.py:307](../vllm/lora/model_manager.py#L307)）。`max_lora_rank` = A/B buffer 的 rank 维上界，决定每层预分配的显存（[base_linear.py:129](../vllm/lora/layers/base_linear.py#L129)）。`max_cpu_loras` = CPU 侧 LRU 容量（`LoRAModelManager.capacity`，[model_manager.py:302](../vllm/lora/model_manager.py#L302)），默认等于 `max_loras`，校验逻辑在 [config/lora.py:116](../vllm/config/lora.py#L116) 要求 `max_cpu_loras >= max_loras`。三者关系：CPU 缓存 ≥ GPU slot，`max_loras` 决定运行时并发。

**Q2：一批请求里 adapter 数超过 `max_loras` 会怎样？**

调度器在 waiting 队列遍历时检查（[scheduler.py:890](../vllm/v1/core/sched/scheduler.py#L890)）：若 `len(scheduled_loras) == max_loras` 且新请求的 adapter 不在集合里，就把它 `pop` 出来塞进 `step_skipped_waiting` 并 `continue`。请求不被抢占、不报错，只是本 step 不调度。running 侧的 `assert`（[scheduler.py:845](../vllm/v1/core/sched/scheduler.py#L845)）保证已调度集合永不超限。

**Q3：`token_lora_mapping` 里存的是 lora id 还是 slot index？在哪转换？**

存的是 **slot index**（0..max_loras-1，-1 表示无 LoRA）。转换发生在 `convert_mapping`（[punica_wrapper/utils.py:54](../vllm/lora/punica_wrapper/utils.py#L54)），它先用 `lora_index_to_id` 建反向表 `lora_id_to_index`，再逐 token 映射。注释里说明这是为了避免对每个 token 做 `list.index()` 的 O(n) 扫描。

**Q4：为什么把 `no_lora_flag` 和 `num_active_loras` 存成 CPU tensor 而不是 Python bool/int？**

三点原因写在 [lora_kernel_metadata.py:21](../vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L21) 的注释里：(1) torch.compile 把 Python 标量 trace 成**常量**；(2) 无法处理动态控制流；(3) `torch.ops` 函数**内部不会被 trace**。所以把它们伪装成 CPU tensor，在 `lora_shrink` / `lora_expand` 这个 custom op 内部读取并 early exit——这样值变化不会触发重新编译或重新 capture CUDA Graph。

**Q5：Punica / SGMV 相比"逐请求 GEMM"到底省在哪？**

三层：(a) launch 次数从 `O(num_reqs × num_layers × 2)` 降到 `O(num_layers × 2)`——grid 第 2 维是 `num_active_loras`（[lora_shrink_op.py:226](../vllm/lora/ops/triton_ops/lora_shrink_op.py#L226)）；(b) 每个 CTA 的 M 维是"该 adapter 的 token 数"而非 1，避免 skinny GEMM，tensor core 利用率上去了；(c) 用 `token_indices_sorted_by_lora_ids` + `lora_token_start_loc` 做 gather，让同一 adapter 的 token 在 kernel 内是连续段。另外 `lora_id == -1` 和 `cta_m_offset >= lora_m_size` 两处 early exit 保证不活跃的 adapter 不占算力。

**Q6：TP 下 LoRA 权重怎么切？默认和 `fully_sharded_loras` 有什么区别？**

默认：column parallel 只切 `lora_b`（按输出维，merged 层左右两半各切一段再拼，[column_parallel_linear.py:107](../vllm/lora/layers/column_parallel_linear.py#L107)），row parallel 只切 `lora_a`（按输入维，[row_parallel_linear.py:32](../vllm/lora/layers/row_parallel_linear.py#L32)）。这样 LoRA 路径**零额外通信**，搭 base layer 原有的 all-gather / all-reduce 顺风车。`fully_sharded_loras=True` 时两层都切：column parallel 在 shrink 后 all-gather rank 维（[column_parallel_linear.py:64](../vllm/lora/layers/column_parallel_linear.py#L64)），row parallel 在 shrink 后 all-reduce 并用 `offset_start = tp_rank * shard_size` 写分片（[row_parallel_linear.py:144](../vllm/lora/layers/row_parallel_linear.py#L144)）。代价是多一次通信，收益是每个 rank 少算 `1/tp` 的 FLOPs。

**Q7：`pin_lora` 做了什么？什么时候用？**

`LRUCacheLoRAModelManager.pin_adapter`（[model_manager.py:1233](../vllm/lora/model_manager.py#L1233)）做两件事：`_registered_adapters.pin()` 把它从 CPU LRU 淘汰队列摘出；`_active_adapters.pin()` 保证它已 activate 到 GPU slot 且不被 `remove_oldest` 选中。用于"必须常驻的高频 adapter"或 A/B 对比。代价是永久占一个 slot，pin 满 `max_loras` 后 `activate_adapter` 会抛 `ValueError("No free lora slots")`（[model_manager.py:331](../vllm/lora/model_manager.py#L331)）。

**Q8：LoRA 和 CUDA Graph 怎么共存？**

三招：(1) 元数据张量全部预分配，graph 里捕获的指针不变，每 step 只 `copy_` 内容（[lora_kernel_metadata.py:131](../vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L131)）；(2) 动态量伪装成 CPU tensor 让 kernel 内部 early exit；(3) 按 active adapter 数抓多份 graph——`get_captured_lora_counts()`（[lora/utils.py:50](../vllm/lora/utils.py#L50)）是 dispatcher 和 punica wrapper 共用的单一事实来源，`specialize_active_lora=False` 时只抓 `max_loras+1` 一份并让 grid 恒为最大值，`True` 时抓 2 的幂 + 上界并在 dispatch 时 `bisect` 向上取整（[cudagraph_dispatcher.py:283](../vllm/v1/cudagraph_dispatcher.py#L283)）。capture 时用 `maybe_dummy_run_with_lora` 造 dummy adapter（[lora_model_runner_mixin.py:252](../vllm/v1/worker/lora_model_runner_mixin.py#L252)），capture 完 `maybe_remove_all_loras`。

**Q9：不同 adapter 的 KV cache 会互相污染吗？**

不会。`_gen_lora_extra_hash_keys`（[kv_cache_utils.py:543](../vllm/v1/core/kv_cache_utils.py#L543)）把 `lora_request.lora_name` 加入 block hash 的 extra keys，且排在拼列表的第一位（[kv_cache_utils.py:612](../vllm/v1/core/kv_cache_utils.py#L612)）。所以 adapter A 和 adapter B 即使 prompt 完全相同也是不同的 block hash。注意用的是 **name 不是 int_id**，与 `LoRARequest.__hash__` 一致。

**Q10：`add_lora` 时如果 adapter 加载失败，已有的 adapter 会丢吗？**

不会。`LRUCacheWorkerLoRAManager.add_adapter`（[worker_manager.py:287](../vllm/lora/worker_manager.py#L287)）的顺序是：先 `_load_adapter`（可能抛异常）→ 再 `remove_adapter(lora.id)` → 再检查容量并 `remove_oldest_adapter()` → 最后 `add_adapter`。注释明确写 "Load the new adapter first to ensure it is actually valid, before evicting any existing adapters"。代价是已加载数量可能瞬时超过 `max_cpu_loras`（代码注释承认了）。

**Q11：为什么 `from_layer` 里 `_all_lora_classes` 的顺序很重要？**

因为 `can_replace_layer` 是按顺序第一个匹配即返回（[utils.py:114](../vllm/lora/utils.py#L114)）。`QKVParallelLinear` 既能被 `QKVParallelLinearWithLoRA`（packed_modules_list 长度 1）也能被 `MergedQKVParallelLinearWithLoRA`（长度 3）匹配，靠的是 `len(packed_modules_list)` 判断；但 `VocabParallelEmbeddingWithLoRA` 必须排在最前，否则 embedding 可能被更通用的类误吞。源码注释就是 "Order matters here: more specific wrappers must be checked before generic merged/column-parallel wrappers"（[utils.py:77](../vllm/lora/utils.py#L77)）。

**Q12：LoRA 请求会被抢占来腾 slot 吗？**

不会。调度器只在 waiting 阶段做 adapter 数量准入（[scheduler.py:890](../vllm/v1/core/sched/scheduler.py#L890)），running 请求一旦被调度就一定有 slot（上一步的 `assert len(scheduled_loras) <= max_loras` 保证了这一点）。抢占逻辑（KV block 不足时触发）与 LoRA 无关；不过抢占时如果被抢占请求本 step 已分配了 encoder/其他资源，代码里有对应的回滚（例如 encoder compute budget 的回滚在 [scheduler.py:769](../vllm/v1/core/sched/scheduler.py#L769)）。

---

## 7. 延伸阅读

- Punica: Multi-Tenant LoRA Serving, arXiv:2310.18547 —— [punica_gpu.py:1](../vllm/lora/punica_wrapper/punica_gpu.py#L1) 文件头
- S-LoRA: Serving Thousands of Concurrent LoRA Adapters, arXiv:2311.03285 —— `fully_sharded_loras` 的理论来源，见 [column_parallel_linear.py:518](../vllm/lora/layers/column_parallel_linear.py#L518) 与 [row_parallel_linear.py:96](../vllm/lora/layers/row_parallel_linear.py#L96) 的注释
- Triton kernel 调优说明：[vllm/lora/ops/triton_ops/README_TUNING.md](../vllm/lora/ops/triton_ops/README_TUNING.md)
- 相关源码：`vllm/lora/lora_model.py`（adapter 权重解析）、`vllm/lora/peft_helper.py`（PEFT 配置校验）、`vllm/lora/resolver.py`（adapter 路径解析）
- 配置项全量说明：`vllm/config/lora.py`
