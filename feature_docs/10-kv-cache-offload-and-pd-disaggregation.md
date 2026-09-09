# KV Cache 卸载与 Prefill-Decode 分离（KV Connector）

> 适用版本：vLLM V1。抽象在 `vllm/distributed/kv_transfer/kv_connector/v1/base.py`，注入点在 `vllm/v1/worker/kv_connector_model_runner_mixin.py`，调度协同在 `vllm/v1/core/sched/scheduler.py`，内置实现在 `vllm/distributed/kv_transfer/kv_connector/v1/`。

## 0. TL;DR

- **是什么**：KV Connector 是一套「KV cache 在实例间/介质间传输与卸载」的插件接口。两大用法：①PD 分离（prefill 与 decode 实例解耦、KV 跨实例传输）；②KV offload（把 KV 卸到 CPU/SSD 省显存）。
- **解决什么**：KV cache 显存是并发上限的瓶颈；PD 不分离时 prefill 长任务会占住 decode 卡、互相干扰；长会话/多轮对话 KV 占显存高。
- **怎么做**：`KVConnectorBase_V1` 定义 register/save/load 生命周期；ModelRunner 通过 `KVConnectorModelRunnerMixin` 在 forward 前后 hook 传输；调度器用 `get_num_new_matched_tokens` / `get_finished` 协调。
- **收益**：PD 分离让 prefill/decode 独立扩缩容、互不干扰、GPU 利用率高；offload 让长会话省显存、并发更高。代价是引入网络/存储带宽瓶颈与一致性约束。

---

## 1. 场景与痛点

### 1.1 KV cache 是并发的硬上限

每请求的 KV cache 占用随序列长度线性增长。显存有限 → 可同时服务的请求数有限 → 排队。当单卡 KV 打满，只能靠抢占（RECOMPUTE，见 `05`），尾延迟飙升。

### 1.2 Prefill 与 Decode 互相干扰

不分离时，长 prefill 和大 batch decode 挤在同一张卡：prefill 占算力、decode 占带宽，彼此抢资源，SLO 难保证。PD 分离把「算 KV 的 prefill 实例」和「自回归 decode 的实例」拆开，各自独立扩缩容。

### 1.3 长会话/多轮占显存

多轮对话、长文档 QA 的 KV 长期驻留显存。offload 到 CPU/SSD 可把「冷」KV 移出显存，需要时再 load 回来。

---

## 2. 核心设计

### 2.1 KVConnectorBase_V1 生命周期接口

```python
# vllm/distributed/kv_transfer/kv_connector/v1/base.py:171
class KVConnectorBase_V1(ABC):

    # vllm/distributed/kv_transfer/kv_connector/v1/base.py:264
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        # 把各 KV cache group 的 GPU 张量注册给 connector（save/load 的目标）

    # vllm/distributed/kv_transfer/kv_connector/v1/base.py:289
    def start_load_kv(self, forward_context, **kwargs) -> None:
        # 本步开始加载 KV（同步 load 须在 forward 前；异步 load 可在 forward 后提交）

    # vllm/distributed/kv_transfer/kv_connector/v1/base.py:307
    def wait_for_layer_load(self, layer_name: str) -> None:
        # 该层 KV 必须已 load 完，才能跑该层 forward（同步语义保证正确性）

    # vllm/distributed/kv_transfer/kv_connector/v1/base.py:321
    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, ...):
        # 该层算完后把 KV 存到远端/CPU/SSD

    # vllm/distributed/kv_transfer/kv_connector/v1/base.py:353
    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], ...]:
        # 通知 connector 哪些请求结束，返回可释放的 KV 位置

    # vllm/distributed/kv_transfer/kv_connector/v1/base.py:224
    def bind_connector_metadata(self, connector_metadata): ...   # 绑定本步传输元数据
    # vllm/distributed/kv_transfer/kv_connector/v1/base.py:236
    def clear_connector_metadata(self): ...                       # 清理元数据（finalize）
    # vllm/distributed/kv_transfer/kv_connector/v1/base.py:450
    def get_num_new_matched_tokens(self, request, ...): ...       # 与远端 KV 命中多少
    # vllm/distributed/kv_transfer/kv_connector/v1/base.py:485
    def update_state_after_alloc(self, request, ...): ...         # 分配后更新 connector 状态
    # vllm/distributed/kv_transfer/kv_connector/v1/base.py:511
    def build_connector_meta(self, scheduler_output): ...         # 构造本步传输元数据
```

### 2.2 ModelRunner 侧的注入（Mixin）

```python
# vllm/v1/worker/kv_connector_model_runner_mixin.py:25
class KVConnectorModelRunnerMixin:
    # _get_kv_connector_output (line 69): 把 KV connector 的整个生命周期包进
    #   execute_model 的 context manager：bind_connector_metadata ->
    #   start_load_kv(forward前/后) -> yield(跑 forward) -> save_kv_layer(逐层) -> wait_for_save/clear
    # maybe_get_kv_connector_output (line 42): 有 kv_transfer_group 才启用
    # finalize_kv_connector (line 55): wait_for_save + clear_connector_metadata
    # kv_connector_no_forward (line 27): 即使本步无计算也要把 KV send/recv 推进
```

关键点：load 与 save 是**按层（layer）粒度**的——`start_load_kv` 启动加载，`wait_for_layer_load(layer)` 保证该层 forward 前 KV 已就位，`save_kv_layer(layer, ...)` 在该层算完后存出。这让 save 与后续层 forward **overlap**（异步传输藏在计算后面），把传输延迟移出关键路径。

### 2.3 调度器协同

调度器在每个 step 调用 `build_connector_meta` 构造传输计划，并在 `get_num_new_matched_tokens`（[vllm/distributed/kv_transfer/kv_connector/v1/base.py:450](../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L450)）告知「该请求有多少 token 的 KV 已在远端/prefix 命中」，从而只算缺失部分（与 prefix caching 思路一致，见 `04`）。`get_finished`（[base.py:353](../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L353)）在请求结束时释放远端 KV。

### 2.4 内置实现

`vllm/distributed/kv_transfer/kv_connector/v1/` 下（以本仓库实际文件为准，常见的有）：shared storage（`shared_storage_connector`）、LMCache、NixL、Mooncake、P2P/NCCL（`p2p`）、offloading（`offloading_connector`）、multi-connector 组合。PD 分离用 NixL/Mooncake/LMCache 做跨实例传输；offload 用 offloading_connector 把 KV 卸到 CPU/SSD（`vllm/v1/kv_offload/` 含 KVCacheOffloadingManager 与各 backend）。工厂在 `vllm/distributed/kv_transfer/kv_connector/factory.py`。

---

## 3. 代码走读

### 3.1 PD 分离的 KV 传输时序

```
[Prefill 实例]
  EngineCore 调度 prefill -> Worker 跑 prefill forward 算 KV
    -> save_kv_layer(layer) 逐层把 KV 推到 KV 传输层(NixL/Mooncake/共享内存)
  KV 到达 Decode 实例的 connector buffer

[Decode 实例]
  EngineCore 收到带 connector_metadata 的请求
    -> start_load_kv(forward后提交异步 load)
    -> execute_model: wait_for_layer_load(layer) 保证该层 KV 就位
    -> 跑 decode forward（用已 load 的 KV，不重算 prefill）
```

load 与 decode forward 的逐层 overlap 是性能关键：decode 跑第 L 层时，第 L-1 层的 KV 已在异步 load。

### 3.2 offload 时序

长会话 KV 驻显存；当显存紧张，offloading_connector 把「冷」KV 块 `save_kv_layer` 到 CPU/SSD（[vllm/v1/kv_offload/](../vllm/v1/kv_offload/)）；下次需要该块时 `start_load_kv` + `wait_for_layer_load` 搬回显存。本质是「KV 的换入换出」，和 OS 的 page cache 思想一致。

---

## 4. 关键数据结构

| 结构 | 作用 | 位置 |
| --- | --- | --- |
| `KVConnectorBase_V1` | 生命周期接口 | [base.py:171](../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L171) |
| `KVConnectorModelRunnerMixin` | ModelRunner hook | [kv_connector_model_runner_mixin.py:25](../vllm/v1/worker/kv_connector_model_runner_mixin.py#L25) |
| `KVConnectorMetadata` | 本步传输计划 | `vllm/v1/core/sched/output.py` |
| `KVConnectorOutput` | 本步传输结果 | [vllm/v1/outputs.py](../vllm/v1/outputs.py) |

---

## 5. 收益与代价

**收益**
- PD 分离：prefill/decode 独立扩缩容，互不干扰；GPU 利用率高，SLO 稳定。
- offload：长会话/多轮省显存，并发上限提高。
- 按层异步传输与计算 overlap，把传输延迟移出关键路径。

**代价 / 限制 / 失效场景**
- **网络/存储带宽成新瓶颈**：PD 分离的 KV 传输量 = KV cache 总大小，跨机需高带宽（NIXL/Mooncake 面向 RDMA）；offload 受 CPU/SSD 带宽限制。
- **与 prefix caching 的交互**：远端 KV 命中等同跨实例共享前缀，需 token 完全一致；命中后只算缺失部分。
- **与 chunked prefill 的交互**：长 prefill 分块时，KV 传输需按块对齐。
- **一致性**：跨实例需保证 token 一致，否则 KV 对不上（PD 分离要求 prefill 与 decode 用词表/模型一致）。
- **与 spec decode / cudagraph**：spec decode 下 KV 传输需谨慎；cudagraph 捕获要注意 load/save 不在图内（piecewise）。

---

## 6. 面试高频问题

**Q1：KV Connector 解决哪两类问题？**
A：①PD 分离（prefill/decode 实例解耦、KV 跨实例传输）；②KV offload（KV 卸到 CPU/SSD 省显存）（[vllm/distributed/kv_transfer/kv_connector/v1/base.py:171](../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L171)）。

**Q2：save/load 为什么按层粒度？**
A：逐层 `save_kv_layer` 后，下一层 forward 可与更早层的 KV 传输 overlap（异步传输藏在计算后），把传输延迟移出关键路径（[kv_connector_model_runner_mixin.py:69](../vllm/v1/worker/kv_connector_model_runner_mixin.py#L69)）。

**Q3：wait_for_layer_load 保证什么？**
A：该层 forward 前，本层所需的 KV 必须已从远端/CPU load 完就位，保证计算正确性（同步语义）（[base.py:307](../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L307)）。

**Q4：PD 分离下 decode 实例为什么不重算 prefill？**
A：prefill 实例已把 KV 传到 decode 的 connector buffer；decode 用 `start_load_kv` + `wait_for_layer_load` 直接拿到 KV 跑 decode，只算新 token。

**Q5：get_num_new_matched_tokens 的作用？**
A：告知调度器该请求有多少 KV 已在远端/prefix 命中，从而只算缺失部分，类似 prefix caching 的命中逻辑（[base.py:450](../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L450)）。

**Q6：KV offload 和 OS page cache 的思想一样吗？**
A：类似——都是「热数据留高速介质、冷数据换出到低速介质、按需换入」；offload 把 KV 块在显存与 CPU/SSD 间换入换出（[vllm/v1/kv_offload/](../vllm/v1/kv_offload/)）。

**Q7：PD 分离的网络瓶颈在哪？**
A：KV 传输量 = KV cache 总大小，跨机需 RDMA 高带宽（NixL/Mooncake 面向此场景），否则传输比 prefill 计算还慢。

**Q8：KV Connector 怎么注入到 ModelRunner？**
A：通过 `KVConnectorModelRunnerMixin`（[kv_connector_model_runner_mixin.py:25](../vllm/v1/worker/kv_connector_model_runner_mixin.py#L25)），在 `execute_model` 里用 context manager 包住 load/save 生命周期。

**Q9：请求结束时怎么清理远端 KV？**
A：调度器调 `get_finished(finished_req_ids)`（[base.py:353](../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L353)）通知 connector 释放该请求的远端 KV 位置。

**Q10：offload connector 和 prefix caching 冲突吗？**
A：不冲突。prefix caching 是「同实例 block 复用」，offload 是「跨介质换入换出」，两者可在不同维度省显存；offload 的块也可被 prefix 命中复用。

---

## 7. 延伸阅读

- 实现：`vllm/distributed/kv_transfer/kv_connector/v1/base.py`、`kv_connector_model_runner_mixin.py`、`factory.py`、`vllm/v1/kv_offload/`
- 调度协同：`vllm/v1/core/sched/scheduler.py`（build_connector_meta / get_num_new_matched_tokens）
- 配套：`03-paged-attention-kv-cache.md`、`04-prefix-caching.md`、`05-scheduler-continuous-batching.md`
- 官方：https://docs.vllm.ai/en/latest/serving/prefill_decode_disaggregation.html
