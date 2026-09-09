# V1 分布式并行（TP / PP / EP / DP）

> 适用版本：vLLM V1。配置在 `vllm/config/parallel.py`，进程组在 `vllm/distributed/parallel_state.py`，执行器在 `vllm/v1/executor/`，Worker 在 `vllm/v1/worker/`。

## 0. TL;DR

- **是什么**：把模型/请求切到多卡/多机，四种并行——TP（张量）、PP（流水线）、EP（专家，MoE）、DP（数据）。
- **解决什么**：单卡放不下大模型（TP/PP）、MoE 专家太多（EP）、要提高吞吐（DP）。
- **怎么做**：`ParallelConfig` 描述切分方式；`initialize_model_parallel` 建 TP/PP/EP/DP 进程组；`MultiprocExecutor` 用共享内存消息总线把指令广播到各 Worker（`collective_rpc`）；Worker 按 rank 切权重与 KV cache。
- **收益**：突破单卡显存/算力上限，按负载选策略；DP 提升吞吐、TP 降延迟、PP 跨节点、EP 服务 MoE。

---

## 1. 场景与痛点

### 1.1 四种并行的取舍

| 策略 | 解决 | 代价 | 典型场景 |
| --- | --- | --- | --- |
| TP | 单卡显存/算力不够，按层内切（列/行并行） | 每步两次 all-reduce，通信密集 | 单机多卡、低延迟 |
| PP | 跨节点放不下、想减通信 | 流水线气泡、需 micro-batch | 多机、超大模型 |
| EP | MoE 专家数爆炸 | all-to-all 通信 | MoE（DeepSeek/Qwen-MoE） |
| DP | 提高吞吐 | 不降单请求延迟、需独立 KV cache | 高 QPS |

### 1.2 为什么需要多进程执行器

每个 rank 是一个进程（或 Ray actor）。前端engine 不能直连每卡，需 `Executor` 抽象：`collective_rpc` 把同一个调用广播到所有 Worker。V1 默认多进程执行器（`MultiprocExecutor`）用共享内存 + 消息总线降低开销。

---

## 2. 核心设计

### 2.1 ParallelConfig

```python
# vllm/config/parallel.py:119
class ParallelConfig:
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    data_parallel_size_local: int
    expert_parallel_size: int          # 或 enable_expert_parallel 推导
    enable_expert_parallel: bool
    distributed_executor_backend: str  # mp / ray
    # ...
```

### 2.2 进程组初始化

```python
# vllm/distributed/parallel_state.py:1755
def init_distributed_environment(...):   # 建 world group

# vllm/distributed/parallel_state.py:1920
def initialize_model_parallel(...):      # 建 TP/PP/EP/DP 子 group

# vllm/distributed/parallel_state.py:2265
def get_tensor_model_parallel_rank() -> int:   # 取当前 rank
```

各 group 独立：TP group 用于层内 all-reduce，PP group 用于 stage 间 send/recv，EP group 用于 all-to-all，DP group 用于多副本间负载均衡/聚合。

### 2.3 Executor

```python
# vllm/v1/executor/multiproc_executor.py:111
class MultiprocExecutor(Executor):
    # 每个 rank 起一个 Worker 进程，共享内存消息总线

# vllm/v1/executor/multiproc_executor.py:375
def collective_rpc(self, method, ...):
    # 把 method 调用广播到所有 Worker，收集返回
```

抽象基类 `Executor`（`vllm/v1/executor/abstract.py`）定义 `execute_model` / `collective_rpc` / `initialize_kv_cache` 等接口；`UniprocExecutor`（单进程调试）、`RayExecutor`、`RayDistributedExecutor` 为变体。

### 2.4 Worker 按 rank 切分

`GPUWorker` / `GPUModelRunner` 在 `determine_available_memory` 和 `initialize_kv_cache` 时，按 TP rank 把注意力 head 数、KV 维度切成 `head_dim * num_heads / tp_size`；KV cache 也按 rank 分配（见 `03`）。PP 下每个 stage 只持有部分层，靠 PP group 的 send/recv 传激活。

### 2.5 通信原语与 custom all-reduce

`vllm/distributed/device_communicators/`：pynccl（NCCL 封装）、shm_broadcast（同机共享内存）、custom_all_reduce（自研、绕开 NCCL 启动开销、适用于同机 NVLink/PCIe 拓扑）。custom all-reduce 在「小张量、高频率」的 TP all-reduce 上明显快于 NCCL，但有拓扑条件（同机、特定连接），不满足条件时自动回退 NCCL。

### 2.6 DP 与 DPCoordinator

`DPAsyncMPClient`（[vllm/v1/engine/core_client.py:369](../vllm/v1/engine/core_client.py#L369)）+ `DPCoordinator`（`vllm/v1/engine/coordinator.py`）把请求散到多个 EngineCore 副本，做负载均衡与结果汇聚。DP attention（`vllm/v1/worker/dp_utils.py`）配合 micro-batching（`vllm/v1/worker/ubatching.py` / `ubatch_utils.py`）让 attention 在 DP 维度并行。

### 2.7 EP / MoE all-to-all

MoE 的 expert 分散在各 rank，token 经 all-to-all 路由到对应专家再 gather 回来。实现见 `vllm/distributed/device_communicators/all2all.py` 与 `vllm/model_executor/layers/fused_moe/`（DeepEP / pplx / naive 等）。EPLB（`vllm/distributed/eplb/`）做专家负载均衡与权重重排，缓解热点。

---

## 3. 代码走读

### 3.1 启动建组

```
set_parallel_resources -> init_distributed_environment (world group)
                      -> initialize_model_parallel (TP/PP/EP/DP 子 group)
                      -> MultiprocExecutor 起各 rank Worker
```

### 3.2 每步执行

```
EngineCore.step -> scheduler.schedule -> executor.execute_model(scheduler_output)
   -> collective_rpc("execute_model", scheduler_output) 广播到所有 Worker
   -> 各 Worker 跑本 rank 的 forward 片（TP 切 head，PP 跑本 stage，EP 跑本专家）
   -> TP all-reduce 合并 / PP send-recv / EP all-to-all
   -> 收集 ModelRunnerOutput 回 EngineCore
```

### 3.3 TP all-reduce 位置

每层 attention/MLP 的列并行后做一次 all-reduce（合并各 rank 的部分和），行并行前无通信。V1 用 custom all-reduce（同机）或 NCCL（跨机）。

---

## 4. 关键数据结构

| 结构 | 字段 | 位置 |
| --- | --- | --- |
| `ParallelConfig` | tp/pp/dp/ep size | [config/parallel.py:119](../vllm/config/parallel.py#L119) |
| `Executor.collective_rpc` | 广播调用 | [multiproc_executor.py:375](../vllm/v1/executor/multiproc_executor.py#L375) |
| `MultiprocExecutor` | 多进程执行器 | [multiproc_executor.py:111](../vllm/v1/executor/multiproc_executor.py#L111) |
| 进程组 getter | get_tensor/pp/ep/dp_rank | [parallel_state.py:2265](../vllm/distributed/parallel_state.py#L2265) |

---

## 5. 收益与代价

**收益**
- 突破单卡限制：TP/PP 放下超大模型，EP 服务 MoE，DP 提吞吐。
- 按负载选策略，灵活扩缩容。

**代价 / 注意点**
- **通信开销**：TP 每步 all-reduce、EP all-to-all、PP 气泡，跨节点带宽成瓶颈。
- **显存**：PP 各 stage 独立 KV cache；TP 切分但仍需副本通信缓冲。
- **调试复杂**：多进程 + 多 group，错误定位难。
- **调度耦合**：DP 下需 DPCoordinator 做负载均衡；PP 下 V1 是否保留 virtual engine 以本仓库为准。

---

## 6. 面试高频问题

**Q1：TP 和 PP 怎么选？**
A：TP 降单请求延迟但通信密集（适合同机 NVLink）；PP 跨节点、气泡大，适合单卡放不下且跨机（[config/parallel.py:119](../vllm/config/parallel.py#L119)）。

**Q2：TP 的 all-reduce 在哪发生？为什么能自定义？**
A：每层列并行后 all-reduce。V1 用 custom all-reduce（同机、绕开 NCCL 启动开销）或 NCCL（跨机），不满足条件自动回退（[vllm/distributed/device_communicators/](../vllm/distributed/device_communicators/)）。

**Q3：EP 为什么用于 MoE？**
A：MoE 专家数多、单 token 只激活少数专家，按专家切到各 rank，token 经 all-to-all 路由+gather（[vllm/model_executor/layers/fused_moe/](../vllm/model_executor/layers/fused_moe/)）。

**Q4：DP 提升吞吐但不降延迟，为什么？**
A：DP 是多个独立副本，每副本服务不同请求，提高总吞吐；单请求仍走单副本的完整 decode，延迟不变。需 DPCoordinator 做负载均衡（[vllm/v1/engine/coordinator.py](../vllm/v1/engine/coordinator.py)）。

**Q5：collective_rpc 是什么？**
A：Executor 把同一个方法调用广播到所有 Worker 并收集返回（[vllm/v1/executor/multiproc_executor.py:375](../vllm/v1/executor/multiproc_executor.py#L375)），是 V1 多进程协同的基础原语。

**Q6：KV cache 在 TP 下怎么分？**
A：按 TP rank 切 head 数，每 rank 只持有 `num_heads/tp_size` 的 KV，KV cache 按 rank 分配（[vllm/v1/worker/gpu_model_runner.py](../vllm/v1/worker/gpu_model_runner.py) 的 `initialize_kv_cache`）。

**Q7：V1 默认执行器是什么？**
A：多进程 `MultiprocExecutor`（[vllm/v1/executor/multiproc_executor.py:111](../vllm/v1/executor/multiproc_executor.py#L111)），用共享内存消息总线；Ray 场景用 `RayExecutor`/`RayDistributedExecutor`。

**Q8：PP 在 V1 怎么传递激活？**
A：靠 PP group 的 send/recv 在 stage 间传激活；每个 stage 只跑部分层（[vllm/distributed/parallel_state.py:1920](../vllm/distributed/parallel_state.py#L1920) 建 PP group）。

**Q9：EPLB 解决什么？**
A：MoE 专家负载不均导致热点，EPLB（`vllm/distributed/eplb/`）做专家负载均衡与权重重排，提升 EP 利用率。

**Q10：同机多卡为什么用 custom all-reduce 不用 NCCL？**
A：custom all-reduce 绕过 NCCL 的启动/同步开销，对 TP 频繁的小张量 all-reduce 更快；但有同机与拓扑条件，不满足则回退 NCCL。

**Q11：DP attention 是什么？**
A：attention 在 DP 维度并行计算（[vllm/v1/worker/dp_utils.py](../vllm/v1/worker/dp_utils.py)），配合 micro-batching（[vllm/v1/worker/ubatching.py](../vllm/v1/worker/ubatching.py)）提升注意力吞吐。

**Q12：initialize_model_parallel 建了哪些 group？**
A：TP / PP / EP / DP 四种子 group，分别服务于层内归约、stage 间传输、专家路由、副本间协同（[vllm/distributed/parallel_state.py:1920](../vllm/distributed/parallel_state.py#L1920)）。

---

## 7. 延伸阅读

- 配置：`vllm/config/parallel.py`、`vllm/config/compilation.py`
- 进程组：`vllm/distributed/parallel_state.py`、`vllm/distributed/device_communicators/`
- 执行器：`vllm/v1/executor/`
- 配套：`03-paged-attention-kv-cache.md`、`10-kv-cache-offload-and-pd-disaggregation.md`
- 官方：https://docs.vllm.ai/en/latest/serving/distributed_serving.html
