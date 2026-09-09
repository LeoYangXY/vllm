# CUDA Graph 捕获与 torch.compile

> 适用版本：vLLM V1。实现在 `vllm/v1/cudagraph_dispatcher.py`、`vllm/config/compilation.py`、`vllm/compilation/`，以及 `vllm/v1/worker/gpu_model_runner.py` 的捕获流程。

## 0. TL;DR

- **是什么**：把一整个 decode step 的 CUDA kernel 序列「录制」成一张图，之后只 replay 不重排（消除 CPU launch 开销）；并用 `torch.compile` 做 FX graph 切分与算子融合。
- **解决什么**：小 batch decode 时，kernel launch + Python/Triton dispatch 开销占比高，GPU 在等 CPU 喂指令；图捕获把这些开销变成一次 replay。
- **怎么做**：`CudagraphDispatcher` 按 batch size 选择 FULL / PIECEWISE / 不捕获；attention 动态形状等不能进图的部分走 piecewise（图外算 attention）；`CompilationConfig` 控制 cudagraph 模式、torch.compile level、splitting_ops。
- **收益**：decode 延迟与 CPU 开销显著下降，小 batch 吞吐提升；但带来显存占用、动态形状限制、与部分 feature 的兼容约束。

---

## 1. 场景与痛点

### 1.1 为什么需要 CUDA Graph

每个 decode step：Python 侧构造输入张量 → 调用各层 forward → 每层多个 CUDA kernel（launch）。kernel launch 是 CPU→GPU 的异步提交，有固定开销。batch 小（低 QPS）时，GPU 每个 step 实际计算量很小，launch 开销占比可达 20%~40%，GPU 大量时间在「等 CPU 排指令」。

CUDA Graph 把「一步内的所有 kernel + 它们的依赖」录成图，之后每步只需一次 `graph.replay()`（一次提交），彻底消除逐 kernel launch 与 Python dispatch。

### 1.2 torch.compile 又解决什么

光有 CUDA Graph 还不够：图里还是一堆小 kernel。torch.compile 把 FX graph 切块（按 `splitting_ops`），对每块做算子融合（RMSNorm+Quant、SiluMul+Quant、AllReduce fusion、attention+quant fusion 等），减少 kernel 数、提升算术强度。两者配合：图捕获固定控制流，compile 融合减小 kernel 数。

---

## 2. 核心设计

### 2.1 CudagraphDispatcher 模式

```python
# vllm/v1/cudagraph_dispatcher.py:15
class CudagraphDispatcher:
    # 模式: FULL / PIECEWISE / FULL_AND_PIECEWISE / NONE
    # dispatch(batch_size, ...) -> 选择本步用哪种 runtime mode
```

- **FULL**：整步进图（最快，但要求所有形状固定、无动态分支）。
- **PIECEWISE**：模型被切成多段，每段单独 capture 成图，attention 等动态部分在图外 eager 执行。兼容更多模型/feature。
- **FULL_AND_PIECEWISE**：两者都捕获，按 batch size 选最优。
- **NONE**：不捕获（调试/不支持时）。

```python
# vllm/v1/cudagraph_dispatcher.py:235
def dispatch(self, batch_size, ..., is_non_cudagraph_batch=False):
    # 根据 batch_size 是否落在 capture sizes 集合，以及 attention 是否支持，
    # 决定走 full graph / piecewise graph / eager
```

`capture_sizes`（`cudagraph_capture_sizes`）是一组预捕获的 batch size，运行时按最近匹配选择。

### 2.2 哪些不能进图（必须 piecewise 或 eager）

- **attention 的动态形状**：不同请求序列长度不同，KV block 形状变化；部分 attention backend（如某些 flash-attn 变体）不支持 graph。V1 通过 `splitting_ops` 把 attention 切成图外段。
- **采样随机数**：每个 step 的 seed 不同，不能固化。
- **spec decode 的树形注意力**：动态结构难以捕获。
- **动态控制流**：如有些请求 finish、batch 成员变化——通过 persistent batch + 固定最大 batch 缓解。

### 2.3 捕获流程

`GPUModelRunner` 在启动期做 dummy_run（用假输入跑一遍、测显存），再按 `capture_sizes` 逐一 capture：

```python
# vllm/v1/worker/gpu_model_runner.py （capture 流程，按本仓库实现）
# initialize_cudagraph_capture() -> _dummy_run() -> 对各 size capture 图
```

捕获时分配独立的权重/激活缓冲（graph 的「静态」输入输出地址固定），不能与常规 forward 混用。

### 2.4 CompilationConfig

```python
# vllm/config/compilation.py:53
class CUDAGraphMode(enum.Enum):
    FULL = 0
    PIECEWISE = 1
    FULL_AND_PIECEWISE = 2
    NONE = 3

# vllm/config/compilation.py:398
class CompilationConfig:
    # level: 0~3（compile 强度，越高融合越多、首次编译越慢）
    # custom_ops / pass_config / splitting_ops / compile_sizes / inductor_compile_config
    # cudagraph_mode / cudagraph_capture_sizes
```

`level` 含义（按本仓库 `CompilationConfig` 注释）：0=不 compile；1=基础；2=更多 fusion；3=全量 fusion（含 attention/quant 融合、collective fusion 等）。越高首次启动越慢（torch.compile 编译耗时），但运行时越快。

### 2.5 torch.compile 与融合 pass

`vllm/compilation/` 下：`backends.py`（`VllmBackend`）、`pass_manager.py`、各类 fusion pass（RMSNorm+Quant、SiluMul+Quant、AllReduce fusion、attention+quant fusion、collective fusion 等）。编译结果按 `VLLM_CACHE_ROOT`/缓存目录落地，避免每次重启重编（首次冷启动慢的根因）。

---

## 3. 代码走读

### 3.1 捕获决策

```python
# vllm/v1/cudagraph_dispatcher.py:235 （示意）
def dispatch(self, batch_size, is_non_cudagraph_batch=False, ...):
    if self.cudagraph_mode == CUDAGraphMode.NONE or is_non_cudagraph_batch:
        return CUDAGRAPH_DISABLED
    if batch_size in self.capture_sizes:
        # 命中预捕获 size -> full 或 piecewise graph replay
        ...
    else:
        # 未命中 -> 部分模式或 eager
        ...
```

### 3.2 切分点 splitting_ops

`CompilationConfig.splitting_ops` 列出「图必须在此断开」的算子（典型如 attention）。torch.compile 把 FX graph 按这些 op 切成多块，每块各自 capture/编译，attention 块不进图（走 eager/piecewise）。

### 3.3 Dummy run 与显存

启动期 `_dummy_run` 用最大 batch / 最长序列的假输入跑一遍，探测峰值显存（含 KV cache 之外的 graph 缓冲），据此确定 `num_gpu_blocks`。所以 **CUDA Graph 的图缓冲会与 KV cache 争抢显存**——`capture_sizes` 越多、size 越大，图占显存越多，可服务的 KV block 越少，需权衡。

---

## 4. 关键数据结构

| 结构 | 字段 | 位置 |
| --- | --- | --- |
| `CudagraphDispatcher` | dispatch / capture_sizes / mode | [cudagraph_dispatcher.py:15](../vllm/v1/cudagraph_dispatcher.py#L15) |
| `CUDAGraphMode` | FULL/PIECEWISE/FULL_AND_PIECEWISE/NONE | [compilation.py:53](../vllm/config/compilation.py#L53) |
| `CompilationConfig` | level/cudagraph_mode/splitting_ops/compile_sizes | [compilation.py:398](../vllm/config/compilation.py#L398) |
| `VllmBackend` | torch.compile 后端 | [vllm/compilation/backends.py](../vllm/compilation/backends.py) |

---

## 5. 收益与代价

**收益**
- 消除逐 kernel launch 与 Python dispatch，decode 延迟与 CPU 开销显著下降。
- 小 batch 吞吐提升（GPU 不再等 CPU）。
- torch.compile 融合减少 kernel 数、提升算术强度。

**代价 / 限制 / 失效场景**
- **显存代价**：每个 capture size 一份图缓冲，与 KV cache 争抢显存；`capture_sizes` 多 → KV block 少。
- **动态形状**：未命中 `capture_sizes` 的 batch size 走 eager，可能变慢；attention 动态形状需 piecewise。
- **与 LoRA**：多 LoRA 权重切换使图难以复用，常走 eager/piecewise（见 `12`）。
- **与多模态/PP/抢占**：encoder 输出动态、PP 的 send/recv、抢占导致的 batch 变动，都会限制图的覆盖。
- **首次启动慢**：torch.compile 编译 + 图捕获耗时（尤其 `level=3`），靠编译缓存缓解但不总是命中。
- **可调试性差**：图内报错栈不直观，需 `CUDA_LAUNCH_BLOCKING=1` 或关图调试。

---

## 6. 面试高频问题

**Q1：CUDA Graph 为什么能加速 decode？**
A：把一步的所有 kernel 录成图，之后一次 `replay()` 替代逐 kernel launch + Python dispatch，消除 CPU 喂指令的开销（小 batch 下占比很高）。

**Q2：FULL 和 PIECEWISE 的区别？**
A：FULL 整步进图（最快但要求全固定形状）；PIECEWISE 按 `splitting_ops` 切块，每块单独图、attention 等动态部分走图外 eager，兼容性更好（[vllm/v1/cudagraph_dispatcher.py:15](../vllm/v1/cudagraph_dispatcher.py#L15)）。

**Q3：为什么 attention 常不能进图？**
A：不同请求序列长度不同、KV block 形状动态变化，部分 attention backend 不支持固化形状；故把 attention 设为 splitting op，图只覆盖其前后稳定部分。

**Q4：CUDA Graph 和 KV cache 显存怎么争抢？**
A：每个 capture size 一份图缓冲占显存，会减少可分配的 KV block 数；`_dummy_run` 探测峰值后定 `num_gpu_blocks`（[vllm/v1/worker/gpu_model_runner.py](../vllm/v1/worker/gpu_model_runner.py)）。

**Q5：torch.compile 的 level 是什么？**
A：`CompilationConfig.level` 0~3 控制融合强度（[vllm/config/compilation.py:398](../vllm/config/compilation.py#L398)），越高融合越多、运行越快，但首次编译越慢。

**Q6：为什么首启慢？怎么缓解？**
A：torch.compile 编译 + 图捕获耗时长；靠 `VLLM_CACHE_ROOT` 下的编译缓存缓解，但模型/配置变了易失效要重编。

**Q7：LoRA 和 CUDA Graph 冲突吗？**
A：多 LoRA 权重切换使图难复用，V1 通常让 LoRA 走 eager/piecewise（见 `12`）。

**Q8：spec decode 下为什么 graph 难用？**
A：树形注意力动态形状难以固化，spec decode 常走 piecewise 或不捕获 attention（见 `07`）。

**Q9：capture_sizes 怎么选？**
A：覆盖典型 batch size（如 1,2,4,8,16,...），运行时按最近匹配选图；太多占显存、太少则常 fallback eager。

**Q10：调试图内报错怎么办？**
A：用 `CUDA_LAUNCH_BLOCKING=1`、或临时 `cudagraph_mode=NONE` 跑 eager 定位；图报错栈不直观。

**Q11：splitting_ops 的作用？**
A：告诉 torch.compile 在哪些算子处把 FX graph 断开，断开后的块各自编译/捕获，动态部分（attention）留图外（[vllm/config/compilation.py:398](../vllm/config/compilation.py#L398)）。

**Q12：为什么 persistent batch 对 CUDA Graph 很重要？**
A：图要求形状固定，persistent batch 用「固定最大 batch + slot 复用」让每步输入张量形状稳定，才能进图（见 `02` / `05`）。

---

## 7. 延伸阅读

- 实现：`vllm/v1/cudagraph_dispatcher.py`、`vllm/config/compilation.py`、`vllm/compilation/`（backends / pass_manager / fusion passes）、`vllm/v1/worker/gpu_model_runner.py`（capture 流程）
- 配套：`05-scheduler-continuous-batching.md`（persistent batch）、`07-speculative-decoding.md`、`12-lora-and-multilora.md`
- 官方：https://docs.vllm.ai/en/latest/performance/cuda_graph.html
