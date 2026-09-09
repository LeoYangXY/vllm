# V1 调度器：Continuous Batching、Chunked Prefill、抢占与异步调度

## 0. TL;DR

- vLLM V1 没有"prefill 阶段 / decode 阶段"的概念。`Scheduler.schedule()` 每个 step 只做一件事：让每个请求的 `num_computed_tokens` 追上它的 `num_tokens_with_spec`，追不上就分几次追（chunked prefill），追上了就停止。见 [vllm/v1/core/sched/scheduler.py:565](../vllm/v1/core/sched/scheduler.py#L565) 的 `NOTE(woosuk)`。
- 一个 step 的调度分两段：先遍历 `self.running`（已有请求的 decode + 未完成的 prefill chunk），再遍历 `waiting`/`skipped_waiting`（新请求与被抢占请求）。[scheduler.py:612](../vllm/v1/core/sched/scheduler.py#L612) 与 [scheduler.py:847](../vllm/v1/core/sched/scheduler.py#L847)。
- 硬约束只有三个：`token_budget`（= `max_num_scheduled_tokens`，默认等于 `max_num_batched_tokens=2048`）、`max_num_running_reqs`（= `max_num_seqs=128`）、KV block 池空闲块数。三者任一不满足就停止准入或触发抢占。
- Chunked prefill 由 `enable_chunked_prefill`（默认 True）+ `max_num_batched_tokens` 自然实现：`num_new_tokens = min(需要量, 剩余 budget)`；`long_prefill_token_threshold` 是额外的单请求每步上限（默认 0，即不限制）。
- V1 **只支持 RECOMPUTE 式抢占**：`_preempt_request()` 直接 `num_computed_tokens = 0` + 释放全部 block + `waiting.prepend_request()`。全仓库 `vllm/v1/` 下不存在 `PreemptionMode`/`swap_out`/`SWAP` 语义（已实际搜索确认），SWAP 是 V0 遗留概念。
- `AsyncScheduler` 通过"输出占位符"（`num_output_placeholders`）让调度器在 GPU 还在跑第 N 步时就按预测状态调度第 N+1 步，从而把 Python 调度开销隐藏在 GPU 执行里；收到真实输出后再做 correction。

---

## 1. 场景与痛点

### 1.1 request-level batching（静态 batch）为什么不行

传统的 "request-level batching"：攒够 N 个请求 → 一起 prefill → 一起 decode → **等最慢的那个请求生成完** → 整批返回 → 再组下一批。它有三个致命问题：

1. **队头阻塞与 GPU 空转**：batch 内某个请求提前遇到 EOS 结束，它占的 batch slot 与显存不能立刻给别人，只能空着陪跑到全批结束。一个 batch 里请求输出长度服从长尾分布时（有的 20 token，有的 2000 token），GPU 实际有效利用率可以低到 `sum(实际输出) / (batch_size × max(输出长度))`，典型值只有 30%~50%。
2. **尾延迟（P99）被最长请求绑架**：TTFT 要等 batch 攒够；单个请求的 latency 被 batch 内最长者决定。交互式场景下这是不可接受的。
3. **无法应对动态到达**：线上流量是泊松到达的，静态 batch 强制"要么等一批、要么浪费 slot"，没有中间态。

### 1.2 只做 iteration-level batching 还不够：长 prefill 的两种破坏

即使做到了"每个 step 重组 batch"，如果允许一个请求的完整 prompt 在一个 step 内一次性 prefill，还会引入两类问题：

- **Decode 被 stall，破坏 TPOT SLO**：一个 32K 的长 prompt，其 prefill 前向的计算量与 FLOPs 是单 token decode 的数千倍。若它与 200 个正在 decode 的请求同处一个 step，这一步的耗时由 prefill 决定，这 200 个请求的 TPOT 全部被拉长。反过来，若把 prefill 独占一个 step，那么这一 step 内所有 decode 请求完全不产出 token——TPOT 直接出现一个断崖。
- **Activation OOM**：prefill 的 activation 形状是 `[num_tokens, hidden]`（还要乘以层数、各类中间张量）。32K token × 8K hidden × 80 层，即使只看 attention 前后的几份大张量，也是 GB 级。不切块就会在长 prompt 上直接 OOM。而 `max_num_batched_tokens` 恰好同时约束了这两件事：它既限定了单步的最大 token 数（因而限定了 activation 峰值），也限定了单步 prefill 能推进多远（因而限定了对 decode 的 stall 时长）。

### 1.3 Python 调度开销成为新瓶颈

V1 把调度做成了纯 Python 的 CPU 工作：每个 step 要遍历 running 列表、做 prefix cache 哈希查询、调 `allocate_slots`、拼 `CachedRequestData`、处理停止与 logprobs。在 7B~8B 模型 + batch=1 + 小 `max_num_batched_tokens` 的场景下，一次 GPU decode 前向可能只有 3~5 ms，而 Python 调度要 1~2 ms——**GPU 有 20%~40% 的时间在等 CPU**。这就是 `AsyncScheduler` 存在的动机：把第 N+1 步的调度提前到第 N 步 GPU 执行期间做完。

---

## 2. 核心设计

### 2.1 一个统一的抽象：`num_computed_tokens` 追赶 `num_tokens_with_spec`

V1 调度器最核心的设计是**消解了 prefill/decode 的二元对立**。源码注释写得非常直白：

```text
There's no "decoding phase" nor "prefill phase" in the scheduler.
Each request just has the num_computed_tokens and num_tokens_with_spec.
num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
At each step, the scheduler tries to assign tokens to the requests so that
each request's num_computed_tokens can catch up its num_tokens_with_spec.
```

即：

- 一个刚入队的 4K prompt 请求：`num_computed_tokens=0`，`num_tokens_with_spec=4000`。若 budget 只剩 512，本 step 就分配 512，于是 `num_computed_tokens=512` —— 这就是 **chunked prefill**。
- 一个已生成 37 个 token 的请求：`num_computed_tokens=4037`，`num_tokens_with_spec=4038`，本 step 分配 1 —— 这就是 **decode**。
- 一个 EAGLE 草稿已就绪的请求：`spec_token_ids` 非空，`num_tokens_with_spec` 已经包含草稿位，本 step 可能分配 `1 + K` —— 这就是 **speculative decoding 的 verify**。

三者共用同一段代码路径，不需要分支。这个抽象让 chunked prefill、prefix caching、spec decode、jump decoding 可以同时叠加而不产生组合爆炸。

### 2.2 请求状态机

`RequestStatus` 定义在 [vllm/v1/request.py:364](../vllm/v1/request.py#L364)：

```text
WAITING ──admit──> RUNNING ──stop/abort──> FINISHED_*
   ^                  │
   │                  │ KV block 不足
   │              PREEMPTED  (prepend 回 waiting 队首)
   └──────────────────┘
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR / WAITING_FOR_REMOTE_KVS /
WAITING_FOR_STREAMING_REQ  ── 阻塞态，放 skipped_waiting 队列
```

注意 `is_finished()` 的实现是 `status > RequestStatus.PREEMPTED`（[request.py:386](../vllm/v1/request.py#L386)），所以 `PREEMPTED` 是一个"可复活"的中间态，而非终止态。

### 2.3 调度主循环流程图

```
              EngineCoreProc.step()   —— 每个 engine step 调用一次
                          |
                          v
        +--------------------------------------------+
        |  Scheduler.schedule(throttle_prefills)     |
        +--------------------------------------------+
                          |
   (0) current_step += 1
       kv_cache_manager.new_step_starts()
                          |
   (1) token_budget = max_num_scheduled_tokens          # 逻辑 token 预算
       input_budget = max_num_batched_tokens            # 物理输入槽位预算
       draft_slots  = spec.max_num_new_slots_for_drafting
                          |
                          v
   (2) ==== 阶段 A：遍历 self.running（decode + 未完成 chunk）====
       while req_index < len(running) and token_budget > 0:
          |-- input_budget <= draft_slots?                 -> break
          |-- 已达 max_tokens（异步调度保护）?               -> continue
          |-- current_step < next_decode_eligible_step?     -> continue
          |-- defer_prefills and is_prefill_chunk?          -> continue
          |-- ec_connector 未就绪?                          -> continue
          |
          num_new_tokens = min(need,
                               long_prefill_token_threshold (若>0),
                               token_budget,
                               input_budget - draft_slots,
                               max_model_len 余量)
          |-- mamba 对齐裁剪 -> encoder 预算裁剪 -> MTP lookahead 裁剪
          |-- num_new_tokens == 0 ?                        -> continue
          |
          allocate_slots() == None ?  ──yes──> 选 victim 抢占
          |                                    （FCFS: running.pop()
          |                                     PRIORITY: max(priority, arrival)）
          |                                    回滚 victim 已扣预算，重试
          no
          |
          记账：scheduled_running_reqs / req_to_new_blocks /
                num_scheduled_tokens / token_budget -= n / input_budget -= n + draft_slots
                          |
                          v
   (3) ==== 阶段 B：waiting / skipped_waiting（新请求 + 被抢占请求）====
       if not preempted_reqs and pause_state == UNPAUSED:
         while (waiting or skipped_waiting) and token_budget > 0:
           |-- len(running) + num_waiting_for_streaming >= max_num_seqs -> break
           |-- 阻塞态 / 有 stale output / LoRA 超配额 -> 移入 skipped_waiting, continue
           |-- prefix cache 查询：num_new_local_computed_tokens
           |-- connector：num_external_computed_tokens（可能走 async load）
           |-- num_new_tokens = num_tokens - num_computed_tokens
           |-- long_prefill_threshold / enable_chunked_prefill / budget 三重裁剪
           |-- allocate_slots() == None ?  -> encoder_cache_manager.free(request); break
           |-- 出队 -> running.append(); status = RUNNING;
               num_computed_tokens = num_computed_tokens
                          |
                          v
   (4) 断言（total <= max_num_scheduled_tokens / len(running) <= max_num_seqs）
       num_common_prefix_blocks（cascade attention）
       组装 SchedulerOutput
       _update_after_schedule(): num_computed_tokens += n; num_in_flight_tokens += n
                          |
                          v
              executor.execute_model(...)  ──GPU──>  update_from_output()
                                                            |
                                                    回到 (0)，下一个 step
```

### 2.4 一个 step 的 batch 组成图

假设 `max_num_batched_tokens = 2048`、`max_num_seqs = 256`、`block_size = 16`、无 spec decode：

```
step N 的 num_scheduled_tokens = {
    "A": 1,      # 已 decode 很久的老请求
    "B": 1,      # decode
    "C": 1,      # decode
    "D": 512,    # 8K prompt，正在跑第 2 个 chunk（running，上一 step 已跑 512）
    "E": 512,    # 3K prompt，本 step 首次准入，切成 chunk（还剩 2560）
    "F": 1021,   # 1021 token prompt，本 step 一次跑完（budget 剩 1021）
}
total_num_scheduled_tokens = 1+1+1+512+512+1021 = 2048  == token_budget

GPU 侧 InputBatch 布局（逻辑上）：

  row 0: [A]  len=1    ─┐
  row 1: [B]  len=1     ├─ decode 行：query_len=1，走 paged decode kernel
  row 2: [C]  len=1    ─┘
  row 3: [D]  len=512  ─┐
  row 4: [E]  len=512   ├─ prefill chunk 行：query_len>1，走 prefill/varlen kernel
  row 5: [F]  len=1021 ─┘
       └────────────────────────────────────────────┘
        total = 2048 tokens，被拼接成一个扁平的 input_ids/positions
```

三种行在同一个 forward 里，靠 `num_scheduled_tokens` 这个 `req_id -> int` 字典区分：模型 runner 据此构造 query 起始偏移、`slot_mapping` 与 attention 元数据。这个字典是 `SchedulerOutput` 里最关键的字段（[output.py:239](../vllm/v1/core/sched/output.py#L239)）。

### 2.5 抢占时序图

```
 step N    running = [A, B, C]        waiting = [D]
           block_pool 空闲块 = 0
                    |
                    v
           allocate_slots(A, num_new_tokens=1) -> None
                    |
                    v
           选 victim：
             FCFS     -> victim = self.running.pop()            # 最晚进入的（LIFO）
             PRIORITY -> victim = max(running, key=(priority, arrival_time))
           从 running 摘掉 victim；若 victim 本 step 已被调度过，
           回滚它占用的 token_budget / input_budget / encoder_compute_budget
                    |
                    v
           _preempt_request(victim=C):
              _free_request_blocks(C)      # 归还 C 的全部 KV block（ref_cnt--）
              encoder_cache_manager.free(C)
              _inflight_prefills.discard(C)
              status            = PREEMPTED
              num_computed_tokens = 0          <== 关键：整段从头重算
              num_stale_output_tokens = num_in_flight_tokens
              num_output_placeholders = 0
              num_preemptions  += 1
              waiting.prepend_request(C)       <== 插到队首，尽快重跑
              reset_preempted_req_ids.add(C)
                    |
                    v
           重试 allocate_slots(A) -> 成功（C 的 block 已归还）
                    |
                    v
 step N+1  waiting 队首就是 C
           C.num_computed_tokens == 0 -> 走完整 prefix cache 查询路径
             命中（prefix 块还在缓存里） -> 只补算尾部
             未命中（块已被别的请求占用） -> 全量重算
```

---

## 3. 代码走读

### 3.1 构造：约束、队列与三块缓存

`Scheduler.__init__`（[scheduler.py:80](../vllm/v1/core/sched/scheduler.py#L80)）里与调度语义直接相关的字段：

```python
# vllm/v1/core/sched/scheduler.py:124-130
# Scheduling constraints.
self.max_num_running_reqs = self.scheduler_config.max_num_seqs
self.max_num_scheduled_tokens = (
    self.scheduler_config.max_num_scheduled_tokens
    if self.scheduler_config.max_num_scheduled_tokens is not None
    else self.scheduler_config.max_num_batched_tokens
)
self.max_model_len = vllm_config.model_config.max_model_len
```

`max_num_scheduled_tokens` 默认回退到 `max_num_batched_tokens`；两者分离是为了 spec decode——模型可能在 batch 里追加 token，所以"调度器能签发的 token 数"可以小于"模型能接收的 token 数"（[vllm/config/scheduler.py:56](../vllm/config/scheduler.py#L56)）。

三个队列（[scheduler.py:202-206](../vllm/v1/core/sched/scheduler.py#L202)）：

```python
self.policy = SchedulingPolicy(self.scheduler_config.policy)
self.waiting = create_request_queue(self.policy)          # 正常等待
self.skipped_waiting = create_request_queue(self.policy)  # 本 step 被临时跳过
self.running: list[Request] = []
```

`waiting` 是 `RequestQueue`（[request_queue.py:20](../vllm/v1/core/sched/request_queue.py#L20) 的 ABC），两种实现：

- `FCFSRequestQueue`（[request_queue.py:75](../vllm/v1/core/sched/request_queue.py#L75)）直接继承 `deque[Request]`，`pop_request = popleft`，`prepend_request = appendleft`。
- `PriorityRequestQueue`（[request_queue.py:131](../vllm/v1/core/sched/request_queue.py#L131)）用 `heapq`，排序键是 `Request.__lt__`（[request.py:350](../vllm/v1/request.py#L350)）：`(priority, arrival_time, request_id, id(self))`。注意它在文档里明确写了"priority queue 没有队首概念"，`prepend_request` 退化成 `add_request`。

`skipped_waiting` 是一个重要但容易被忽略的设计：一个 waiting 请求可能因为**瞬时约束**（结构化输出 grammar 还没编译完、远程 KV 还在传、streaming 输入还没到、LoRA 配额打满、有 stale output 在飞）而无法调度。如果原地 `continue`，while 循环会死循环；如果直接丢弃，请求会被饿死。V1 的做法是把它挪到 `skipped_waiting`，下个 step 由 `_select_waiting_queue_for_scheduling()`（[scheduler.py:2321](../vllm/v1/core/sched/scheduler.py#L2321)）重新比较两队队首，决定先消费谁。

### 3.2 `schedule()` 开头：预算初始化与 DP prefill 节流

```python
# vllm/v1/core/sched/scheduler.py:581-610
req_to_new_blocks: dict[str, KVCacheBlocks] = {}
num_scheduled_tokens: dict[str, int] = {}
token_budget = self.max_num_scheduled_tokens
spec = self.vllm_config.speculative_config
draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
input_budget = self.scheduler_config.max_num_batched_tokens
if self._pause_state == PauseState.PAUSED_ALL:
    # Do not schedule any requests when paused.
    token_budget = 0
...
self.kv_cache_manager.new_step_starts()

# DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
# all prefill compute unless saturated.
defer_prefills = (
    throttle_prefills and not self.prefill_capacity_bound
) and any(not r.is_prefill_chunk for r in self.running)
```

三个预算：

| 变量 | 初值 | 扣减方式 | 语义 |
|---|---|---|---|
| `token_budget` | `max_num_scheduled_tokens` | `-= num_new_tokens` | 调度器能签发的 token 数 |
| `input_budget` | `max_num_batched_tokens` | `-= num_new_tokens + draft_slots` | 模型实际接收的输入槽位数（含 spec decode 的草稿位） |
| `encoder_compute_budget` | `max_num_encoder_input_tokens` | `-= num_encoder_embeds` | 多模态 encoder 的 token 预算 |

`input_budget` 比 `token_budget` 多扣一个 `draft_slots`，是因为 spec decode 下模型前向会额外接收 `draft_slots` 个草稿 query 行，而调度器"签发"的 token 数并不包含它们。

### 3.3 阶段 A：RUNNING 列表

```python
# vllm/v1/core/sched/scheduler.py:612-616
# First, schedule the RUNNING requests.
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    if input_budget <= draft_slots:
        break
```

注意这里用的是 `while req_index < len(self.running)` 而不是 `for request in self.running`，因为抢占会在循环中**原地修改 `self.running`**（`pop`），需要手工维护游标（见 3.6 的 `victim_index < req_index` 修正）。

每个 running 请求要过几道 `continue` 闸门：

```python
# vllm/v1/core/sched/scheduler.py:619-645
if (
    request.num_output_placeholders > 0
    and request.num_computed_tokens + 2 - request.num_output_placeholders
    >= request.num_prompt_tokens + request.max_tokens
):
    # Async scheduling: Avoid scheduling an extra step when we are sure that
    # the previous step has reached request.max_tokens. ...
    req_index += 1
    continue

if self.current_step < request.next_decode_eligible_step:
    # V2+PP+async: enforce `pp_size` steps between same-req decodes ...
    req_index += 1
    continue

if defer_prefills and request.is_prefill_chunk:
    # DP prefill balancing: defer this in-progress prefill chunk ...
    req_index += 1
    continue
```

然后计算本 step 该给它多少 token：

```python
# vllm/v1/core/sched/scheduler.py:658-676
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
num_new_tokens = min(
    num_new_tokens, token_budget, input_budget - draft_slots
)

# Make sure the input position does not exceed the max model len.
# This is necessary when using spec decoding.
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len
    - request.num_computed_tokens
    - self.num_sampled_tokens_per_step,
)
```

这一段就是 continuous batching 的全部秘密：**"需要多少"被 `token_budget` 截断**。需要 1 个（decode）就给 1 个；需要 30K（长 prefill）就给 `min(30K, 剩余 budget)`，剩下的下个 step 再来。

`num_output_placeholders` 是异步调度专用的乐观计数（见 3.11），同步调度下恒为 0。

### 3.4 阶段 A 的抢占循环

```python
# vllm/v1/core/sched/scheduler.py:729-792
with record_function_or_nullcontext("schedule: allocate_slots"):
    while True:
        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            num_lookahead_tokens=self.num_lookahead_tokens,
        )

        if new_blocks is not None:
            # The request can be scheduled.
            break

        # The request cannot be scheduled.
        # Preempt the lowest-priority request.
        if self.policy == SchedulingPolicy.PRIORITY:
            preempted_req = max(
                self.running,
                key=lambda r: (r.priority, r.arrival_time),
            )
            victim_index = self.running.index(preempted_req)
            del self.running[victim_index]
            if victim_index < req_index:
                req_index -= 1

            if preempted_req in scheduled_running_reqs:
                preempted_req_id = preempted_req.request_id
                scheduled_running_reqs.remove(preempted_req)
                restored = num_scheduled_tokens.pop(preempted_req_id)
                token_budget += restored
                input_budget += restored + draft_slots
                req_to_new_blocks.pop(preempted_req_id)
                scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                    preempted_req_id, None
                )
                if preempted_encoder_inputs:
                    num_embeds_to_restore = sum(...)
                    encoder_compute_budget += num_embeds_to_restore
        else:
            preempted_req = self.running.pop()

        self._preempt_request(
            preempted_req,
            scheduled_timestamp,
            drop_stale_output=self.requires_kv_delivery,
        )
        preempted_reqs.append(preempted_req)
        if preempted_req == request:
            # No more request to preempt. Cannot schedule this request.
            break

if new_blocks is None:
    # Cannot schedule this request.
    break
```

几个必须讲清的细节：

- **FCFS 下 victim 是 `running.pop()`**，即最晚进入 running 的请求（LIFO）。直觉上"后到的请求进度最少，重算代价最低"，而且它大概率是最后一个被 admit 的 prefill，回头重算的 token 数最少。
- **PRIORITY 下 victim 是 `(priority, arrival_time)` 最大者**，即优先级数值最大（最低优先级）且最晚到达的。这符合直觉：抢占最低优先级的。
- **victim 可能就是 `request` 自己**（`running.pop()` 正好弹出当前 request）。此时 `preempted_req == request`，直接 `break`，`new_blocks` 仍是 `None`，外层 `break` 退出整个 running 循环。
- **回滚逻辑**：如果 victim 在本 step 已经被调度过（在 `scheduled_running_reqs` 里），必须把它占用的 token/encoder 预算、block、spec tokens 全部退还，否则预算会算错。这是 V1 里一处很容易被漏掉的复杂度。
- **`break` vs `continue`**：分配失败时是 `break`（停止整个 running 循环），而不是 `continue`。因为 block 池已经没有空闲块了，继续遍历后面的请求只会徒劳地反复抢占。

成功分配后的记账：

```python
# vllm/v1/core/sched/scheduler.py:794-802
scheduled_running_reqs.append(request)
prefill_scheduled |= request.is_prefill_chunk
request_id = request.request_id
req_to_new_blocks[request_id] = new_blocks
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
input_budget -= num_new_tokens + draft_slots
req_index += 1
```

### 3.5 阶段 B：WAITING 队列

```python
# vllm/v1/core/sched/scheduler.py:847-863
# Next, schedule the WAITING requests.
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    step_skipped_waiting = create_request_queue(self.policy)

    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        if input_budget <= draft_slots:
            break
        # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
        # in `running` but still hold a model-runner request slot.
        num_running = len(self.running) + self.num_waiting_for_streaming_input
        if num_running >= self.max_num_running_reqs:
            break

        request_queue = self._select_waiting_queue_for_scheduling()
        assert request_queue is not None
        request = request_queue.peek_request()
```

三个要点：

1. **`if not preempted_reqs`**：本 step 只要发生过抢占，就**完全不再准入新请求**。这是防止"抢占-准入-再抢占"震荡的关键闸门。
2. **`peek_request()` 而非 `pop_request()`**：先试探，只有 `allocate_slots` 成功后才真正出队（[scheduler.py:1201](../vllm/v1/core/sched/scheduler.py#L1201)）。失败时请求留在队里。
3. **`max_num_seqs` 只在这里检查**。running 列表不会被 `max_num_seqs` 削减（它最多等于历史 admission 数），所以 `max_num_seqs` 实际是"并发准入上限"而非"batch 大小上限"。

prefix cache 查询只在 `num_computed_tokens == 0` 时做（[scheduler.py:911](../vllm/v1/core/sched/scheduler.py#L911)），因为已经算过一部分的请求（KV transfer 恢复的、streaming 恢复的）其 `num_computed_tokens` 已经非零：

```python
# vllm/v1/core/sched/scheduler.py:911-918
if request.num_computed_tokens == 0:
    did_prefix_cache_lookup = True
    (
        new_computed_blocks,
        num_new_local_computed_tokens,
        request.shared_prefix_boundary,
        hit_diverged,
    ) = self._get_local_prefix_cache_hit(request)
```

`num_external_computed_tokens` 来自 KV connector（远程实例 / 外部存储的命中）。它在 [scheduler.py:929-963](../vllm/v1/core/sched/scheduler.py#L929) 里与本地命中做对齐仲裁：只有当远程命中**严格超过**本地完整命中（含 sub-block 尾巴）时，才丢弃本地 tail 并改为外部加载，否则保留本地 tail、不加载外部。这段逻辑注释里明确解释了原因是避免与 copy-on-write 竞争。

### 3.6 Chunked Prefill：三重切分约束

waiting 请求本 step 的 token 数：

```python
# vllm/v1/core/sched/scheduler.py:1028-1033
request_token_budget = min(token_budget, input_budget - draft_slots)
# Number of tokens to be scheduled.
# We use `request.num_tokens` instead of
# `request.num_prompt_tokens` to consider the resumed
# requests, which have output tokens.
num_new_tokens = request.num_tokens - num_computed_tokens
```

然后是 `long_prefill_token_threshold` 与 `enable_chunked_prefill` 两道闸门：

```python
# vllm/v1/core/sched/scheduler.py:1059-1074
threshold = self.scheduler_config.long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold

# chunked prefill has to be enabled explicitly to allow
# pooling requests to be chunked
if (
    not self.scheduler_config.enable_chunked_prefill
    and num_new_tokens > request_token_budget
):
    # If chunked_prefill is disabled,
    # we can stop the scheduling here.
    break

num_new_tokens = min(num_new_tokens, request_token_budget)
assert num_new_tokens > 0
```

语义总结：

| 配置 | 默认 | 作用 |
|---|---|---|
| `enable_chunked_prefill` | `True`（[config/scheduler.py:116](../vllm/config/scheduler.py#L116)） | 关掉后，请求要么**整段**塞进剩余 budget，要么本 step 不调度（`break`，连后面的请求也不看了）。同时在 `verify_max_model_len`（[config/scheduler.py:295](../vllm/config/scheduler.py#L295)）里会强制 `max_num_batched_tokens >= max_model_len` |
| `max_num_batched_tokens` | `2048`（[config/scheduler.py:42](../vllm/config/scheduler.py#L42)） | 单步总 token 上限，即 chunk 的实际大小上限 |
| `long_prefill_token_threshold` | `0`（[config/scheduler.py:70](../vllm/config/scheduler.py#L70)），0 表示不限制 | 单请求单步 token 上限。设成一个小于 `max_num_batched_tokens` 的值（如 512），可以让一个 step 里同时容纳更多个长 prefill 的 chunk，**提高长 prompt 之间的公平性**，代价是单个长 prompt 的 TTFT 变长 |
| `disable_chunked_mm_input` | `False`（[config/scheduler.py:149](../vllm/config/scheduler.py#L149)） | 不允许把多模态 item 切成两半，chunk 边界要回退到 mm item 起点之前 |

`long_prefill_token_threshold` 的经典用法：`max_num_batched_tokens=8192, long_prefill_token_threshold=2048`。这样一个 step 里最多 4 个长 prefill 各推进 2048，而不是一个 8192 的长 prompt 独占全部 budget 饿死其他人。

注意 encoder-decoder 模型会在构造时强制关掉 chunked prefill：

```python
# vllm/config/scheduler.py:274-282
if is_encoder_decoder:
    # Chunked prefill should be disabled for encoder-decoder models.
    self.disable_chunked_mm_input = True
    self.enable_chunked_prefill = False
    self.long_prefill_token_threshold = 0
```

### 3.7 Token Budget 的扣减顺序：decode 优先

代码里"先 running 后 waiting"的结构（3.3 / 3.5）就是 decode 优先的实现。这个顺序带来的性质：

1. **已 in-flight 的请求不会被饿死**。running 里的每个请求至少能分到 1 个 token（除非 budget 真的被前面的 prefill chunk 吃光——注意 running 循环里没有"给 running 预留最小 budget"的逻辑，所以理论上一个超长 chunk 可以把后面的 decode 全挡掉，这正是 `long_prefill_token_threshold` 要解决的问题）。
2. **新请求填满剩余 budget**。waiting 循环在 `token_budget > 0` 时持续准入，直到 budget 耗尽或 `max_num_seqs` 打满。
3. **抢占只发生在 running 阶段**。waiting 阶段的 `allocate_slots` 失败是 `break`（[scheduler.py:1166](../vllm/v1/core/sched/scheduler.py#L1166)），不会去抢占 running——因为 waiting 请求本来就没有 KV block，抢占别人给它腾地方会立刻导致颠簸。

还有一个隐式的"公平性"补偿：被抢占的请求会 `prepend` 到 waiting 队首，所以下个 step 它一定第一个被重新准入，不会真的被饿死。

### 3.8 抢占：`_preempt_request`

```python
# vllm/v1/core/sched/scheduler.py:1472-1513
def _preempt_request(
    self, request: Request, timestamp: float, drop_stale_output: bool = False
) -> None:
    """Preempt a request and put it back to the waiting queue.

    NOTE: The request should be popped from the running queue outside of this
    method.
    ...
    """
    assert request.status == RequestStatus.RUNNING, (
        "Only running requests can be preempted"
    )
    self._free_request_blocks(request)
    self.encoder_cache_manager.free(request)
    self._inflight_prefills.discard(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0
    if request.spec_token_ids:
        request.spec_token_ids = []
    # Async scheduling: mark all in-flight output as stale. ...
    request.drop_stale_output = drop_stale_output or (
        request.drop_stale_output and request.num_stale_output_tokens > 0
    )
    request.num_stale_output_tokens = request.num_in_flight_tokens
    request.num_output_placeholders = 0
    request.num_preemptions += 1
    if self.log_stats:
        request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

    # Put the request back to the waiting queue.
    self.waiting.prepend_request(request)
    self.reset_preempted_req_ids.add(request.request_id)
```

关键点：

- **`num_computed_tokens = 0`**：V1 是纯 RECOMPUTE 抢占，没有 swap。释放的 block 回到 block pool 后，如果 prefix cache 仍保留对应 hash（block 的 `ref_cnt` 归零但还在 free 队列里未被复用），重新调度时 `get_computed_blocks` 可以命中，**实际只补算尾部**。这就是为什么抢占的代价在实践中远低于理论值。
- **`num_stale_output_tokens = num_in_flight_tokens`**：异步调度下，GPU 上可能还有这个请求若干 step 的 token 没回来。这些 token 仍会投递（不能丢，否则 spec decode 接受率统计会乱），但不能再改动已经归零的计数器。
- **`waiting.prepend_request(request)`**：不是 `add_request`。抢占的请求应尽快重试。
- **`reset_preempted_req_ids`**：本 step 被抢占的 id 集合，会被放进 `SchedulerOutput.preempted_req_ids`（[output.py:267](../vllm/v1/core/sched/output.py#L267)，注释写明 "Only used for v2 model runner"），供 v2 runner 清理 persistent batch 里的状态。

另一个抢占入口是 `reset_prefix_cache(reset_running_requests=True)`：

```python
# vllm/v1/core/sched/scheduler.py:2700-2716
if reset_running_requests:
    timestamp = time.monotonic()
    # Preempt in reverse order so the requests will be added back to the
    # running queue in FIFO order.
    while self.running:
        request = self.running.pop()
        self._preempt_request(request, timestamp, drop_stale_output=True)
    self.prev_step_scheduled_req_ids.clear()
```

（权重热更新场景用，配合 `drop_stale_output=True`，因为同 step 内 resume 会导致 token 乱序。）

### 3.9 `_update_after_schedule`：为什么调度后再推进 `num_computed_tokens`

```python
# vllm/v1/core/sched/scheduler.py:1515-1541
def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
    # Advance the number of computed tokens for the request AFTER
    # the request is scheduled.
    # 1. The scheduler_output of the current step has to include the
    #    original number of scheduled tokens to determine input IDs.
    # 2. Advance the number of computed tokens here allowing us to
    #    schedule the prefill request again immediately in the next
    #    scheduling step.
    # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
    #    computed tokens will be adjusted in update_from_output.
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens
    for req_id, num_scheduled_token in num_scheduled_tokens.items():
        request = self.requests[req_id]
        request.num_computed_tokens += num_scheduled_token
        request.num_in_flight_tokens += num_scheduled_token
        ...
        request.is_prefill_chunk = request.num_computed_tokens < (
            request.num_tokens + request.num_output_placeholders
        )
        scheduler_output.has_structured_output_requests |= (
            request.use_structured_output and not request.is_prefill_chunk
        )
        if not request.is_prefill_chunk:
            self._inflight_prefills.discard(request)
```

这个顺序很讲究：`num_scheduled_tokens` 里存的是**本 step 要算的 token 数**（用于构造 input_ids），而 `num_computed_tokens` 立刻加上去，使得下一个 step 的 `num_new_tokens = num_tokens - num_computed_tokens` 天然就是"还剩多少"。若 spec token 被拒绝，在 `update_from_output` 里回滚（[scheduler.py:1980-1984](../vllm/v1/core/sched/scheduler.py#L1980)）。

`is_prefill_chunk` 是一个布尔缓存，供下一 step 的 `defer_prefills` 判断和 structured output 的 bitmask 计算使用。

### 3.10 `update_from_output`：回收、停止检测、释放

主循环（[scheduler.py:1933](../vllm/v1/core/sched/scheduler.py#L1933)）按 `num_scheduled_tokens` 遍历：

```python
# vllm/v1/core/sched/scheduler.py:1931-1963
stopped_running_reqs: set[Request] = set()
stopped_preempted_reqs: set[Request] = set()
for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
    assert num_tokens_scheduled > 0
    request = self.requests.get(req_id)
    output_is_stale = False
    if request is not None:
        request.num_in_flight_tokens -= num_tokens_scheduled
        # Drain any stale share (see _preempt_request) in lockstep.
        if request.num_stale_output_tokens > 0:
            output_is_stale = True
            request.num_stale_output_tokens -= num_tokens_scheduled
            assert request.num_stale_output_tokens >= 0
    if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
        continue
    if request is None or request.is_finished():
        # The request is already finished. This can happen if the
        # request is aborted while the model is executing it (e.g.,
        # in pipeline parallelism or in async scheduling).
        continue
```

spec decode 的拒绝回滚：

```python
# vllm/v1/core/sched/scheduler.py:1969-1984
if scheduled_spec_token_ids and (
    generated_token_ids or self.num_sampled_tokens_per_step == 0
):
    num_draft_tokens = len(scheduled_spec_token_ids)
    num_sampled = self.num_sampled_tokens_per_step
    num_accepted = max(len(generated_token_ids) - num_sampled, 0)
    num_rejected = num_draft_tokens - num_accepted
    # Rejections roll back num_computed_tokens (and, under async
    # scheduling, num_output_placeholders, which covers the spec
    # tokens). A stale rejection count predates the preemption
    # rollback and must not apply.
    if not output_is_stale:
        if request.num_computed_tokens > 0:
            request.num_computed_tokens -= num_rejected
        if request.num_output_placeholders > 0:
            request.num_output_placeholders -= num_rejected
```

停止检测与释放：

```python
# vllm/v1/core/sched/scheduler.py:2022-2033
# Check for stop and update request status.
if new_token_ids:
    new_token_ids, stopped = self._update_request_with_output(
        request, new_token_ids, is_stale=output_is_stale
    )
elif request.pooling_params and pooler_output is not None:
    # Pooling stops as soon as there is output.
    request.status = RequestStatus.FINISHED_STOPPED
    stopped = True
```

```python
# vllm/v1/core/sched/scheduler.py:2122-2133
if stopped:
    # Capture finish_reason BEFORE _handle_stopped_request, which may
    # reset the status to WAITING for streaming requests that continue.
    finish_reason = request.get_finished_reason()
    finished = self._handle_stopped_request(request)
    if finished:
        kv_transfer_params, ec_transfer_params = self._free_request(request)

    if status_before_stop == RequestStatus.RUNNING:
        stopped_running_reqs.add(request)
    else:
        stopped_preempted_reqs.add(request)
```

`_free_request`（[scheduler.py:2566](../vllm/v1/core/sched/scheduler.py#L2566)）→ `_free_blocks`（[scheduler.py:2595](../vllm/v1/core/sched/scheduler.py#L2595)）→ `_free_request_blocks`（[scheduler.py:2608](../vllm/v1/core/sched/scheduler.py#L2608)）最终调 `kv_cache_manager.free(request)`。注意 `_free_blocks` 还会 `del self.requests[request_id]`，这是请求从调度器彻底消失的时刻。

`_free_request_blocks` 里的 `defer_block_free` 分支值得留意：

```python
# vllm/v1/core/sched/scheduler.py:2608-2621
def _free_request_blocks(self, request: Request):
    """Free the request's KV blocks, deferring the return to the block
    pool when an in-flight GPU step may still write them.
    """
    if not self.defer_block_free or (
        # Last scheduled step already processed: no in-flight write remains
        request.last_sched_seq <= self.processed_step_seq
    ):
        self.kv_cache_manager.free(request)
        return
    blocks = self.kv_cache_manager.pop_blocks_for_free(request)
    if blocks:
        self.deferred_frees.append((self.sched_step_seq, blocks))
```

这是异步调度 / PP 场景下的正确性补丁：GPU 上可能还有一个 step 正在写这些 block，此时归还给 pool 会被别的请求立刻复用，读到脏数据。所以用一个 `(fence_seq, blocks)` 的 FIFO 延迟到 `processed_step_seq >= fence_seq` 再真正释放（[scheduler.py:2634](../vllm/v1/core/sched/scheduler.py#L2634) 的 `_drain_deferred_frees`）。

### 3.11 停止与终止：V1 里没有 `StopChecker`

**这一点必须说明白**：任务描述里提到的 `StopChecker` 在当前代码库中不存在。我在 `vllm/sequence.py` 和 `vllm/v1/` 全目录下搜索 `StopChecker` 均无匹配（V0 遗留已移除）。V1 的停止判定被拆成了**两处、两个进程**：

**(a) Token 级停止 —— 在 scheduler 进程内，同步判定**

```python
# vllm/v1/core/sched/utils.py:94-136
def check_stop(request: Request, max_model_len: int) -> bool:
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True

    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    # Note(arpera):
    # Order of checks is important for min_tokens ...
    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    repetition_detection = sampling_params.repetition_detection
    if repetition_detection is not None and (
        check_sequence_repetition(...)
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False
```

- **`max_tokens`**：`request.max_tokens` 在 `Request.__init__` 里从 `sampling_params.max_tokens` 拷贝（[request.py:113](../vllm/v1/request.py#L113)），pooling 模型强制为 1。
- **`ignore_eos`**：不是在这里判断的。`SamplingParams.update_from_generation_config` 里 `if not self.ignore_eos: self._eos_token_id = eos_token_id`（[vllm/sampling_params.py:692](../vllm/sampling_params.py#L692)），即 `ignore_eos=True` 时 `_eos_token_id` 保持 `None`，于是 `last_token_id == sampling_params.eos_token_id` 永远为 `False`，EOS 自然不触发停止。这是一个"把开关编码进数据"的巧妙做法。
- **`min_tokens` 的顺序**被注释特别强调（PR 47489 / 49521 / 51299）：`min_tokens` 检查必须在 stop_token_ids 之后、repetition 之前。

**(b) stop string —— 在前端（API server / output processor）进程内，异步判定**

```python
# vllm/v1/engine/output_processor.py:695-701
# 2) Detokenize the token ids into text and perform stop checks.
stop_string = req_state.detokenizer.update(
    new_token_ids, finish_reason == FinishReason.STOP
)
if stop_string:
    finish_reason = FinishReason.STOP
    stop_reason = stop_string
```

实际匹配在 [vllm/v1/engine/detokenizer.py:96](../vllm/v1/engine/detokenizer.py#L96) 的 `BaseIncrementalDetokenizer.update`，最终调 [detokenizer.py:310](../vllm/v1/engine/detokenizer.py#L310) 的 `check_stop_strings`。它只扫描"新增字符"窗口（`new_char_count`），并在多个 stop string 同时命中时选**文本中最早完成**的那个，保证结果与"逐 token 追加"一致（spec decode 一步多个 token 时这一点很重要）。

因为 stop string 是在前端判定的，它**不能直接让 engine 停**，而是由前端回调 `scheduler.finish_requests(req_id, FINISHED_STOPPED)`（见 `SchedulerInterface.finish_requests` 的文档，[interface.py:153-156](../vllm/v1/core/sched/interface.py#L153)）。这意味着命中 stop string 的请求会在 engine 里**多生成一个 token** 才被终止，然后被截断。

**logprobs 与 prompt_logprobs**：

```python
# vllm/v1/core/sched/scheduler.py:2136-2148
if (
    request.sampling_params is not None
    and request.sampling_params.num_logprobs is not None
    and logprobs
):
    new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
```

`prompt_logprobs` 在 [scheduler.py:2154](../vllm/v1/core/sched/scheduler.py#L2154) 获取，并且有一个不变式断言：partial prefill 的 step 不会产出 prompt logprobs（[scheduler.py:2181-2183](../vllm/v1/core/sched/scheduler.py#L2181)）。

### 3.12 与 KVCacheManager / EncoderCacheManager 的协同

**`allocate_slots` 是唯一的准入闸门。** 签名与返回（[kv_cache_manager.py:360](../vllm/v1/core/kv_cache_manager.py#L360)）：

```python
def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: KVCacheBlocks | None = None,
    num_lookahead_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
    full_sequence_must_fit: bool = False,
    reserved_blocks: int = 0,
    has_scheduled_reqs: bool = True,
) -> KVCacheBlocks | None:
```

返回 `KVCacheBlocks`（每组 KV cache group 一个 block id 列表）或 `None`。失败路径有三条：

```python
# vllm/v1/core/kv_cache_manager.py:479-486
watermark_blocks = 0
# The watermark is applied to waiting/preempted requests only, and only
# when there's at least one request already scheduled.
if has_scheduled_reqs and request.status in (
    RequestStatus.WAITING,
    RequestStatus.PREEMPTED,
):
    watermark_blocks = self.watermark_blocks
```

```python
# vllm/v1/core/kv_cache_manager.py:488-504
if full_sequence_must_fit:
    # First check and fail if the full request sequence won't fit.
    full_num_tokens = min(request.num_tokens, self.max_model_len)
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(...)
    required_blocks = num_blocks_to_allocate + watermark_blocks
    if required_blocks > self.block_pool.get_num_free_blocks():
        return None
```

```python
# vllm/v1/core/kv_cache_manager.py:537-543
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks:
    # Cannot allocate new blocks
    return None
```

三个参数都对应一个调度策略旋钮：

- `watermark`（[config/scheduler.py:178](../vllm/config/scheduler.py#L178)，默认 0.0）：给 waiting/preempted 准入保留的空闲块比例。**只对 WAITING/PREEMPTED 生效**，已经 running 的请求继续扩块不受限制——这是为了防止 running 请求自己把自己饿死。默认 0 意味着关闭；在 KV cache 紧张、抢占频繁的场景下调到 0.01~0.05 可以显著降低抢占率（代价是可用显存变少）。
- `full_sequence_must_fit`（来自 `scheduler_reserve_full_isl`，[config/scheduler.py:172](../vllm/config/scheduler.py#L172)，默认 True）：准入时检查**整个序列**能否放下，而不只是第一个 chunk。这是防止 chunked prefill 过度准入导致 KV cache thrashing 的关键——否则一个 100K 的请求只要第一 chunk 能放下就进来了，跑到一半必然触发抢占。
- `reserved_blocks`：给其他 in-flight prefill 预留的块数，用于异步 KV 加载的准入门控，避免死锁。

**EncoderCacheManager**（[encoder_cache_manager.py:19](../vllm/v1/core/encoder_cache_manager.py#L19)）是第二道闸门。调度器通过 `_try_schedule_encoder_inputs`（[scheduler.py:1667](../vllm/v1/core/sched/scheduler.py#L1667)）决定本 step 编码哪些多模态 item：

```python
# vllm/v1/core/sched/scheduler.py:1757-1792
# If no encoder input chunking is allowed, we do not want to
# partially schedule a multimodal item. ...
if (
    self.scheduler_config.disable_chunked_mm_input
    and num_computed_tokens < start_pos
    and (num_computed_tokens + num_new_tokens)
    < (start_pos + num_encoder_tokens)
):
    num_new_tokens = max(
        0, start_pos - (num_computed_tokens + shift_computed_tokens)
    )
    break
if not self.encoder_cache_manager.can_allocate(
    request, i, encoder_compute_budget, num_embeds_to_schedule
):
    # The encoder cache is full or the encoder budget is exhausted.
    # NOTE(woosuk): We assume that the encoder input tokens should
    # be processed altogether, as the encoder usually uses
    # bidirectional attention.
    if num_computed_tokens + shift_computed_tokens < start_pos:
        # We only schedule the decoder tokens just before the
        # encoder input.
        num_new_tokens = start_pos - (
            num_computed_tokens + shift_computed_tokens
        )
    else:
        # Because of prefix caching, num_computed_tokens is greater
        # than start_pos even though its encoder input is not
        # available. In this case, we can't schedule any token for
        # the request in this step.
        num_new_tokens = 0
    break
```

关键约束：**一个多模态 item 必须整体编码**（因为 vision encoder 通常做双向注意力，切一半语义就错了）。所以 encoder 预算不足时，调度器会把 `num_new_tokens` **回退到该 item 的起点之前**，而不是切穿它。若因为 prefix caching 导致 `num_computed_tokens` 已经越过 `start_pos` 但 encoder 输出不在缓存里，则本 step 该请求一个 token 都调度不了（`num_new_tokens = 0`）。

`max_num_encoder_input_tokens` 不是独立配置项：

```python
# vllm/config/scheduler.py:284-285
self.max_num_encoder_input_tokens = self.max_num_batched_tokens
self.encoder_cache_size = self.max_num_batched_tokens
```

但如果最大的多模态 embedding 超过这个值，会被 `MultiModalBudget` 覆盖（[scheduler.py:247-250](../vllm/v1/core/sched/scheduler.py#L247)）。

`allocate_slots` 失败时，调度器必须撤销 encoder cache 的触碰：

```python
# vllm/v1/core/sched/scheduler.py:1166-1173
if new_blocks is None:
    # The request cannot be scheduled.

    # NOTE: we need to untouch the request from the encode cache
    # manager
    if request.has_encoder_inputs:
        self.encoder_cache_manager.free(request)
    break
```

### 3.13 异步调度 `AsyncScheduler`

**动机**。同步调度下，engine core 的 busy loop 是：

```
[CPU: schedule()] -> [GPU: forward] -> [CPU: update_from_output()] -> [CPU: schedule()] -> ...
```

小模型、低 batch、低 `max_num_batched_tokens` 时，GPU forward 只有几 ms，而 Python 的 schedule + update 可能 1~2 ms，GPU 利用率直接掉 20%+。

**实现机制**。`AsyncScheduler`（[async_scheduler.py:12](../vllm/v1/core/sched/async_scheduler.py#L12)）只 override 了两个方法，思路是"**用预测状态提前跑**"：

```python
# vllm/v1/core/sched/async_scheduler.py:19-49
def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
    super()._update_after_schedule(scheduler_output)
    spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
    self._spec_token_placeholders = [
        -1
    ] * scheduler_output.num_spec_tokens_to_schedule
    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests[req_id]
        if request.is_prefill_chunk:
            continue

        scheduler_output.pending_structured_output_tokens |= (
            request.use_structured_output and request.num_output_placeholders > 0
        )
        # The request will generate num_sampled_tokens_per_step new tokens
        # plus num_spec_tokens in this scheduling step. ...
        cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
        request.num_output_placeholders += (
            self.num_sampled_tokens_per_step + cur_num_spec_tokens
        )
        # Add placeholders for the new draft/spec tokens.
        # We will update the actual spec token ids in the worker process.
        request.spec_token_ids = self._spec_token_placeholders

        if self.use_v2_model_runner:
            request.next_decode_eligible_step = self.current_step + self.pp_size
```

调度完第 N 步后，立刻给每个请求加上"我预测它下一步会多出 `1 + K` 个 token"的占位符（`num_output_placeholders`）。于是第 N+1 步的 `schedule()` 在 GPU 还在跑第 N 步时就能基于这个乐观状态做决策——block 预留、`num_tokens_with_spec` 计算、停止判断全部按预测值走。GPU 回来后再 correction：

```python
# vllm/v1/core/sched/async_scheduler.py:51-70
def _update_request_with_output(
    self, request: Request, new_token_ids: list[int], is_stale: bool = False
) -> tuple[list[int], bool]:
    status_before_update = request.status
    new_token_ids, stopped = super()._update_request_with_output(
        request, new_token_ids
    )

    # Placeholders were zeroed at preemption; a stale delivery must not
    # decrement them (it would underflow).
    if not is_stale:
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

    # Cache the new tokens. Preempted requests should be skipped.
    if status_before_update == RequestStatus.RUNNING:
        self.kv_cache_manager.cache_blocks(
            request, request.num_computed_tokens - request.num_output_placeholders
        )
    return new_token_ids, stopped
```

即：真实 token 数回来后扣掉占位符；同时只有 `RUNNING` 状态的请求才把新 token 提交进 prefix cache（被抢占的请求状态是 `PREEMPTED`，它的输出要丢弃，不能污染缓存）。

**调度器类的选择**：

```python
# vllm/config/scheduler.py:212-220
def get_scheduler_cls(self) -> type["SchedulerInterface"]:
    if self.scheduler_cls is None:
        if self.async_scheduling:
            from vllm.v1.core.sched.async_scheduler import AsyncScheduler

            return AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler

        return Scheduler
```

**限制**（在 `VllmConfig.__post_init__` 里强制，[vllm/config/vllm.py:1270-1350](../vllm/config/vllm.py#L1270)）：

- 显式开启时若与 spec decode 冲突（非 EAGLE / MTP / draft_model / ngram_gpu / dspark），直接 `raise ValueError`。
- `disable_padded_drafter_batch=True` 不兼容。
- executor backend 不支持就报错（CPU 平台直接强制关闭，[vllm/platforms/cpu.py:269](../vllm/platforms/cpu.py#L269)）。
- ROCm DeepEP high-throughput DBO 不兼容。
- 默认（`None`）时：pooling 模型关闭；非上述 spec decode 方法的关闭；executor 不支持的关闭；否则**打开**（[vllm/config/vllm.py:1349-1350](../vllm/config/vllm.py#L1349)）。
- 开启后 `max_concurrent_batches` 至少为 2（[vllm/config/vllm.py:577-588](../vllm/config/vllm.py#L577)），即允许 2 个 batch 同时在飞；`max_in_flight_tokens = max_concurrent_batches * max_num_batched_tokens`（[vllm/config/vllm.py:590](../vllm/config/vllm.py#L590)）。
- 与 spec decode 同开时会强制关闭 cascade attention（[vllm/config/vllm.py:1365-1375](../vllm/config/vllm.py#L1365)）。

**engine core 侧的双 batch 队列**：

```python
# vllm/v1/engine/core.py:630-645
def step_with_batch_queue(
    self,
) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
    """Schedule and execute batches with the batch queue.
    Note that if nothing to output in this step, None is returned.

    The execution flow is as follows:
    1. Try to schedule a new batch if the batch queue is not full.
    If a new batch is scheduled, directly return an empty engine core
    output. In other words, fulfilling the batch queue has a higher priority
    than getting model outputs.
    2. If there is no new scheduled batch, meaning that the batch queue
    is full or no other requests can be scheduled, we block until the first
    batch in the job queue is finished.
    3. Update the scheduler from the output.
    """
```

策略是"填满 batch queue 优先于取输出"：只要队列没满就继续 schedule + 非阻塞 execute；队列满了才阻塞等最老的那个 batch 返回。这是让调度和 GPU 真正 overlap 的机制。

### 3.14 调度相关 metrics

`make_stats()`（[scheduler.py:2762](../vllm/v1/core/sched/scheduler.py#L2762)）每个 step 生成 `SchedulerStats`：

```python
# vllm/v1/core/sched/scheduler.py:2786-2798
return SchedulerStats(
    num_running_reqs=len(self.running),
    num_waiting_reqs=len(self.waiting),
    num_skipped_waiting_reqs=len(self.skipped_waiting),
    kv_cache_usage=self.kv_cache_manager.usage,
    prefix_cache_stats=prefix_cache_stats,
    connector_prefix_cache_stats=connector_prefix_cache_stats,
    kv_cache_eviction_events=eviction_events,
    spec_decoding_stats=spec_stats,
    kv_connector_stats=connector_stats_payload,
    cudagraph_stats=cudagraph_stats,
    perf_stats=perf_stats,
)
```

`SchedulerStats` 定义在 [vllm/v1/metrics/stats.py:186](../vllm/v1/metrics/stats.py#L186)，`kv_cache_usage` 来自 `KVCacheManager.usage`（[kv_cache_manager.py:210](../vllm/v1/core/kv_cache_manager.py#L210)，即 `block_pool.get_usage()`）。

Prometheus 暴露（[vllm/v1/metrics/loggers.py:1118-1133](../vllm/v1/metrics/loggers.py#L1118)）：

| 指标 | 含义 | 来源 |
|---|---|---|
| `vllm:num_requests_running` | `num_running_reqs` | `len(self.running)` |
| `vllm:num_requests_waiting` | `num_waiting_reqs + num_skipped_waiting_reqs` | waiting + skipped |
| `vllm:num_requests_waiting_by_reason{reason=capacity}` | `num_waiting_reqs` | 真正排队等容量 |
| `vllm:num_requests_waiting_by_reason{reason=deferred}` | `num_skipped_waiting_reqs` | 被瞬时约束推迟 |
| `vllm:kv_cache_usage_perc` | `kv_cache_usage` | block pool 使用率 |
| `vllm:num_preemptions` | 累计抢占次数 | counter |
| `vllm:request_num_preemptions` | 单请求被抢占次数直方图（buckets 1,2,3,4,5,10,20） | `Request.num_preemptions` |

调度器侧还有两个给外部查询的方法：`get_request_counts()`（[scheduler.py:2467](../vllm/v1/core/sched/scheduler.py#L2467)，返回 `(len(running), len(waiting)+len(skipped_waiting))`）和 `get_kv_cache_usage()`（[scheduler.py:2471](../vllm/v1/core/sched/scheduler.py#L2471)）。

---

## 4. 关键数据结构

### 4.1 `Request`（[vllm/v1/request.py:59](../vllm/v1/request.py#L59)）

| 字段 | 类型 | 含义 | 定义处 |
|---|---|---|---|
| `request_id` | `str` | 请求唯一 id | [request.py:82](../vllm/v1/request.py#L82) |
| `status` | `RequestStatus` | 状态机当前状态，初始 `WAITING` | [request.py:98](../vllm/v1/request.py#L98) |
| `num_computed_tokens` | `int` | **已算过的 token 数**（含 prefix cache 命中 + 外部 KV）。调度器的核心游标 | [request.py:182](../vllm/v1/request.py#L182) |
| `num_cached_tokens`（PrefillStats） | `int` | 本次 prefill 中"没实际算"的 token 数 = local + external cached | [metrics/stats.py:275](../vllm/v1/metrics/stats.py#L275) |
| `num_output_tokens` | `property` | `len(self._output_token_ids)`，**不含**占位符 | [request.py:296](../vllm/v1/request.py#L296) |
| `num_tokens` | `property` | `len(self._all_token_ids)` = prompt + output | [request.py:288](../vllm/v1/request.py#L288) |
| `num_tokens_with_spec` | `property` | `num_tokens + len(spec_token_ids)`，调度追赶目标 | [request.py:292](../vllm/v1/request.py#L292) |
| `num_output_placeholders` | `int` | 异步调度下的乐观未回 token 数；同步调度恒为 0 | [request.py:160](../vllm/v1/request.py#L160) |
| `num_in_flight_tokens` | `int` | 已调度但输出未处理的 token 数 | [request.py:171](../vllm/v1/request.py#L171) |
| `num_stale_output_tokens` | `int` | 被抢占时在飞的输出 token 数，回来后按 step 排空 | [request.py:163](../vllm/v1/request.py#L163) |
| `is_prefill_chunk` | `bool` | 本请求当前是否处于"非最后一个 prefill chunk" | [request.py:198](../vllm/v1/request.py#L198) |
| `num_preemptions` | `int` | 被抢占次数，进入 `vllm:request_num_preemptions` | [request.py:210](../vllm/v1/request.py#L210) |
| `priority` | `int` | 越小越优先；`__lt__` 的第一关键字 | [request.py:350](../vllm/v1/request.py#L350) |
| `arrival_time` | `float` | 到达时间，`__lt__` 的第二关键字 | [request.py:96](../vllm/v1/request.py#L96) |
| `next_decode_eligible_step` | `int` | V2+PP+async 下强制同请求 decode 间隔 `pp_size` 步 | [request.py:175](../vllm/v1/request.py#L175) |
| `max_tokens` | `int` | 来自 `sampling_params.max_tokens`（pooling 为 1） | [request.py:109-113](../vllm/v1/request.py#L109) |

### 4.2 `SchedulerOutput`（[vllm/v1/core/sched/output.py:227](../vllm/v1/core/sched/output.py#L227)）

| 字段 | 类型 | 含义 |
|---|---|---|
| `scheduled_new_reqs` | `list[NewRequestData]` | 本 step 首次调度的请求；worker 会把它们的 prompt 数据缓存下来 |
| `scheduled_cached_reqs` | `CachedRequestData` | 之前调度过的请求，只发 diff（新 block ids / token ids / num_computed_tokens） |
| `num_scheduled_tokens` | `dict[str, int]` | **核心**：每个请求本 step 算几个 token |
| `total_num_scheduled_tokens` | `int` | `sum(num_scheduled_tokens.values())` |
| `scheduled_spec_decode_tokens` | `dict[str, list[int]]` | 每请求的草稿 token |
| `scheduled_encoder_inputs` | `dict[str, list[int]]` | 每请求本 step 要编码的 mm item 下标 |
| `num_common_prefix_blocks` | `list[int]` | 每组 KV cache 的公共前缀块数，供 cascade attention |
| `finished_req_ids` | `set[str]` | 上一 step 到本 step 之间结束的请求，通知 worker 清理 |
| `preempted_req_ids` | `set[str] \| None` | 本 step 被抢占的请求（注释：Only used for v2 model runner） |
| `free_encoder_mm_hashes` | `list[str]` | 本 step 要从 encoder cache 驱逐的 mm hash |
| `new_block_ids_to_zero` | `list[int] \| None` | 新分配的 block，worker 在使用前要清零 |

`CachedRequestData`（[output.py:133](../vllm/v1/core/sched/output.py#L133)）是"diff"传输的关键：`new_block_ids` 里 `None` 表示该请求本 step 没有新 block，worker 直接复用之前的 block table。

### 4.3 `SchedulerConfig` 调度相关字段（[vllm/config/scheduler.py](../vllm/config/scheduler.py)）

| 字段 | 默认 | 定义处 |
|---|---|---|
| `max_num_batched_tokens` | `2048` | [scheduler.py:49](../vllm/config/scheduler.py#L49) |
| `max_num_scheduled_tokens` | `None`（回退到上者） | [scheduler.py:56](../vllm/config/scheduler.py#L56) |
| `max_num_seqs` | `128` | [scheduler.py:63](../vllm/config/scheduler.py#L63) |
| `max_num_queued_reqs` | `None` | [scheduler.py:74](../vllm/config/scheduler.py#L74)（API server 侧 503 阀门） |
| `long_prefill_token_threshold` | `0` | [scheduler.py:70](../vllm/config/scheduler.py#L70) |
| `enable_chunked_prefill` | `True` | [scheduler.py:116](../vllm/config/scheduler.py#L116) |
| `disable_chunked_mm_input` | `False` | [scheduler.py:149](../vllm/config/scheduler.py#L149) |
| `policy` | `"fcfs"` | [scheduler.py:141](../vllm/config/scheduler.py#L141) |
| `scheduler_reserve_full_isl` | `True` | [scheduler.py:172](../vllm/config/scheduler.py#L172) |
| `watermark` | `0.0` | [scheduler.py:178](../vllm/config/scheduler.py#L178) |
| `prefill_schedule_interval` | `1` | [scheduler.py:185](../vllm/config/scheduler.py#L185)（DP 对齐） |
| `async_scheduling` | `None`（自动） | [scheduler.py:190](../vllm/config/scheduler.py#L190) |
| `stream_interval` | `1` | [scheduler.py:195](../vllm/config/scheduler.py#L195) |

---

## 5. 收益与代价

### 5.1 收益

| 机制 | 收益 | 代价 / 限制 |
|---|---|---|
| Continuous batching | 消除队头阻塞与 batch 空转；请求随到随处理、随完随退出。GPU 利用率从静态 batch 的 30~50% 提升到接近饱和 | 每个 step 都要重跑一遍调度决策，Python 开销变成新瓶颈（→ AsyncScheduler） |
| Chunked prefill | 把长 prompt 的 activation 峰值钉在 `max_num_batched_tokens` 上；把 decode 的 TPOT 抖动限制在一个 chunk 的时长内 | 长 prompt 的 TTFT 变长（要跨多个 step）；chunk 边界会切断 prefix cache 的块对齐（部分块无法及时入缓存） |
| `scheduler_reserve_full_isl` | 防止过度准入导致 KV thrashing | 准入更保守，长请求排队更久 |
| `watermark` | 减少抢占率 | 减少可用 KV cache |
| 抢占 | 在 KV 打满时仍能保证系统不死锁、有进展 | 被抢占请求要重算（若无 prefix 命中，代价 = 已算 token 数的全部 FLOPs） |
| AsyncScheduler | 消除 Python 调度造成的 GPU 气泡，小模型 / 低延迟场景收益显著 | 状态变成"乐观"的，需要占位符 + correction + 延迟释放 block 三套补丁；兼容性限制多 |

### 5.2 失效场景 / 需要注意的坑

1. **`max_num_batched_tokens` 太小**：decode 吞吐上不去（每 step 的 token 数被人为压低，GPU 并行度不足）。
2. **`max_num_batched_tokens` 太大**：长 prefill 独占时间变长，TPOT 抖动变大，且 activation 峰值上升可能 OOM。经验值：吞吐优先 8192+；延迟优先 2048~4096 配合 `long_prefill_token_threshold`。
3. **`max_num_seqs` 打满但 KV 没满**：此时抢占不会发生（抢占只在 `allocate_slots` 返回 None 时触发），新请求只会在 waiting 里排队。所以 `max_num_seqs` 太小会导致 KV cache 用不满 —— 监控 `vllm:kv_cache_usage_perc` 长期远低于 1.0 而 `vllm:num_requests_waiting` 长期很高，就是这个症状。
4. **`enable_chunked_prefill=False` + `max_num_batched_tokens < max_model_len`**：直接启动报错（[config/scheduler.py:295](../vllm/config/scheduler.py#L295)）。
5. **抢占风暴**：KV cache 接近满且 `scheduler_reserve_full_isl=False` 时，长请求被准入 → 跑到一半抢占别人 → 别人重算又占块 → 再次抢占。`num_preemptions` 直方图出现大量 >5 的样本就是信号。
6. **异步调度下的 stale output**：被抢占请求的在飞 token 仍会投递给客户端（`drop_stale_output=False` 时），表现为"请求被抢占后输出出现重复/回退片段"。这是设计权衡（不丢是为了不扰动 spec decode 接受率统计）。

---

## 6. 面试高频问题

**Q1：Continuous batching 和 static/request-level batching 的本质区别是什么？**
A：Static batching 在**请求级**组批，一批请求必须一起结束才释放资源；continuous batching 在**迭代级（每个 forward step）** 组批，每个 step 重新决定哪些请求参与、各算几个 token。vLLM V1 的实现里没有"prefill 阶段/decode 阶段"的概念，只有一个不变量：让每个请求的 `num_computed_tokens` 追上 `num_tokens_with_spec`。见 [scheduler.py:565](../vllm/v1/core/sched/scheduler.py#L565) 的 `NOTE(woosuk)`。请求结束立即从 `self.running` 移除并被 `_free_request` 释放 block（[scheduler.py:2566](../vllm/v1/core/sched/scheduler.py#L2566)），空出的 slot 下一个 step 就能给新请求。

**Q2：`Scheduler.schedule()` 的两个阶段分别做什么？为什么要这个顺序？**
A：阶段 A 遍历 `self.running`（[scheduler.py:612](../vllm/v1/core/sched/scheduler.py#L612)），处理 decode 和未完成的 prefill chunk；阶段 B 遍历 `waiting`/`skipped_waiting`（[scheduler.py:847](../vllm/v1/core/sched/scheduler.py#L847)），准入新请求与被抢占请求。顺序体现了 **decode 优先**：已 in-flight 的请求先拿到 token budget，剩余 budget 才给新 prefill。同时阶段 B 有 `if not preempted_reqs` 的前置条件（[scheduler.py:848](../vllm/v1/core/sched/scheduler.py#L848)）——本 step 发生过抢占就不再准入新请求，避免抢占-准入震荡。

**Q3：chunked prefill 在 V1 里是怎么实现的？需要额外开关吗？**
A：核心就一行 `num_new_tokens = min(num_new_tokens, request_token_budget)`（[scheduler.py:1073](../vllm/v1/core/sched/scheduler.py#L1073)）。"需要量"是 `request.num_tokens - num_computed_tokens`，被剩余 budget 截断后剩下的部分下个 step 再来。开关是 `enable_chunked_prefill`（默认 True，[config/scheduler.py:116](../vllm/config/scheduler.py#L116)）；关掉后若请求整段塞不进剩余 budget 就 `break`（[scheduler.py:1065-1071](../vllm/v1/core/sched/scheduler.py#L1065)）。`long_prefill_token_threshold` 是额外的单请求每步上限（默认 0 = 不限），用于提高长 prompt 之间的公平性。

**Q4：`long_prefill_token_threshold` 和 `max_num_batched_tokens` 有什么区别？什么时候该调它？**
A：`max_num_batched_tokens` 是**全局**单步 token 上限；`long_prefill_token_threshold` 是**单请求**单步上限（[scheduler.py:1059-1061](../vllm/v1/core/sched/scheduler.py#L1059)）。当 `max_num_batched_tokens` 很大（如 8192）时，一个 8K prompt 会独占整个 step，其他长 prefill 全部饿死。设 `long_prefill_token_threshold=2048` 后，一个 step 可以并行推进 4 个长 prefill，提升长请求的公平性与整体 TTFT 的 P99，代价是单个超长 prompt 的 TTFT 变长。

**Q5：V1 支持哪些抢占策略？SWAP 还在吗？**
A：**只支持 RECOMPUTE**。全仓库 `vllm/v1/` 下不存在 `PreemptionMode` / `swap_out` / `SWAP` 语义（已实际搜索确认），SWAP 是 V0 遗留概念。`_preempt_request`（[scheduler.py:1472](../vllm/v1/core/sched/scheduler.py#L1472)）的做法是：`_free_request_blocks` 释放全部 block、`num_computed_tokens = 0`、`status = PREEMPTED`、`waiting.prepend_request(request)` 插回队首。注意"重算"不一定是真的全量重算：如果 prefix cache 里对应 hash 块还没被复用，重新准入时 `get_computed_blocks` 会命中，只需补算尾部。

**Q6：抢占的触发条件是什么？谁会成为 victim？**
A：唯一触发点是 `kv_cache_manager.allocate_slots()` 返回 `None`（[scheduler.py:731](../vllm/v1/core/sched/scheduler.py#L731) / [kv_cache_manager.py:541](../vllm/v1/core/kv_cache_manager.py#L541)）。`max_num_seqs` 打满**不会**触发抢占，只是停止准入（[scheduler.py:857](../vllm/v1/core/sched/scheduler.py#L857)）。victim 选择：FCFS 策略下是 `self.running.pop()`（最晚进入 running 的，[scheduler.py:778](../vllm/v1/core/sched/scheduler.py#L778)）；PRIORITY 策略下是 `max(running, key=lambda r: (r.priority, r.arrival_time))`（[scheduler.py:744](../vllm/v1/core/sched/scheduler.py#L744)）。若 victim 本 step 已被调度过，会回滚它的 token/encoder 预算与 block（[scheduler.py:758-776](../vllm/v1/core/sched/scheduler.py#L758)）。

**Q7：`max_num_seqs` 到底限制什么？和 `max_num_batched_tokens` 谁先被打满？**
A：`max_num_seqs` → `self.max_num_running_reqs`，只在 waiting 准入时检查 `len(self.running) + num_waiting_for_streaming_input >= max_num_running_reqs` 就 `break`（[scheduler.py:856-858](../vllm/v1/core/sched/scheduler.py#L856)）。它限制的是**并发在飞的请求数**，不限制单 step 的 token 数。反过来 `max_num_batched_tokens` 限制 token 数不限制请求数。纯 decode 负载下先打满 `max_num_seqs`（每请求 1 token，128 个请求才 128 token）；长 prompt 负载下先打满 `max_num_batched_tokens`。调参时要两个一起看，并用 `vllm:kv_cache_usage_perc` 判断是否 KV 才是真正的瓶颈。

**Q8：为什么 `_update_after_schedule` 要在 `schedule()` 的最后才推进 `num_computed_tokens`？**
A：源码注释给了三条理由（[scheduler.py:1516-1524](../vllm/v1/core/sched/scheduler.py#L1516)）：(1) `SchedulerOutput.num_scheduled_tokens` 必须保留"本 step 要算几个 token"的原值，worker 靠它切 input_ids；(2) 立刻推进后，下一个 step 的 `num_new_tokens = num_tokens - num_computed_tokens` 天然等于"还剩多少"，prefill 请求可以立刻被再次调度；(3) 若 spec token 被拒，在 `update_from_output` 里回滚（[scheduler.py:1981-1984](../vllm/v1/core/sched/scheduler.py#L1981)）。

**Q9：异步调度（AsyncScheduler）解决什么问题？怎么解决的？**
A：解决"Python 调度开销导致 GPU 空转"。小模型 + 低延迟场景下单步 GPU forward 只有几 ms，Python schedule + update 占 1~2 ms。做法是：调度完第 N 步后，立刻给每个请求加 `num_output_placeholders += num_sampled_tokens_per_step + cur_num_spec_tokens`（[async_scheduler.py:39](../vllm/v1/core/sched/async_scheduler.py#L39)），即"预测它下一步会多出 1+K 个 token"；基于这个乐观状态，第 N+1 步的 `schedule()` 可以在 GPU 跑第 N 步时提前做完。真实输出回来后在 `_update_request_with_output` 里扣掉占位符并做 correction（[async_scheduler.py:62](../vllm/v1/core/sched/async_scheduler.py#L62)）。engine core 侧配合 `step_with_batch_queue`（[core.py:630](../vllm/v1/engine/core.py#L630)），策略是"填满 batch queue 优先于取输出"。

**Q10：异步调度有哪些已知限制？**
A：(1) 与 spec decode 同时使用时，只允许 EAGLE / MTP / draft_model / ngram_gpu / dspark，否则 `raise ValueError`（[vllm/config/vllm.py:1280-1291](../vllm/config/vllm.py#L1280)）；(2) 与 `disable_padded_drafter_batch=True` 不兼容；(3) executor backend 不支持时报错（CPU 平台直接强制关闭，[cpu.py:267](../vllm/platforms/cpu.py#L267)）；(4) ROCm DeepEP HT DBO 不兼容；(5) pooling 模型默认关闭（性能反而变差，[vllm/config/vllm.py:1303-1312](../vllm/config/vllm.py#L1303)）；(6) 与 spec decode 同开会强制关 cascade attention（[vllm/config/vllm.py:1365](../vllm/config/vllm.py#L1365)）。另外它需要 `max_concurrent_batches >= 2`（[vllm/config/vllm.py:577](../vllm/config/vllm.py#L577)）。

**Q11：异步调度下被抢占的请求，它的在飞输出怎么处理？**
A：`_preempt_request` 里 `request.num_stale_output_tokens = request.num_in_flight_tokens`（[scheduler.py:1505](../vllm/v1/core/sched/scheduler.py#L1505)）。这些 token 在后续 step 随 `update_from_output` 回来时会被标记为 `output_is_stale`（[scheduler.py:1940](../vllm/v1/core/sched/scheduler.py#L1940)），**默认仍然投递给客户端**（不能丢，否则扰动 spec decode 接受率统计），但不会改动已经归零的 `num_computed_tokens` / `num_output_placeholders`（[scheduler.py:1980](../vllm/v1/core/sched/scheduler.py#L1980)、[async_scheduler.py:61](../vllm/v1/core/sched/async_scheduler.py#L61)）。若 `drop_stale_output=True`（`reset_prefix_cache` 或需要 KV 交付的 connector 场景），则整段丢弃（[scheduler.py:1957-1959](../vllm/v1/core/sched/scheduler.py#L1957)）。

**Q12：chunked prefill 与 prefix caching 如何交互？`num_external_computed_tokens` 是什么？**
A：请求首次准入时（`num_computed_tokens == 0`）才做 prefix cache 查询（[scheduler.py:911](../vllm/v1/core/sched/scheduler.py#L911)），得到 `num_new_local_computed_tokens`，它直接减少本 step 需要算的 token 数。chunk 边界若不在 block 边界上，最后一个不完整 block 暂时无法入缓存，要等后续 chunk 填满——这也是为什么 `_mamba_block_aligned_split`（[scheduler.py:415](../vllm/v1/core/sched/scheduler.py#L415)）在某些情况下会强制把 chunk 截到 block 对齐位置。`num_external_computed_tokens` 是 KV connector 报告的**外部缓存命中**（远程实例 / 外部存储）token 数（[scheduler.py:929](../vllm/v1/core/sched/scheduler.py#L929)），它与本地命中的仲裁规则是：只有远程命中严格超过本地完整命中（含 sub-block 尾）时才丢弃本地 tail 并改为外部加载（[scheduler.py:943-963](../vllm/v1/core/sched/scheduler.py#L943)）。

**Q13：`waiting` 和 `skipped_waiting` 两个队列有什么区别？为什么要两个？**
A：`waiting` 放的是"资源上还没轮到"的请求；`skipped_waiting` 放的是"本 step 被**瞬时约束**挡住"的请求：结构化输出 grammar 未编译完、远程 KV 未到、streaming 输入未到、LoRA 配额打满、有 stale output 在飞（[scheduler.py:866-903](../vllm/v1/core/sched/scheduler.py#L866)、[scheduler.py:2308](../vllm/v1/core/sched/scheduler.py#L2308) 的 `_is_blocked_waiting_status`）。不分成两个队列的话，while 循环里 `continue` 会死循环，直接丢弃又会饿死请求。消费顺序由 `_select_waiting_queue_for_scheduling`（[scheduler.py:2321](../vllm/v1/core/sched/scheduler.py#L2321)）决定：FCFS 下优先 `skipped_waiting`（它们等得更久）；PRIORITY 下比较两队队首。这两个队列在 metrics 上分别对应 `vllm:num_requests_waiting_by_reason` 的 `capacity` 和 `deferred` 标签（[loggers.py:1127](../vllm/v1/metrics/loggers.py#L1127)）。

**Q14：多模态请求的调度有什么额外限制？**
A：三个。(1) **一个 mm item 必须整体编码**（vision encoder 用双向注意力，切一半语义错），`_try_schedule_encoder_inputs` 在 encoder 预算不足时会把 `num_new_tokens` 回退到该 item 的 `start_pos` 之前（[scheduler.py:1773-1792](../vllm/v1/core/sched/scheduler.py#L1773)）；若因 prefix caching 导致 `num_computed_tokens` 已越过 `start_pos` 但 encoder 输出不在缓存，本 step 一个 token 都调度不了。(2) `disable_chunked_mm_input=True` 时不允许 chunk 边界落在 mm item 中间（[scheduler.py:1760](../vllm/v1/core/sched/scheduler.py#L1760)）。(3) `max_num_encoder_input_tokens` 独立预算，默认等于 `max_num_batched_tokens`（[config/scheduler.py:284](../vllm/config/scheduler.py#L284)），但可被 `MultiModalBudget` 覆盖。另外 `allocate_slots` 失败时要显式 `encoder_cache_manager.free(request)` 撤销触碰（[scheduler.py:1171](../vllm/v1/core/sched/scheduler.py#L1171)）。

**Q15：V1 里停止条件在哪里判定？`StopChecker` 还存在吗？**
A：**不存在 `StopChecker`**——我在 `vllm/sequence.py` 和 `vllm/v1/` 全目录搜索均无匹配，这是 V0 遗留。V1 拆成两处：token 级停止在 scheduler 进程内的 `check_stop()`（[vllm/v1/core/sched/utils.py:94](../vllm/v1/core/sched/utils.py#L94)），判断 EOS / `stop_token_ids` / `max_tokens`+`max_model_len` / `min_tokens` / repetition；stop **string** 在前端进程的 `BaseIncrementalDetokenizer.update`（[detokenizer.py:96](../vllm/v1/engine/detokenizer.py#L96)）与 `check_stop_strings`（[detokenizer.py:310](../vllm/v1/engine/detokenizer.py#L310)）。因为 stop string 在前端判定，engine 并不知道，所以命中后会**多生成一个 token** 才由前端通过 `finish_requests` 终止并截断。`ignore_eos` 的实现是让 `_eos_token_id` 保持 `None`（[sampling_params.py:692](../vllm/sampling_params.py#L692)），于是 `check_stop` 的 EOS 分支永不命中。

**Q16：`scheduler_reserve_full_isl` 和 `watermark` 分别防什么？**
A：`scheduler_reserve_full_isl`（默认 True）在准入时检查**整个序列**能否放进 KV cache（`full_sequence_must_fit=True`，[kv_cache_manager.py:488](../vllm/v1/core/kv_cache_manager.py#L488)），而不只是第一个 chunk。它防的是 chunked prefill 的过度准入：否则一个 100K 请求只要首 chunk 放得下就进来了，跑到一半必然触发抢占，把别人一起拖下水（KV thrashing）。`watermark`（默认 0.0，即关闭）在准入 waiting/preempted 请求时额外要求保留一定比例的空闲块（[kv_cache_manager.py:479-486](../vllm/v1/core/kv_cache_manager.py#L479)），它防的是频繁抢占。注意 watermark **只对 WAITING/PREEMPTED 生效**，已 running 的请求继续扩块不受限——否则 running 请求会把自己饿死。

---

## 7. 延伸阅读

### 7.1 源码文件清单

| 文件 | 内容 |
|---|---|
| [vllm/v1/core/sched/scheduler.py](../vllm/v1/core/sched/scheduler.py) | 调度器主体，`schedule()` / `_preempt_request` / `update_from_output` |
| [vllm/v1/core/sched/async_scheduler.py](../vllm/v1/core/sched/async_scheduler.py) | `AsyncScheduler`，占位符与 correction |
| [vllm/v1/core/sched/interface.py](../vllm/v1/core/sched/interface.py) | `SchedulerInterface` ABC 与 `PauseState` |
| [vllm/v1/core/sched/output.py](../vllm/v1/core/sched/output.py) | `SchedulerOutput` / `NewRequestData` / `CachedRequestData` |
| [vllm/v1/core/sched/request_queue.py](../vllm/v1/core/sched/request_queue.py) | `FCFSRequestQueue` / `PriorityRequestQueue` |
| [vllm/v1/core/sched/utils.py](../vllm/v1/core/sched/utils.py) | `check_stop` / `remove_all` / 重复检测 |
| [vllm/v1/request.py](../vllm/v1/request.py) | `Request` 与 `RequestStatus` 状态机 |
| [vllm/v1/core/kv_cache_manager.py](../vllm/v1/core/kv_cache_manager.py) | `KVCacheManager.allocate_slots` / `get_computed_blocks` / `usage` |
| [vllm/v1/core/encoder_cache_manager.py](../vllm/v1/core/encoder_cache_manager.py) | 多模态 encoder cache 的 allocate/free |
| [vllm/v1/engine/core.py](../vllm/v1/engine/core.py) | `EngineCoreProc.step` / `step_with_batch_queue` |
| [vllm/v1/engine/detokenizer.py](../vllm/v1/engine/detokenizer.py) | stop string 判定（前端进程） |
| [vllm/v1/engine/output_processor.py](../vllm/v1/engine/output_processor.py) | 输出处理与 finish_reason 组装 |
| [vllm/v1/metrics/stats.py](../vllm/v1/metrics/stats.py) | `SchedulerStats` / `PrefillStats` |
| [vllm/v1/metrics/loggers.py](../vllm/v1/metrics/loggers.py) | Prometheus 指标定义与上报 |
| [vllm/config/scheduler.py](../vllm/config/scheduler.py) | `SchedulerConfig` 全部调度相关配置项 |
| [vllm/config/vllm.py](../vllm/config/vllm.py) | `max_concurrent_batches` / `max_in_flight_tokens` / async scheduling 兼容性检查 |

### 7.2 官方链接

- vLLM 官方文档 · Prefix Caching 设计：<https://docs.vllm.ai/en/latest/design/prefix_caching.html>
- vLLM 官方文档 · Hybrid KV Cache Manager：<https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html>
- vLLM 官方文档 · Automatic Prefix Caching：<https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html>
- vLLM 官方文档 · Optimization and Tuning：<https://docs.vllm.ai/en/latest/configuration/optimization.html>
- vLLM 官方文档 · Metrics：<https://docs.vllm.ai/en/latest/design/metrics.html>
- vLLM Blog · vLLM v0.6.0 性能更新（chunked prefill + prefix caching 的官方量化数据）：<https://blog.vllm.ai/2024/09/05/perf-update.html>
- Orca: A Distributed Serving System for Transformer-Based Generative Models（continuous batching 的原始论文）：<https://www.usenix.org/conference/osdi22/presentation/yu>
- SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills：<https://arxiv.org/abs/2308.16369>
