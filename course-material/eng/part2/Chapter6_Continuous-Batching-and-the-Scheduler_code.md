# Chapter 6 Continuous Batching and the Scheduler (Code Implementation)

Requests A and B are generating when C arrives. If A finishes before B, can C use the freed slot immediately? Continuous batching answers this by deciding, between computation rounds, which requests enter, continue, or leave.

This chapter implements a runnable scheduler to accompany the [concept article](./Chapter6_Continuous-Batching-and-the-Scheduler.md). To make request lifecycles easy to observe, the backend generates integers using a fixed rule, without loading a model or allocating a real KV cache. Once the scheduling loop is clear, the same interface can connect to a model forward pass and cache management.

## 1 Learning Objectives

1. Organize requests with waiting and running queues.
2. Implement FIFO admission, prefill priority, completion, and cancellation.
3. Explain dynamic batching from a trace and distinguish request limits, token budgets, and physical memory.

## 2 Watch a Batch Change

Only the Python 3.12 standard library is required. Run from the repository root:

```bash
python3 examples/part2/chapter06/scheduler.py
```

A and B arrive initially; C arrives after the first round. At most two requests can run at once. A, B, and C generate 2, 5, and 2 tokens respectively:

```text
tick=0 prefill batch=['A', 'B'] reserved=12
tick=1 decode  batch=['A', 'B'] reserved=8
tick=2 prefill batch=['C'] reserved=11
tick=3 decode  batch=['B', 'C'] reserved=8
tick=4 decode  batch=['B'] reserved=8
tick=5 decode  batch=['B'] reserved=0
A: [3, 4] (length)
B: [6, 7, 8, 9, 10] (length)
C: [7, 8] (length)
```

Focus on `tick=2`: A has left, C starts prefill, and B is still active. C does not wait for the entire initial batch to finish. Because this scheduler prioritizes prefill, B pauses for one round and resumes decoding alongside C at `tick=3`.

`tick` counts logical rounds, and `reserved` tracks token capacity reserved for active requests. Neither measures elapsed time or physical memory usage.

## 3 Request State and Budgets

### 3.1 Each Request Owns Its Output

`Request` stores its uid, prompt, output limit, optional EOS, and its own `output` list. The normal state sequence is waiting → running → finished. Waiting and running requests can also be cancelled.

The backend interface, `engine(batch, phase)`, returns one new token per request in batch order. The scheduler uses that order to write tokens back to their requests. The example's `toy_step` computes `(last token + 1) % 100`, so you can check every output by hand. It simulates a batch interface without running a batched neural-network computation.

### 3.2 Check Three Separate Limits

| Parameter | What it controls | Default |
| --- | --- | --- |
| `max_running` | Number of active requests | 2 |
| `prefill_budget` | Total prompt tokens admitted in one round | 8 |
| `token_capacity` | Total capacity reserved for active requests | 16 |

Each request reserves `len(prompt) + max_new_tokens`. A reserves `2+2=4` and B reserves `3+5=8`, totaling 12. When A finishes, the total drops to 8. The `reserved` property sums the running requests directly, so removing a request releases its logical reservation.

Reserving the maximum length up front is a conservative policy that is easy to follow. Physical memory also depends on layer count, KV heads, data type, and temporary tensors. This counter does not replace a memory allocator.

## 4 Three Tasks per Round

### 4.1 Try to Admit the First Waiting Request

`submit` adds a request to waiting. `step` checks available request slots, reserved capacity, and this round's prefill budget. A request moves to running only when all three limits allow it.

Admission is strictly FIFO: if the first request cannot fit yet, stop admitting requests instead of skipping to a smaller one. `submit` rejects a prompt larger than the entire prefill budget or a reservation larger than total capacity; otherwise that request would block the queue forever. Handling longer prompts requires adding chunked prefill.

### 4.2 Choose the Batch

The selection logic is two lines:

```python
phase = "prefill" if admitted else "decode"
batch = admitted or list(self.running)
```

If this round admitted requests, run prefill for those requests. Otherwise, advance every running request by one decode step. Prefill receives the complete prompt, while decode receives the token generated in the previous round. The toy backend only needs the last token; a real model backend also needs to retain each request's KV cache.

`list(self.running)` takes a snapshot because committing this round's outputs may remove completed requests. Iterating over the original list while removing elements could skip the next request.

### 4.3 Commit Outputs and Remove Completed Requests

Write each returned token to its request, then check EOS followed by the output limit. All completion paths call `_finish`, which removes the request from waiting or running and records the reason.

Cancellation uses the same path. `cancel(uid)` returns `False` for unknown or already completed requests. A request for zero output tokens finishes during submission. If the backend raises an exception or returns the wrong number of tokens, the batch finishes with reason `error`, and the exception propagates to the caller.

The example retains request history for printing. A long-running service should also remove historical records after their results are consumed.

## 5 Verify the Scheduler and Connect a Real Engine

```bash
python3 -m unittest examples.part2.chapter06.test_scheduler -v
```

Tests check that C reuses A's slot, each request's output matches the hand-computed result, the first waiting request is never bypassed, and capacity and prefill budgets each take effect. They also cover EOS, zero output, cancellation in both queues, and backend failure. Once all work finishes, running should be empty and reserved capacity should be zero.

To connect a real model, keep the scheduling order and backend interface, then add three operations: create each request's KV during prefill, read its KV during decode, and release cache slots on completion. The backend handles tensor layouts for different request lengths and paged storage. [Chapter 7](./Chapter7_Paged-KV-Cache-and-Memory-Management.md) continues with cache management.

Read the pinned mini-sglang version in this order:

| Action in this chapter | Entry point in mini-sglang `9a91cfa` |
| --- | --- |
| Receive requests | `Scheduler._process_one_msg` |
| Choose prefill or decode | `Scheduler._schedule_next_batch` |
| Commit outputs and release resources | `Scheduler._process_last_data`, `_free_req_resources` |

Start with `normal_loop`, then read the overlapping execution path. In a real asynchronous engine, cancellation must wait for computations still using a cache before releasing it; this synchronous simulator has no such in-flight references. A continuous stream of new requests can also delay decode under a prefill-first policy, so fairness needs evaluation with real workloads.

The complete program is `examples/part2/chapter06/scheduler.py`. The course site includes that same file directly:

::: details Complete program: scheduler.py

<<< @/examples/part2/chapter06/scheduler.py

:::

## 6 Summary and Exercises

### 6.1 Summary

Continuous batching chooses participants again each round. Waiting controls admission, running tracks ongoing generation, and one completion path releases slots. Verify the trace with a deterministic backend before connecting a model and KV cache so scheduling errors and computation errors can be investigated separately.

### 6.2 Exercises

1. Set `max_running` to 1 and rerun the example. In which round can C first start?
2. Set `token_capacity` to 8. Why might the next request have to wait even when a request slot is free?
3. Cancel B after the first round. How does C's admission time change?
4. If short requests keep arriving, which latency metric of an existing request can prefill priority affect?

## References

- [mini-sglang 9a91cfa: scheduler loop](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py)
- [Chapter 6: Continuous Batching and the Scheduler concepts](./Chapter6_Continuous-Batching-and-the-Scheduler.md)
- [Chapter 7: Paged KV Cache and Memory Management](./Chapter7_Paged-KV-Cache-and-Memory-Management.md)
