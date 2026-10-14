# Chapter 6: Continuous Batching and the Scheduler

FIFO admission, dynamic batches, and request completion in a minimal scheduler. This example runs independently of the other chapters.

[English tutorial](../../../course-material/eng/part2/Chapter6_Continuous-Batching-and-the-Scheduler_code.md) · [中文教程](../../../course-material/ch/part2/第6章_ContinuousBatching与调度_代码.md)

Run from the repository root with Python 3.12:

```bash
python examples/part2/chapter06/scheduler.py
python -m unittest examples.part2.chapter06.test_scheduler -v
```

Uses only the Python standard library. A deterministic integer rule simulates the batch interface; there is no neural-network tensor batch or real KV allocation. Token capacity is a logical reservation, and tick is a round counter.

The default example and tests need no model download or GPU.
