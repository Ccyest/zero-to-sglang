# 第 6 章：连续批处理与调度

用最小调度器理解 FIFO、动态组批与请求结束。本目录独立运行，不依赖其他章节示例。

[阅读中文教程](../../../course-material/ch/part2/第6章_ContinuousBatching与调度_代码.md)

在仓库根目录运行，使用 Python 3.12：

```bash
python examples/part2/chapter06/scheduler.py
python -m unittest examples.part2.chapter06.test_scheduler -v
```

只使用 Python 标准库。后端按固定规则生成整数，模拟批接口；没有神经网络张量批处理或真实 KV 分配。token 容量是逻辑预留，tick 是轮次。

默认运行和测试均不下载模型、不需要 GPU。
