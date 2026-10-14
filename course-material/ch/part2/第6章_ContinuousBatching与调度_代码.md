# 第 6 章 连续批处理与调度（代码实现）

A、B 正在生成时，C 又来了。A 比 B 先结束，C 能否马上使用空出来的位置？连续批处理的关键，就是在每轮计算之间重新决定谁进入、谁继续、谁退出。

本章用一个可运行的调度器回答这个问题，对应[概念篇](./第6章_ContinuousBatching与调度.md)。为了集中观察请求生命周期，后端只按固定规则生成整数，不加载模型、不分配真实 KV。读完调度主循环后，再把这个接口连接到模型前向和缓存管理。

## 1 本章学习目标

1. 用 waiting 和 running 两个队列组织请求。
2. 实现 FIFO 接入、prefill 优先、结束与取消。
3. 从运行轨迹解释动态组批，区分请求数、token 预算和真实显存。

## 2 先看一次动态组批

只需 Python 3.12 标准库。在仓库根目录运行：

```bash
python3 examples/part2/chapter06/scheduler.py
```

A、B 初始进入，C 在第一轮结束后到达。最多同时运行两条请求，A/B/C 分别生成 2/5/2 个 token：

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

重点看 `tick=2`：A 已退出，C 开始 prefill，B 还没有结束。C 不必等待整个初始 batch 完成。这里选择 prefill 优先，因此 B 暂停一轮，到 `tick=3` 才与 C 一起 decode。

`tick` 是逻辑轮次，`reserved` 是活动请求的 token 容量预留，二者都不是实测时间或显存大小。

## 3 请求和预算各自保存什么

### 3.1 每条请求保留自己的输出

`Request` 保存 uid、prompt、输出长度上限、可选 EOS，以及自己的 `output`。正常状态变化为 waiting → running → finished；排队或运行中的请求也可以取消。

后端接口 `engine(batch, phase)` 按 batch 顺序返回每条请求的一个新 token，调度器再按相同顺序写回输出。示例里的 `toy_step` 使用 `(最后一个 token + 1) % 100`，所以结果可以直接手算。它模拟批接口，没有执行神经网络张量批处理。

### 3.2 三种约束分别检查

| 参数 | 控制什么 | 默认值 |
| --- | --- | --- |
| `max_running` | 同时活动的请求条数 | 2 |
| `prefill_budget` | 一轮新接入 prompt 的 token 总数 | 8 |
| `token_capacity` | 所有活动请求的容量预留总和 | 16 |

每条请求预留 `len(prompt) + max_new_tokens`。例如 A 预留 `2+2=4`，B 预留 `3+5=8`，合计 12；A 结束后降到 8。`reserved` 直接对 running 求和，请求移出后就不再占用这份逻辑容量。

整段预留是便于理解的保守策略。真实显存还受层数、KV 头数、数据类型和临时张量影响，这个整数不能代替显存分配器。

## 4 每轮只做三件事

### 4.1 从队首尝试接入

`submit` 将请求加入 waiting。`step` 从队首检查请求槽位、容量预留和本轮 prefill 预算，全部满足才移入 running。

这里采用严格 FIFO：队首暂时放不下，就停止接入，不绕过它寻找较小请求。prompt 本身超过一轮预算，或整条请求超过总容量时，`submit` 直接报错，否则它会永远占住队首。长 prompt 的分块处理需要后续加入 chunked prefill。

### 4.2 选择本轮 batch

选择逻辑只有两行：

```python
phase = "prefill" if admitted else "decode"
batch = admitted or list(self.running)
```

本轮接入了新请求，就对这些请求 prefill；否则，让所有 running 请求各 decode 一步。prefill 输入完整 prompt，decode 输入上轮生成的 token。模拟后端只需要最后一个 token；真正的模型后端还要在请求之间保存各自的 KV。

`list(self.running)` 创建快照，因为本轮提交输出时可能移除已结束请求。若直接遍历并删除原列表中的元素，就可能跳过下一条请求。

### 4.3 写回输出，移除结束请求

后端返回后，将每个 token 写入对应请求，先检查 EOS，再检查最大输出长度。所有结束路径统一调用 `_finish`，把请求移出 waiting 或 running，并记录原因。

取消也走这个出口。`cancel(uid)` 对未知或已结束请求返回 `False`；零输出请求在提交时就完成；后端抛错或返回数量不对时，本批请求以 `error` 结束，再向调用方抛出异常。

示例保留历史请求方便打印；长期运行的服务还应在结果消费后清理历史记录。

## 5 运行验证，连接真实引擎

```bash
python3 -m unittest examples.part2.chapter06.test_scheduler -v
```

测试检查 C 能复用 A 的槽位、各请求输出与手算结果一致、队首不会被绕过，以及容量与 prefill 预算分别生效。还覆盖 EOS、零输出、两种队列的取消和后端失败。完成后 running 应为空、reserved 应归零。

接入真实模型时，可以保留调度顺序和后端接口，再补充三件事：prefill 建立每条请求的 KV，decode 读取自己的 KV，结束时归还缓存槽位。不同长度请求的张量布局与分页存储由后端处理，[第 7 章](./第7章_PagedKVCache与显存管理.md)继续讨论缓存管理。

对照固定版本的 mini-sglang 时，可按下面的顺序阅读：

| 本章动作 | mini-sglang `9a91cfa` 的入口 |
| --- | --- |
| 接收请求 | `Scheduler._process_one_msg` |
| 选择 prefill 或 decode | `Scheduler._schedule_next_batch` |
| 提交输出、结束回收 | `Scheduler._process_last_data`、`_free_req_resources` |

先读 `normal_loop`，再读重叠执行。真实异步引擎取消请求时，必须等仍在使用缓存的计算结束后再释放；本章同步模拟器没有这类在途引用。持续的新请求也可能让 prefill 优先策略推迟 decode，需要用真实负载评估公平性。

完整程序在 `examples/part2/chapter06/scheduler.py`，课程站点直接引入同一份文件：

::: details 完整程序：scheduler.py

<<< @/examples/part2/chapter06/scheduler.py

:::

## 6 总结与测试题

### 6.1 课程总结

连续批处理在每轮重新选择请求。waiting 管接入，running 管继续生成，统一结束出口归还槽位。先用确定性后端确认调度轨迹，再接入模型和 KV，可以分别定位调度错误与计算错误。

### 6.2 测试题

1. 将 `max_running` 改成 1，再运行程序。C 最早在哪一轮开始？
2. 将 `token_capacity` 改成 8，为什么即使有空闲请求槽位，也不一定能接入下一条请求？
3. 在第一轮后取消 B，C 的接入时间会如何变化？
4. 若持续接入短请求，prefill 优先可能影响已有请求的哪项延迟？

## 参考资料

- [mini-sglang 9a91cfa：调度主循环](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py)
- [第 6 章：Continuous Batching 与调度概念篇](./第6章_ContinuousBatching与调度.md)
- [第 7 章：Paged KV Cache 与显存管理](./第7章_PagedKVCache与显存管理.md)
