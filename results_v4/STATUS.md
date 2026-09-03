# 批次 v4 运行状态

- 主机并行度：`WORKERS=35`（探测到 128 核）
- 每组规模：7 权重档 × 5 种子 = 35 次运行，`--iters 400 --eps 8`
- 最后刷新：2026-09-04 06:05:10 CST
- 提交：`9b83330`

| 组 | 名称 | 状态 | 耗时 | 产出 |
|---|---|---|---|---|
| 1 | 基线（实测标定制度参数） | 完成 | 0h36m | curves.csv objectives.csv runs.json  |
| 2 | 冷却期 0 档（核心主张压力测试） | 完成 | 0h38m | curves.csv objectives.csv runs.json  |
| 3 | 现行法定窗口 2+1 年（条文对照） | 完成 | 0h42m | curves.csv objectives.csv runs.json  |

> 尚未全部完成。已完成的组其结果即可用，情景之间的**标量化回报不可比**
> （分母按情景而异），跨情景只能比 `objectives.csv` 的原始量纲值。
