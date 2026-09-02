# TP-MORL — 时序多目标强化学习的存量城市更新决策（深圳光明区）

**Temporal-Priority Multi-Objective Reinforcement Learning for Urban Renewal Scheduling**

## 这个项目要解决什么

现有把强化学习用于国土空间规划的工作（pSO、MOGNNAC、LUP-PPO 等）都在回答同一个问题：
*"理想状态下这块地该是什么功能？"* 它们把所有地块当成**同一时刻可以自由重新分配的白板**，
忽略了产权到期、更新计划有效期、分期实施顺位这些真实约束——而这恰恰是**已建成区域**
与**从零规划新区**最本质的区别。

本项目改问另一个问题：*"这块地现在值不值得动，还是再等 5 年？"*
引入时序维度后，agent 学到的不是"最优静态方案"，而是
**在有限时间窗口和法定顺序约束下的可执行行动日程表**。

方法上的定位：**空间耦合的多目标最优停止问题**（spatially coupled multi-objective optimal stopping）。

## 目录结构

```
data/
  raw/                    原始栅格，144 MB，不入 git → scripts/fetch_raw.sh 拉取
  interim/                中间产物，不入 git
  processed/
    gm_dataset_v1/        光明区 100 m 决策格网数据集（1.2 MB，入库）
      raster_5m/          LU_geo.tif — 补齐 EPSG:2383 投影后的 5 m 用地栅格
      grid_100m/          10 个 171×161 数组：类别 / 概率 / 动作掩码 / 地理条件
      zones_v0/           制度通道分区 + 717 个候选更新单元 + 更新压力代理
      tables/             制度事件表 + pSO 的 UUM / CM / CCM 矩阵
docs/                     方案、评估与竞品核查文档
src/tpmorl/
  data/                   数据构建（build_zones.py）
  env/                    MDP 环境（待建）
  objectives/             多目标函数（待建，复用 pSO objs.py）
  agents/                 策略网络与训练（待建）
figures/                  论文级图件
notebooks/                探索性分析
refs/                     参考文献与竞品材料
scripts/                  数据拉取脚本
```

## 快速开始

```bash
bash scripts/fetch_raw.sh                          # 拉取 pSO 原始栅格
python -m tpmorl.data.build_zones                  # 重建 zones_v0
```

## 核心设计决策（详见 docs/）

| 决策 | 结论 | 依据 |
|---|---|---|
| 决策格网分辨率 | **100 m** | 已批更新单元拆除面积中位 4.4 ha；500 m 格会让 94% 的单元装不满一格 |
| 功能区划分依据 | **制度通道 × 时序状态**，取代用地类型 | 全市 269 条已批单元中 39.0% 为工业类，且村类中位容积率 7.24 vs 工业 6.47 |
| 动作对象 | **717 个候选更新单元**，非 12,454 个格子 | 候选体格数中位 11，全部落在实证 1–30 格区间 |
| 坐标系 | EPSG:2383（深圳独立坐标系） | 由同目录 road.tif 转写并逐字节验证 |

## 数据来源

- 用地与地理栅格：[codeRimoe/pSO](https://github.com/codeRimoe/pSO)（pMOLU/GMCase）
- 制度事件数据：深圳市政府数据开放平台（城市更新单元计划 729 条、已批单元规划 269 条）
- 历史建筑点位：光明区历史建筑名录 10 处

## 状态

- [x] 空间底图（t=0）完整，投影已补齐
- [x] 制度通道分区 v0 与候选更新单元
- [ ] 时序状态机与年度动作掩码
- [ ] 52 个光明区更新单元的边界落格
- [ ] MDP 环境与多目标奖励
- [ ] 策略训练与行动日程表输出
