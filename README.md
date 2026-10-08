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
      temporal_v0/ reward_v0/ rl_v0/ scale_v2/   时序标定、奖励与尺度（src 读取）
      metro/              地铁 6 号线、6 号线支线、13 号线北延站点与单元距离（P1，来源 OSM）
docs/                     方案、设计依据、数据评估文档（P1服务器说明.md 为当前服务器说明）
src/tpmorl/               环境（env/、rl/）、目标（objectives/）、评价（eval/）
scripts/                  现役实验脚本：exp_temporal_gate / exp_fqi_local / exp_prescreen_ablation /
                          exp_field_sensitivity / exp_rho_sweep / exp_timing_world / export_schedules /
                          build_table_main / check_conformity_gate / audit_manuscript_numbers / p1_*.py /
                          p2_generalize（迁移与历史观测）/ p2_timing_space / p2_announced_greedy
paper_ijgis/              figures/ 与 figsrc/ 作图脚本、LaTeX 与 Word 构建脚本（审稿期间稿件正文不入公开仓库）
results_det/ results_figures/ results_fqi/ results_ablation*/ results_field_sens/
results_rho*/ results_p1/ results_p2/ 论文用到的结果
refs/                     参考文献与竞品材料
启动实验_P1服务器.sh       P1 服务器作业（见 docs/P1服务器说明.md）
_archive/2026-10-02/      过期内容归档（旧版本结果、脚本、文档、数据输出），见其中 ARCHIVE_INDEX.md
```

## 快速开始

论文结果的复现步骤（环境、种子、每张表和图对应的脚本）见 **[REPRODUCE.md](REPRODUCE.md)**（English）。

```bash
conda env create -f environment.yml && conda activate tpmorl
export PYTHONPATH=src
python scripts/audit_manuscript_numbers.py        # 核对稿件中的全部数字
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

- [x] 空间底图、制度通道分区与 717 个候选更新单元
- [x] 时序机会场、审批管线与年度容量
- [x] 排期规则（贪心、滚动规划、价值学习策略）与参照排期
- [x] IJGIS 稿件对应的全部实验、表格与图（见 REPRODUCE.md）
