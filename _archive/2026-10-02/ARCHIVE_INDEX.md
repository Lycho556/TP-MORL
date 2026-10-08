# 归档索引（2026-10-02）

共 149 项，约 2624 MB。只移动、未删除；每项在本目录下保持原相对路径。
其中 131 项被 git 跟踪，已用 `git mv` 移动（变更已暂存，提交后历史随文件保留）；其余为未跟踪文件，用普通 mv 移动。

**归档后的检查（均通过）**：16 个现役脚本中，12 个通过导入测试，audit_manuscript_numbers、check_conformity_gate、build_table_main、p1_collect 4 个直接运行通过（build_table_main 在临时目录重建的 table_main.csv 与仓库版本逐字节相同；check_conformity_gate 复现 6876.25/7054.04）；主配置逐位复现（参考值 6876.246054424286、持续贪心 0.7745224826431258、轨迹贪心 0.8935554552319573）；稿件数字审计 41/41；论文作图脚本 fig2–fig6、figA1 均能重新出图。

**恢复**：单项恢复用 `git mv _archive/2026-10-02/<路径> <路径>`（git 跟踪的）或 `mv`（未跟踪的）；全部恢复运行 `bash _archive/2026-10-02/restore.sh`。

## 旧实验结果目录（50 项，17.9 MB）



| 原路径 | 大小 | git |
|---|---|---|
| `results_bcppo_A` | 0.0 MB | 是 |
| `results_bcppo_A_rem` | 0.0 MB | 是 |
| `results_bcppo_B` | 0.0 MB | 是 |
| `results_cfcredit` | 0.0 MB | 是 |
| `results_cfcredit_n1` | 0.0 MB | 是 |
| `results_cfcredit_n2` | 0.0 MB | 是 |
| `results_cfcredit_n3` | 0.0 MB | 是 |
| `results_cfcredit_n6` | 0.0 MB | 是 |
| `results_cfcredit_real` | 0.0 MB | 是 |
| `results_curve` | 0.0 MB | 是 |
| `results_fqi_L4` | 0.0 MB | 是 |
| `results_g5_anticip` | 0.0 MB | 是 |
| `results_g5_fs` | 0.0 MB | 是 |
| `results_g5_gae` | 0.0 MB | 是 |
| `results_g5_mc` | 0.0 MB | 是 |
| `results_g5b_anticip` | 0.0 MB | 是 |
| `results_g5b_fs` | 0.0 MB | 是 |
| `results_g5b_gae` | 0.0 MB | 是 |
| `results_g5b_mc` | 0.0 MB | 是 |
| `results_gate` | 0.4 MB | 是 |
| `results_gm_bc_s0` | 0.1 MB | 是 |
| `results_gm_bc_s1` | 0.1 MB | 是 |
| `results_gm_bc_s2` | 0.1 MB | 是 |
| `results_gm_bconly_s0` | 0.1 MB | 是 |
| `results_gm_bconly_s1` | 0.1 MB | 是 |
| `results_gm_bconly_s2` | 0.1 MB | 是 |
| `results_sgate_static` | 0.1 MB | 是 |
| `results_sgate_unit` | 0.1 MB | 是 |
| `results_tgate` | 0.0 MB | 是 |
| `results_tgate_dec` | 0.1 MB | 是 |
| `results_tgate_floor` | 0.0 MB | 是 |
| `results_tgate_fs4` | 0.1 MB | 是 |
| `results_tgate_it400` | 0.0 MB | 是 |
| `results_tgate_k20` | 0.1 MB | 是 |
| `results_tgate_k20_fs4` | 0.1 MB | 是 |
| `results_tgate_k20_s1` | 0.1 MB | 是 |
| `results_tgate_k20_s2` | 0.1 MB | 是 |
| `results_v11` | 0.5 MB | 是 |
| `results_v12` | 0.3 MB | 是 |
| `results_v15` | 0.4 MB | 是 |
| `results_v16` | 0.4 MB | 是 |
| `results_v4` | 0.0 MB | 是 |
| `results_v6` | 0.3 MB | 是 |
| `results_v7` | 0.3 MB | 是 |
| `results_v8` | 0.3 MB | 是 |
| `results_wa` | 6.2 MB | 是 |
| `results_wa_exact` | 6.1 MB | 是 |
| `results_world` | 0.1 MB | 是 |
| `results_ww` | 0.8 MB | 是 |
| `results_p1/_rolling_resetbug` | 0.1 MB | 否 |

原因：
- 早期版本或已被取代的实验输出，当前论文与 P1 流程不读取
- 滚动时域首版：列表种子分支有 env.reset 缺陷，已用修正版替换

## 运行日志（12 项，2465.0 MB）



| 原路径 | 大小 | git |
|---|---|---|
| `results_ablation_A1050.log` | 0.0 MB | 否 |
| `results_ablation_Aall.log` | 0.0 MB | 否 |
| `results_ablation_Aall_s12.log` | 0.0 MB | 否 |
| `results_ablation_B.log` | 0.0 MB | 否 |
| `results_figures_export.log` | 0.0 MB | 否 |
| `results_rho_run.log` | 0.0 MB | 否 |
| `results_field_sens/amp_high.log` | 0.0 MB | 否 |
| `results_field_sens/amp_low.log` | 0.0 MB | 否 |
| `results_field_sens/onset_early.log` | 0.0 MB | 否 |
| `results_field_sens/onset_late.log` | 955.8 MB | 否 |
| `results_field_sens/win_broad.log` | 1509.2 MB | 否 |
| `results_field_sens/win_narrow.log` | 0.0 MB | 否 |

原因：
- 场敏感性实验逐步日志（论文数字取自同目录 *_runs.csv）
- 顶层运行日志（对应结果已整理进各 results 目录）

## 顶层零散文件与空目录（10 项，2.7 MB）



| 原路径 | 大小 | git |
|---|---|---|
| `_view_001_acc_scores.png` | 0.3 MB | 是 |
| `_view_001_growth_no_timing_effect.png` | 0.2 MB | 是 |
| `_view_001_v18_three_gates.png` | 0.1 MB | 是 |
| `_view_002_deferral_payoff.png` | 0.1 MB | 是 |
| `deferral_payoff.png` | 0.2 MB | 是 |
| `v18_three_gates.png` | 0.2 MB | 是 |
| `figs` | 0.5 MB | 是 |
| `figures` | 1.2 MB | 是 |
| `logs` | 0.0 MB | 否 |
| `notebooks` | 0.0 MB | 否 |

原因：
- 早期数据集/奖励概览图；论文图件在 paper_ijgis/figures/
- 空目录
- 顶层零散预览图（早期诊断）

## 旧启动脚本（7 项，0.2 MB）

原因：旧批次启动脚本（v12–v20）；当前为 启动实验_P1服务器.sh

| 原路径 | 大小 | git |
|---|---|---|
| `启动实验_v12.sh` | 0.0 MB | 是 |
| `启动实验_v13.sh` | 0.0 MB | 是 |
| `启动实验_v14.sh` | 0.0 MB | 是 |
| `启动实验_v15.sh` | 0.0 MB | 是 |
| `启动实验_v16.sh` | 0.0 MB | 是 |
| `启动实验_v17.sh` | 0.0 MB | 是 |
| `启动实验_v20服务器.sh` | 0.0 MB | 是 |

## 不再调用的脚本（22 项，0.2 MB）



| 原路径 | 大小 | git |
|---|---|---|
| `scripts/audit_dynamics.py` | 0.0 MB | 是 |
| `scripts/baselines.py` | 0.0 MB | 是 |
| `scripts/bench_cpu_vs_gpu.py` | 0.0 MB | 是 |
| `scripts/check_historical_data.py` | 0.0 MB | 是 |
| `scripts/diag_horizon.py` | 0.0 MB | 是 |
| `scripts/eval_metrics.py` | 0.0 MB | 是 |
| `scripts/exp_bc_ppo.py` | 0.0 MB | 是 |
| `scripts/exp_counterfactual.py` | 0.0 MB | 是 |
| `scripts/exp_counterfactual_credit.py` | 0.0 MB | 是 |
| `scripts/exp_external_temporal.py` | 0.0 MB | 是 |
| `scripts/exp_opt_quality.py` | 0.0 MB | 是 |
| `scripts/exp_planning_shock.py` | 0.0 MB | 是 |
| `scripts/exp_srv_mechanism.py` | 0.0 MB | 是 |
| `scripts/exp_srv_scenarios.py` | 0.0 MB | 是 |
| `scripts/exp_timing_gate.py` | 0.0 MB | 是 |
| `scripts/exp_waiting_advantage.py` | 0.0 MB | 是 |
| `scripts/verify_budget_identity.py` | 0.0 MB | 是 |
| `scripts/verify_opportunity_neutral.py` | 0.0 MB | 是 |
| `scripts/run_batch_v4.sh` | 0.0 MB | 是 |
| `scripts/run_batch_v6.sh` | 0.0 MB | 是 |
| `scripts/run_batch_v11.sh` | 0.0 MB | 是 |
| `scripts/pick_gpu.sh` | 0.0 MB | 是 |

原因：
- 旧批次/GPU 调度脚本
- 现役流程不调用（依赖图检查）；PPO/BC、v20 服务器阶段、早期诊断

## data 下的旧实验输出（19 项，137.5 MB）

原因：旧版本实验输出（v4–v17、预算/增长情景、v15 基线与评价）；src 与现役脚本不读取

| 原路径 | 大小 | git |
|---|---|---|
| `data/processed/gm_dataset_v1/exp_v4` | 3.3 MB | 是 |
| `data/processed/gm_dataset_v1/exp_v6` | 0.6 MB | 是 |
| `data/processed/gm_dataset_v1/exp_v7` | 8.3 MB | 是 |
| `data/processed/gm_dataset_v1/exp_v8` | 4.8 MB | 是 |
| `data/processed/gm_dataset_v1/exp_v11` | 21.8 MB | 是 |
| `data/processed/gm_dataset_v1/exp_v12` | 17.0 MB | 是 |
| `data/processed/gm_dataset_v1/exp_v15` | 39.3 MB | 是 |
| `data/processed/gm_dataset_v1/exp_v16` | 41.7 MB | 是 |
| `data/processed/gm_dataset_v1/exp_v17` | 0.0 MB | 否 |
| `data/processed/gm_dataset_v1/exp_opt_v1` | 0.4 MB | 是 |
| `data/processed/gm_dataset_v1/exp_smoke` | 0.0 MB | 是 |
| `data/processed/gm_dataset_v1/rl_budget_B2400` | 0.0 MB | 是 |
| `data/processed/gm_dataset_v1/rl_budget_B500` | 0.0 MB | 是 |
| `data/processed/gm_dataset_v1/rl_budget_B900` | 0.0 MB | 是 |
| `data/processed/gm_dataset_v1/rl_growth_G0.05` | 0.0 MB | 是 |
| `data/processed/gm_dataset_v1/rl_growth_G0.10` | 0.0 MB | 否 |
| `data/processed/gm_dataset_v1/rl_v0_sel` | 0.0 MB | 否 |
| `data/processed/gm_dataset_v1/baselines_v15` | 0.0 MB | 是 |
| `data/processed/gm_dataset_v1/eval_v15` | 0.1 MB | 是 |

## 带版本号的过程文档（29 项，0.4 MB）

原因：带版本号的过程记录（v4–v20），已被后续版本取代

| 原路径 | 大小 | git |
|---|---|---|
| `docs/honest_assessment_v0.md` | 0.0 MB | 是 |
| `docs/honest_assessment_v1.md` | 0.0 MB | 是 |
| `docs/honest_assessment_v2.md` | 0.0 MB | 是 |
| `docs/honest_assessment_v3.md` | 0.0 MB | 是 |
| `docs/rerun_v2.md` | 0.0 MB | 是 |
| `docs/reward_defect_v1.md` | 0.0 MB | 是 |
| `docs/v15环境改造_实施记录.md` | 0.0 MB | 是 |
| `docs/v16改动_实施记录.md` | 0.0 MB | 是 |
| `docs/v17改动_实施记录.md` | 0.0 MB | 是 |
| `docs/v18三道门槛_本地结论.md` | 0.0 MB | 是 |
| `docs/v19_收口_BC微调窗口测试.md` | 0.0 MB | 是 |
| `docs/v19简化框架_本地结论.md` | 0.0 MB | 是 |
| `docs/v20服务器实验_说明.md` | 0.0 MB | 是 |
| `docs/大白话_v15实验与故事线.md` | 0.0 MB | 是 |
| `docs/大白话_v16实验与故事线.md` | 0.0 MB | 是 |
| `docs/大白话_v17实验与故事线.md` | 0.0 MB | 是 |
| `docs/基线与上界_v15.md` | 0.0 MB | 是 |
| `docs/实施类指标_v15.md` | 0.0 MB | 是 |
| `docs/建议条目审计与改动清单_v15.md` | 0.0 MB | 是 |
| `docs/建议条目对照审计_v15.csv` | 0.0 MB | 是 |
| `docs/批次v4_查验_v5.md` | 0.0 MB | 是 |
| `docs/服务器批次_v4.md` | 0.0 MB | 是 |
| `docs/服务器批次_v6.md` | 0.0 MB | 是 |
| `docs/服务器批次_v11.md` | 0.0 MB | 是 |
| `docs/生态目标诊断_v4.md` | 0.0 MB | 是 |
| `docs/目标对偏好的响应强度_v12数据.csv` | 0.0 MB | 是 |
| `docs/修复记录_2026-09-12.md` | 0.0 MB | 是 |
| `docs/择时机制_诊断_v1.md` | 0.0 MB | 是 |
| `docs/立项与记分分离_实施记录_v1.md` | 0.0 MB | 是 |

## 保留在原位的现役内容

- `src/`、`scripts/` 中被现役流程调用的 16 个脚本，外加 `fetch_raw.sh`、`make_learned_scale.py`（数据来源与尺度生成）
- `data/processed/gm_dataset_v1/` 的 grid_100m、raster_5m、tables、zones_v0、temporal_v0、reward_v0、rl_v0、scale_v2、metro
- `paper_ijgis/`、`refs/`、设计与数据评估文档、`docs/P1服务器说明.md`
- `results_det/`、`results_figures/`、`results_fqi/`、`results_ablation*/`、`results_field_sens/*_runs.csv`、`results_rho*/`、`results_p1/`
- `启动实验_P1服务器.sh`、`.tex_bundle/`（本地 TeX 包，已在 .gitignore）
