# P7 验证阶段：E3 排序学习 + E2 保守偏离（开跑前提交）

本文件是**探索阶段**的事先说明，不是测试登记。只用验证情景 201–210 和观测地铁场；测试情景 1–10 与主情景在登记前不评价（`scripts/p7_common.guard_fields` 会直接拒绝）。选定方法后另写 `docs/P7/registration_<名字>.md` 再跑测试。

## 共同设置（全部沿用 P3 冻结方法，`scripts/p3_formal_test.FROZEN`）

- 状态：论文状态（H = 1）+ 已公布规划预测两列 `f_now = F_t(u,t)`、`f_best = max_{y≥t} F_t(u,y)`
- 训练：情景 101–120，每种行为规则（随机、持续贪心、轨迹贪心）2 回合，采样种子 `seed·10007 + k`，训练种子 0–4
- 公布提前量 4 年，M = 20，listing draw 7，`BACKUP_GAMMA = 1.0`
- 单线程（`TPMORL_NJOBS=1`、`OMP_NUM_THREADS=1`），作业之间进程并行

## E3 排序学习（`scripts/p7_rank.py` → `results_p7_rank/`）

每个种子一个作业，在**同一批**训练数据上拟合三个模型，决策都是每年选得分最高的 3 个：

| 名称 | 模型 |
|---|---|
| `p3` | 论文的学习修正（对照）：双树集成拟合 `EV[u,t] − f_now`，一步目标 |
| `rank_scratch` | LightGBM LambdaMART，特征同 p3 |
| `rank_resid` | 同上，但以 `κ·z(f_now)`（组内标准化，κ = 1）作 `init_score`：从规划排序出发学修正 |

- 一组 = 一个（情景, 行为规则, 回合, 年份）的候选集；相关度标签 = 组内 `EV[u,t]` 的五分位档（0–4）
- LightGBM：400 棵树，学习率 0.05，31 叶，`min_child_samples=20`，单线程，`deterministic=True`
- 通过标准（方案原文）：验证情景上均值高于 p3，**且**种子间方差更小

## E2 保守偏离（`scripts/p7_safe.py` → `results_p7_safe/`）

不确定性 = 5 个种子 p3 模型之间的分歧：`ΔQ_s = Q_s − f_now`，`m = mean_s ΔQ_s`，`sd = std_s ΔQ_s`（ddof = 1）。

| 变体 | 决策 |
|---|---|
| E2a `safe_lam{λ}`，λ ∈ {0, 0.5, 1, 2} | 每年选 `f_now + m − λ·sd` 最高的 3 个；λ = 0 即 5 个 p3 模型取平均 |
| E2b `safe_swap_z{z}`，z ∈ {1, 2} | 从规划排序的前 3 出发，只有当未选候选的下界 `f_now + m − z·sd` 高于已选单元的上界 `f_now + m + z·sd` 时才替换（只换不乱） |
| `safe_swap_zinf` | z = ∞，必须逐位等于规划排序（公平性检查） |

**选择规则（事先固定）**：每个族内取验证情景 201–210 上均值最高的变体；与最高值相差不超过 0.002 的，取更保守的（λ 或 z 更大）。

**地铁场**：已被 `scripts/p3_metro_test.py` 用过，不再是未触碰的测试场。所有变体都报告，但**不参与选择**。通过标准（方案原文）：地铁场上不再输给规划排序（p3 为 0.9164 对 0.9408）。

## 统计与检查

- 每个情景先对种子取平均（E2 本身是集成，无需再平均），再减去基准（规划排序、p3）；报告均值差、95% t 区间、Wilcoxon 检验、赢的情景数。区间全在 0 以上才写 "improves on"。
- 泄漏检查：种子 0 作业对情景 201 运行 `p3_formal_test.leak_check`，未来场值换成随机数后决策输入必须不变，且反向对照必须敏感。
- 复现检查：p3 对照与 `results_p3_explore`（验证集）、`results_p3_metro`（地铁场）的旧结果比对；同一平台应逐位一致，跨平台记 PASS_TOL。
- 公平性：`rule_replay` 与 `safe_swap_zinf` 都必须逐位等于 `rank_announced`。

## 运行

服务器（仓库根目录）：`nohup bash 启动实验_P7验证.sh > ~/p7val.log 2>&1 &`。每个作业结束自动提交推送（标题 `[p7-val] ...`），全部完成后打标签 `p7-val-complete-<批次>`。模型文件写在 `~/tpmorl_p7_models/`，不入库。
