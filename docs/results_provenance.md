# 稿件数字来源表（结果 → 文件 → 脚本）

代码版本：所有价值策略结果来自提交 0caf6bf（FQI 回溯去掉重复折扣，`scripts/p1_common.BACKUP_GAMMA = 1`）；2026-10-13 在本机（macOS arm64，scikit-learn 1.9.1）全部重跑。规则类结果与此改动无关，未变。旧结果备份在 `_archive/gamma095_backup_20261013/`。本表写于本地 HEAD b6affc1。
所有数字由 `scripts/audit_manuscript_numbers.py` 从下列文件重算并逐条核对稿件（102 项，0 缺失）。

| 稿件位置 | 指标 | 实验 | 结果文件 | 生成脚本 | 场景 / 种子 / 抽样 |
|---|---|---|---|---|---|
| 摘要, §5.1–5.2, Table 4, Fig 5 | 0.770 / 0.861 / 0.901 / 0.968 等 | 主实验（训练场景，主场） | results_p1/tables/main_M20.csv, results_p1/rolling/rolling_results.csv | scripts/p1_rolling_horizon.py, scripts/p1_build_tables.py | 主场；listing draws 0,1,2,3,4,7；M=20 |
| §5.3, Fig 6a | 价值策略 0.904，配对差 +0.002（17/30） | 主实验，学习规则 | results_p1/fqi/main_M20_s*.csv | scripts/p1_fqi_job.py | 训练种子 0–4 × 6 个 listing draws |
| §5.4, Table D6–D7, Fig 6b–c | 0.865 vs 0.871，−0.006 [−0.017, 0.006]，21/50，4/10，主场景 0.863 vs 0.865；措辞 does not beat | 登记的正式迁移测试 v2（修正回溯，服务器，tag p2-formal-v2-complete-20261014_130838） | results_p2_formal_v2/test/gen_s*_h1.csv, SUMMARY.md；v1（旧回溯，0.873）保留在 results_p2_formal/ | 启动实验_P2正式测试.sh, scripts/p2_formal_summary.py | 训练场 101–200；验证 201–210；测试 1–10 + 主场；种子 0–4；listing draw 7 |
| §5.4 第 4 段, Table D6–D7 | 0.832 / 0.831 / H=1–5 0.844–0.856 | 早期迁移运行（较少训练经验） | results_p2/gen_s*.csv, results_p2/gen_s*_h{1,2,3,5}.csv | scripts/p2_generalize.py | 训练场 101–120；测试 1–10；种子 0–4 |
| D.5 | 验证 H=0/1/2：0.841/0.868/0.864（v2） | 正式测试第一阶段 | results_p2_formal_v2/hval_h*/ | 同上 | 验证场 201–210，种子 0 |
| D.5（仅 v1） | 试验 D 与服务器差异（均值 <0.001，单场 ≤0.018） | 跨机器复现检查（旧回溯） | results_p2_val/D/gen_s0_h1.csv vs results_p2_formal/hval_h1/gen_s0_h1.csv | 同一代码版本 | 验证场 201–210，种子 0 |
| §5.2 末段, §6.2, Table D8 | 信息消融 0.871→0.880 / 0.900 / 0.938 / 0.963；提前量 8、12 年 | 信息消融（只计算，不训练） | results_p2_diag/waiting_value.csv, results_p2_diag/announce_lead.csv | scripts/diag_waiting_value.py, scripts/diag_announce_lead.py | 测试场 1–10；listing draw 7；M=20 |
| §5.2, Table D3–D5 | 初筛、无准入闸门、地铁、场敏感性 | 敏感性 | results_p1/tables/*.csv | scripts/p1_build_tables.py | 见各表注 |

注意：主实验的公告规则 0.861 是主场 6 个 listing draws 的均值；迁移实验的 0.871 是 10 个测试场、listing draw 7 的均值。两者是不同评价集合，稿件 §4.2 和 §5.4 都写明了。

## 2026-10-16 补充
| 稿件位置 | 指标 | 结果文件 | 脚本 |
|---|---|---|---|
| §3.1 命题 1，附录 D.6 | 当前/公布预测的可分离性（最大相对偏差 0）；滚动规划与逐年排序逐单元相同 42/42（两种预测各一次） | results_p2/equivalence_check.csv, results_p2/separability_check.csv | scripts/p2_equivalence_check.py |
| §5.2，Figure 6（时机走廊） | 站点 1.5 km 内"最好现在开工"49% vs 36%，差 0.13 [0.06, 0.20]，p<0.001；情景场对照 34% vs 32%，p=0.62；各距离带时机价值 41–45% | results_p2/start_now_by_distance.csv, results_p2/timing_stake_by_distance.csv, results_p2/timing_stake_units.csv | scripts/p3_timing_space_metro.py；作图 paper_ijgis/figsrc/fig7_timing_corridors.py |
