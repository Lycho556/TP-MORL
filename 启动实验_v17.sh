#!/usr/bin/env bash
# ============================================================================
#  TP-MORL 批次 v17 —— 一条命令跑完全部实验
#
#  在服务器上只需要敲这一句：
#
#      bash ~/TP-MORL/启动实验_v17.sh
#
#  想挂后台断开 ssh 也不中断：
#
#      nohup bash ~/TP-MORL/启动实验_v17.sh > ~/v17.log 2>&1 &
#      tail -f ~/v17.log
#
#  脚本自己会做：git pull → 环境自检 → 逐组训练 → 每组跑完顺手算规则基线
#  → **每组结束立即 commit + push + 打标签**（备注里带组名、状态、耗时）
#  → 断点续跑（已完成的组自动跳过）→ 全部完成后重算实施类指标、出主判据、
#  打标签 batch-v17-complete。
#
#  ---------------------------------------------------------------------------
#  v17 为什么存在
#
#  v16 的主判据成立（补齐生命周期观测后 LSR 降、CRH 升），但**次判据不成立**：
#
#      主组立项年份分布与均摊无法区分 —— 卡方 p=0.999，立项重心 6.98
#      对均摊基准 7.00（results_v16/quicklook.txt）。十六个组里没有一组
#      出现"前移"。规则基线 frontload 的 CRH_T=0.613 反而高于策略的 0.301。
#
#  也就是说：**到 v16 为止，强化学习没有学到"时间维度上什么时候建什么合适"。**
#
#  原因不在优化器，而在环境。诊断（docs/择时机制_诊断_v1.md §3，以及
#  env_gym 顶部"非平稳性"一节）早已写明：
#
#      候选集是平稳的。预算结转只改变"能做多少"，不改变"何时做更好"；
#      成本与收益都与面积线性、无规模报酬，故攒钱无回报。于是明年的机会集
#      不优于今年，而 γ<1 始终把动作往前推 —— 等待永不严格占优。
#
#  策略学不到择时，不是学不会，而是**没有择时可学**。此前唯一的非平稳通道
#  FAR_GROWTH 是**全局同步**的：所有单元同时变好，只改变"整体早做还是晚做"，
#  不改变"先做哪个、后做哪个"，而后者才是真正的择时问题。
#
#  v17 做四件事：
#
#      (1) **四张逐年变化的机会场**（tpmorl/env/opportunity.py），逐单元异步：
#            上位规划 PlanningOpportunity(i,t)     政府什么时候开始重视这里
#            基础设施 InfrastructureOpportunity(i,t) 这个地方什么时候成熟
#            建筑老化 BuildingAge(i,t) → 更新必要性  什么时候更新变得必要
#            实施条件 ImplementationReadiness(i,t)  什么时候真正容易做成
#          前三者走**价值通道**（时机乘子作用在交付价值上，规划读立项年、
#          设施与老化读建成年）；实施条件走**概率通道**（调制逐年批准风险率，
#          条件成熟的年份更容易批得下来、更不容易失效）。
#          于是同一个地块在第 2 年与第 8 年不再是同一个项目。
#
#      (2) **观测新增八维**（N_FEAT 22 → 30）：四项当期水平 + 四项趋势。
#          趋势那四维不是冗余——见下面"门槛实验"一节。
#
#      (3) **决策期 15 年 → 25 年**。制度前置期（审批中位 3 年 + 次年开工 +
#          建设 5 年）把 15 年窗口砍成前 8 年，择时几乎没有活动空间。
#          按状态机蒙特卡洛重算（scripts/diag_horizon.py，2 万次重复）：
#            T=15：期内建成概率 0.70 的立项年只有第 1–5 年，第 10–15 年
#                  立项**必然**不能在期内交付（结构性死区占决策期 40%）
#            T=25：0.70 的立项年扩到第 1–15 年，死区收缩到 24%
#          即"可以真正选时点的年份"从 5 年扩到 15 年，是三倍。
#
#      (4) **择时质量指标**（tpmorl/eval/metrics.timing_quality）。CRH/LSR 只
#          管"赶不赶得上"，一个把项目尽早塞进管道的策略在这两项上表现最好，
#          而那恰是 v16 的问题。新增四项管"挑得准不准"：
#            opp_at_init / opp_lift  立项时点的机会场取值，及其相对"同一批
#                                    地块全期均值"的提升（参照系刻意排除选址）
#            ramp_post_onset         "会变好"型地块在其爬升之后立项的占比，
#                                    与均匀立项的零假设并列报告
#            timing_regret           与单元独立最优时点的折现价值差距（松下界）
#
#  ---------------------------------------------------------------------------
#  门槛实验：跑这一批之前必须知道的事（scripts/exp_timing_gate.py）
#
#  在把机会场接进真实环境之前，先在一个**最优解可以穷举算出**的三地块最小
#  情景上问了一件事：掩码指针 + PPO + 逐年奖励这套学习器，**有没有能力**学会等？
#  情景：B 现在就好且以后不变（应第 0 年动）、A 现在一般而第 onset 年起变好
#  （应等到 onset）、C 一直不好（低优先，且不该为它一直等）。
#
#  实测结论（十一个档，每档 3–5 个种子，判据为"A 是否等过 onset / B 是否没被
#  一起推后 / 折现回报是否达到穷举最优的 95%"）：
#
#      **没有任何一档同时满足三条判据。** 试过的方向与各自的失败方式：
#        奖励改到立项年计（去掉 lead 年延迟）      立项年不变（回报比 0.743）
#        熵系数 ×5                                 同上，逐位相同
#        迭代数 300 → 1000                         同上，逐位相同
#        折现率 0.95 → 0.99                        略差（0.706）
#        熵系数 ×20                                **训练崩掉**：3 个种子里
#                                                  1 个整回合不立项，均值 0.491
#        机会改善从阶跃改为平滑爬升                 立项年反而更早（第 1 年）
#        幅度加大到 10 倍                          立项年更早（第 0 年）
#        观测加入未来水平/趋势（前瞻 1/3 年）       立项年不变
#        每轮回合数 8 → 32（降梯度方差）           **1/3 种子学会了等**（A=7
#                                                  正是最优年），但该种子把
#                                                  B、C 也一起推后 —— 退化解
#        势函数型整形 PBRS + 爬升场                 唯一把回报做到穷举最优 95%
#                                                  以上的一档（3/3 种子，均值
#                                                  0.967；同一场形与前瞻档的
#                                                  未整形对照均值 0.930），
#                                                  但立项年仍未推后
#        势函数型整形 PBRS + 阶跃场                 不稳定（均值 0.491）
#
#  两条结论，直接决定这一批该怎么读：
#
#    (a) **环境改造是必要的，但不是充分的。** 把"等"变得有价值之后，这套学习器
#        仍然倾向于按当期价值降序尽快填满配额。因此本批次**不能**预设"开了机会场
#        就会学到择时"；主判据必须写成可以被否证的形式（见下）。
#    (b) 因此把两个算法侧因子**当作一等实验因子放进组表**（第 6、7 组），而不是
#        当成既定修复悄悄开着：每轮回合数 32 是门槛实验里唯一出现过"等"的方向，
#        势函数整形是唯一把回报做到最优 95% 的方向。两者都只是部分有效。
#
#  ---------------------------------------------------------------------------
#  主判据（本批次要回答的问题，写成可否证的形式）
#
#    H1 机会场是"等待有价值"的载体：第 1 组（场开+观测开）相对第 4 组
#       （场关+观测关，= v16 口径）在 opp_lift > 0、ramp_post_onset 显著高于
#       其零假设、timing_regret 下降。
#    H2 这段信息必须看得见：第 1 组相对第 2 组（同场但观测置零，**共用分母**，
#       故标量回报也可直接相减）在上述择时指标上更优。
#    H3 趋势是载体中的关键：第 1 组（前瞻 3 年）相对第 5 组（前瞻 0 年，只给
#       当期水平）更优。门槛实验预测这一条可能**不成立**——若不成立，说明
#       真实环境里"低水平"已经足以蕴含"会变好"（因为场有三型构成），
#       那是一个关于场的设定的结论，要如实写。
#    H4 延长决策期带来择时自由度：第 1 组（T=25）相对第 14 组（同场 T=15）
#       的 timing_regret 更低、立项年份分布更不像均摊。
#
#  **任何一条不成立都要照实写。** v16 的教训是次判据不成立时曾被放在附录，
#  这一版把它抬到主判据里。
#
#  ---------------------------------------------------------------------------
#  口径声明（论文必须写，不得含糊）
#
#  四个机会场**目前全部是情景参数，没有任何实证标定**。单元表
#  zones_v0/candidate_units.csv 只有 uid/通道/格数/row/col/压力/距路m 七列，
#  既没有建成年份，也没有历年规划定位或设施投用年份。因此：
#    * 场的形状（三型构成 + 逻辑斯蒂爬升）是假设；
#    * 场的幅度 a_* 是情景参数，必须扫（第 8、9 组）；
#    * 结论只能写成条件句："若上位规划与基础设施按 X 幅度在 Y 年内改善，则
#      最优策略呈现 Z 的时序结构"，**不得**写成"实证表明光明区应推迟某些项目"。
#  另外，任一幅度 > 0 时 Floor 不再是纯粹的计容建筑面积，而是"按时机加权的
#  交付价值（以建面计量）"，报告时必须写明。
#  替换为实证数据所需的清单见 docs/大白话_v17实验与故事线.md 末节。
#
#  ---------------------------------------------------------------------------
#  与既往批次的可比性
#
#  * **幅度全 0 且整形关闭时，环境动态与 v16 逐位等价**，已机械验证：
#    scripts/verify_opportunity_neutral.py 用固定种子的动作序列在两版源码上
#    各跑一遍，逐年事件/Floor/Cost/奖励的 sha 摘要相同（066724d07d0379ad）。
#    自检里会再跑一次；不一致即中止整批。
#  * **网络权重不可跨版载入**：观测 22 → 30 维改变了第一层参数量，torch 的
#    初始化抽样序列随之改变。这与 v15 的 16 → 22 是同一类情况，处理方式相同
#    —— v17 的结论一律由**批次内**对照给出，不与 v16 的图混画。
#  * 四个幅度与场的形状参数**都入分母缓存键**（改变可达上界）；观测开关、
#    前瞻年数、势函数整形强度**刻意不入键**（参考策略不读观测与整形项，
#    可达上界按构造相同），故这几对组共用分母、标量回报可直接相减。
#
#  ---------------------------------------------------------------------------
#  环境变量（都有默认值，一般不用管）
#
#    RES=results_v17     结果与日志目录
#    BASE=$DS/exp_v17    各组输出目录
#    ITERS=400 EPS=8     每次运行的迭代数与每轮回合数（第 6 组自带 EPS=32）
#    HORIZON=25          决策期长度；评价期一律 auto（= T+τ+建设+1）
#    WORKERS=            并行进程数，默认 min(核数-2, 35)
#    ONLY="1 2 3"        只跑指定组；PHASE=1/2/3 是它的便捷写法
#    FORCE=1             忽略断点续跑，强制重跑
#    PUSH=0              不推 github（默认推）
#    NOBASE=1            跳过规则基线
# ============================================================================
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || { echo "找不到仓库目录 $ROOT"; exit 1; }

export PYTHONPATH=src
export OMP_NUM_THREADS=1      # 必须：否则 torch 线程与进程池争核，实测慢数倍

DS="data/processed/gm_dataset_v1"
BASE="${BASE:-$DS/exp_v17}"
RES="${RES:-results_v17}"
PUSH="${PUSH:-1}"
ITERS="${ITERS:-400}"
EPS="${EPS:-8}"
HORIZON="${HORIZON:-25}"
FORCE="${FORCE:-0}"
NOBASE="${NOBASE:-0}"
FIXED="${FIXED:-0,0.1}"       # 固定分母覆盖的增长率集合，与 v11–v16 一致
RUNS_PER_GROUP=35             # 7 权重档 × 5 种子；并行度超过它没有收益
NPROC="$( (OMP_NUM_THREADS= nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 10) )"
WORKERS="${WORKERS:-$(( NPROC > 4 ? NPROC - 2 : 2 ))}"
[ "$WORKERS" -gt "$RUNS_PER_GROUP" ] && WORKERS="$RUNS_PER_GROUP"

# 主组的机会场幅度。**情景参数**，取值理由：
#   规划 0.8 / 设施 0.6 —— 让"等到规划与设施到位"的价值乘子最高约 2.9 倍，
#     与门槛实验里能让等待严格占优的量级同阶（那里 lo→hi 是 2.7 倍）。
#   老化 0.3 —— 更新必要性是渐变量，作用应弱于前两者。
#   实施条件 0.4 —— 走概率通道，0.4 时 ready 从 0.5 升到 1.0 使批准率提高约
#     1.4 倍；取到 1.0 会让条件差的年份批准率归零，那是"行政冻结"而非"难推动"。
#   前瞻 3 年 —— 法定图则与设施计划确实提前公布，3 年是保守取值。
FIELD="--a-plan 0.8 --a-infra 0.6 --a-age 0.3 --a-ready 0.4 --foresight 3"

# ---------------------------------------------------------------------------
# 实验组：编号 | 目录 | 中文名 | 额外参数 | 每轮回合数（可留空=用 $EPS）
#
# 1–4  **2×2 主判据**：机会场开/关 × 机会场观测开/关。
#      第 1、2 组是唯一一对可直接相减标量化回报的组（观测开关不入分母缓存键，
#      两组共用分母），别动它们除 --no-obs-opportunity 之外的任何参数。
#      第 4 组 = v16 口径在 v17 代码上的锚点（场关 + 观测关），用于确认
#      "换了代码但没换设定"时结果没有漂移。
# 5    信息消融：前瞻 0 年（只给当期水平，不给趋势）。检验 H3。
# 6–7  **算法侧因子**（门槛实验里唯一两个有过效果的方向，作一等因子而非默认开启）：
#      第 6 组把每轮回合数从 8 提到 32（降梯度方差），第 7 组开势函数型整形。
#      两者都不入分母缓存键，故都与第 1 组共用分母、可直接相减标量回报。
# 8–9  幅度扫描：低幅度与高幅度。幅度是情景参数，必须扫。
# 10–13 四个场逐一单开：把"哪一类机会在起作用"分离开。第 13 组尤其要看——
#      实施条件走的是概率通道而非价值通道，若只有它起效，故事线完全不同。
# 14–15 决策期对照：同场 T=15（检验 H4），以及 T=15 的 v16 口径锚点。
# 16   换场种子：同样的设定、另一张规划安排图。**这一组是可信度的关键**——
#      若结论只在某一张场上成立，那是过拟合到一张随机图，不是发现。
# 17   配额 6（窗内立项位放松）。
# 18   爬升时间常数 6 年（更缓的机会改善）。门槛实验显示场的**形状**会改变
#      可学习性（阶跃档下等待在 onset 前逐年净亏，最优解隔着多步壁垒）。
#
# `--horizon-eval auto` 写在**每组的参数里**而不是全局命令行，是为了让每组
# 解析出的评价期一眼可审。auto = T + tau_valid + tau_ext + max(建设年限) + 1，
# 故 T=25 的组得 36、T=15 的组得 26。
# **不能全批写死一个常数**——写死会把某些组的管道截断在期内，那正是 v14 修的错误。
# ---------------------------------------------------------------------------
SCENARIOS=(
  "1|main|主组：机会场开 + 观测开（T=25，前瞻 3 年）|$FIELD --horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3|"
  "2|noobs|消融：同主组但机会场八维观测置零（共用分母，可直接相减）|$FIELD --horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3 --no-obs-opportunity|"
  "3|nofield|场关但观测开：\"看得见但没得做\"，检验这八维本身不是噪声|--horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3|"
  "4|v16base|场关且观测关（= v16 口径在 T=25 上的锚点）|--horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3 --no-obs-opportunity|"
  "5|fs0|信息消融：前瞻 0 年，只给当期水平不给趋势|--a-plan 0.8 --a-infra 0.6 --a-age 0.3 --a-ready 0.4 --foresight 0 --horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3|"
  "6|eps32|算法臂：每轮 32 个回合（降梯度方差；门槛实验里唯一出现过\"等\"的方向）|$FIELD --horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3|32"
  "7|shaping|算法臂：势函数型奖励整形（不改变最优策略集，只进标量奖励）|$FIELD --horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3 --reward-shaping 1.0|"
  "8|ampLow|低幅度：规划 0.3 设施 0.2 老化 0.1 实施条件 0.2|--a-plan 0.3 --a-infra 0.2 --a-age 0.1 --a-ready 0.2 --foresight 3 --horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3|"
  "9|ampHigh|高幅度：规划 1.5 设施 1.2 老化 0.6 实施条件 0.6|--a-plan 1.5 --a-infra 1.2 --a-age 0.6 --a-ready 0.6 --foresight 3 --horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3|"
  "10|planOnly|只开上位规划场（价值通道，读立项年）|--a-plan 0.8 --foresight 3 --horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3|"
  "11|infraOnly|只开基础设施场（价值通道，读建成年）|--a-infra 0.6 --foresight 3 --horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3|"
  "12|ageOnly|只开建筑老化场（更新必要性，读建成年）|--a-age 0.3 --foresight 3 --horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3|"
  "13|readyOnly|只开实施条件场（**概率通道**：调制逐年批准率）|--a-ready 0.4 --foresight 3 --horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3|"
  "14|T15|决策期对照：同场但 T=15（检验延长决策期是否带来择时自由度）|$FIELD --horizon 15 --horizon-eval auto --budget 2700 --carry 3|"
  "15|T15base|T=15 的 v16 口径锚点（场关 + 观测关）|--horizon 15 --horizon-eval auto --budget 2700 --carry 3 --no-obs-opportunity|"
  "16|seed2|换场种子：同设定、另一张规划安排图（检验结论不依赖某一张场）|$FIELD --field-seed 20260918 --horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3|"
  "17|quota6|年度配额 6（窗内立项位放松）|$FIELD --horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3 --quota 6|"
  "18|ramp6|爬升时间常数 6 年（更缓的机会改善；场的形状影响可学习性）|$FIELD --ramp-years 6 --horizon $HORIZON --horizon-eval auto --budget 2700 --carry 3|"
)
NGROUP=${#SCENARIOS[@]}

# PHASE 是 ONLY 的便捷写法。分三段是为了让"主判据"能先拿到：
#   1 = 2×2 主判据 + 信息消融 + 两个算法臂（1–7），这七组跑完就能回答 H1/H2/H3
#   2 = 幅度与单通道分离（8–13）
#   3 = 期限 / 场种子 / 配额 / 形状（14–18），含 H4 与可信度检验
PHASE="${PHASE:-}"
case "$PHASE" in
  1) ONLY="${ONLY:-1 2 3 4 5 6 7}" ;;
  2) ONLY="${ONLY:-8 9 10 11 12 13}" ;;
  3) ONLY="${ONLY:-14 15 16 17 18}" ;;
  "") : ;;
  *) echo "PHASE 只能是 1、2 或 3（当前 '$PHASE'）"; exit 2 ;;
esac

ONLY="${ONLY:-}"
mkdir -p "$RES" "$BASE"
STATUS="$RES/STATUS.md"

want_group () {                     # 空 ONLY = 全跑
  [ -z "$ONLY" ] && return 0
  local g; for g in $ONLY; do [ "$g" = "$1" ] && return 0; done; return 1
}

# ---------------------------------------------------------------------------
# push_now —— v11 的教训在这里
#   v11 那次九组的提交全部没能进 main：跑批期间本地另一条线也在推 main，
#   两边分叉，`git push origin main` 每次被拒（非快进），只有 --tags 成功，
#   结果数据只能靠标签找回来。这一版把兜底做厚：
#     先 fetch + rebase（带 autostash），rebase 冲突就中止改用 merge，
#     merge 再冲突就对结果目录取本地版本——结果文件本来就是本机唯一产出，
#     取本地不会丢别人的东西。三次都失败才放弃，且一定会打标签兜底。
# ---------------------------------------------------------------------------
push_now () {
  [ "$PUSH" != "1" ] && { echo "[push] PUSH=0，跳过"; return 0; }
  git add -A
  git diff --cached --quiet && { echo "[push] 无改动，跳过"; return 0; }
  git commit -q -m "$1" || return 1
  local try
  for try in 1 2 3; do
    if git push origin HEAD:main 2>&1 | tail -2; then
      git ls-remote --exit-code origin main >/dev/null 2>&1 && { echo "[push] 成功"; return 0; }
    fi
    echo "[push] 第 $try 次被拒，尝试与远端合并后重推"
    git fetch origin main || true
    if ! git rebase --autostash origin/main; then
      git rebase --abort 2>/dev/null || true
      if ! git merge --no-edit origin/main; then
        echo "[push] merge 冲突，对结果目录取本地版本"
        git checkout --ours -- "$RES" "$BASE" 2>/dev/null || true
        git add -A && git commit -q --no-edit || true
      fi
    fi
    sleep 5
  done
  echo "[push] 三次均失败——结果已 commit 在本地，标签仍会打上，可事后手工 push"
  return 1
}

tag_now () {                        # $1 标签名  $2 说明
  [ "$PUSH" != "1" ] && { echo "[tag] PUSH=0，跳过"; return 0; }
  git tag -f "$1" -m "$2" >/dev/null || return 1
  git push -f origin "$1" >/dev/null 2>&1 \
    && echo "[tag] 已标注 $1" || echo "[tag] $1 推送失败，请手工 git push -f origin $1"
}

write_status () {
  {
    echo "# 批次 v17 运行状态"
    echo
    echo "**这批实验回答的问题**：v16 的主判据成立，但**次判据不成立**——策略的"
    echo "立项年份分布与均摊无法区分（卡方 p=0.999，重心 6.98 对基准 7.00）。"
    echo "也就是说到 v16 为止，RL 没有学到\"什么时候建什么合适\"。原因在环境："
    echo "候选集是平稳的，明年的机会集不优于今年，而折现始终把动作往前推，"
    echo "于是**没有择时可学**。v17 引入四张逐年变化的机会场（上位规划 / 基础设施 /"
    echo "建筑老化 / 实施条件），并把决策期从 15 年延长到 25 年。"
    echo
    echo "**主判据（写成可否证的形式）**"
    echo
    echo "- H1 机会场是载体：第 1 组相对第 4 组（v16 口径锚点）opp_lift > 0、"
    echo "  ramp_post_onset 显著高于其零假设、timing_regret 下降。"
    echo "- H2 信息必须看得见：第 1 组相对第 2 组（同场、观测置零、**共用分母**）更优。"
    echo "- H3 趋势是关键：第 1 组（前瞻 3 年）相对第 5 组（前瞻 0 年）更优。"
    echo "- H4 延长决策期带来择时自由度：第 1 组（T=25）相对第 14 组（T=15）更优。"
    echo
    echo "**门槛实验的预先警告**：在一个最优解可穷举的三地块最小情景上，十一个档"
    echo "（含改奖励时点、熵系数、迭代数、场形状、幅度、每轮回合数、势函数整形）"
    echo "**没有一档**同时满足\"学会等 + 不把别的项目一起推后 + 回报达最优 95%\"。"
    echo "故本批次不预设\"开了机会场就会学到择时\"；任一主判据不成立都要照实写。"
    echo "细节见 \`scripts/exp_timing_gate.py\` 的模块文档与 \`results_gate/\`。"
    echo
    echo "- 主机并行度：\`WORKERS=$WORKERS\`（探测到 $NPROC 核）"
    echo "- 每组规模：7 权重档 × 5 种子 = 35 次运行，\`--iters $ITERS --eps $EPS\`"
    echo "  （第 6 组自带 \`--eps 32\`，是一个刻意的算法侧因子）"
    echo "- 决策期 \`HORIZON=$HORIZON\`，评价期一律 \`auto\`（T=25 得 36，T=15 得 26）"
    echo "- 固定分母：\`--fixed-scale $FIXED\`，分母版本 \`R9\`（**不递增**：机会场的"
    echo "  幅度与形状参数全部入缓存键，既不会复用错误旧值，场全关的默认档还能"
    echo "  命中既有缓存）"
    echo "- 最后刷新：$(date '+%F %T %Z')　提交：\`$(git rev-parse --short HEAD)\`"
    echo
    echo "| 组 | 名称 | 状态 | 耗时 | 标签 |"
    echo "|---|---|---|---|---|"
    cat "$RES/.rows" 2>/dev/null
    echo
    if [ -f "$RES/.alldone" ]; then
      echo "## 全部实验已完成（${NGROUP}/${NGROUP}），已打标签 \`batch-v17-complete\`"
    else
      echo "> 尚未全部完成。已完成的组其结果即可用；重跑本脚本会自动跳过已完成的组。"
    fi
    echo
    echo "**动力学与口径自证**：每个 run 的 \`diag\` 记录**实际生效值**而非命令行"
    echo "意图。v17 新增 \`opp_active/a_plan_eff/a_infra_eff/a_age_eff/a_ready_eff/\`"
    echo "\`obs_opportunity_eff/foresight_eff/field_seed_eff/reward_shaping_eff\` 与"
    echo "三型实际单元数 \`n_kind_now/n_kind_ramp/n_kind_never\`，以及择时质量指标"
    echo "\`opp_at_init/opp_lift/ramp_post_onset/ramp_post_onset_null/timing_regret\`。"
    echo "其中 \`obs_opportunity_eff\` 与 \`reward_shaping_eff\` 尤其要看：这两个开关"
    echo "**刻意不进分母缓存键**（为的是与主组共用分母），因此光看键分不出这几组，"
    echo "只有这两个字段能证明消融/整形确实生效。"
    echo
    echo "**择时质量指标在 run 内算，不事后重建**：机会场依赖（单元数, row/col,"
    echo "场种子, 形状参数），事后重建一旦有一处不符就会给出看起来合理却错的数。"
    echo
    echo "**跨组比较的限制**：机会场幅度与形状、预算、制度参数、配额、审批时长、"
    echo "建设年限、资金口径、T/T_eval 都进分母缓存键，各组各有各的分母，"
    echo "**组间只能比原始量纲值与无量纲指标（CRH/LSR/ACD/择时质量/立项年份分布），"
    echo "不能比标量化回报**。可比标量回报的例外只有三对：第 1 组 vs 第 2 组"
    echo "（观测开关）、第 1 组 vs 第 6 组（每轮回合数）、第 1 组 vs 第 7 组"
    echo "（势函数整形）——这三个因子都不入缓存键。"
    echo
    echo "**口径声明**：四个机会场是**情景参数，无实证标定**（单元表没有建成年份、"
    echo "历年规划定位与设施投用年份）。任一幅度 > 0 时 Floor 不再是纯粹的计容建面，"
    echo "而是\"按时机加权的交付价值（以建面计量）\"。结论只能写成条件句。"
  } > "$STATUS"
}

# ---------------------------------------------------------------------------
echo "==================== TP-MORL 批次 v17 ===================="
echo "决策期 $HORIZON 年　结果目录 $RES　输出 $BASE"
echo "机会场主组幅度：$FIELD"
echo

if [ "$PUSH" = "1" ]; then
  echo "--- git pull"
  git pull --rebase --autostash origin main 2>&1 | tail -3 || echo "（pull 失败，用本地版本继续）"
fi

echo "--- 环境自检"
python3 - <<'PY' || { echo "环境自检未通过，已中止。"; exit 1; }
import importlib, os, sys
for m in ("numpy", "scipy", "pandas", "torch"):
    importlib.import_module(m)
for f in ("scripts/exp_opt_quality.py", "scripts/baselines.py",
          "scripts/eval_metrics.py", "scripts/exp_timing_gate.py",
          "scripts/verify_opportunity_neutral.py",
          "src/tpmorl/env/opportunity.py"):
    assert os.path.exists(f), f"缺文件 {f}"

from tpmorl.rl import env_gym as EG, scenario as SC
from tpmorl.env import schedule as S, opportunity as O
import numpy as np

# ---- v14/v15/v16 的口径修复必须仍在位（缺任何一样，整批会悄悄退回旧口径）----
assert hasattr(EG.RenewalEnv, "__init__") and "T_eval" in EG.RenewalEnv.__init__.__code__.co_varnames, \
    "T_eval 口径缺失：立项与记分分离未在位"
for fn in ("remaining_valid", "remaining_build", "expected_years_to_delivery"):
    assert hasattr(S.RenewalSchedule, fn), f"v15 生命周期查询缺 {fn}"
assert hasattr(EG, "OBS_LIFECYCLE"), "v15 生命周期观测开关缺失"

# ---- v17 专项 ----
assert EG.N_FEAT == 30, f"N_FEAT 应为 30（22 生命周期 + 8 机会场），实际 {EG.N_FEAT}"
assert EG.N_PAIR_FEAT == 30 + 12 + 2 + 3 + 1, f"配对特征宽度不符：{EG.N_PAIR_FEAT}"

# (a) 幅度全 0 + 整形 0 时，价值乘子与批准率调制必须是**精确的** 1.0
SC.reset()
assert not O.active(), "出厂默认应为机会场关闭"
assert O.tag() == "", f"场全关时分母键后缀应为空，实际 {O.tag()!r}"
f0 = O.OpportunityField(5, 15, T_total=26)
assert f0.value_mult(0, 0, 9) == 1.0 and f0.hazard_mult(3) == 1.0, "场关闭时不中性"
assert EG.REWARD_SHAPING == 0.0, "整形默认应为 0"

# (b) 四个幅度都必须入分母缓存键；观测开关、前瞻、整形强度都必须**不入**键
SC.reset(); SC.apply(horizon=25, horizon_eval="auto")
k0 = SC.inst_tag()
for kw in (dict(a_plan=0.8), dict(a_infra=0.6), dict(a_age=0.3), dict(a_ready=0.4),
           dict(field_seed=20260918, a_plan=0.8), dict(ramp_years=6, a_plan=0.8),
           dict(share_now=0.2, share_ramp=0.5, a_plan=0.8)):
    SC.reset(); SC.apply(horizon=25, horizon_eval="auto", **kw)
    assert SC.inst_tag() != k0, f"{kw} 未进分母缓存键——对照组会静默复用基线分母"
for kw in (dict(obs_opportunity=False), dict(foresight=3), dict(reward_shaping=1.0),
           dict(obs_lifecycle=False)):
    SC.reset(); SC.apply(horizon=25, horizon_eval="auto", **kw)
    assert SC.inst_tag() == k0, f"{kw} 进了分母缓存键——本该与主组共用分母"
keys = set()
for kw in (dict(a_plan=0.8), dict(a_infra=0.8), dict(a_age=0.8), dict(a_ready=0.8)):
    SC.reset(); SC.apply(horizon=25, horizon_eval="auto", **kw)
    keys.add(SC.inst_tag())
assert len(keys) == 4, f"四个幅度的缓存键发生碰撞：{sorted(keys)}"

# (c) 非法值必须当场拦住，而不是跑完才发现
for bad in (dict(a_plan=-0.1), dict(a_ready=1.5), dict(foresight=-1),
            dict(share_now=0.7, share_ramp=0.5), dict(share_ramp=0.0),
            dict(ramp_years=0), dict(onset_lo=0.8, onset_hi=0.2),
            dict(reward_shaping=-1)):
    SC.reset()
    try:
        SC.apply(horizon=25, horizon_eval="auto", **bad)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError(f"非法情景参数未被拦住：{bad}")

# (d) 评价期 auto 必须随 T 走：T=25 得 36、T=15 得 26
for T, want in ((25, 36), (15, 26)):
    SC.reset(); SC.apply(horizon=T, horizon_eval="auto")
    assert SC.horizon_eval() == want, f"T={T} 的 auto 评价期应为 {want}，实际 {SC.horizon_eval()}"

# (e) 三型构成必须都非空——"一直不好"型缺了的话，"一直等"就是最优解，
#     退化解会冒充择时能力
SC.reset(); SC.apply(horizon=25, horizon_eval="auto", a_plan=0.8)
f1 = O.OpportunityField(717, 25, T_total=36)
e = f1.eff()
assert min(e["n_kind_now"], e["n_kind_ramp"], e["n_kind_never"]) > 0, f"三型有空档：{e}"
# 场必须**真的**随时间变化，且不同单元不同步（否则退化成 FAR_GROWTH 那种全局同步）
assert f1.plan[:, 0].std() > 0 and np.abs(f1.plan[:, -1] - f1.plan[:, 0]).max() > 0.5, \
    "规划场没有逐年变化或逐单元差异，非平稳性没有真正建立"
assert f1.plan_onset.std() > 0.5, "爬升起始年几乎相同——机会场是全局同步的，等于没做"
print("  机会场：三型 %d/%d/%d，片区 %d，爬升起始年 std=%.2f"
      % (e["n_kind_now"], e["n_kind_ramp"], e["n_kind_never"],
         e["infra_clusters_eff"], f1.plan_onset.std()))
print("  分母键：场全关 %s；主组 " % k0, end="")
SC.reset(); SC.apply(horizon=25, horizon_eval="auto", a_plan=0.8, a_infra=0.6,
                     a_age=0.3, a_ready=0.4, foresight=3)
print(SC.inst_tag())
SC.reset()
print("  python", sys.version.split()[0], "| 依赖与数据齐备")
print("  N_FEAT=%d（22 生命周期 + 8 机会场）| 四幅度入键、观测/前瞻/整形不入键、无碰撞"
      % EG.N_FEAT)
PY

# 1b) 机会场关闭时**动态与 v16 逐位等价**的机械回归。
#     这是 v17 唯一能让历史结果不作废的凭据，故写成硬校验而不是文档里的一句话。
echo "--- 场关闭时与 v16 逐位等价"
if git rev-parse -q --verify batch-v16-complete >/dev/null; then
  rm -rf /tmp/v16_ref && mkdir -p /tmp/v16_ref
  git archive batch-v16-complete src | tar -x -C /tmp/v16_ref
  PYTHONPATH=/tmp/v16_ref/src python3 scripts/verify_opportunity_neutral.py --tag v16ref >/dev/null
  python3 scripts/verify_opportunity_neutral.py --tag v17cur >/dev/null
  python3 scripts/verify_opportunity_neutral.py --compare /tmp/neutral_v16ref.json /tmp/neutral_v17cur.json \
    || { echo "场关闭时改变了环境动态，已中止（历史结果的可比性是硬要求）。"; exit 1; }
else
  echo "（本地没有 batch-v16-complete 标签，跳过逐位回归；建议先 git fetch --tags）"
fi

# 1c) 资金记账恒等式回归：三条恒等式写死在脚本里，不成立即非零退出。
echo "--- 资金记账恒等式"
python3 scripts/verify_budget_identity.py \
  || { echo "资金记账恒等式不成立，已中止。"; exit 1; }

# 1d) 择时门槛实验：很快（分钟级），且它的结论决定这一批怎么读，故每次跑批
#     之前重跑一遍，把结果连同批次一起提交。
echo "--- 择时门槛实验（最小情景，最优解可穷举）"
python3 scripts/exp_timing_gate.py --out "$RES/gate" --label main_gate \
    --ramp-years 3.0 --foresight 3 --leads 1 --T 10 --onset 4 \
    --iters 300 --seeds 0 1 2 2>&1 | tail -6 | tee -a "$RES/quicklook.txt" \
  || echo "（门槛实验失败，不阻塞主批次）"


: > "$RES/.rows"
rm -f "$RES/.alldone"
FAILED=0; RAN=0; SKIP=0

for spec in "${SCENARIOS[@]}"; do
  IFS='|' read -r NUM DIR NAME ARGS GEPS <<< "$spec"
  EPSG="${GEPS:-$EPS}"        # 逐组可覆盖每轮回合数（见第 6 组）
  OUT="$BASE/$DIR"; LOG="$RES/g${NUM}_${DIR}.log"; TAG="v17-g${NUM}-${DIR}"

  if ! want_group "$NUM"; then
    printf '| %s | %s | 跳过（ONLY） | | |\n' "$NUM" "$NAME" >> "$RES/.rows"; continue
  fi
  # 断点续跑：已有完整结果就不重跑
  if [ "$FORCE" != "1" ] && [ -f "$OUT/objectives.csv" ]; then
    n=$(( $(wc -l < "$OUT/objectives.csv") - 1 ))
    if [ "$n" -ge "$RUNS_PER_GROUP" ]; then
      echo "===== 第 $NUM 组已完成（$n 条记录），跳过。要重跑请 FORCE=1"
      printf '| %s | %s | 已完成（续跑跳过） | | %s |\n' "$NUM" "$NAME" "\`$TAG\`" >> "$RES/.rows"
      RAN=$((RAN+1)); SKIP=$((SKIP+1)); continue
    fi
  fi

  echo; echo "===== 第 $NUM/$NGROUP 组：$NAME　($(date '+%F %T'))"
  echo "      参数：$ARGS      每轮回合 $EPSG      输出：$OUT"
  mkdir -p "$OUT"
  t0=$(date +%s)

  # 这里**不再**单独预建分母。v11/v12 的脚本在这一步跑 `python3 -m tpmorl.rl.scale`，
  # 但启用 --fixed-scale 时 worker 实际读的是 `fixed_*.json`，而该命令建的是
  # `ref_*.csv`——预建的和用的不是同一个文件，fixed_ 仍留给 35 个 worker 并发去建，
  # 这正是动力学污染的触发条件；日志里打印的又是 ref_ 路径，把问题盖住了一轮。
  # 现在由 exp_opt_quality.main() 在起进程池之前预建**实际会被读的那个**文件。
  echo "### 主扫描  $(date '+%F %T')" > "$LOG"

  # 主扫描。不要用 grep/tee 过滤 stdout，进度输出会被管道缓冲住看不到。
  # --growth 不在这里写死：各组自己的 $ARGS 决定（本批次全为默认 0.0）。
  python3 scripts/exp_opt_quality.py --dataset "$DS" --out "$OUT" \
      --iters "$ITERS" --eps "$EPSG" --workers "$WORKERS" \
      --fixed-scale "$FIXED" $ARGS >> "$LOG" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 )); hm="$(( dt/3600 ))h$(( (dt%3600)/60 ))m"

  # 规则基线：同一组情景参数下重跑一遍手写规则策略与松上界。
  # 很快（无训练），且放在每组之后落盘，这样即便后面的组失败，本组的
  # 「策略 vs 基线」也已经齐了。$ARGS 直接复用——baselines.py 接受同一套场景参数，
  # 这样基线与训练**不可能**跑在不同的情景上（各写一份参数就会有这种风险）。
  if [ "$NOBASE" != "1" ] && [ $rc -eq 0 ]; then
    if [ "$FORCE" = "1" ] || [ ! -f "$OUT/baseline/baselines.csv" ]; then
      echo "### 规则基线  $(date '+%F %T')" >> "$LOG"
      python3 scripts/baselines.py --dataset "$DS" --out "$OUT/baseline" \
          $ARGS >> "$LOG" 2>&1 || echo "!!! 第 $NUM 组基线失败（不影响主扫描结果）"
    fi
  fi

  if [ $rc -eq 0 ] && [ -f "$OUT/objectives.csv" ]; then
    ST="完成"; RAN=$((RAN+1)); TAGCELL="\`$TAG\`"
  else
    ST="**失败**（退出码 ${rc}，见 \`$LOG\`）"; FAILED=$((FAILED+1)); TAGCELL="—"
    echo "!!! 第 $NUM 组失败，退出码 ${rc}；继续下一组"
    tail -15 "$LOG"
  fi
  printf '| %s | %s | %s | %s | %s |\n' "$NUM" "$NAME" "$ST" "$hm" "$TAGCELL" >> "$RES/.rows"
  write_status
  push_now "批次 v17 第 $NUM/$NGROUP 组：${NAME}（${ST}，耗时 ${hm}）"
  [ "$ST" = "完成" ] && tag_now "$TAG" "v17 第 $NUM 组 ${NAME}：${ITERS} iters × ${EPSG} eps × ${RUNS_PER_GROUP} 次，耗时 ${hm}"
done

# ---------------------------------------------------------------------------
# 全部跑完后直接出主判据，不用等人回来手工算
if [ $FAILED -eq 0 ] && [ $RAN -eq $NGROUP ]; then
  echo; echo "--- 全部完成，重算实施类指标"
  python3 scripts/eval_metrics.py --batch "$BASE" --out "$RES/metrics" \
      2>&1 | tail -20 | tee -a "$RES/quicklook.txt"

  echo; echo "--- 主判据：择时是否真的学到了"
  python3 - "$BASE" "$RES" <<'PY' 2>&1 | tee -a "$RES/quicklook.txt"
import sys, os, json, glob
import numpy as np, pandas as pd
from scipy import stats
BASE, RES = sys.argv[1], sys.argv[2]

def diag_of(g, key):
    """从 runs.json 读**生效值**——不看组名，也不看缓存键。

    观测开关与整形强度刻意不入分母缓存键，故光看键分不出这几组，
    只有 diag 里的生效值能证明消融/整形确实生效。
    """
    p = os.path.join(BASE, g, "runs.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    vs = {r.get("diag", {}).get(key) for r in d.get("runs", [])}
    vs = {v for v in vs if v is not None}
    if len(vs) == 1:
        return vs.pop()
    return "混杂:" + str(sorted(map(str, vs))) if vs else None

def tq_of(g):
    """该组的择时质量指标（逐 run 记在 diag 里，这里取均值与标准差）。"""
    p = os.path.join(BASE, g, "runs.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    rows = [r.get("diag", {}) for r in d.get("runs", [])]
    ks = ("opp_at_init", "opp_lift", "ramp_post_onset",
          "ramp_post_onset_null", "timing_regret")
    out = {}
    for k in ks:
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))
             and r[k] == r[k]]
        out[k] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                  len(v))
    err = [r.get("timing_quality_error") for r in rows if r.get("timing_quality_error")]
    if err:
        out["_error"] = err[0]
    return out

groups = sorted(os.path.basename(p) for p in glob.glob(os.path.join(BASE, "*"))
                if os.path.isdir(p))

# ---- 0) 各组生效值一览。先证明消融真的生效，再谈判据 ----
print("各组生效值（须与组名一致）：")
print("  %-10s %8s %8s %8s %8s %6s %8s" %
      ("组", "场开", "观测", "前瞻", "整形", "T", "T_eval"))
for g in groups:
    print("  %-10s %8s %8s %8s %8s %6s %8s" %
          (g, diag_of(g, "opp_active"), diag_of(g, "obs_opportunity_eff"),
           diag_of(g, "foresight_eff"), diag_of(g, "reward_shaping_eff"),
           diag_of(g, "t_dec_eff"), diag_of(g, "t_eval_eff")))

# ---- 1) 择时质量总表 ----
print("\n择时质量（均值 ± 标准差，n 为有效 run 数）：")
print("  %-10s %16s %16s %16s %16s" %
      ("组", "立项时机会场", "相对随机提升", "会变好型在爬升后", "择时后悔（松下界）"))
for g in groups:
    t = tq_of(g)
    if not t:
        continue
    if "_error" in t:
        print("  %-10s 指标计算出错：%s" % (g, t["_error"]))
        continue
    def f(k):
        m, s, n = t[k]
        return "—" if n == 0 else f"{m:.4f}±{s:.4f}"
    post, null = t["ramp_post_onset"][0], t["ramp_post_onset_null"][0]
    print("  %-10s %16s %16s %16s %16s" %
          (g, f("opp_at_init"), f("opp_lift"),
           ("—" if post != post else f"{post:.3f}(零假设{null:.3f})"),
           f("timing_regret")))

# ---- 2) 立项年份分布：是否仍与均摊无法区分 ----
#      这是 v16 栽的那一条，所以放在最显眼的位置。
print("\n立项年份分布（v16 就是这一条不成立）：")
print("  %-10s %8s %10s %10s %12s" % ("组", "T", "立项重心", "均摊基准", "卡方 p"))
for g in groups:
    fs = sorted(glob.glob(os.path.join(BASE, g, "rec_*.csv")))
    if not fs:
        continue
    R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    R = R[R["unit"] >= 0]
    T = diag_of(g, "t_dec_eff")
    T = int(T) if isinstance(T, (int, float)) else int(R["year"].max()) + 1
    cnt = np.bincount(R["year"].astype(int), minlength=T)[:T]
    exp = np.full(T, cnt.sum() / T)
    p = stats.chisquare(cnt, exp).pvalue if cnt.sum() >= T else float("nan")
    print("  %-10s %8d %10.2f %10.2f %12.3g" %
          (g, T, float((np.arange(T) * cnt).sum() / max(cnt.sum(), 1)),
           (T - 1) / 2.0, p))

# ---- 3) H1–H4 的逐条判定。**任一条不成立都照实打印"不成立"** ----
def cmp_pair(a, b, key, better="high"):
    """两组在某个择时指标上的差。配对不了就返回 None（不硬凑）。"""
    ta, tb = tq_of(a), tq_of(b)
    if not ta or not tb:
        return None
    ma, mb = ta[key][0], tb[key][0]
    if ma != ma or mb != mb:
        return None
    d = ma - mb
    ok = d > 0 if better == "high" else d < 0
    return ma, mb, d, ok

print("\n主判据逐条判定：")
HYP = [("H1 机会场是载体", "main", "v16base"),
       ("H2 信息必须看得见", "main", "noobs"),
       ("H3 趋势是关键", "main", "fs0"),
       ("H4 延长决策期有用", "main", "T15")]
for name, a, b in HYP:
    line = [name + f"（{a} vs {b}）"]
    for key, better in (("opp_lift", "high"), ("ramp_post_onset", "high"),
                        ("timing_regret", "low")):
        r = cmp_pair(a, b, key, better)
        if r is None:
            line.append(f"{key}=无法配对")
        else:
            ma, mb, d, ok = r
            line.append(f"{key}: {ma:.4f} vs {mb:.4f}（差 {d:+.4f}，"
                        f"{'符合' if ok else '**不符合**'}）")
    print("  " + "\n      ".join(line))

# ---- 4) 可直接相减标量回报的三对（这三个因子都不入分母缓存键）----
mp = os.path.join(RES, "metrics", "实施类指标_逐运行.csv")
if os.path.exists(mp):
    R = pd.read_csv(mp, encoding="utf-8-sig")
    R.columns = [c.lstrip("\ufeff") for c in R.columns]
    for a, b, what in (("main", "noobs", "机会场观测"),
                       ("main", "eps32", "每轮回合数"),
                       ("main", "shaping", "势函数整形")):
        if not {a, b} <= set(R["组"].unique()):
            continue
        x = R[R["组"] == a].set_index(["alpha", "seed"])
        y = R[R["组"] == b].set_index(["alpha", "seed"])
        idx = x.index.intersection(y.index)
        if len(idx) < 3 or "scalar_return" not in R.columns:
            continue
        dx = x.loc[idx, "scalar_return"] - y.loc[idx, "scalar_return"]
        t, p = stats.ttest_rel(x.loc[idx, "scalar_return"], y.loc[idx, "scalar_return"])
        print(f"\n{what}（{a} vs {b}，共用分母故可相减）：配对 {len(idx)} 对，"
              f"差均值 {dx.mean():+.4f}，配对 t 检验 p={p:.3g}")
else:
    print("\n（未找到实施类指标逐运行表，跳过标量回报对照）")

print("\n提醒：以上判定只回答\"这一批的设定下择时结构有没有出现\"。四个机会场是"
      "**情景参数、无实证标定**，故任何成立的结论都只能写成条件句；"
      "任何不成立的结论都必须照实写进正文，不能放附录。")
PY

  touch "$RES/.alldone"
  write_status
  push_now "批次 v17 全部 ${NGROUP} 组完成：指标重算 + 主判据（择时质量 / 立项年份分布 / H1–H4）"
  tag_now "batch-v17-complete" "v17 全部 ${NGROUP} 组完成：机会场 + 决策期 ${HORIZON} 年 + 择时质量指标"
  echo; echo "===== 全部完成。主判据见 $RES/quicklook.txt，状态见 $STATUS"
else
  write_status
  push_now "批次 v17 部分完成：成功 ${RAN}/${NGROUP}，失败 ${FAILED}"
  echo; echo "===== 有 ${FAILED} 组失败。修好后直接重跑本脚本，已完成的组会自动跳过。"
  exit 1
fi
