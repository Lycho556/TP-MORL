#!/usr/bin/env bash
# ============================================================================
#  TP-MORL 批次 v16 —— 一条命令跑完全部实验
#
#  在服务器上只需要敲这一句：
#
#      bash ~/TP-MORL/启动实验_v16.sh
#
#  想挂后台断开 ssh 也不中断：
#
#      nohup bash ~/TP-MORL/启动实验_v16.sh > ~/v16.log 2>&1 &
#      tail -f ~/v16.log
#
#  脚本自己会做：git pull → 环境自检 → 逐组训练 → 每组跑完顺手算规则基线
#  → commit+push+打标签 → 断点续跑（已完成的组自动跳过）→ 全部完成后重算
#  实施类指标、出主判据、打标签 batch-v16-complete。
#
#  ---------------------------------------------------------------------------
#  v15 为什么存在
#
#  v14 修好了时间口径（决策期 15 年 / 评价期 26 年分离），奖励里第一次有了
#  时序梯度。但 v14 的观测**没有告诉策略它需要什么才能用上这个梯度**：
#  策略要判断"现在立项还赶得上在期内建成吗"，而 v14 的观测里只有
#      F[:,12] 当前状态内**已用**比例（已用，不是剩余）
#      F[:,14] 全局 t/T（全局，无法区分建设年限不同的单元）
#  两者都不足以回答那个问题。于是"大量在期望上已来不及的时点立项"这一现象，
#  在 v14 里既无法被策略避免，也无法被归因——策略不是不想早动，是看不见。
#
#  v15 做了四件事（逐条的取证与判定见 docs/建议条目审计与改动清单_v15.md）：
#
#      (1) **观测补齐生命周期剩余量**。新增 6 维逐单元特征（16..21）：
#          剩余有效年、剩余建设年、期望交付年数、slack、期内可交付标志、
#          在建管道占比。其中 slack = (T−1−t) − E[交付所需年数]，是"现在立项
#          还赶得上吗"的直接答案。N_FEAT 16 → 22。
#          这些量由状态机解析给出（`remaining_valid` / `remaining_build` /
#          `expected_years_to_delivery`），**不偷看本回合的随机实现**——期望
#          口径按 hazard 反向递推，与 step() 的真实转移顺序对齐。
#
#      (2) **承诺预算与可用预算分离**，并新增 staged 资金口径。旧口径
#          upfront 下立项当年全额扣款，且**失效单元花掉了预算却从不进入 Cost
#          目标**（实测差额 4900.0/13480.0，即 36%，且逐位等于未完工单元成本）。
#          staged 下立项付 20% 前期款、余额在实施期内按年等额支付，Cost 改按实际
#          支付年计入，于是 Cost 总额 ≡ 支付总额（实测 10982.0 逐位相等）。
#          以上数字的固定复现口径见 scripts/verify_budget_identity.py（种子 7），
#          该脚本把三条恒等式写成硬校验，可当回归测试跑。
#          upfront 仍是默认且与 v14 **逐位等价**（已用 git archive 取旧包对跑
#          验证），故 v14 的结果不作废。
#
#      (3) **场景参数扩展**：年度配额 `--quota`、平均审批时长 `--tau-approval`、
#          逐通道建设年限 `--build-years-by-channel`、资金口径 `--budget-mode`。
#          四者都改变可达上界，**都已入分母缓存键**（已逐个回归，见下）。
#
#      (4) **评价基线与实施类指标**。v14 之前只有标量化回报，它把"更新哪里"
#          与"什么时候"混在一个数里。v15 新增
#          规则基线族（myopic / greedy_crh / frontload / evenspread / backload）
#          与一个**松上界** oracle，以及三层评价指标（标量层 / 实施层 / 制度层），
#          核心实施层指标是 CRH（期内交付率）、LSR（窗外立项率）、ACD（决策到
#          交付时长）。见 docs/基线与上界_v15.md、docs/实施类指标_v15.md。
#
#  ---------------------------------------------------------------------------
#  v15 要回答什么：主判据
#
#      主判据（本批次成立与否看这一条）：
#          **补齐生命周期观测之后，窗外立项率 LSR 下降、期内交付率 CRH 上升。**
#      判据的对照是**第 2 组（消融组）**：它与第 1 组除"那 6 维特征被置零"
#      之外参数逐字相同、种子相同、分母共用，于是两组之差可以干净地归因到
#      这段信息本身。
#
#      为什么用批次内消融而不是拿 v14 比：v15 与 v14 在默认口径下动力学逐位
#      等价、分母缓存键逐字相同，原则上可比；但跨批次比对要连带解释代码、
#      随机数、依赖版本的全部差异，而批次内消融只差一个布尔量。后者可辩护得多。
#      消融是**置零而非删列**：位宽、网络形状、参数量、优化器状态全都不变。
#      （实测：两组观测的差异恰为第 16..21 列，0..15 列逐位相同。）
#
#      消融开关**刻意不进分母缓存键**。参考策略是手写规则、不读观测，两组的
#      可达上界按构造完全相同；共用同一套分母，两组的标量化回报才能直接相减。
#      若误入键，消融组会另建一套数值相同但路径不同的分母，白跑一遍还会让人
#      误以为两组不可比。实测两组键均为 `V3E2DAY1-52-53-55-5T15X26`。
#      代价是：**光看缓存键分不出消融组**，所以每个 run 的 diag 里记了
#      `obs_lifecycle_eff`——这是唯一能证明消融确实生效的字段，看这个，别看键。
#
#      次判据：立项年份分布是否继续前移（均年 < 均匀期望 7.0，卡方拒绝均匀）；
#      策略的 CRH 能否超过 myopic（规则基线里最强的那个），与 oracle 的差距
#      有多大。注意 oracle 是**松**上界（只去掉审批不确定性这一个维度），
#      它与策略的差距不能读作"策略还差多少"。
#
#  ---------------------------------------------------------------------------
#  分母版本仍是 R9，**不递增**，这是有意的
#
#      规矩是"凡改动会改变可达上界、又不改变缓存键的东西，必须递增版本号"。
#      v15 新增的四个场景参数**全部入键**（逐个回归过：默认档键与改动前逐字
#      相同；四个参数各自产生互不相同的键；无一被静默漏掉）。因此
#        · 默认档各组会命中 v14 已建好的分母缓存，且那些值仍然正确（动力学
#          逐位等价），省掉重建时间；
#        · 新场景各组是全新的键，必然新建，不存在复用旧值的风险。
#      两种情形都不需要递增版本号。消融开关不改可达上界，同理不需要。
#
#  ---------------------------------------------------------------------------
#  跨组比较的限制（照例，写在这里免得事后误读）
#
#      budget / carry / 制度参数 / T / T_eval / 配额 / 审批时长 / 建设年限 /
#      资金口径 都进分母缓存键，所以这些组各有各的分母，**组间只能比
#      objectives.csv 的原始量纲值与无量纲指标（CRH / LSR / ACD / 立项年份
#      分布），不能比标量化回报**。
#      唯一的例外是**第 1 组与第 2 组**：分母共用，标量化回报可直接相减。
#      主判据本来就用的是无量纲指标，不受此限。
#
#      单组约 37 分钟，16 组合计约 10 小时。可分两阶段：
#          PHASE=1  只跑第 1–8 组（主组 + 消融 + 资金口径 + 预算扫描，约 5 小时）
#          PHASE=2  只跑第 9–16 组（制度扫描，约 5 小时）
#      不设 PHASE 则依次跑完全部 16 组。
#      **第 1、2 组是主判据，任何情况下都要先跑**（PHASE=1 已包含）。
#
#  ---------------------------------------------------------------------------
#  可调环境变量（都有合理默认，一般不用动）
#      WORKERS=64      并行度，默认 = 核数-2，上限 35
#      ITERS=400       每次运行的训练轮数
#      EPS=8           每轮的 episode 数
#      PUSH=0          不推 GitHub（本地调试用）
#      ONLY="1 2"      只跑指定组号
#      FORCE=1         忽略已完成标记，全部重跑
#      NOBASE=1        跳过每组的规则基线（基线很快，一般不用跳）
#  ---------------------------------------------------------------------------
#  正式跑之前建议先干跑三分钟验管线（不写正式目录、不推送）：
#      PUSH=0 ITERS=2 EPS=1 BASE=/tmp/dry15 RES=/tmp/dry15res \
#          bash ~/TP-MORL/启动实验_v16.sh
# ============================================================================
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || { echo "找不到仓库目录 $ROOT"; exit 1; }

export PYTHONPATH=src
export OMP_NUM_THREADS=1      # 必须：否则 torch 线程与进程池争核，实测慢数倍

DS="data/processed/gm_dataset_v1"
BASE="${BASE:-$DS/exp_v16}"
RES="${RES:-results_v16}"
PUSH="${PUSH:-1}"
ITERS="${ITERS:-400}"
EPS="${EPS:-8}"
FORCE="${FORCE:-0}"
NOBASE="${NOBASE:-0}"
FIXED="${FIXED:-0,0.1}"       # 固定分母覆盖的增长率集合，与 v11/v13/v14 一致
RUNS_PER_GROUP=35             # 7 权重档 × 5 种子；并行度超过它没有收益
NPROC="$( (OMP_NUM_THREADS= nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 10) )"
WORKERS="${WORKERS:-$(( NPROC > 4 ? NPROC - 2 : 2 ))}"
[ "$WORKERS" -gt "$RUNS_PER_GROUP" ] && WORKERS="$RUNS_PER_GROUP"

# ---------------------------------------------------------------------------
# 实验组：编号 | 目录 | 中文名 | 额外参数
#
# 1–2  主判据：主组与消融组。**两组除 --no-obs-lifecycle 外逐字相同**，
#      这是本批次唯一一对可以直接相减标量化回报的组，别动它们的参数。
# 3    资金口径 staged（Cost 按支付年计入，Cost ≡ 支付总额）。
# 4–7  预算维度：结转上限与年度总额，与 v14 同参数便于纵向看。
# 8–10 平均审批时长 τ_A 扫描。给出后逐年条件批准率整体换成常数风险率 1/τ_A，
#      是**对照而非更精确的估计**（默认档的实测向量才是标定值）。
# 11–12 年度配额扫描。
# 13–14 建设年限：全通道同值 3 年，与逐通道分档。
#      **分档无实证依据**（单元表没有更新方式字段），只能作情景参数报告。
# 15   法定窗口 tau_valid=2 tau_ext=1。
# 16   规划期 20 年。
#
# `--horizon-eval auto` 写在**每组的参数里**而不是全局命令行，是为了让每组
# 解析出的评价期一眼可审。auto = T + tau_valid + tau_ext + max(建设年限) + 1，
# 故各组不同：主组 26，build3 与 statutory 24，horizon20 31。
# **不能全批写死一个常数**——写死会把某些组的管道截断在期内，那正是 v14 修的错误。
# ---------------------------------------------------------------------------
SCENARIOS=(
  "1|main|主组：年度总额 2700（瓶颈交给配额）+ 生命周期观测全开|--budget 2700 --carry 3 --horizon-eval auto"
  "2|ablation|消融：同主组但生命周期 6 维特征置零（共用分母）|--budget 2700 --carry 3 --horizon-eval auto --no-obs-lifecycle"
  "3|poor|旧财力锚：年度总额 900 + 观测全开（= v15 主组，供 2x2 交互）|--budget 900 --carry 3 --horizon-eval auto"
  "4|poorAbl|旧财力锚的消融：900 + 观测置零（= v15 消融组）|--budget 900 --carry 3 --horizon-eval auto --no-obs-lifecycle"
  "5|richQ6|钱松且配额松：2700 + 配额 6 + 观测全开（窗内占用率仍有余量的一档）|--budget 2700 --carry 3 --horizon-eval auto --quota 6"
  "6|richQ6Abl|钱松且配额松的消融：2700 + 配额 6 + 观测置零|--budget 2700 --carry 3 --horizon-eval auto --quota 6 --no-obs-lifecycle"
  "7|bytype|拆除基数按现状用地类别（均值保持）+ 观测全开|--budget 2700 --carry 3 --horizon-eval auto --cell-cost-mode bytype"
  "8|bytypeAbl|拆除基数按类别 + 观测置零（隔离成本结构与信息的交互）|--budget 2700 --carry 3 --horizon-eval auto --cell-cost-mode bytype --no-obs-lifecycle"
  "9|budget1500|年度总额 1500（资金—占用率曲线的中档）|--budget 1500 --carry 3 --horizon-eval auto"
  "10|quota2|年度配额 2（更紧）|--budget 2700 --carry 3 --horizon-eval auto --quota 2"
  "11|tauA1|平均审批 1 年（最快）|--budget 2700 --carry 3 --horizon-eval auto --tau-approval 1"
  "12|tauA5|平均审批 5 年（最慢）|--budget 2700 --carry 3 --horizon-eval auto --tau-approval 5"
  "13|staged|资金口径 staged：立项付 20pct，Cost 按支付年计入|--budget 2700 --carry 3 --horizon-eval auto --budget-mode staged --stage-init 0.2"
  "14|carry1|结转上限 1 年（最紧）|--budget 2700 --carry 1 --horizon-eval auto"
  "15|buildmix|建设年限逐通道分档（情景参数，无实证依据）|--budget 2700 --carry 3 --horizon-eval auto --build-years-by-channel 1:5,2:3,3:5,5:3"
  "16|statutory|法定窗口 tau_valid=2 tau_ext=1|--budget 2700 --carry 3 --horizon-eval auto --tau-valid 2 --tau-ext 1"
)
NGROUP=${#SCENARIOS[@]}

# PHASE 是 ONLY 的便捷写法：1 = 主判据+资金+预算(1-8)，2 = 制度扫描(9-16)。
PHASE="${PHASE:-}"
case "$PHASE" in
  1) ONLY="${ONLY:-1 2 3 4 5 6 7 8}" ;;
  2) ONLY="${ONLY:-9 10 11 12 13 14 15 16}" ;;
  "") : ;;
  *) echo "PHASE 只能是 1 或 2（当前 '$PHASE'）"; exit 2 ;;
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
    echo "# 批次 v16 运行状态"
    echo
    echo "**这批实验回答的问题**：v14 修好时间口径后奖励里有了时序梯度，但观测"
    echo "没有告诉策略\"现在立项还赶得上吗\"。v15 给观测补上生命周期剩余量与 slack"
    echo "（6 维逐单元特征），并新增规则基线与实施类指标。"
    echo
    echo "**主判据**：补齐观测后**窗外立项率 LSR 下降、期内交付率 CRH 上升**。"
    echo "对照是第 2 组消融组——与第 1 组除那 6 维特征被置零外参数逐字相同、"
    echo "种子相同、**分母共用**，故两组之差可干净归因到这段信息本身。"
    echo "消融是置零而非删列，位宽与网络形状不变。"
    echo
    echo "- 主机并行度：\`WORKERS=$WORKERS\`（探测到 $NPROC 核）"
    echo "- 每组规模：7 权重档 × 5 种子 = 35 次运行，\`--iters $ITERS --eps $EPS\`"
    echo "- 固定分母：\`--fixed-scale $FIXED\`，分母版本 \`R9\`（**不递增**：v15 的"
    echo "  四个新场景参数全部入缓存键，既不会复用错误旧值，默认档还能命中 v14"
    echo "  已建好且仍然正确的缓存）"
    echo "- 最后刷新：$(date '+%F %T %Z')　提交：\`$(git rev-parse --short HEAD)\`"
    echo
    echo "| 组 | 名称 | 状态 | 耗时 | 标签 |"
    echo "|---|---|---|---|---|"
    cat "$RES/.rows" 2>/dev/null
    echo
    if [ -f "$RES/.alldone" ]; then
      echo "## 全部实验已完成（${NGROUP}/${NGROUP}），已打标签 \`batch-v16-complete\`"
    else
      echo "> 尚未全部完成。已完成的组其结果即可用；重跑本脚本会自动跳过已完成的组。"
    fi
    echo
    echo "**动力学与口径自证**：每个 run 的 \`diag\` 记录**实际生效值**而非命令行"
    echo "意图——\`far_growth_eff/budget_eff/carry_eff\`、\`t_dec_eff/t_eval_eff\`，"
    echo "v15 另加 \`obs_lifecycle_eff/budget_mode_eff/stage_init_eff/quota_eff/\`"
    echo "\`tau_valid_eff/tau_ext_eff/hazard_eff/build_years_eff/gamma_eff/inst_tag_eff\`。"
    echo "其中 \`obs_lifecycle_eff\` 尤其要看：消融开关**刻意不进分母缓存键**"
    echo "（为的是让消融组与主组共用分母），因此光看键分不出消融组，"
    echo "只有这个字段能证明消融确实生效。"
    echo "\`one_run\` 在建完分母后立即断言动力学与入参一致，\`run_episode\` 在回合"
    echo "结束时断言管道已排空（S1/S2/S3 为 0），任一不符当场失败。"
    echo
    echo "**跨组比较的限制**：预算/制度参数/配额/审批时长/建设年限/资金口径/T/T_eval"
    echo "都进分母缓存键，各组各有各的分母，**组间只能比原始量纲值与无量纲指标"
    echo "（CRH/LSR/ACD/立项年份分布），不能比标量化回报**。"
    echo "**唯一例外是第 1 组与第 2 组**（分母共用，可直接相减）。"
    echo "主判据用的就是无量纲指标，不受此限。"
  } > "$STATUS"
}

# ---------------------------------------------------------------------------
echo "==================== TP-MORL 批次 v16 ===================="
echo "仓库：$ROOT"
echo "开始：$(date '+%F %T')   WORKERS=$WORKERS   ITERS=$ITERS   EPS=$EPS   PUSH=$PUSH"

# 0) 先同步代码，避免跑的是旧版
if [ "$PUSH" = "1" ]; then
  echo "--- git pull"
  git pull --rebase --autostash origin main 2>&1 | tail -3 || echo "（pull 失败，用本地版本继续）"
fi

# 1) 环境自检：宁可现在报错，也不要跑到第 5 小时才发现缺东西
echo "--- 环境自检"
python3 - <<'PY' || { echo "环境自检未通过，已中止。"; exit 1; }
import importlib, sys, os
for m in ("numpy", "scipy", "pandas", "torch"):   # 本仓库的环境是自写的，不依赖 gym
    importlib.import_module(m)
import tpmorl.rl.train_ppo, tpmorl.rl.scale, tpmorl.objectives.uis  # noqa
ds = "data/processed/gm_dataset_v1"
need = ["grid_100m/L0_class.npy", "grid_100m/action_mask.npy",
        "tables/UUM.csv", "tables/CM.csv", "tables/CCM.csv",
        "tables/gm_renewal_units.csv"]
miss = [f for f in need if not os.path.exists(os.path.join(ds, f))]
if miss:
    print("缺少数据文件：", miss); sys.exit(1)
for f in ("scripts/exp_opt_quality.py", "scripts/baselines.py", "scripts/eval_metrics.py"):
    if not os.path.exists(f):
        print("缺少脚本：", f); sys.exit(1)

# ---- v14 的口径修复必须仍在位（缺任何一样，整批会悄悄退回旧口径）----
from tpmorl.rl import scenario, scale
from tpmorl.rl import env_gym as EG
from tpmorl.env import schedule as S
from tpmorl.rl.env_gym import RenewalEnv
assert hasattr(scenario, "auto_horizon_eval"), \
    "scenario 缺 auto_horizon_eval：代码是旧版，--horizon-eval auto 不会生效"
assert scale.REF_VER == "R9", f"分母版本是 {scale.REF_VER}，应为 R9"
assert "T_eval" in RenewalEnv.__init__.__code__.co_varnames, \
    "RenewalEnv 不接受 T_eval：代码是旧版"

# ---- v15 专项 ----
# (a) 观测位宽与消融开关
assert EG.N_FEAT == 22, f"N_FEAT={EG.N_FEAT}，应为 22（v15 观测未在位）"
assert EG.OBS_LIFECYCLE is True, "OBS_LIFECYCLE 出厂值应为 True"
# (b) 状态机的剩余时间查询
for fn in ("remaining_valid", "remaining_build", "expected_years_to_delivery"):
    assert hasattr(S.RenewalSchedule, fn), f"状态机缺 {fn}：v15 观测无法构造"
# (c) 消融组与主组**必须共用分母缓存键**——这是主判据可比的前提
scenario.reset(); scenario.apply(budget=900, carry=3, growth=0.0,
                                 horizon=15, horizon_eval="auto")
k_main, te_main = scenario.inst_tag(), scenario.horizon_eval()
scenario.reset(); scenario.apply(budget=900, carry=3, growth=0.0,
                                 horizon=15, horizon_eval="auto", obs_lifecycle=False)
k_abl = scenario.inst_tag()
assert k_main == k_abl, \
    f"消融组的分母键 {k_abl} 与主组 {k_main} 不同——消融进了缓存键，两组会各建一套分母"
assert EG.OBS_LIFECYCLE is False, "obs_lifecycle=False 未生效"
scenario.reset()
assert EG.OBS_LIFECYCLE is True, "reset() 未复原 OBS_LIFECYCLE：会发生静默继承"
# (d) 四个新场景参数必须各自入键、互不相撞、且不被静默漏掉
keys = {}
for name, kw in (("默认", {}),
                 ("配额6", dict(quota=6)),
                 ("审批3", dict(tau_approval=3)),
                 ("分档", dict(build_years_by_channel="1:5,2:3,3:5,5:3")),
                 ("staged", dict(budget_mode="staged", stage_init=0.2))):
    scenario.reset()
    scenario.apply(budget=900, carry=3, growth=0.0, horizon=15,
                   horizon_eval="auto", **kw)
    keys[name] = scenario.inst_tag()
scenario.reset()
assert keys["默认"] == k_main, "默认档的键与改动前不一致"
if len(set(keys.values())) != len(keys):
    print("分母缓存键发生碰撞：", keys); sys.exit(1)
# (e) auto 解析：各组不同，不能写死常数
def _auto(**kw):
    scenario.reset()
    scenario.apply(budget=900, carry=3, growth=0.0, horizon=kw.pop("horizon", 15),
                   horizon_eval="auto", **kw)
    v = scenario.horizon_eval(); scenario.reset(); return v
assert te_main == 26, f"主组 auto 解析成 {te_main}，应为 26"
assert _auto(horizon=20) == 31, "horizon20 组 auto 应为 31"
assert _auto(build_years=3) == 24, "build3 组 auto 应为 24"
assert _auto(tau_valid=2, tau_ext=1) == 24, "statutory 组 auto 应为 24"
# (f) 资金口径默认值必须仍是 upfront（否则默认档与 v14 不再等价）
assert EG.BUDGET_MODE == "upfront", \
    f"BUDGET_MODE 出厂值是 {EG.BUDGET_MODE}，应为 upfront（默认档须与 v14 逐位等价）"
# (g) 配额必须真正到达采样器。这一条是实测缺陷的回归：QUOTA 曾被 from-import 按值
#     快照，于是 --quota 2 在建分母阶段被断言打断、--quota 6 静默按 3 跑；而诊断字段
#     读的是活值，所以**日志是对的、行为是错的**，查日志查不出来，必须查使用点。
_qmax = []
for q in (2, 3, 6):
    scenario.reset()
    scenario.apply(budget=900, carry=3, growth=0.0, horizon=15,
                   horizon_eval="auto", quota=(None if q == 3 else q))
    e = RenewalEnv(ds, T=15, T_eval=scenario.horizon_eval(),
                   weights=__import__("numpy").ones(11) / 11,
                   scale=__import__("numpy").ones(11), seed=0)
    e.reset(seed=0)
    assert int(e.quota) == q, f"--quota {q} 未到达环境（env.quota={e.quota}）"
    assert int(e.env.quota) == q, f"--quota {q} 未到达状态机"
    # 光断言属性相等还不够——要的是**实际行为**受配额约束。用贪心可付性策略跑
    # 满决策期，逐年实际选中数必须不超过配额。这一条同时能抓住"放宽配额却静默
    # 按出厂值跑"（那种情形下单年最大选中数会卡在 3）。
    import numpy as _np
    mx = 0
    for _ in range(15):
        _X, _meta, _cost, _u = e.pairs()
        _left, _used, _acts = e.budget, set(), []
        for _i in range(len(_meta)):
            _uu = _meta[_i][0]
            if _uu < 0 or _uu in _used or _cost[_i] > _left + 1e-6:
                continue
            _acts.append(_meta[_i]); _used.add(_uu); _left -= _cost[_i]
            if len(_acts) >= q:
                break
        mx = max(mx, len(_acts))
        e.step(_acts)
    assert mx <= q, f"单年实际选中 {mx} 个 > 配额 {q}：配额未真正约束行为"
    if q > 3:
        assert mx > 3, f"--quota {q} 下单年最大选中数仍为 {mx}——疑似静默按出厂值 3 跑"
    _qmax.append((q, mx))
scenario.reset()
# (h) 情景参数的取值校验必须在位，否则手误（配额 0、建设年限 0/负/超大）会跑完
#     才发现指标全空；建设年限超大还曾因 clock 位宽而静默永不完工。
for bad in (dict(quota=0), dict(quota=-1), dict(build_years=0), dict(build_years=-1),
            dict(build_years=1000)):
    scenario.reset()
    try:
        scenario.apply(budget=900, carry=3, growth=0.0, horizon=15, **bad)
    except ValueError:
        pass
    else:
        print("情景参数校验缺失，未拦住：", bad); sys.exit(1)
scenario.reset()
# (i) 实施类指标必须读诊断里的生效值，否则审批时序与建设年限那几组算错
# 用源码级检查而不是导入：eval_metrics 是带 argparse 的入口脚本，导入它会触发
# 参数解析。只要确认它确实引用了这三个生效值字段即可。
_src = open("scripts/eval_metrics.py", encoding="utf-8").read()
for key in ("hazard_eff", "build_years_eff", "tau_valid_eff"):
    assert key in _src, f"eval_metrics 不读 diag.{key}：该参数那几组的实施类指标会算错"
# (j) 实施类指标模块可导入
import tpmorl.eval.metrics  # noqa
# ---- v16 专项 ----
# (k) 拆除基数口径：flat 必须逐位等于旧口径，bytype 必须**均值保持**。
#     均值保持是这一档可解释的前提：若归一化写错，bytype 同时改了成本水平与成本
#     结构，两个效应无法分离，而分母、预算咬合、可达上界全对成本水平敏感。
from tpmorl.objectives import reward as RW
from tpmorl.rl.env_gym import RenewalEnv as _RE
import numpy as _np
scenario.reset()
_e = _RE(ds, T=15, T_eval=26, seed=0); _e.reset()
assert set(_np.round(_e.cell_base, 9)) == {float(RW.CELL_COST)}, \
    "flat 档的逐类基数不是常数：与既往结果不再逐位等价"
_PC_flat = _e.PC.copy()
scenario.apply(budget=2700, carry=3, growth=0.0, horizon=15,
               horizon_eval="auto", cell_cost_mode="bytype")
_k_bt = scenario.inst_tag()
_e2 = _RE(ds, T=15, T_eval=26, seed=0); _e2.reset()
_tot_flat = RW.CELL_COST * float(_np.asarray(_e2.ncell, float).sum())
_tot_bt = float(sum(_e2.cell_base[f] * c for h in _e2.hist0 for f, c in h.items()))
assert abs(_tot_bt - _tot_flat) < 1e-6, \
    f"bytype 未保持均值：Σ基数×格数 {_tot_bt:.1f} != flat 的 {_tot_flat:.1f}"
assert "KB" in _k_bt and _k_bt != k_main, \
    f"bytype 未入分母缓存键（{_k_bt}）：会静默复用 flat 档的分母"
assert not _np.allclose(_e2.PC, _PC_flat), "bytype 档 PC 未改变，开关没生效"
scenario.reset()
_e3 = _RE(ds, T=15, T_eval=26, seed=0); _e3.reset()
assert _np.array_equal(_e3.PC, _PC_flat), "reset() 未复原拆除基数口径：会静默继承"
# (l) 主组预算 2700 下窗内瓶颈应当是**配额**而不是资金——这是 v16 的设计意图，
#     若不成立则整批仍在测 v15 已证明测不出来的东西，必须当场停。
_cheap = float(_np.median(_e3.PC.min(axis=1)))
_cap_money = 5.0 * 2700.0 / _cheap
assert _cap_money > 5 * 3, (
    f"预算 2700 下窗内资金上限仅 {_cap_money:.1f} 个 < 配额上限 15 个："
    "瓶颈仍是资金，主判据依旧无活动空间，请再提高 --budget")
print("  拆除基数 flat 逐位中性、bytype 均值保持且入键", _k_bt)
print("  主组窗内瓶颈 = 配额（资金可支持 %.1f 个 > 配额 15 个；单元中位成本 %.0f）"
      % (_cap_money, _cheap))
print("  python", sys.version.split()[0], "| 依赖与数据齐备")
print("  v14 口径修复在位（R9, auto=26/31/24）；v15 观测在位（N_FEAT=22）")
print("  消融组与主组共用分母键", k_main, "；四个新场景参数均入键且无碰撞")
print("  配额实测生效（配额, 单年最大实际选中数）:", _qmax,
      "| 情景参数非法值均被拦住")
PY

# 1b) 资金记账恒等式回归：三条恒等式写死在脚本里，不成立即非零退出。
#     放在这里而不是放在文档里，是因为那几个绝对金额曾经因为没记种子与策略而
#     不可复现（交叉检查方复现不出），现在它是代码而不是文字。
echo "--- 资金记账恒等式"
python3 scripts/verify_budget_identity.py \
  || { echo "资金记账恒等式不成立，已中止。"; exit 1; }

: > "$RES/.rows"
rm -f "$RES/.alldone"
FAILED=0; RAN=0; SKIP=0

for spec in "${SCENARIOS[@]}"; do
  IFS='|' read -r NUM DIR NAME ARGS <<< "$spec"
  OUT="$BASE/$DIR"; LOG="$RES/g${NUM}_${DIR}.log"; TAG="v16-g${NUM}-${DIR}"

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
  echo "      参数：$ARGS      输出：$OUT"
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
      --iters "$ITERS" --eps "$EPS" --workers "$WORKERS" \
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
  push_now "批次 v16 第 $NUM/$NGROUP 组：${NAME}（${ST}，耗时 ${hm}）"
  [ "$ST" = "完成" ] && tag_now "$TAG" "v16 第 $NUM 组 ${NAME}：${ITERS} iters × ${EPS} eps × ${RUNS_PER_GROUP} 次，耗时 ${hm}"
done

# ---------------------------------------------------------------------------
# 全部跑完后直接出主判据，不用等人回来手工算
if [ $FAILED -eq 0 ] && [ $RAN -eq $NGROUP ]; then
  echo; echo "--- 全部完成，重算实施类指标"
  python3 scripts/eval_metrics.py --batch "$BASE" --out "$RES/metrics" \
      2>&1 | tail -20 | tee -a "$RES/quicklook.txt"

  echo; echo "--- 主判据：消融对照（第 1 组 vs 第 2 组）"
  python3 - "$BASE" "$RES" <<'PY' 2>&1 | tee -a "$RES/quicklook.txt"
import sys, os, json, glob
import numpy as np, pandas as pd
from scipy import stats
BASE, RES = sys.argv[1], sys.argv[2]

# ---- 1) 消融对照。两组分母共用，故这里**可以**比标量化回报 ----
mp = os.path.join(RES, "metrics", "实施类指标_逐运行.csv")
if not os.path.exists(mp):
    print("未找到实施类指标输出，跳过主判据"); sys.exit(0)
R = pd.read_csv(mp, encoding="utf-8-sig")
R.columns = [c.lstrip("\ufeff") for c in R.columns]

def eff_flag(g):
    """从 runs.json 读**生效**的消融开关——不看组名，也不看缓存键。"""
    p = os.path.join(BASE, g, "runs.json")
    if not os.path.exists(p): return None
    d = json.load(open(p, encoding="utf-8"))
    vs = {r.get("diag", {}).get("obs_lifecycle_eff") for r in d.get("runs", [])}
    return vs.pop() if len(vs) == 1 else ("混杂:" + str(sorted(map(str, vs))))

print("各组生效的 obs_lifecycle（须与组名一致，main=True / ablation=False）：")
for g in sorted(R["组"].unique()):
    print(f"  {g:12s} obs_lifecycle_eff = {eff_flag(g)}")

if {"main", "ablation"} <= set(R["组"].unique()):
    a = R[R["组"] == "main"].set_index(["alpha", "seed"])
    b = R[R["组"] == "ablation"].set_index(["alpha", "seed"])
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    assert eff_flag("main") is True and eff_flag("ablation") is False, \
        "两组的 obs_lifecycle_eff 不是 True/False——消融没生效，对照无意义"
    print(f"\n配对样本 {len(idx)} 对（权重档 × 种子逐对配齐）")
    print("| 指标 | 主组 | 消融组 | 差(主−消融) | 配对 t 检验 p | 期望方向 |")
    print("|---|---|---|---|---|---|")
    for col, want in (("LSR", "下降"), ("CRH_T", "上升"),
                      ("ACD_T", "下降"), ("expire_rate_T", "下降"),
                      ("标量回报", "上升")):
        if col not in a.columns: continue
        x, y = a[col].astype(float).values, b[col].astype(float).values
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            print(f"| {col} | | | | 有效样本不足 | {want} |"); continue
        d = x[ok] - y[ok]
        p = stats.ttest_rel(x[ok], y[ok]).pvalue
        print(f"| {col} | {x[ok].mean():.4g} | {y[ok].mean():.4g} | "
              f"{d.mean():+.4g} | {p:.3g} | {want} |")
    print("\n注：LSR 下降且 CRH_T 上升 = 主判据成立。标量回报一栏之所以可比，"
          "是因为这两组**共用同一套分母**（消融开关不入缓存键）；其余任意两组之间"
          "都不可比这一列。")
else:
    print("main / ablation 两组不全，无法出主判据")

# ---- 2) 次判据：立项年份是否前移 ----
print("\n--- 次判据：立项年份分布（均匀期望 7.0）")
print("| 组 | 均年 | 卡方 p | 前移 |")
print("|---|---|---|---|")
for g in sorted(R["组"].unique()):
    ys = []
    for f in glob.glob(os.path.join(BASE, g, "runs", "rec_*.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig")
        d = d[d["unit"] >= 0]          # unit=-1 是「到此为止」占位行，必须滤掉
        if "year" in d: ys.append(d["year"].values)
    if not ys: continue
    y = np.concatenate(ys)
    y = y[y < 15]                       # 只看决策期内的立项
    cnt = np.bincount(y, minlength=15)[:15]
    p = stats.chisquare(cnt).pvalue if cnt.sum() > 0 else np.nan
    print(f"| {g} | {y.mean():.2f} | {p:.3g} | "
          f"{'是' if y.mean() < 7.0 and p < 0.05 else '否'} |")

# ---- 3) 与规则基线比 CRH ----
print("\n--- 策略 vs 规则基线（期内交付率 CRH_T；oracle 是**松**上界，"
      "只去掉审批不确定性这一个维度，差距不能读作\"策略还差多少\"）")
G = pd.read_csv(os.path.join(RES, "metrics", "实施类指标_分组.csv"),
                encoding="utf-8-sig")
G.columns = [c.lstrip("\ufeff") for c in G.columns]
for g in sorted(G["组"].unique()):
    bp = os.path.join(BASE, g, "baseline", "baselines.csv")
    row = G[G["组"] == g].iloc[0]
    line = f"  {g:12s} 策略 CRH_T={float(row['CRH_T']):.3f}"
    if os.path.exists(bp):
        B = pd.read_csv(bp, encoding="utf-8-sig")
        B.columns = [c.lstrip("\ufeff") for c in B.columns]
        s = B.groupby("policy")["crh_T"].mean()
        line += "".join(f"  {k}={v:.3f}" for k, v in s.items())
    print(line)
PY

  touch "$RES/.alldone"; write_status
  push_now "批次 v16 全部 $NGROUP 组完成，含实施类指标与主判据"
  tag_now "batch-v16-complete" "v16 全部 $NGROUP 组完成"
else
  echo; echo "--- 未全部完成（成功 $RAN/$NGROUP，失败 $FAILED，续跑跳过 $SKIP），不出主判据"
  write_status
fi

echo; echo "==================== 结束 $(date '+%F %T') ===================="
echo "状态表：$STATUS"
[ -f "$RES/quicklook.txt" ] && echo "主判据：$RES/quicklook.txt"
exit $(( FAILED > 0 ? 1 : 0 ))
