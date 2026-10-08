#!/usr/bin/env bash
# ============================================================================
#  TP-MORL 批次 v14 —— 一条命令跑完全部实验
#
#  在服务器上只需要敲这一句：
#
#      bash ~/TP-MORL/启动实验_v14.sh
#
#  想挂后台断开 ssh 也不中断：
#
#      nohup bash ~/TP-MORL/启动实验_v14.sh > ~/v14.log 2>&1 &
#      tail -f ~/v14.log
#
#  脚本自己会做：git pull → 环境自检 → 逐组训练 → 每组跑完 commit+push+打标签
#  → 断点续跑（已完成的组自动跳过）→ 全部完成后打 batch-v14-complete。
#
#  ---------------------------------------------------------------------------
#  v14 为什么存在：一处**口径错误**与一处**既有缺陷**，二者各自独立地要求重跑
#
#      (1) 立项与记分不分离（口径错误，不是设计取舍）。
#          旧口径把决策期与评价期绑成同一个闭区间 T=15：第 15 年之后不再记分，
#          于是**第 5 年以后立项的项目其交付永远不会被算进回报**——最坏路径要
#          5(有效期)+1(开工)+5(建设)=11 年才建成。这不是"模型认为晚立项不好"，
#          而是"模型看不见晚立项的成果"。诊断与实务依据见
#          docs/跨期结转_口径修正_v1.md，实施记录见
#          docs/立项与记分分离_实施记录_v1.md。
#          v14 新增**评价期** T_eval：决策期仍是 15 年（第 15 年起不再立项、
#          不再进钱），但状态机继续推进到管道排空，晚立项的交付按**真实年份**
#          折现后计入。`--horizon-eval auto` 按本组的 T/有效期/建设年限自动取
#          最短排空期（主组 26，horizon20 组 31，法定窗口与建设 3 年组 24）。
#          **不能全批写死一个常数**——写死 26 会把 horizon20 组的管道截断在期内，
#          那正是本次要修的错误的翻版。代码里已有硬断言在回合结束时检查管道
#          是否真排空，取短了当场失败，不会跑完 10 小时才发现。
#
#      (2) 分母的规划期缺陷（既有，v11/v13 的 horizon20 组已受影响）。
#          `scale._rollout` 建参考策略 env 时既不传 T 也不传 T_eval，一律用模块
#          默认 T=15。于是 `--horizon 20` 组的分母是**15 年**的可达上界，而该组
#          实跑 20 年——缓存键里写着 T20、内容却是 T=15 的值，该组全部目标的
#          归一化值被系统性抬高。实测 T=20 的分母比 T=15 大 1.2–1.5 倍。
#          启用尾部评价后同一缺陷会再放大一次：实测尾部口径的分母比闭区间大
#          1.45–2.06 倍，不修则归一化值整体虚高 45–106%。
#          已修（读 scenario 登记的 T/T_eval），分母版本号递增到 **R9**——必须
#          递增，因为 `T20` 这个键下已存在 R8 的**错误**缓存文件，不递增会被
#          静默复用。T=15 闭区间各组的分母值不受此修复影响（新旧逐行等价），
#          重建后应逐位相同，可据此自查本次修复无副作用。
#
#      目标集合**不变**：仍是既有的 11 项（Gdp/Eco/Res/Emp/Aec/E2r/Cpt/Floor
#      + Cost/Disrupt/Expire）。v14 只改时间口径与分母，不改目标定义，故
#      v13 与 v14 的差异可直接归因于这两处修复本身。
#
#      单组约 37 分钟，16 组合计约 10 小时。可分两阶段：
#          PHASE=1  只跑第 1–8 组（预算扫描 + 口径对照，约 5 小时）
#          PHASE=2  只跑第 9–16 组（制度扫描，约 5 小时）
#      不设 PHASE 则依次跑完全部 16 组。
#
#  ---------------------------------------------------------------------------
#  v14 要回答什么
#      v13 之前的一致发现是个**负面结论**：策略学会了"更新哪里"，没学会
#      "什么时候"——立项年份分布与均匀分布无法区分，且干净组里晚立项数量与
#      标量化回报的相关（去掉权重档混杂后）≈ 0，即**奖励里根本没有时序梯度**。
#      旧口径下这是必然的：晚立项的交付被截断，早立项的收益又只差一点折现，
#      两头都没信号。
#
#      修好之后信号第一次存在了：早立项的交付折现少、晚立项的交付照算但折现多，
#      于是"何时动"成为一个有梯度的真决策。v14 的主判据因此是
#          **立项年份分布是否开始前移（均年 < 均匀期望 7.0，且卡方拒绝均匀）**。
#      第 8 组是**闭区间口径对照**：与第 2 组除记分口径外参数逐字相同，
#      用来把"前移"归因到口径修正本身，而不是别的什么。
#
#      注意：第 2 组与第 8 组的分母不同（T15X26 vs T15），**两组之间只能比
#      objectives.csv 的原始量纲值与立项年份分布，不能比标量化回报**。
#      好在主判据（立项年份分布）本来就是无量纲的。
#      （v13 的第 8 组 `base` 与第 2 组 `carry3` 参数逐字相同、种子与代码也相同，
#        结果必然逐位一致——白跑一组 37 分钟。v14 把这一格改成了口径对照。）
#  ---------------------------------------------------------------------------
#  可调环境变量（都有合理默认，一般不用动）
#      WORKERS=64      并行度，默认 = 核数-2，上限 35
#      ITERS=400       每次运行的训练轮数
#      EPS=8           每轮的 episode 数
#      PUSH=0          不推 GitHub（本地调试用）
#      ONLY="1 3"      只跑指定组号
#      FORCE=1         忽略已完成标记，全部重跑
#  ---------------------------------------------------------------------------
#  正式跑之前建议先干跑三分钟验管线（不写正式目录、不推送）：
#      PUSH=0 ITERS=2 EPS=1 BASE=/tmp/dry14 RES=/tmp/dry14res \
#          bash ~/TP-MORL/启动实验_v14.sh
# ============================================================================
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || { echo "找不到仓库目录 $ROOT"; exit 1; }

export PYTHONPATH=src
export OMP_NUM_THREADS=1      # 必须：否则 torch 线程与进程池争核，实测慢数倍

DS="data/processed/gm_dataset_v1"
BASE="${BASE:-$DS/exp_v14}"
RES="${RES:-results_v14}"
PUSH="${PUSH:-1}"
ITERS="${ITERS:-400}"
EPS="${EPS:-8}"
FORCE="${FORCE:-0}"
FIXED="${FIXED:-0,0.1}"       # 固定分母覆盖的增长率集合，与 v11/v13 一致
RUNS_PER_GROUP=35             # 7 权重档 × 5 种子；并行度超过它没有收益
NPROC="$( (OMP_NUM_THREADS= nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 10) )"
WORKERS="${WORKERS:-$(( NPROC > 4 ? NPROC - 2 : 2 ))}"
[ "$WORKERS" -gt "$RUNS_PER_GROUP" ] && WORKERS="$RUNS_PER_GROUP"

# ---------------------------------------------------------------------------
# 实验组：编号 | 目录 | 中文名 | 额外参数
#
# 1–4 是结转上限扫描，5–6 是年度总额扫描，7 是两者同时放开的上界参照。
# 8   是**闭区间口径对照**（唯一不加 --horizon-eval 的一组）。
# 9–16 是制度扫描，参数与 v11/v13 逐字一致。
#
# `--horizon-eval auto` 写在**每组的参数里**而不是全局命令行，就是为了让第 8 组
# 的"不加"一眼可见、可审。auto 会按该组的 T/有效期/建设年限各自解析。
# ---------------------------------------------------------------------------
SCENARIOS=(
  "1|carry1|结转上限 1 年（最紧）|--budget 900 --carry 1 --horizon-eval auto"
  "2|carry3|结转上限 3 年（主组）|--budget 900 --carry 3 --horizon-eval auto"
  "3|carry6|结转上限 6 年|--budget 900 --carry 6 --horizon-eval auto"
  "4|carryInf|结转无上限（钱可以一直攒）|--budget 900 --carry 999 --horizon-eval auto"
  "5|budget600|年度总额 600（更紧）|--budget 600 --carry 3 --horizon-eval auto"
  "6|budget1500|年度总额 1500（更松）|--budget 1500 --carry 3 --horizon-eval auto"
  "7|loose|上界参照：总额 1500 且结转无上限|--budget 1500 --carry 999 --horizon-eval auto"
  "8|closed|口径对照：闭区间记分（与第 2 组仅差记分口径）|--budget 900 --carry 3"
  "9|g010|容积率年增 10%（本组设计如此，非污染）|--budget 900 --carry 3 --growth 0.1 --horizon-eval auto"
  "10|statutory|法定窗口 tau_valid=2 tau_ext=1|--budget 900 --carry 3 --tau-valid 2 --tau-ext 1 --horizon-eval auto"
  "11|relax2|冷却期 2 年|--budget 900 --carry 3 --cooldown 2 --horizon-eval auto"
  "12|relax5|冷却期 5 年|--budget 900 --carry 3 --cooldown 5 --horizon-eval auto"
  "13|build3|建设周期 3 年|--budget 900 --carry 3 --build-years 3 --horizon-eval auto"
  "14|gamma090|折现率 0.90|--budget 900 --carry 3 --gamma 0.9 --horizon-eval auto"
  "15|gamma0926|折现率 0.926|--budget 900 --carry 3 --gamma 0.926 --horizon-eval auto"
  "16|horizon20|规划期 20 年|--budget 900 --carry 3 --horizon 20 --horizon-eval auto"
)
NGROUP=${#SCENARIOS[@]}

# PHASE 是 ONLY 的便捷写法：1 = 预算扫描+口径对照(1-8)，2 = 制度扫描(9-16)。
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
    echo "# 批次 v14 运行状态"
    echo
    echo "**这批实验回答的问题**：修好\"立项与记分不分离\"这处口径错误之后，"
    echo "奖励里第一次存在时序梯度（早立项折现少、晚立项交付照算但折现多）。"
    echo "主判据是**立项年份分布是否开始前移**（均年 < 均匀期望 7.0，且卡方拒绝均匀）。"
    echo "第 8 组是闭区间口径对照，与第 2 组除记分口径外参数逐字相同，"
    echo "用来把前移归因到口径修正本身。"
    echo
    echo "- 主机并行度：\`WORKERS=$WORKERS\`（探测到 $NPROC 核）"
    echo "- 每组规模：7 权重档 × 5 种子 = 35 次运行，\`--iters $ITERS --eps $EPS\`"
    echo "- 固定分母：\`--fixed-scale $FIXED\`，分母版本 \`R9\`"
    echo "- 最后刷新：$(date '+%F %T %Z')　提交：\`$(git rev-parse --short HEAD)\`"
    echo
    echo "| 组 | 名称 | 状态 | 耗时 | 标签 |"
    echo "|---|---|---|---|---|"
    cat "$RES/.rows" 2>/dev/null
    echo
    if [ -f "$RES/.alldone" ]; then
      echo "## 全部实验已完成（${NGROUP}/${NGROUP}），已打标签 \`batch-v14-complete\`"
    else
      echo "> 尚未全部完成。已完成的组其结果即可用；重跑本脚本会自动跳过已完成的组。"
    fi
    echo
    echo "**动力学与口径自证**：每个 run 的 \`diag\` 记录 \`far_growth_eff/budget_eff/"
    echo "carry_eff\` 与 \`t_dec_eff/t_eval_eff\`，即**实际生效**的动力学与时间口径而非"
    echo "命令行意图（光看 \`horizon_eval\` 的意图值分不出 \`auto\` 解析成了几）。"
    echo "\`one_run\` 在建完分母后立即断言动力学三者与入参一致，\`run_episode\` 在回合"
    echo "结束时断言管道已排空（S1/S2/S3 为 0），任一不符当场失败。"
    echo "v11/v12 的污染之所以跑完 4 小时才被发现，正是缺这两样。"
    echo
    echo "**跨组比较的限制**：\`budget\`/\`carry\`/制度参数/\`T\`/\`T_eval\` 都进分母缓存键，"
    echo "所以各组各有各的分母，**只能比 \`objectives.csv\` 的原始量纲值与立项年份分布，"
    echo "不能比标量化回报**。第 2 组与第 8 组也一样（分母分别是 \`T15X26\` 与 \`T15\`）。"
    echo "好在 v14 的主判据（立项年份分布、盲区率、预算利用率）本来就是无量纲的。"
  } > "$STATUS"
}

# ---------------------------------------------------------------------------
echo "==================== TP-MORL 批次 v14 ===================="
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

# v14 专项：确认跑的是修复后的代码，不是服务器上残留的旧 checkout。
# 这三样缺任何一样，整批结果都会悄悄退回旧口径——必须现在就响。
from tpmorl.rl import scenario, scale
from tpmorl.rl.env_gym import RenewalEnv
assert hasattr(scenario, "auto_horizon_eval"), \
    "scenario 缺 auto_horizon_eval：代码是旧版，--horizon-eval auto 不会生效"
assert scale.REF_VER == "R9", f"分母版本是 {scale.REF_VER}，应为 R9（否则会复用错误缓存）"
assert "T_eval" in RenewalEnv.__init__.__code__.co_varnames, \
    "RenewalEnv 不接受 T_eval：代码是旧版"
scenario.reset()
scenario.apply(budget=900, carry=3, growth=0.0, horizon=15, horizon_eval="auto")
assert scenario.horizon_eval() == 26, f"主组 auto 解析成 {scenario.horizon_eval()}，应为 26"
scenario.reset()
scenario.apply(budget=900, carry=3, growth=0.0, horizon=20, horizon_eval="auto")
assert scenario.horizon_eval() == 31, f"horizon20 组 auto 解析成 {scenario.horizon_eval()}，应为 31"
scenario.reset()
print("  python", sys.version.split()[0], "| 依赖与数据齐备 | 口径修复已在位（R9, auto=26/31）")
PY

: > "$RES/.rows"
rm -f "$RES/.alldone"
FAILED=0; RAN=0; SKIP=0

for spec in "${SCENARIOS[@]}"; do
  IFS='|' read -r NUM DIR NAME ARGS <<< "$spec"
  OUT="$BASE/$DIR"; LOG="$RES/g${NUM}_${DIR}.log"; TAG="v14-g${NUM}-${DIR}"

  if ! want_group "$NUM"; then
    printf '| %s | %s | 跳过（ONLY） | | |\n' "$NUM" "$NAME" >> "$RES/.rows"; continue
  fi
  # 断点续跑：已有完整结果就不重跑
  if [ "$FORCE" != "1" ] && [ -f "$OUT/objectives.csv" ]; then
    n=$(( $(wc -l < "$OUT/objectives.csv") - 1 ))
    if [ "$n" -ge "$RUNS_PER_GROUP" ]; then
      echo "===== 第 $NUM 组已完成（$n 条记录），跳过。要重跑请 FORCE=1"
      printf '| %s | %s | 已完成（续跑跳过） | | \\`%s\\` |\n' "$NUM" "$NAME" "$TAG" >> "$RES/.rows"
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
  # 现在由 exp_opt_quality.main() 在起进程池之前预建**实际会被读的那个**文件，
  # 并把它的路径打进日志。v14 的键里多了 X{T_eval}，故每组都会重建一次分母。
  echo "### 主扫描  $(date '+%F %T')" > "$LOG"

  # 主扫描。不要用 grep/tee 过滤 stdout，进度输出会被管道缓冲住看不到。
  # --growth 不在这里写死：各组自己的 $ARGS 决定（默认 0.0，仅第 9 组为 0.1）。
  python3 scripts/exp_opt_quality.py --dataset "$DS" --out "$OUT" \
      --iters "$ITERS" --eps "$EPS" --workers "$WORKERS" \
      --fixed-scale "$FIXED" $ARGS >> "$LOG" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 )); hm="$(( dt/3600 ))h$(( (dt%3600)/60 ))m"

  if [ $rc -eq 0 ] && [ -f "$OUT/objectives.csv" ]; then
    ST="完成"; RAN=$((RAN+1)); TAGCELL="\`$TAG\`"
  else
    ST="**失败**（退出码 ${rc}，见 \`$LOG\`）"; FAILED=$((FAILED+1)); TAGCELL="—"
    echo "!!! 第 $NUM 组失败，退出码 ${rc}；继续下一组"
    tail -15 "$LOG"
  fi
  printf '| %s | %s | %s | %s | %s |\n' "$NUM" "$NAME" "$ST" "$hm" "$TAGCELL" >> "$RES/.rows"
  write_status
  push_now "批次 v14 第 $NUM/$NGROUP 组：${NAME}（${ST}，耗时 ${hm}）"
  [ "$ST" = "完成" ] && tag_now "$TAG" "v14 第 $NUM 组 ${NAME}：${ITERS} iters × ${EPS} eps × ${RUNS_PER_GROUP} 次，耗时 ${hm}"
done

# ---------------------------------------------------------------------------
# 全部跑完后直接出主判据，不用等人回来手工算
if [ $FAILED -eq 0 ] && [ $RAN -eq $NGROUP ]; then
  echo; echo "--- 全部完成，直接算主判据（立项年份是否开始前移）"
  python3 - "$BASE" "$RES" <<'PY' 2>&1 | tee -a "$RES/quicklook.txt"
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy import stats
BASE, RES = sys.argv[1], sys.argv[2]
# 目录 | 中文名 | 是否启用尾部评价
GRP = [("carry1", "结转1年", 1), ("carry3", "结转3年", 1), ("carry6", "结转6年", 1),
       ("carryInf", "结转无上限", 1), ("budget600", "总额600", 1),
       ("budget1500", "总额1500", 1), ("loose", "总额1500+无上限", 1),
       ("closed", "闭区间口径对照", 0),
       ("g010", "容积率+10%", 1), ("statutory", "法定窗口", 1),
       ("relax2", "冷却2年", 1), ("relax5", "冷却5年", 1),
       ("build3", "建设3年", 1), ("gamma090", "折现0.90", 1),
       ("gamma0926", "折现0.926", 1), ("horizon20", "规划期20年", 1)]
rows = []
for d, cn, tail in GRP:
    ys, T = [], None
    for f in sorted(glob.glob(f"{BASE}/{d}/runs/rec_*.csv")):
        # encoding 必须带 -sig：rec 文件表头有 BOM，否则第一列名会变成 "\ufeffep"
        r = pd.read_csv(f, encoding="utf-8-sig")
        r = r[r.ep == r.ep.min()]          # 确定性评估，5 个 episode 完全一致，取一个即可
        T = int(r.year.max()) + 1
        # 每一行 unit 非空且 >=0 = 该年立项一个更新单元
        # （不要用 stopped 当非动作标记，它不是；unit=-1 是"当年未立项"的占位行）
        ys += r.loc[r.unit.notna() & (r.unit >= 0), "year"].astype(int).tolist()
    if not ys:
        continue
    obs = np.bincount(ys, minlength=T)[:T]
    p = stats.chisquare(obs).pvalue
    unif = (T - 1) / 2
    mean_y = float(np.mean(ys))
    # 后段立项占比：最后 5 个决策年。旧口径下这些项目的交付必被截断。
    late = float(np.mean(np.asarray(ys) >= T - 5))
    rows.append(dict(组=cn, 口径=("尾部" if tail else "闭区间"), T=T, n立项=len(ys),
                     均年=round(mean_y, 2), 均匀期望=round(unif, 2),
                     前移=round(unif - mean_y, 2),
                     后段5年占比=round(late, 3), 卡方p=round(float(p), 4)))
t = pd.DataFrame(rows)
print("\n===== v14 主判据：立项年份是否开始前移 =====")
print("（前移 > 0 表示均年早于均匀期望；卡方 p < 0.05 表示分布可与均匀区分）")
print(t.to_string(index=False))
t.to_csv(os.path.join(RES, "v14_timing_quicklook.csv"), index=False, encoding="utf-8-sig")

tl = t[t.口径 == "尾部"]
cl = t[t.口径 == "闭区间"]
nsig = int((tl.卡方p < 0.05).sum())
print(f"\n>>> 尾部口径 {len(tl)} 组中有 {nsig} 组可与均匀分布区分"
      f"（均年前移中位数 {tl.前移.median():+.2f} 年）")
if len(cl):
    c = cl.iloc[0]
    m = t[(t.组 == "结转3年")]
    if len(m):
        m = m.iloc[0]
        print(f">>> 口径对照（其余参数逐字相同）：")
        print(f"      闭区间  均年 {c.均年:.2f}  后段5年占比 {c.后段5年占比:.3f}  卡方p {c.卡方p:.4f}")
        print(f"      尾部    均年 {m.均年:.2f}  后段5年占比 {m.后段5年占比:.3f}  卡方p {m.卡方p:.4f}")
        print(f"      差值    均年 {m.均年 - c.均年:+.2f} 年  后段占比 {m.后段5年占比 - c.后段5年占比:+.3f}")
if nsig >= len(tl) * 0.5:
    print("\n>>> 多数组出现择时行为：口径修正让奖励第一次带上时序梯度，"
          "\"何时动\"成为可学的决策。这是 v14 的主要结论。")
else:
    print("\n>>> 多数组仍与均匀无法区分：口径不是唯一障碍，"
          "需回看奖励分解里交付/成本那一组是否仍净负（见 docs/择时机制_诊断_v1.md）。")
print("\n注意：各组分母不同（键含 budget/carry/制度/T/T_eval），"
      "上表各列均为无量纲或原始年份，可跨组比；**标量化回报不可跨组比**。")
PY
  touch "$RES/.alldone"; write_status
  push_now "批次 v14 全部实验完成（${NGROUP}/${NGROUP} 组，其中续跑跳过 ${SKIP} 组）"
  tag_now "batch-v14-complete" "批次 v14：${NGROUP} 组实验全部完成"
  echo; echo "===== 批次 v14 全部完成  $(date '+%F %T')"
  echo "主判据见 $RES/v14_timing_quicklook.csv 与 $RES/quicklook.txt"
else
  write_status; push_now "批次 v14 部分完成（成功 $RAN / 失败 ${FAILED}）"
  echo; echo "===== 批次结束  $(date '+%F %T')  成功 $RAN  失败 $FAILED"
  echo "未全部完成，**未**打完成标签。修好后重跑本脚本，已完成的组会自动跳过。"
fi
