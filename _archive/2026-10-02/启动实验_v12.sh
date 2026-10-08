#!/usr/bin/env bash
# ============================================================================
#  TP-MORL 批次 v12 —— 一条命令跑完全部实验
#
#  在服务器上只需要敲这一句：
#
#      bash ~/TP-MORL/启动实验_v12.sh
#
#  想挂后台断开 ssh 也不中断：
#
#      nohup bash ~/TP-MORL/启动实验_v12.sh > ~/v12.log 2>&1 &
#      tail -f ~/v12.log
#
#  脚本自己会做：git pull → 环境自检 → 逐组训练 → 每组跑完 commit+push+打标签
#  → 断点续跑（已完成的组自动跳过）→ 全部完成后打 batch-v12-complete。
#
#  ---------------------------------------------------------------------------
#  v12 要回答什么
#      v11 九组给出了一个一致的负面结论：策略学会了"更新哪里"，没学会"什么时候"，
#      九组的立项年份分布全部与均匀分布无法区分（卡方 p ≥ 0.968），
#      且九组年度预算利用率都在 0.966–0.991。
#      这指向一个解释：**年度预算额度把择时自由度锁死了**——钱每年都花光，
#      每年能立几个项是被预算机械定死的，不存在"今年少立、攒钱明年多立"。
#
#      但这还只是解释，不是证据。v12 就是去拿证据：
#      **放开结转上限和年度总额，看立项分布会不会开始前移。**
#      如果会 → 是制度约束，不是算法缺陷，这是本项目最有分量的政策结论。
#      如果不会 → 是策略学不到时序信号，得回去改奖励设计。
#
#      这是唯一能分离这两种解释的实验，比 v11 已跑的任何一组都重要。
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
#      PUSH=0 ITERS=2 EPS=1 BASE=/tmp/dry12 RES=/tmp/dry12res \
#          bash ~/TP-MORL/启动实验_v12.sh
# ============================================================================
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || { echo "找不到仓库目录 $ROOT"; exit 1; }

export PYTHONPATH=src
export OMP_NUM_THREADS=1      # 必须：否则 torch 线程与进程池争核，实测慢数倍

DS="data/processed/gm_dataset_v1"
BASE="${BASE:-$DS/exp_v12}"
RES="${RES:-results_v12}"
PUSH="${PUSH:-1}"
ITERS="${ITERS:-400}"
EPS="${EPS:-8}"
FORCE="${FORCE:-0}"
FIXED="${FIXED:-0,0.1}"       # 固定分母覆盖的增长率集合，与 v11 一致
RUNS_PER_GROUP=35             # 7 权重档 × 5 种子；并行度超过它没有收益
NPROC="$( (OMP_NUM_THREADS= nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 10) )"
WORKERS="${WORKERS:-$(( NPROC > 4 ? NPROC - 2 : 2 ))}"
[ "$WORKERS" -gt "$RUNS_PER_GROUP" ] && WORKERS="$RUNS_PER_GROUP"

# ---------------------------------------------------------------------------
# 实验组：编号 | 目录 | 中文名 | 额外参数
#
# 1–4 是结转上限扫描（第一优先，v12 的主线）
# 5–6 是年度总额扫描（第二优先，与结转互为对照）
# 7   是两者同时放开的上界参照——如果连它都还是均匀分布，那就一定不是预算的锅
# ---------------------------------------------------------------------------
SCENARIOS=(
  "1|carry1|结转上限 1 年（最紧）|--budget 900 --carry 1"
  "2|carry3|结转上限 3 年（v11 主组口径，同批次对照）|--budget 900 --carry 3"
  "3|carry6|结转上限 6 年|--budget 900 --carry 6"
  "4|carryInf|结转无上限（钱可以一直攒）|--budget 900 --carry 999"
  "5|budget600|年度总额 600（更紧）|--budget 600 --carry 3"
  "6|budget1500|年度总额 1500（更松）|--budget 1500 --carry 3"
  "7|loose|上界参照：总额 1500 且结转无上限|--budget 1500 --carry 999"
)
NGROUP=${#SCENARIOS[@]}

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
    echo "# 批次 v12 运行状态"
    echo
    echo "**这批实验回答的问题**：v11 九组发现立项时间全部呈均匀分布、预算利用率全在 0.97–0.99，"
    echo "指向\"年度预算额度锁死了择时自由度\"。v12 放开结转上限与年度总额，"
    echo "看立项分布会不会开始前移——这是唯一能分离\"策略学不会择时\"与\"制度不让择时\"的实验。"
    echo
    echo "- 主机并行度：\`WORKERS=$WORKERS\`（探测到 $NPROC 核）"
    echo "- 每组规模：7 权重档 × 5 种子 = 35 次运行，\`--iters $ITERS --eps $EPS\`"
    echo "- 固定分母：\`--fixed-scale $FIXED\`"
    echo "- 最后刷新：$(date '+%F %T %Z')　提交：\`$(git rev-parse --short HEAD)\`"
    echo
    echo "| 组 | 名称 | 状态 | 耗时 | 标签 |"
    echo "|---|---|---|---|---|"
    cat "$RES/.rows" 2>/dev/null
    echo
    if [ -f "$RES/.alldone" ]; then
      echo "## 全部实验已完成（${NGROUP}/${NGROUP}），已打标签 \`batch-v12-complete\`"
    else
      echo "> 尚未全部完成。已完成的组其结果即可用；重跑本脚本会自动跳过已完成的组。"
    fi
    echo
    echo "**跨组比较的限制**：\`budget\` 与 \`carry\` 都进分母缓存键，所以七组各有各的分母，"
    echo "**只能比 \`objectives.csv\` 的原始量纲值与立项年份分布，不能比标量化回报**。"
    echo "好在 v12 的主判据（立项年份分布、盲区率、预算利用率）本来就是无量纲的，不受影响。"
  } > "$STATUS"
}

# ---------------------------------------------------------------------------
echo "==================== TP-MORL 批次 v12 ===================="
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
print("  python", sys.version.split()[0], "| 依赖与数据齐备")
PY

: > "$RES/.rows"
rm -f "$RES/.alldone"
FAILED=0; RAN=0; SKIP=0

for spec in "${SCENARIOS[@]}"; do
  IFS='|' read -r NUM DIR NAME ARGS <<< "$spec"
  OUT="$BASE/$DIR"; LOG="$RES/g${NUM}_${DIR}.log"; TAG="v12-g${NUM}-${DIR}"

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

  # 分母预建单独跑一次并记进日志：情景键必须可事后核对，防止串组
  {
    echo "### 分母预建  $(date '+%F %T')"
    python3 -m tpmorl.rl.scale --dataset "$DS" $ARGS --growth 0
    echo; echo "### 主扫描  $(date '+%F %T')"
  } > "$LOG" 2>&1

  # 主扫描。不要用 grep/tee 过滤 stdout，进度输出会被管道缓冲住看不到。
  python3 scripts/exp_opt_quality.py --dataset "$DS" --out "$OUT" \
      --iters "$ITERS" --eps "$EPS" --workers "$WORKERS" \
      --growth 0 --fixed-scale "$FIXED" $ARGS >> "$LOG" 2>&1
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
  push_now "批次 v12 第 $NUM/$NGROUP 组：${NAME}（${ST}，耗时 ${hm}）"
  [ "$ST" = "完成" ] && tag_now "$TAG" "v12 第 $NUM 组 ${NAME}：${ITERS} iters × ${EPS} eps × ${RUNS_PER_GROUP} 次，耗时 ${hm}"
done

# ---------------------------------------------------------------------------
# 全部跑完后直接出主判据，不用等人回来手工算
if [ $FAILED -eq 0 ] && [ $RAN -eq $NGROUP ]; then
  echo; echo "--- 全部完成，直接算主判据（立项年份是否仍均匀）"
  python3 - "$BASE" "$RES" <<'PY' 2>&1 | tee -a "$RES/quicklook.txt"
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy import stats
BASE, RES = sys.argv[1], sys.argv[2]
GRP = [("carry1","结转1年"),("carry3","结转3年"),("carry6","结转6年"),
       ("carryInf","结转无上限"),("budget600","总额600"),
       ("budget1500","总额1500"),("loose","总额1500+无上限")]
rows = []
for d, cn in GRP:
    ys, T = [], None
    for f in sorted(glob.glob(f"{BASE}/{d}/runs/rec_*.csv")):
        # encoding 必须带 -sig：rec 文件表头有 BOM，否则第一列名会变成 "\ufeffep"
        r = pd.read_csv(f, encoding="utf-8-sig")
        r = r[r.ep == r.ep.min()]          # 确定性评估，5 个 episode 完全一致，取一个即可
        T = int(r.year.max()) + 1
        # 每一行 unit 非空 = 该年立项一个更新单元（不要用 stopped 当非动作标记，它不是）
        ys += r.loc[r.unit.notna(), "year"].astype(int).tolist()
    if not ys:
        continue
    obs = np.bincount(ys, minlength=T)[:T]
    p = stats.chisquare(obs).pvalue
    rows.append(dict(组=cn, n立项=len(ys), 均年=round(float(np.mean(ys)), 2),
                     均匀期望=round((T-1)/2, 2), 卡方p=round(float(p), 4)))
t = pd.DataFrame(rows)
print("\n===== v12 主判据：立项年份是否仍与均匀分布无法区分 =====")
print(t.to_string(index=False))
t.to_csv(os.path.join(RES, "v12_timing_quicklook.csv"),
         index=False, encoding="utf-8-sig")
if (t.卡方p > 0.05).all():
    print("\n>>> 七组全部拒绝不了均匀假设：放开预算约束**没有**带来择时行为。")
    print(">>> 结论指向策略侧（奖励里没有有效的时序梯度），不是制度侧。")
else:
    bad = t[t.卡方p <= 0.05]
    print(f"\n>>> 有 {len(bad)} 组偏离均匀：{list(bad.组)}")
    print(">>> 说明放开预算后择时行为出现了——制度约束假说得到支持，这是主要结论。")
PY
  touch "$RES/.alldone"; write_status
  push_now "批次 v12 全部实验完成（${NGROUP}/${NGROUP} 组，其中续跑跳过 ${SKIP} 组）"
  tag_now "batch-v12-complete" "批次 v12：${NGROUP} 组实验全部完成"
  echo; echo "===== 批次 v12 全部完成  $(date '+%F %T')"
  echo "主判据见 $RES/v12_timing_quicklook.csv 与 $RES/quicklook.txt"
else
  write_status; push_now "批次 v12 部分完成（成功 $RAN / 失败 ${FAILED}）"
  echo; echo "===== 批次结束  $(date '+%F %T')  成功 $RAN  失败 $FAILED"
  echo "未全部完成，**未**打完成标签。修好后重跑本脚本，已完成的组会自动跳过。"
fi
