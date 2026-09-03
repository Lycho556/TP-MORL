#!/usr/bin/env bash
# TP-MORL 批次 v4 —— 服务器完整实验驱动脚本
#
# 用法（在仓库根目录）：
#     bash scripts/run_batch_v4.sh                 # 全部 6 组，按序
#     bash scripts/run_batch_v4.sh 1 2             # 只跑第 1、2 组
#     WORKERS=24 bash scripts/run_batch_v4.sh      # 指定并行度
#
# 正式跑之前先干跑一次，确认管线与 push 之外的环节都通（约 2 分钟）：
#     PUSH=0 ITERS=2 EPS=1 BASE=/tmp/dry RES=/tmp/dryres \
#         bash scripts/run_batch_v4.sh 1
#
# 设计约定（改动前先读 docs/服务器批次_v4.md）：
#   * 每组独立目录、独立日志；**每组跑完立刻 commit + push**，
#     所以中途断掉也不丢已完成的组。
#   * 单组失败不终止整批：记为 FAIL，继续下一组。
#   * 状态写在 results_v4/STATUS.md，每组结束刷新一次。
#   * 全部 6 组成功后打 git tag `batch-v4-complete` 作为完成标注。
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=src
export OMP_NUM_THREADS=1          # 必须：否则 torch 线程与进程池争核，实测慢数倍

DS="data/processed/gm_dataset_v1"
BASE="${BASE:-$DS/exp_v4}"      # 干跑时可指向 /tmp
RES="${RES:-results_v4}"
PUSH="${PUSH:-1}"               # PUSH=0 只跑不提交，用于干跑
ITERS="${ITERS:-400}"
EPS="${EPS:-8}"
NPROC="$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 10) )"
WORKERS="${WORKERS:-$(( NPROC > 4 ? NPROC - 2 : 2 ))}"

# 情景组：编号|目录名|中文名|附加参数
SCENARIOS=(
  "1|base|基线（实测标定制度参数）|"
  "2|cool0|冷却期 0 档（核心主张压力测试）|--cooldown 0"
  "3|statutory|现行法定窗口 2+1 年（条文对照）|--tau-valid 2 --tau-ext 1"
  "4|cool1|冷却期 1 档|--cooldown 1"
  "5|cool3|冷却期 3 档|--cooldown 3"
  "6|build3|建设年限 3 年（访谈前旧值，敏感性）|--build-years 3"
)

WANT=("$@")
mkdir -p "$RES" "$BASE"
STATUS="$RES/STATUS.md"

want_group () {           # 无参数 = 全跑
  [ ${#WANT[@]} -eq 0 ] && return 0
  local g
  for g in "${WANT[@]}"; do [ "$g" = "$1" ] && return 0; done
  return 1
}

write_status () {
  {
    echo "# 批次 v4 运行状态"
    echo
    echo "- 主机并行度：\`WORKERS=$WORKERS\`（探测到 $NPROC 核）"
    echo "- 每组规模：7 权重档 × 5 种子 = 35 次运行，\`--iters $ITERS --eps $EPS\`"
    echo "- 最后刷新：$(date '+%F %T %Z')"
    echo "- 提交：\`$(git rev-parse --short HEAD)\`"
    echo
    echo "| 组 | 名称 | 状态 | 耗时 | 产出 |"
    echo "|---|---|---|---|---|"
    cat "$RES/.rows" 2>/dev/null
    echo
    if [ -f "$RES/.alldone" ]; then
      echo "## 全部实验已完成"
      echo
      echo "6 组全部成功，已打标签 \`batch-v4-complete\`。"
      echo "解读前先照 \`docs/rerun_v2.md\` 的验收判据核对基线组。"
    else
      echo "> 尚未全部完成。已完成的组其结果即可用，情景之间的**标量化回报不可比**"
      echo "> （分母按情景而异），跨情景只能比 \`objectives.csv\` 的原始量纲值。"
    fi
  } > "$STATUS"
}

push_now () {             # $1 = commit message
  if [ "$PUSH" != "1" ]; then echo "[push] PUSH=0，跳过"; return 0; fi
  git add -A
  if git diff --cached --quiet; then echo "[push] 无改动，跳过"; return 0; fi
  git commit -q -m "$1" || return 1
  for try in 1 2 3; do
    if git push origin HEAD:main; then echo "[push] 成功"; return 0; fi
    echo "[push] 第 $try 次失败，30s 后重试"; sleep 30
    git pull --rebase origin main || true
  done
  echo "[push] 三次均失败——结果已 commit 在本地，请手工 push"
  return 1
}

: > "$RES/.rows"
rm -f "$RES/.alldone"
FAILED=0; RAN=0
echo "批次 v4 开始  $(date '+%F %T')  WORKERS=$WORKERS  ITERS=$ITERS"

for spec in "${SCENARIOS[@]}"; do
  IFS='|' read -r NUM DIR NAME ARGS <<< "$spec"
  want_group "$NUM" || { printf '| %s | %s | 跳过 | | |\n' "$NUM" "$NAME" >> "$RES/.rows"; continue; }

  OUT="$BASE/$DIR"; LOG="$RES/g${NUM}_${DIR}.log"
  echo; echo "===== 第 $NUM/6 组：$NAME  ($(date '+%F %T'))"
  echo "      参数：${ARGS:-（默认）}   输出：$OUT"
  mkdir -p "$OUT"
  t0=$(date +%s)

  # 1) 先单独预建并记录分母：情景键必须出现在日志里，事后可核对没有串组。
  #    exp_opt_quality 主进程也会建，但那时已在跑，日志里不便核对。
  {
    echo "### 分母预建  $(date '+%F %T')"
    python3 -m tpmorl.rl.scale --dataset "$DS" --budget 900 --carry 3 --growth 0 $ARGS
    echo; echo "### 主扫描  $(date '+%F %T')"
  } > "$LOG" 2>&1

  # 2) 主扫描。不要用 grep/tee 过滤 stdout，进度输出会被管道缓冲住。
  python3 scripts/exp_opt_quality.py --dataset "$DS" --out "$OUT" \
      --iters "$ITERS" --eps "$EPS" --workers "$WORKERS" \
      --budget 900 --carry 3 --growth 0 $ARGS >> "$LOG" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 )); hm="$(( dt/3600 ))h$(( (dt%3600)/60 ))m"

  if [ $rc -eq 0 ] && [ -f "$OUT/objectives.csv" ]; then
    ST="完成"; RAN=$((RAN+1))
    NF="$(ls "$OUT" | tr '\n' ' ')"
  else
    ST="**失败**（退出码 $rc，见 \`$LOG\`）"; FAILED=$((FAILED+1)); NF="—"
    echo "!!! 第 $NUM 组失败，退出码 $rc；继续下一组"
  fi
  printf '| %s | %s | %s | %s | %s |\n' "$NUM" "$NAME" "$ST" "$hm" "$NF" >> "$RES/.rows"
  write_status
  push_now "批次 v4 第 $NUM/6 组：$NAME（$ST，耗时 $hm）"
done

echo; echo "===== 批次结束  $(date '+%F %T')  成功 $RAN  失败 $FAILED"
if [ $FAILED -eq 0 ] && [ $RAN -eq 6 ]; then
  touch "$RES/.alldone"; write_status
  push_now "批次 v4 全部实验完成（6/6 组）"
  [ "$PUSH" = "1" ] || { echo "[tag] PUSH=0，跳过标签"; exit 0; }
  git tag -f batch-v4-complete -m "批次 v4：6 组情景实验全部完成"
  git push -f origin batch-v4-complete || echo "[tag] push 失败，请手工推标签"
  echo "已标注完成：tag batch-v4-complete"
else
  write_status; push_now "批次 v4 部分完成（成功 $RAN / 失败 $FAILED）"
  echo "未全部完成，**未**打完成标签。"
fi
