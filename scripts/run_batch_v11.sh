#!/usr/bin/env bash
# TP-MORL 批次 v11 —— 服务器完整实验驱动脚本
#
# 用法（在仓库根目录）：
#     bash scripts/run_batch_v11.sh                # 全部 9 组，按序
#     bash scripts/run_batch_v11.sh 1 2            # 只跑第 1、2 组
#     WORKERS=64 bash scripts/run_batch_v11.sh     # 指定并行度
#
# 正式跑之前先干跑一次（约 3 分钟，只验管线不验结果）：
#     PUSH=0 ITERS=2 EPS=1 BASE=/tmp/dry11 RES=/tmp/dry11res \
#         bash scripts/run_batch_v11.sh 1 7
#
# 与 v6 的差异（改动前先读 docs/服务器批次_v11.md）：
#   * 全组统一用**情景无关固定分母** `--fixed-scale 0,0.1`（P0-1 成果）。
#     增长率不再改变权重，g0 与 g0.1 两臂可直接比。
#   * 新增 γ 敏感性（第 7、8 组）与规划期延长（第 9 组）。
#   * 每组跑完 commit + push + **单独打标签** `v11-g{N}-{名}`，
#     所以每一项实验在 GitHub 上都能单独定位。
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=src
export OMP_NUM_THREADS=1          # 必须：否则 torch 线程与进程池争核，实测慢数倍

DS="data/processed/gm_dataset_v1"
BASE="${BASE:-$DS/exp_v11}"
RES="${RES:-results_v11}"
PUSH="${PUSH:-1}"
ITERS="${ITERS:-400}"
EPS="${EPS:-8}"
FIXED="${FIXED:-0,0.1}"           # 固定分母覆盖的增长率集合
# GNU nproc 会遵守上面刚设成 1 的 OMP_NUM_THREADS，必须临时清空再探测
NPROC="$( (OMP_NUM_THREADS= nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 10) )"
RUNS_PER_GROUP=35                 # 7 权重档 × 5 种子；并行度超过它没有收益
WORKERS="${WORKERS:-$(( NPROC > 4 ? NPROC - 2 : 2 ))}"
[ "$WORKERS" -gt "$RUNS_PER_GROUP" ] && WORKERS="$RUNS_PER_GROUP"

# 编号|目录名|中文名|附加参数
SCENARIOS=(
  "1|base|主结果：容积率不增长（g=0）|--growth 0"
  "2|g010|增长反事实：容积率年增 10%（g=0.1）|--growth 0.1"
  "3|statutory|现行法定窗口 2+1 年（条文对照）|--growth 0 --tau-valid 2 --tau-ext 1"
  "4|relax2|宽松对照：失效后 2 年可重报|--growth 0 --cooldown 2"
  "5|relax5|宽松对照：失效后 5 年可重报|--growth 0 --cooldown 5"
  "6|build3|建设年限 3 年（综合整治档敏感性）|--growth 0 --build-years 3"
  "7|gamma090|折现率 0.90（更短视）|--growth 0 --gamma 0.90"
  "8|gamma0926|折现率 0.926（财政部社会折现率 8%）|--growth 0 --gamma 0.926"
  "9|horizon20|规划期延长至 20 年（检验零梯度盲区假说）|--growth 0 --horizon 20"
)
NGROUP=${#SCENARIOS[@]}

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
    echo "# 批次 v11 运行状态"
    echo
    echo "- 主机并行度：\`WORKERS=$WORKERS\`（探测到 $NPROC 核）"
    echo "- 每组规模：7 权重档 × 5 种子 = 35 次运行，\`--iters $ITERS --eps $EPS\`"
    echo "- 固定分母：\`--fixed-scale $FIXED\`（情景无关，增长率不再改变权重）"
    echo "- 最后刷新：$(date '+%F %T %Z')"
    echo "- 提交：\`$(git rev-parse --short HEAD)\`"
    echo
    echo "| 组 | 名称 | 状态 | 耗时 | 标签 |"
    echo "|---|---|---|---|---|"
    cat "$RES/.rows" 2>/dev/null
    echo
    if [ -f "$RES/.alldone" ]; then
      echo "## 全部实验已完成（${NGROUP}/${NGROUP}）"
      echo
      echo "已打标签 \`batch-v11-complete\`。"
    else
      echo "> 尚未全部完成。已完成的组其结果即可用。"
    fi
    echo
    echo "**跨组比较的限制**：固定分母只消除了*增长率*这一维，制度参数（有效期、冷却期、"
    echo "建设年限、折现率、规划期）仍进分母缓存键。所以第 1、2 组之间标量化回报可比，"
    echo "第 3–9 组与主组之间**只能比 \`objectives.csv\` 的原始量纲值**，不能比标量化回报。"
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

tag_now () {              # $1 = 标签名  $2 = 说明
  if [ "$PUSH" != "1" ]; then echo "[tag] PUSH=0，跳过"; return 0; fi
  git tag -f "$1" -m "$2" || return 1
  git push -f origin "$1" >/dev/null 2>&1 \
    && echo "[tag] 已标注 $1" \
    || echo "[tag] $1 推送失败，请手工 git push -f origin $1"
}

: > "$RES/.rows"
rm -f "$RES/.alldone"
FAILED=0; RAN=0
echo "批次 v11 开始  $(date '+%F %T')  WORKERS=$WORKERS  ITERS=$ITERS  固定分母=$FIXED"

for spec in "${SCENARIOS[@]}"; do
  IFS='|' read -r NUM DIR NAME ARGS <<< "$spec"
  want_group "$NUM" || { printf '| %s | %s | 跳过 | | |\n' "$NUM" "$NAME" >> "$RES/.rows"; continue; }

  OUT="$BASE/$DIR"; LOG="$RES/g${NUM}_${DIR}.log"; TAG="v11-g${NUM}-${DIR}"
  echo; echo "===== 第 $NUM/$NGROUP 组：$NAME  ($(date '+%F %T'))"
  echo "      参数：$ARGS   输出：$OUT"
  mkdir -p "$OUT"
  t0=$(date +%s)

  # 1) 先单独预建并记录分母：情景键必须出现在日志里，事后可核对没有串组。
  {
    echo "### 分母预建  $(date '+%F %T')"
    python3 -m tpmorl.rl.scale --dataset "$DS" --budget 900 --carry 3 $ARGS
    echo; echo "### 主扫描  $(date '+%F %T')"
  } > "$LOG" 2>&1

  # 2) 主扫描。不要用 grep/tee 过滤 stdout，进度输出会被管道缓冲住。
  python3 scripts/exp_opt_quality.py --dataset "$DS" --out "$OUT" \
      --iters "$ITERS" --eps "$EPS" --workers "$WORKERS" \
      --budget 900 --carry 3 --fixed-scale "$FIXED" $ARGS >> "$LOG" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 )); hm="$(( dt/3600 ))h$(( (dt%3600)/60 ))m"

  if [ $rc -eq 0 ] && [ -f "$OUT/objectives.csv" ]; then
    ST="完成"; RAN=$((RAN+1)); TAGCELL="\`$TAG\`"
  else
    ST="**失败**（退出码 ${rc}，见 \`$LOG\`）"; FAILED=$((FAILED+1)); TAGCELL="—"
    echo "!!! 第 $NUM 组失败，退出码 ${rc}；继续下一组"
  fi
  printf '| %s | %s | %s | %s | %s |\n' "$NUM" "$NAME" "$ST" "$hm" "$TAGCELL" >> "$RES/.rows"
  write_status
  push_now "批次 v11 第 $NUM/$NGROUP 组：${NAME}（${ST}，耗时 ${hm}）"
  [ "$ST" = "完成" ] && tag_now "$TAG" "v11 第 $NUM 组 ${NAME}：${ITERS} iters × ${EPS} eps × 35 次，耗时 ${hm}"
done

echo; echo "===== 批次结束  $(date '+%F %T')  成功 $RAN  失败 $FAILED"
if [ $FAILED -eq 0 ] && [ $RAN -eq $NGROUP ]; then
  touch "$RES/.alldone"; write_status
  push_now "批次 v11 全部实验完成（${NGROUP}/${NGROUP} 组）"
  tag_now "batch-v11-complete" "批次 v11：${NGROUP} 组实验全部完成"
else
  write_status; push_now "批次 v11 部分完成（成功 $RAN / 失败 ${FAILED}）"
  echo "未全部完成，**未**打完成标签。"
fi
