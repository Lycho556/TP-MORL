#!/usr/bin/env bash
# 选一块最空闲的 GPU，打印其编号。带原子锁，避免两个任务同时抢同一块。
#
#   用法：  CUDA_VISIBLE_DEVICES=$(bash scripts/pick_gpu.sh) python train.py
#           bash scripts/pick_gpu.sh --status      # 只看现状不分配
#           MIN_FREE_MB=20000 bash scripts/pick_gpu.sh
#
# 评分：空闲显存越多越好，利用率越低越好。显存是硬门槛（装不下就直接跳过），
# 利用率只作次序参考——利用率是瞬时采样，单看它会把刚启动还没喂满的任务误判成空闲。
set -u
LOCK="${GPU_LOCK:-/tmp/gpu_pick.lock}"
CLAIM_DIR="${GPU_CLAIM_DIR:-/tmp/gpu_claims}"
MIN_FREE_MB="${MIN_FREE_MB:-8000}"     # 少于这个空闲显存就不考虑
CLAIM_TTL="${CLAIM_TTL:-120}"          # 认领记录的有效期（秒），防止进程没起来把卡锁死

command -v nvidia-smi >/dev/null 2>&1 || { echo "ERR:无 nvidia-smi" >&2; exit 1; }

query () {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader,nounits 2>/dev/null
}

if [ "${1:-}" = "--status" ]; then
  printf "%-4s %-12s %-12s %-8s %s\n" 卡号 已用MB 总MB 利用率 最近认领
  query | while IFS=, read -r idx used total util; do
    idx=$(echo $idx|tr -d ' '); used=$(echo $used|tr -d ' ')
    total=$(echo $total|tr -d ' '); util=$(echo $util|tr -d ' ')
    c="—"
    if [ -f "$CLAIM_DIR/$idx" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$CLAIM_DIR/$idx" 2>/dev/null || echo 0) ))
      c="${age}s 前 by $(cat "$CLAIM_DIR/$idx" 2>/dev/null)"
    fi
    printf "%-4s %-12s %-12s %-8s %s\n" "$idx" "$used" "$total" "${util}%" "$c"
  done
  exit 0
fi

mkdir -p "$CLAIM_DIR"
exec 9>"$LOCK"
flock -w 30 9 || { echo "ERR:取锁超时" >&2; exit 1; }

NOW=$(date +%s)
BEST=""; BEST_SCORE=-1
while IFS=, read -r idx used total util; do
  idx=$(echo $idx|tr -d ' '); used=$(echo $used|tr -d ' ')
  total=$(echo $total|tr -d ' '); util=$(echo $util|tr -d ' ')
  free=$(( total - used ))
  [ "$free" -lt "$MIN_FREE_MB" ] && continue
  # 跳过刚被别人认领、但显存还没涨上来的卡（进程正在初始化）
  if [ -f "$CLAIM_DIR/$idx" ]; then
    age=$(( NOW - $(stat -c %Y "$CLAIM_DIR/$idx" 2>/dev/null || echo 0) ))
    [ "$age" -lt "$CLAIM_TTL" ] && continue
  fi
  score=$(( free - util * 100 ))       # 空闲显存为主，利用率作次序微调
  if [ "$score" -gt "$BEST_SCORE" ]; then BEST_SCORE=$score; BEST=$idx; fi
done < <(query)

[ -n "$BEST" ] || { echo "ERR:没有满足 ${MIN_FREE_MB}MB 空闲的 GPU" >&2; exit 2; }
echo "${USER:-unknown}:$$" > "$CLAIM_DIR/$BEST"
echo "$BEST"
