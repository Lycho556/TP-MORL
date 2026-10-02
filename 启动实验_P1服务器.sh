#!/usr/bin/env bash
# =============================================================================
# 启动实验_P1服务器.sh —— P1 审稿补充实验中需要训练 FQI 的部分（服务器后台跑）
#
# 一句话启动（在服务器仓库根目录）：
#     nohup bash 启动实验_P1服务器.sh > ~/p1.log 2>&1 &
#
# 本地已完成、不需要在服务器上重跑的：
#     滚动时域基准（scripts/p1_rolling_horizon.py，3 分钟）
#     空间统计与 7 个空间目标（scripts/p1_spatial.py）
#
# 作业组（JOB_GROUPS 环境变量选择，默认全部）：
#   main     主配置 FQI：M ∈ {10,20,50,0} × 训练种子 0–4，每个模型在列表种子
#            7,0,1,2,3,4 上评估（不重训），并跑监督评分器与随机基准      20 个作业
#   noconf   去掉符合性闸门后所有规则重跑：M=20 × 种子 0–4                   5 个作业
#   metro    地铁实测基础设施场：M ∈ {20,0} × 种子 0–4                      10 个作业
#   oos      样本外：在主场训练，种子 0–4，评估场种子 1–5，并在每个场上
#            各自训练一次作对照（--insample-too）                            5 个作业
#   variant  六个场变体，M=20 × 种子 0–4（单线程确定性重跑）               30 个作业
#
# 资源：每个作业单线程、峰值 < 1.2 GB。M=20 约 5–6 分钟/作业；
#       M=0（无初筛）约 15–40 分钟/作业；oos 约 30 分钟/作业（含 5 次场内重训）。
#       NPROC 默认取 CPU 核数 − 2（上限 32）。整组大约 6–10 CPU 小时。
#
# 环境变量：
#     JOB_GROUPS="main noconf"  只跑指定组（注意不能用 GROUPS，那是 bash 内置变量）
#     NPROC=16              并行作业数
#     FORCE=1               忽略已有 CSV 重跑（默认断点续跑：已有 CSV 的作业跳过）
#     PY=/path/to/python    指定 python（默认当前环境 python）
#     PUSH_EACH=0           关闭逐作业提交（默认开：每个作业一结束就 commit+push 它自己的结果）
#     AUTO_COMMIT=0         关闭收尾的汇总提交（默认开）
#     NO_PUSH=1             只 commit 不 push
#
# 逐作业发布：32 路并行会同时撞 git，所以 add/commit/push 整段用 flock 串行化，
# 否则 index.lock 冲突会让一部分作业的结果推不上去。提交备注里带该作业各规则的
# ratio 摘要。运行记录写在 results_p1/fqi_runs/<批次>/（logs/ 被 .gitignore 忽略，
# 放那里的东西 git add 会直接报错、整条提交失败）。
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1
export PYTHONPATH=src TPMORL_NJOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY="${PY:-python}"
JOB_GROUPS="${JOB_GROUPS:-main noconf metro oos variant}"
FORCE="${FORCE:-0}"
NC=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)
NPROC="${NPROC:-$(( NC > 3 ? (NC - 2 < 32 ? NC - 2 : 32) : 1 ))}"
OUT=results_p1/fqi
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/p1_${STAMP}"
mkdir -p "$OUT" "$LOGDIR"
RUNDIR="results_p1/fqi_runs/${STAMP}"          # 入库的运行记录
STATUS_MD="$RUNDIR/STATUS.md"; FAILDIR="$RUNDIR/fail_logs"
GIT_LOCK="$LOGDIR/.git.lock"
CODE_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
PUSH_EACH="${PUSH_EACH:-1}"; NO_PUSH="${NO_PUSH:-0}"
mkdir -p "$RUNDIR"
printf '# P1 服务器作业状态\n\n批次 %s，代码 `%s`\n\n| 作业 | 状态 | 用时 | 时间 |\n|---|---|---|---|\n' \
  "$STAMP" "$CODE_SHA" > "$STATUS_MD"
say() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# ---------------------------------------------------------------- 自检
say "自检：依赖 / 数据 / 默认配置逐位复现"
"$PY" - <<'PY' || { echo "[中止] 自检未通过"; exit 1; }
import importlib, os, sys
for m in ("numpy", "pandas", "scipy", "sklearn", "torch", "joblib"):
    importlib.import_module(m)
need = ["data/processed/gm_dataset_v1/zones_v0/candidate_units.csv",
        "data/processed/gm_dataset_v1/metro/stations.csv",
        "data/processed/gm_dataset_v1/metro/unit_station_distance.csv",
        "scripts/p1_common.py", "scripts/p1_fqi_job.py", "scripts/p1_rolling_horizon.py"]
miss = [p for p in need if not os.path.exists(p)]
assert not miss, f"缺文件：{miss}"
sys.path.insert(0, "scripts")
import numpy as np, p1_common as C
G, env_fn, EV, EVm, elig, v, _ = C.setup(20)
_, vs = G.run_policy(env_fn(), EV, EVm, "myopic", quota=3, rng=np.random.default_rng(0))
_, vt = G.run_policy(env_fn(), EV, EVm, "forecast", quota=3, rng=np.random.default_rng(0))
ok = (v == 6876.246054424286 and vs / v == 0.7745224826431258 and vt / v == 0.8935554552319573)
print("  reference", v, "persistence", vs / v, "trajectory", vt / v, "->", "PASS" if ok else "FAIL")
import sklearn; print("  sklearn", sklearn.__version__, "numpy", np.__version__)
assert ok, "默认配置没有逐位复现：检查代码版本与依赖版本（本地为 sklearn 1.9.1 / numpy 2.4.6）"
PY

# ---------------------------------------------------------------- 作业清单
JOBS="$LOGDIR/jobs.txt"; : > "$JOBS"
LS="--listing-seeds 7 0 1 2 3 4"
add() {  # add <csv-name> <args...>
  local name="$1"; shift
  if [ "$FORCE" != "1" ] && [ -s "$OUT/$name.csv" ]; then return; fi
  echo "$name|$*" >> "$JOBS"
}
for g in $JOB_GROUPS; do
  case "$g" in
    main)    for M in 20 10 50 0; do for s in 0 1 2 3 4; do
               add "main_M${M}_s${s}" --mode main --seed $s --prescreen $M --with-supervised --with-random $LS
             done; done ;;
    noconf)  for s in 0 1 2 3 4; do
               add "noconf_M20_s${s}" --mode noconf --seed $s --prescreen 20 --with-supervised --with-random $LS
             done ;;
    metro)   for M in 20 0; do for s in 0 1 2 3 4; do
               add "metro_M${M}_s${s}" --mode metro --seed $s --prescreen $M --with-supervised --with-random $LS
             done; done ;;
    oos)     for s in 0 1 2 3 4; do
               add "oos_M20_s${s}" --mode oos --seed $s --prescreen 20 --eval-field-seeds 1 2 3 4 5 --insample-too --with-supervised --with-random
             done ;;
    variant) for v in amp_low amp_high win_narrow win_broad onset_early onset_late; do for s in 0 1 2 3 4; do
               add "variant-${v}_M20_s${s}" --mode variant --variant $v --seed $s --prescreen 20 --with-random
             done; done ;;
    *) echo "未知组 $g"; exit 1 ;;
  esac
done
N=$(wc -l < "$JOBS" | tr -d ' ')
say "共 $N 个作业，并行 $NPROC，日志 $LOGDIR/"
[ "$N" = "0" ] && { say "无待跑作业"; }

NJOBS_TOTAL="$N"

# ---------------------------------------------------------------- git 发布
# push_main：推到 origin/main 并以 SHA 比对确认落地（ls-remote 只读成功证明不了推上去了）。
# 被拒就 rebase 重推；若 rebase 带进了别人对 src/ scripts/ 的改动，大声告警——
# 后续作业会用到新代码，冻结协议被破坏。调用方必须已持有 GIT_LOCK。
push_main() {
  [ "$NO_PUSH" = "1" ] && return 0
  local try L R before
  for try in 1 2 3 4 5; do
    git push -q origin HEAD:main >/dev/null 2>&1
    L=$(git rev-parse HEAD); R=$(git ls-remote origin main 2>/dev/null | cut -f1)
    [ -n "$R" ] && [ "$L" = "$R" ] && return 0
    before=$(git rev-parse HEAD)
    git pull -q --rebase --autostash origin main >/dev/null 2>&1 || git rebase --abort >/dev/null 2>&1
    git diff --quiet "$before" HEAD -- src scripts 2>/dev/null \
      || echo "[告警] 跑批期间远端改了 src/ 或 scripts/，后续作业将使用新代码" | tee -a "$LOGDIR/STATUS.txt"
    sleep $(( try * 5 ))
  done
  return 1
}

# publish_job：一个作业结束立即提交它自己的产物，备注里带各规则的 ratio 摘要。
publish_job() {  # publish_job <作业名> <OK|FAIL> <秒>
  [ "$PUSH_EACH" = "1" ] || return 0
  local name="$1" st="$2" sec="$3"
  (
    flock -w 900 9 || { echo "[git] $name 取锁超时，本次不提交（结果仍在磁盘，收尾会补）"; exit 0; }
    local files=() note done_n
    [ -s "$OUT/$name.csv" ] && files+=("$OUT/$name.csv")
    [ -s "$OUT/${name}_schedule.json" ] && files+=("$OUT/${name}_schedule.json")
    if [ "$st" = "FAIL" ]; then            # 失败日志进库，否则远端看不到失败原因
      mkdir -p "$FAILDIR"; tail -n 80 "$LOGDIR/$name.log" > "$FAILDIR/$name.log" 2>/dev/null
      files+=("$FAILDIR/$name.log")
    fi
    printf '| %s | %s | %s 秒 | %s |\n' "$name" "$st" "$sec" "$(date '+%F %T')" >> "$STATUS_MD"
    files+=("$STATUS_MD")
    note=$("$PY" - "$OUT/$name.csv" <<'PYN' 2>/dev/null
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
if not {"rule", "ratio"} <= set(d.columns): sys.exit()
g = d[d["rule"] != "reference"].groupby("rule")["ratio"].agg(["mean", "min", "max", "count"])
print("各规则相对参照（ratio）："); print(g.round(4).to_string())
PYN
)
    git add -- "${files[@]}" >/dev/null 2>&1
    git diff --cached --quiet && exit 0
    done_n=$(grep -c '^OK' "$LOGDIR/STATUS.txt" 2>/dev/null); done_n=${done_n:-0}
    git commit -q -m "P1 作业 ${name}：${st}（${sec} 秒）

${note:-（无可摘要的 CSV）}

批次 ${STAMP}，已完成 ${done_n}/${NJOBS_TOTAL}，代码 ${CODE_SHA}
单线程确定性运行（TPMORL_NJOBS=1）；默认配置自检逐位通过后才开跑。" || exit 0
    if push_main; then echo "[git] $name 已推送 $(git rev-parse --short HEAD)"
    else echo "[git] $name 推送 5 次未落地，已 commit 在本地，收尾会再推"; fi
  ) 9>"$GIT_LOCK"
}

run_one() {
  local line="$1"; local name="${line%%|*}"; local args="${line#*|}"
  local t0=$(date +%s) st dt
  if "$PY" scripts/p1_fqi_job.py $args --out "$OUT" > "$LOGDIR/$name.log" 2>&1; then st=OK; else st=FAIL; fi
  dt=$(( $(date +%s) - t0 ))
  if [ "$st" = OK ]; then
    echo "OK   $name ${dt}s" | tee -a "$LOGDIR/STATUS.txt"
  else
    echo "FAIL $name ${dt}s (see $LOGDIR/$name.log)" | tee -a "$LOGDIR/STATUS.txt"
  fi
  publish_job "$name" "$st" "$dt"
}
export -f run_one push_main publish_job
export PY OUT LOGDIR STAMP RUNDIR STATUS_MD FAILDIR GIT_LOCK CODE_SHA PUSH_EACH NO_PUSH NJOBS_TOTAL
# 先单独跑复现作业（main M20 s0）：失败即中止，避免整批白跑
if grep -q '^main_M20_s0|' "$JOBS"; then
  say "复现作业 main_M20_s0（基准 0.9095038725980372；跨平台容差 ${P1_REPRO_TOL:-0.01}，逐位相同记 PASS，容差内记 PASS_TOL）"
  run_one "$(grep '^main_M20_s0|' "$JOBS")"
  grep -q "^OK   main_M20_s0" "$LOGDIR/STATUS.txt" || { echo "[中止] 复现失败"; exit 1; }
  grep -v '^main_M20_s0|' "$JOBS" > "$JOBS.rest"; mv "$JOBS.rest" "$JOBS"
fi
tr '\n' '\0' < "$JOBS" | xargs -0 -P "$NPROC" -I{} bash -c 'run_one "$@"' _ {}

# ---------------------------------------------------------------- 汇总
say "汇总"
"$PY" scripts/p1_collect.py --fqi "$OUT" | tee "$LOGDIR/summary.txt"
grep -c "^FAIL" "$LOGDIR/STATUS.txt" >/dev/null && say "有失败作业：$(grep '^FAIL' "$LOGDIR/STATUS.txt" | wc -l)"

if [ "${AUTO_COMMIT:-1}" = "1" ]; then
  # 原写法 `git add results_p1 "$LOGDIR"`：logs/ 在 .gitignore 里，git add 会对整条命令报错，
  # 结果是什么都没提交。改为把 STATUS.txt 与汇总复制进入库的运行记录目录。
  cp "$LOGDIR/STATUS.txt" "$LOGDIR/summary.txt" "$RUNDIR/" 2>/dev/null
  OKN=$(grep -c '^OK' "$LOGDIR/STATUS.txt" 2>/dev/null); OKN=${OKN:-0}
  FAILN=$(grep -c '^FAIL' "$LOGDIR/STATUS.txt" 2>/dev/null); FAILN=${FAILN:-0}
  (
    flock -w 900 9 || exit 0
    git add -- results_p1/fqi "$RUNDIR" >/dev/null 2>&1
    if ! git diff --cached --quiet; then
      git commit -q -m "P1 服务器批次 ${STAMP} 收尾：成功 ${OKN} / 失败 ${FAILN}

$(head -n 40 "$LOGDIR/summary.txt" 2>/dev/null)

代码 ${CODE_SHA}；汇总表 results_p1/fqi/summary_by_config.csv、summary_oos.csv"
    fi
    if push_main; then
      say "收尾已推送 $(git rev-parse --short HEAD)"
      if [ "$FAILN" = "0" ] && [ "$NO_PUSH" != "1" ]; then
        git tag -f "p1-complete-${STAMP}" -m "P1 服务器批次 ${STAMP}：${OKN} 个作业全部成功" >/dev/null 2>&1
        git push -q -f origin "p1-complete-${STAMP}" >/dev/null 2>&1 && say "已打标签 p1-complete-${STAMP}"
      fi
    else
      say "[告警] 收尾推送失败，结果已 commit 在本地：git push origin HEAD:main"
    fi
  ) 9>"$GIT_LOCK"
fi
say "完成。逐作业结果已随跑随推；状态表 $STATUS_MD"
