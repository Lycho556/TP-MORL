#!/usr/bin/env bash
# P2 formal transfer test (see docs/P2正式测试说明.md). Uses only existing flags of
# scripts/p2_generalize.py; no new method, no change to src/.
#
#   Stage 1  choose the history window H on validation fields 201-210
#            (100 training scenarios x 2 episodes, seed 0, H = 0 / 1 / 2)
#   Stage 2  formal test on fields 1-10 (+ main field), seeds 0-4, chosen H, run once
#   Stage 3  summary + self-checks -> results_p2_formal/SUMMARY.md
#
# Options (environment variables)
#   PY=python          interpreter of the conda env (numpy pandas scipy scikit-learn joblib)
#   NPROC=<n>          parallel jobs (default: min(cores-1, free GB / 5, 5)); each job ~4.5 GB
#   H_FIXED=<h>        skip stage 1 and use this H (e.g. H_FIXED=1)
#   FORCE=1            rerun jobs whose log already says "done in"
#   AUTO_PUSH=1        commit + push after EVERY job, after stage 1, and at the end (default 1;
#                      AUTO_PUSH=0 leaves git untouched). Each commit message carries the job's
#                      per-rule ratio summary. git is serialised with flock; a push counts as
#                      landed only when the remote SHA equals HEAD; rejected pushes rebase and retry.
#   SMOKE=1            tiny settings to test the pipeline (minutes, results meaningless)
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
PY=${PY:-python}
OUT=${OUT:-results_p2_formal}
FORCE=${FORCE:-0}
AUTO_PUSH=${AUTO_PUSH:-1}
export PYTHONPATH=src TPMORL_NJOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

NTRAIN=100; EPEACH=2; SEEDS="0 1 2 3 4"; HS="0 1 2"
VAL="201 202 203 204 205 206 207 208 209 210"; TEST="1 2 3 4 5 6 7 8 9 10"
if [ "${SMOKE:-0}" = 1 ]; then
  NTRAIN=2; EPEACH=1; SEEDS="0 1"; HS="0 1"; VAL="201 202"; TEST="1 2"; OUT=${OUT}_smoke; AUTO_PUSH=0
fi

if [ -z "${NPROC:-}" ]; then
  cores=$( (nproc || getconf _NPROCESSORS_ONLN) 2>/dev/null )
  memgb=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo 2>/dev/null || echo 16)
  NPROC=$(( cores - 1 )); m=$(( memgb / 5 )); [ $m -lt $NPROC ] && NPROC=$m
  [ $NPROC -gt 5 ] && NPROC=5; [ $NPROC -lt 1 ] && NPROC=1
fi
mkdir -p "$OUT"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
CODE_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo none)
STAMP=$(date +%Y%m%d_%H%M%S)
GIT_LOCK="$(git rev-parse --git-dir)/p2_publish.lock"     # inside .git: never committed

# ---------------------------------------------------------------- git publishing
# push_branch: push HEAD to origin/$BRANCH; landed only if the remote SHA equals HEAD
# (a successful ls-remote alone proves nothing). On rejection rebase and retry; warn loudly
# if the rebase brought in someone else's src/ or scripts/ changes (frozen protocol broken).
push_branch(){
  set +e
  local try L R before
  for try in 1 2 3 4 5; do
    git push -q origin "HEAD:$BRANCH" >/dev/null 2>&1
    L=$(git rev-parse HEAD); R=$(git ls-remote origin "refs/heads/$BRANCH" 2>/dev/null | cut -f1)
    [ -n "$R" ] && [ "$L" = "$R" ] && return 0
    before=$L
    git pull -q --rebase --autostash origin "$BRANCH" >/dev/null 2>&1 || git rebase --abort >/dev/null 2>&1
    git diff --quiet "$before" HEAD -- src scripts 2>/dev/null \
      || echo "[$(date '+%F %T')] WARN: src/ or scripts/ changed on origin/$BRANCH during the run" >> "$OUT/STATUS.txt"
    sleep $(( try * 5 ))
  done
  return 1
}
# commit_push <message> <files...>   (caller holds no lock; this takes it)
commit_push(){
  [ "$AUTO_PUSH" = 1 ] || return 0
  local msg=$1; shift
  (
    set +e
    flock -w 1800 9 || { echo "[git] lock timeout, not committed: ${msg%%$'\n'*}"; exit 0; }
    local f; for f in "$@"; do [ -e "$f" ] && git add -- "$f" >/dev/null 2>&1; done
    git diff --cached --quiet && exit 0
    git commit -q -m "$msg" || exit 0
    if push_branch; then echo "[git] pushed $(git rev-parse --short HEAD): ${msg%%$'\n'*}"
    else echo "[git] WARN push not landed after 5 tries (committed locally): ${msg%%$'\n'*}"; fi
  ) 9>"$GIT_LOCK"
}
# job_note <dir> <seed>: per-rule ratio summary of one job, for the commit message.
# Python passed via -c (not a heredoc): an exported function whose heredoc is followed by
# `|| true` is re-serialised by `export -f` into invalid syntax in the child shells.
JOB_NOTE_PY='
import sys, glob, os, pandas as pd
d, s = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(os.path.join(d, f"gen_s{s}.csv")) + glob.glob(os.path.join(d, f"gen_s{s}_h*.csv")))
if not fs: sys.exit()
x = pd.read_csv(fs[0])
print("ratio by rule over this job fields:")
print(x.groupby("rule")["ratio"].agg(["mean", "min", "max", "count"]).round(4).to_string())
if {"value_based_multi", "rh_announced"} <= set(x["rule"]):
    k = lambda r: x[x["rule"] == r].drop_duplicates("eval_field").set_index("eval_field")["ratio"]
    dd = (k("value_based_multi") - k("rh_announced")).dropna()
    print(f"value_based_multi - rh_announced: mean {dd.mean():+.4f}, higher on {(dd > 0).sum()}/{len(dd)} fields")
'
job_note(){ "$PY" -c "$JOB_NOTE_PY" "$1" "$2" 2>/dev/null; return 0; }
# publish_job <dir> <seed> <H> <OK|FAIL> <seconds>: called by every job right after it ends
publish_job(){
  local dir=$1 seed=$2 h=$3 st=$4 secs=$5 stage
  case "$(basename "$dir")" in
    hval_h*) stage="stage 1 (validation fields 201-210)" ;;
    test)    stage="stage 2 (test fields 1-10 + main)" ;;
    *)       stage="$dir" ;;
  esac
  commit_push "P2 formal ${stage} H=${h} seed ${seed}: ${st} ($(( secs / 60 )) min)

$(job_note "$dir" "$seed")

run ${STAMP}, code ${CODE_SHA}, branch ${BRANCH}; NTRAIN=${NTRAIN} EPEACH=${EPEACH}
single-threaded (TPMORL_NJOBS=1); protocol fixed in docs/P2正式测试说明.md" \
    "$dir/gen_s${seed}.csv" "$dir/gen_s${seed}_h${h}.csv" "$dir/log_s${seed}.txt" "$OUT/STATUS.txt"
}
export -f push_branch commit_push job_note publish_job
export PY OUT BRANCH CODE_SHA STAMP GIT_LOCK AUTO_PUSH NTRAIN EPEACH JOB_NOTE_PY

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OUT/STATUS.txt"; }
log "start: NPROC=$NPROC NTRAIN=$NTRAIN EPEACH=$EPEACH SEEDS=[$SEEDS] git=$(git rev-parse --short HEAD 2>/dev/null || echo none)"

# one job = one line "<dir> <seed> <H> <fields...> [--baselines]"
run_jobs(){
  local jobfile=$1
  # strip trailing blanks first: with -L, a line ending in a blank is CONTINUED onto the next
  # line, so "<dir> <seed> <H> <fields> " (empty $extra) would merge two jobs into one command
  # (H=1+H=2 in stage 1, seeds 1-4 in stage 2) and argparse rejects the stray tokens.
  grep -v '^$' "$jobfile" | sed 's/[[:space:]]*$//' | xargs -P "$NPROC" -L 1 bash -c '
    dir=$0; seed=$1; h=$2; shift 2
    mkdir -p "$dir"; lg="$dir/log_s${seed}.txt"
    if [ "'"$FORCE"'" != 1 ] && [ -f "$lg" ] && grep -q "done in" "$lg"; then echo "skip $dir s$seed (done)"; exit 0; fi
    t0=$(date +%s)
    if '"$PY"' scripts/p2_generalize.py --seed "$seed" --n-train '"$NTRAIN"' --ep-each '"$EPEACH"' \
        --history "$h" --test-fields "$@" --out "$dir" > "$lg" 2>&1; then st=OK; else st=FAIL; fi
    if [ "$st" = OK ]; then echo "OK   $dir s$seed"; else echo "FAIL $dir s$seed (see $lg)"; fi
    publish_job "$dir" "$seed" "$h" "$st" $(( $(date +%s) - t0 ))
    [ "$st" = OK ]'
}

# ---------------------------------------------------------------- stage 1: choose H on validation
if [ -n "${H_FIXED:-}" ]; then
  H=$H_FIXED; log "stage 1 skipped, H fixed to $H"
else
  : > "$OUT/jobs_stage1.txt"
  for h in $HS; do
    extra=""; [ "$h" = "$(echo $HS | awk '{print $1}')" ] && extra="--baselines"
    echo "$OUT/hval_h$h 0 $h $VAL $extra" >> "$OUT/jobs_stage1.txt"
  done
  log "stage 1: $(wc -l < "$OUT/jobs_stage1.txt") validation jobs"
  run_jobs "$OUT/jobs_stage1.txt" | tee -a "$OUT/STATUS.txt"
  dirs=""; for h in $HS; do dirs="$dirs $OUT/hval_h$h"; done
  $PY scripts/p2_formal_summary.py --select-h $dirs --h-values $HS --fields $VAL > "$OUT/stage1_selection.txt"
  H=$(tail -n 1 "$OUT/stage1_selection.txt")
  cat "$OUT/stage1_selection.txt" | tee -a "$OUT/STATUS.txt"
  # self-check: the pilot (results_p2_val/D, H = 1, seed 0) must be reproduced exactly
  if [ "${SMOKE:-0}" != 1 ] && [ -f results_p2_val/D/gen_s0_h1.csv ]; then
    $PY - "$OUT/hval_h1/gen_s0_h1.csv" results_p2_val/D/gen_s0_h1.csv <<'EOF' | tee -a "$OUT/STATUS.txt"
import sys, pandas as pd
a, b = (pd.read_csv(p) for p in sys.argv[1:3])
k = lambda d: d[d.rule == "value_based_multi"].set_index("eval_field").ratio
d = (k(a) - k(b)).abs().max()
v = "PASS (bit-identical)" if d < 1e-9 else "PASS_TOL (cross-platform ExtraTrees, <= 0.01)" if d <= 0.01 else "WARN: check code version"
print(f"self-check vs pilot D (local run, H = 1, seed 0): max |diff| = {d:.2e} -> {v}")
EOF
  fi
fi
commit_push "P2 formal stage 1 done: chosen H = $H

$(cat "$OUT/stage1_selection.txt" 2>/dev/null)
$(grep -h "self-check vs pilot" "$OUT/STATUS.txt" 2>/dev/null | tail -1)

run ${STAMP}, code ${CODE_SHA}, branch ${BRANCH}" \
  "$OUT/stage1_selection.txt" "$OUT/jobs_stage1.txt" "$OUT/STATUS.txt" || true
log "stage 2: formal test with H = $H"

# ---------------------------------------------------------------- stage 2: formal test
: > "$OUT/jobs_stage2.txt"
for s in $SEEDS; do
  extra=""; [ "$s" = 0 ] && extra="--baselines"
  echo "$OUT/test $s $H $TEST $extra" >> "$OUT/jobs_stage2.txt"
done
run_jobs "$OUT/jobs_stage2.txt" | tee -a "$OUT/STATUS.txt"

# ---------------------------------------------------------------- stage 3: summary and checks
$PY scripts/p2_formal_summary.py --runs "$OUT/test" --fields $TEST --out "$OUT/SUMMARY.md" > /dev/null
if [ "${SMOKE:-0}" != 1 ] && ls results_p2/gen_s0.csv > /dev/null 2>&1; then
  $PY - "$OUT/test" <<'EOF' | tee -a "$OUT/STATUS.txt"
import glob, sys, pandas as pd
new = pd.concat([pd.read_csv(f) for f in glob.glob(sys.argv[1] + "/gen_s*.csv")])
old = pd.concat([pd.read_csv(f) for f in glob.glob("results_p2/gen_s?.csv")])
k = lambda d: d[(d.seed == -1) & (d.rule == "rh_announced")].drop_duplicates("eval_field").set_index("eval_field").ratio
d = (k(new) - k(old)).dropna().abs().max()
print(f"self-check announced baselines vs results_p2: max |diff| = {d:.2e} -> {'PASS' if d < 1e-9 else 'WARN'}")
EOF
fi
log "done; summary in $OUT/SUMMARY.md"
cat "$OUT/SUMMARY.md" | head -20

if [ "$AUTO_PUSH" = 1 ]; then
  commit_push "P2 formal transfer test complete (H=$H)

$(head -n 40 "$OUT/SUMMARY.md" 2>/dev/null)

$(grep -h "self-check" "$OUT/STATUS.txt" 2>/dev/null)

run ${STAMP}, code ${CODE_SHA}, branch ${BRANCH}" \
    "$OUT" || true
  if [ "$(git rev-parse HEAD)" = "$(git ls-remote origin "refs/heads/$BRANCH" 2>/dev/null | cut -f1)" ]; then
    git tag -f "p2-formal-complete-${STAMP}" -m "P2 formal transfer test, H=$H" >/dev/null 2>&1 || true
    git push -q -f origin "p2-formal-complete-${STAMP}" >/dev/null 2>&1 \
      && log "tagged p2-formal-complete-${STAMP}" || log "WARN: tag push failed"
  else
    log "WARN: final push not landed; results committed locally, run: git push origin HEAD:$BRANCH"
  fi
fi
