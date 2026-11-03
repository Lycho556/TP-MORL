#!/usr/bin/env bash
# P7 validation runs: E3 learning to rank + E2 conservative deviation (see docs/P7/validation_plan.md).
# EXPLORATORY: validation draws 201-210 and the observed metro field only. Test draws 1-10 and the
# main field are refused by the scripts themselves (p7_common.guard_fields).
#
#   Stage A  scripts/p7_rank.py, one job per training seed (0-4): P3 control + two LambdaMART rankers,
#            evaluated on 201-210; seed 0 also runs the rule baselines and the leakage check.
#            Each job saves its P3 model to $P7_MODEL_DIR (outside the repo, never committed).
#   Stage B  scripts/p7_safe.py: five-seed ensemble with uncertainty brakes, on 201-210 and metro.
#   Stage C  scripts/p7_summary.py -> results_p7_rank/SUMMARY.md, results_p7_safe/SUMMARY.md
#
# Every job commits + pushes its own results at the end (commit title "[p7-val] ...", per-rule
# ratio summary and self-checks in the message); a final commit carries both SUMMARY.md files and the
# tag p7-val-complete-<run>. git is serialised with flock; a push counts as landed only when the
# remote SHA equals HEAD; rejected pushes rebase and retry.
#
# Options (environment variables)
#   PY=python      interpreter (numpy pandas scipy scikit-learn joblib lightgbm)
#   NPROC=5        parallel jobs in stage A (each ~1 GB, ~6-10 min)
#   FORCE=1        rerun jobs whose log already says "done in"
#   AUTO_PUSH=0    leave git untouched
#   SMOKE=1        tiny settings (2 training draws, 2 validation draws, seeds 0-1), no push
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
NPROC=${NPROC:-5}
FORCE=${FORCE:-0}
AUTO_PUSH=${AUTO_PUSH:-1}
export PYTHONPATH=src TPMORL_NJOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export P7_MODEL_DIR=${P7_MODEL_DIR:-$HOME/tpmorl_p7_models}

RANK=results_p7_rank; SAFE=results_p7_safe; SEED_LIST="0 1 2 3 4"; JOB_ARGS=""; SAFE_ARGS=""
if [ "${SMOKE:-0}" = 1 ]; then
  RANK=${RANK}_smoke; SAFE=${SAFE}_smoke; SEED_LIST="0 1"; AUTO_PUSH=0
  JOB_ARGS="--n-train 2 --val-fields 201 202 --metro-listing 7"
  SAFE_ARGS="--val-fields 201 202 --metro-listing 7"
  export P7_MODEL_DIR=$P7_MODEL_DIR/smoke
fi
FIRST_SEED=${SEED_LIST%% *}
mkdir -p "$RANK" "$SAFE"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
CODE_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo none)
STAMP=$(date +%Y%m%d_%H%M%S)
GIT_LOCK="$(git rev-parse --git-dir)/p7_publish.lock"     # inside .git: never committed
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$RANK/STATUS.txt"; }

# ---------------------------------------------------------------- git publishing (as in the P2 launcher)
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
      || echo "[$(date '+%F %T')] WARN: src/ or scripts/ changed on origin/$BRANCH during the run" >> "$RANK/STATUS.txt"
    sleep $(( try * 5 ))
  done
  return 1
}
commit_push(){   # commit_push <message> <files...>
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
# note <csv>: mean ratio per rule and field set, for the commit message (python -c, not a heredoc:
# an exported function whose heredoc is followed by an operator breaks under export -f)
NOTE_PY='
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
g = d.groupby(["field_set", "rule"])["ratio"].agg(["mean", "min", "max", "count"]).round(4)
print("ratio by field set and rule:"); print(g.to_string())
'
note(){ "$PY" -c "$NOTE_PY" "$1" 2>/dev/null; grep -hE "self-check|fairness|leak check on|chosen on" "$2" 2>/dev/null; return 0; }

run_rank_job(){  # run_rank_job <seed>
  local s=$1 lg="$RANK/log_s$1.txt" extra="" st t0 mins
  if [ "$FORCE" != 1 ] && [ -f "$lg" ] && grep -q "done in" "$lg"; then echo "skip rank s$s (done)"; return 0; fi
  [ "$s" = "$FIRST_SEED" ] && extra="--baselines --leak-check"
  t0=$(date +%s)
  if "$PY" scripts/p7_rank.py --seed "$s" $extra $JOB_ARGS --out "$RANK" > "$lg" 2>&1; then st=OK; else st=FAIL; fi
  mins=$(( ($(date +%s) - t0) / 60 ))
  echo "[$(date '+%F %T')] $st E3 rank seed $s (${mins} min)" | tee -a "$RANK/STATUS.txt"
  commit_push "[p7-val] E3 rank seed ${s}: ${st} (${mins} min)

$(note "$RANK/rank_s$s.csv" "$lg")

run ${STAMP}, code ${CODE_SHA}, branch ${BRANCH}; validation draws 201-210 only; P3 control retrained
single-threaded (TPMORL_NJOBS=1); plan in docs/P7/validation_plan.md" \
    "$RANK/rank_s$s.csv" "$lg" "$RANK/leak_check.csv" "$RANK/STATUS.txt"
  [ "$st" = OK ]
}
export -f push_branch commit_push note run_rank_job
export PY RANK SAFE BRANCH CODE_SHA STAMP GIT_LOCK AUTO_PUSH FORCE JOB_ARGS FIRST_SEED NOTE_PY

log "start: NPROC=$NPROC seeds=[$SEED_LIST] code=$CODE_SHA branch=$BRANCH models=$P7_MODEL_DIR"
# without flock every commit_push would be skipped silently and nothing would reach GitHub
if [ "$AUTO_PUSH" = 1 ] && ! command -v flock >/dev/null 2>&1; then
  log "ERROR: flock not found; per-job pushes would all be skipped. Install util-linux or use AUTO_PUSH=0"; exit 1
fi
"$PY" -c "import lightgbm, sklearn, numpy; print('  lightgbm', lightgbm.__version__, 'sklearn', sklearn.__version__, 'numpy', numpy.__version__)" \
  | tee -a "$RANK/STATUS.txt"

# ---------------------------------------------------------------- stage A: E3 (+ P3 control, models for E2)
log "stage A: E3 rank jobs for seeds [$SEED_LIST]"
printf '%s\n' $SEED_LIST | xargs -P "$NPROC" -I{} bash -c 'run_rank_job "$1"' _ {}

# ---------------------------------------------------------------- stage B: E2
for s in $SEED_LIST; do
  [ -f "$P7_MODEL_DIR/p3_s$s.joblib" ] || { log "ERROR: model $P7_MODEL_DIR/p3_s$s.joblib missing (rerun stage A with FORCE=1)"; exit 1; }
done
log "stage B: E2 safe deviation"
t0=$(date +%s)
if "$PY" scripts/p7_safe.py --seeds $SEED_LIST $SAFE_ARGS --rank-dir "$RANK" --out "$SAFE" > "$SAFE/log.txt" 2>&1; then st=OK; else st=FAIL; fi
mins=$(( ($(date +%s) - t0) / 60 ))
log "$st E2 safe (${mins} min)"
commit_push "[p7-val] E2 safe deviation: ${st} (${mins} min)

$(note "$SAFE/safe.csv" "$SAFE/log.txt")

run ${STAMP}, code ${CODE_SHA}, branch ${BRANCH}; five-seed P3 ensemble; variants chosen on 201-210 only;
metro reported, not used to choose" \
  "$SAFE/safe.csv" "$SAFE/selection.txt" "$SAFE/log.txt" "$RANK/STATUS.txt" || true
[ "$st" = OK ] || exit 1

# ---------------------------------------------------------------- stage C: summary
"$PY" scripts/p7_summary.py --rank-dir "$RANK" --safe-dir "$SAFE" > "$SAFE/summary_stdout.txt" 2>&1
log "done; summaries in $RANK/SUMMARY.md and $SAFE/SUMMARY.md"
if [ "$AUTO_PUSH" = 1 ]; then
  BEFORE=$(git rev-parse HEAD)
  commit_push "[p7-val] E2 + E3 validation complete

$(grep -E '^\| (p3|rank_|safe_)' "$RANK/SUMMARY.md" "$SAFE/SUMMARY.md" 2>/dev/null | cut -c1-220)

run ${STAMP}, code ${CODE_SHA}, branch ${BRANCH}" \
    "$RANK" "$SAFE" || true
  # tag only if the final commit was actually made AND landed (otherwise the tag would sit on an old commit)
  if [ "$(git rev-parse HEAD)" != "$BEFORE" ] && \
     [ "$(git rev-parse HEAD)" = "$(git ls-remote origin "refs/heads/$BRANCH" 2>/dev/null | cut -f1)" ]; then
    git tag -f "p7-val-complete-${STAMP}" -m "P7 validation E2 + E3" >/dev/null 2>&1 || true
    # echo, not log: writing STATUS.txt after the final commit would leave the tree dirty
    git push -q -f origin "p7-val-complete-${STAMP}" >/dev/null 2>&1 \
      && echo "[$(date '+%F %T')] tagged p7-val-complete-${STAMP}" || echo "[$(date '+%F %T')] WARN: tag push failed"
  else
    echo "[$(date '+%F %T')] WARN: final commit not made or not landed; no completion tag. Check: git status; git push origin HEAD:$BRANCH"
  fi
fi
