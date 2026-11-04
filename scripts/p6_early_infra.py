# -*- coding: utf-8 -*-
"""p6_early_infra.py -- A2: retrain the learned value correction on scenario fields whose
infrastructure programme ends early (commissioning onsets mapped to model years [-1, 3], peaks
about years 1-8, as the observed metro openings), everything else as the frozen protocol
(draws 101-120, two episodes per behaviour rule, one-step target, lead 4, M = 20).
Pre-registration: results_p6_shift/PREREGISTRATION.md (written before running).
Evaluation, learned vs ranking on announced plans:
  metro    observed metro field, listing draws 7,0,1,2,3,4
  test     the ten original test draws 1-10 (listing 7), original generator
  early    test draws 1-10 with the same early-onset mapping (listing 7)
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/p6_early_infra.py --seed S"""
import argparse, os, sys, time
import joblib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p1_common as P
from exp_fqi_local import make_trees
from tpmorl.rl import fqi
import p3_explore_rulefeat as E
from p3_metro_test import metro_setup
from p1_fqi_job import ev_value
from tpmorl.env import opportunity as OPP
EARLY = (-1.0, 3.0)
LISTING = [7, 0, 1, 2, 3, 4]
OUT = os.path.join(HERE, "..", "results_p6_shift")

def train(seed):
    path = os.path.join(OUT, "models_early", f"s{seed}.joblib")
    if os.path.exists(path): return joblib.load(path)
    OPP.INFRA_ONSET_YEARS = EARLY
    data = []
    for k in range(101, 121):
        C, ann, efn = E.field_setup(k, 1)
        EV = C["EV"]
        reward = (lambda Q: (lambda u, t: float(Q[u, min(t, Q.shape[1] - 1)])))(EV)
        data += fqi.collect(efn, reward, {"random": None, "myopic": C["EVm"], "temporal_greedy": EV},
                            n_ep_each=2, seed=seed * 10007 + k)
    qa, qb, _ = fqi.fit_double(data, P.BACKUP_GAMMA, lambda: E.ResidualModel(make_trees(seed=seed)()),
                               verbose=False, n_iter=1)
    OPP.INFRA_ONSET_YEARS = None
    os.makedirs(os.path.dirname(path), exist_ok=True); joblib.dump((qa, qb), path)
    return qa, qb

def evaluate(name, C, ann, efn, qa, qb, seed, ls):
    init, _, _ = fqi.rollout(efn, qa, qb, None)
    b = {x["rule"]: x["ratio"] for x in E.baselines(C, ann, name)}
    return dict(eval=name, seed=seed, listing=ls, learned=ev_value(C["EV"], init) / C["v_orc"],
                rank_announced=b["rank_announced"], trajectory=b.get("trajectory", np.nan), rh_full=b["rh_full"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); a = ap.parse_args()
    OPP.HISTORY = 1; OPP.INFRA_ANNOUNCE_LEAD = 4.0
    t0 = time.time(); qa, qb = train(a.seed); print(f"[s{a.seed}] trained {time.time()-t0:.0f}s", flush=True)
    rows = []
    for ls in LISTING:
        C, ann, efn = metro_setup(ls, "metro"); rows.append(evaluate("metro", C, ann, efn, qa, qb, a.seed, ls))
    print(f"[s{a.seed}] metro diff {np.mean([r['learned']-r['rank_announced'] for r in rows]):+.4f}", flush=True)
    for k in range(1, 11):
        C, ann, efn = E.field_setup(k, 1); rows.append(evaluate(f"test{k}", C, ann, efn, qa, qb, a.seed, 7))
    OPP.INFRA_ONSET_YEARS = EARLY
    for k in range(1, 11):
        C, ann, efn = E.field_setup(k, 1); rows.append(evaluate(f"early{k}", C, ann, efn, qa, qb, a.seed, 7))
    OPP.INFRA_ONSET_YEARS = None
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"a2_rows_s{a.seed}.csv"), index=False)
    print(f"[s{a.seed}] done {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
