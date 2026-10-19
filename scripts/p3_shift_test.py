# -*- coding: utf-8 -*-
"""p3_shift_test.py — the frozen method of p3_formal_test.py, trained on scenario fields whose
announced infrastructure layer leads the commissioned layer by a cluster-specific lead drawn
uniformly in [3, 7] years (opportunity.ANNOUNCE_SHIFT_RANGE), instead of exactly 4 years.

Motivation: trained on the exact 4-year shift, the policy fell below ranking on announced plans in
the observed metro field, whose announced layer is the construction start (about 4-6 years before
opening, varying by line). The range [3, 7] is set in advance and not tuned. Everything else is as
in p3_formal_test.py (training draws 101-120, seeds 0-4). The announced forecast and the rules are
unchanged. Evaluation, once, with the standard fields (exact 4-year shift):
  validation draws 201-210, test draws 1-10 (listing draw 7); main field and metro field (listing
  draws 7, 0, 1, 2, 3, 4).

Usage (repo root):
  OMP_NUM_THREADS=1 PYTHONPATH=src TPMORL_NJOBS=1 python scripts/p3_shift_test.py --seed 0
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1_common as P                                                    # noqa: E402
from p1_fqi_job import row, ev_value                                      # noqa: E402
from exp_fqi_local import make_trees                                      # noqa: E402
from tpmorl.rl import fqi                                                 # noqa: E402
import p3_explore_rulefeat as E                                           # noqa: E402
from p3_formal_test import FROZEN                                         # noqa: E402
from p3_metro_test import metro_setup, LISTING                            # noqa: E402

SHIFT = (3.0, 7.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="results_p3_shift")
    a = ap.parse_args()
    f = FROZEN
    from tpmorl.env import opportunity as OPP
    OPP.HISTORY = int(f["history"]); OPP.INFRA_ANNOUNCE_LEAD = float(f["lead"])
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time(); data = []
    OPP.ANNOUNCE_SHIFT_RANGE = SHIFT                      # training scenarios only
    for k in [f["train_base"] + i for i in range(f["n_train"])]:
        C, ann, efn = E.field_setup(k, 1)
        EV = C["EV"]
        reward = (lambda Q: (lambda u, t: float(Q[u, min(t, Q.shape[1] - 1)])))(EV)
        data += fqi.collect(efn, reward, {"random": None, "myopic": C["EVm"], "temporal_greedy": EV},
                            n_ep_each=f["ep_each"], seed=a.seed * 10007 + k)
    OPP.ANNOUNCE_SHIFT_RANGE = None                       # evaluation on the standard fields
    mk = make_trees(seed=a.seed)
    qa, qb, _ = fqi.fit_double(data, P.BACKUP_GAMMA, lambda: E.ResidualModel(mk()),
                               verbose=False, n_iter=f["n_iter"])
    print(f"[s{a.seed}] trained in {time.time() - t0:.0f}s", flush=True)
    del data
    rows, path = [], os.path.join(a.out, f"shift_s{a.seed}.csv")
    jobs = [("validation", j, 7) for j in range(201, 211)] + [("test", j, 7) for j in range(1, 11)] \
        + [("main", None, ls) for ls in LISTING] + [("metro", None, ls) for ls in LISTING]
    for setname, j, ls in jobs:
        if setname == "metro":
            C, ann, efn = metro_setup(ls, "metro")
        elif setname == "main":
            C, ann, efn = metro_setup(ls, "cluster")
        else:
            C, ann, efn = E.field_setup(j, 1)
        init, _, _ = fqi.rollout(efn, qa, qb, None, collect_q=True)
        name = setname if j is None else f"field{j}"
        r = row(C, "rl_shift", init, ev_value(C["EV"], init), name, a.seed, 0.0)
        r.update(set=setname, listing_seed=ls); rows.append(r)
        pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[s{a.seed}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
