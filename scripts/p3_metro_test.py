# -*- coding: utf-8 -*-
"""p3_metro_test.py — the frozen method of p3_formal_test.py applied to the observed metro field.

The policy is trained exactly as in the registered test (scenario field draws 101-120, seeds 0-4)
and evaluated without retraining on the main configuration with the infrastructure field rebuilt
from the observed metro network (infra_mode = "metro"), on the six listing draws of the paper
(7, 0, 1, 2, 3, 4). In the metro field the announced layer peaks at the start of construction, not
four years before opening, so the shift between announced and commissioned layers differs by line.

Usage (repo root):
  OMP_NUM_THREADS=1 PYTHONPATH=src TPMORL_NJOBS=1 python scripts/p3_metro_test.py --seed 0 [--baselines]
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
from p1_rolling_horizon import forecast_tables                            # noqa: E402
from exp_fqi_local import make_trees                                      # noqa: E402
from tpmorl.rl import fqi                                                 # noqa: E402
import p3_explore_rulefeat as E                                           # noqa: E402
from p3_formal_test import FROZEN                                         # noqa: E402

LISTING = [7, 0, 1, 2, 3, 4]


def metro_setup(listing_seed, infra_mode="metro"):
    G, env_fn, EV, EVm, elig, v_orc, oplan = P.setup(prescreen=20, infra_mode=infra_mode,
                                                     listing_seed=listing_seed)
    C = dict(G=G, env_fn=env_fn, EV=EV, EVm=EVm, elig=elig, v_orc=v_orc, oplan=oplan)
    ann = forecast_tables(env_fn(), "announced", True)
    return C, ann, E.wrap_env_fn(env_fn, ann)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--field", default="metro", choices=["metro", "main"],
                    help="metro = observed metro field; main = main scenario field (cluster infrastructure)")
    ap.add_argument("--out", default="results_p3_metro")
    a = ap.parse_args()
    f = FROZEN
    from tpmorl.env import opportunity as OPP
    OPP.HISTORY = int(f["history"]); OPP.INFRA_ANNOUNCE_LEAD = float(f["lead"])
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time(); data = []
    for k in [f["train_base"] + i for i in range(f["n_train"])]:
        C, ann, efn = E.field_setup(k, 1)
        EV = C["EV"]
        reward = (lambda Q: (lambda u, t: float(Q[u, min(t, Q.shape[1] - 1)])))(EV)
        data += fqi.collect(efn, reward, {"random": None, "myopic": C["EVm"], "temporal_greedy": EV},
                            n_ep_each=f["ep_each"], seed=a.seed * 10007 + k)
    mk = make_trees(seed=a.seed)
    qa, qb, _ = fqi.fit_double(data, P.BACKUP_GAMMA, lambda: E.ResidualModel(mk()),
                               verbose=False, n_iter=f["n_iter"])
    print(f"[s{a.seed}] trained in {time.time() - t0:.0f}s", flush=True)
    del data
    rows, path = [], os.path.join(a.out, f"{a.field}_s{a.seed}.csv")
    mode = "metro" if a.field == "metro" else "cluster"
    for ls in LISTING:
        C, ann, efn = metro_setup(ls, mode)
        init, _, _ = fqi.rollout(efn, qa, qb, None, collect_q=True)
        r = row(C, "residual_one_step", init, ev_value(C["EV"], init), a.field, a.seed, 0.0)
        r["listing_seed"] = ls; rows.append(r)
        ri, _ = E.conservative_rollout(efn, qa, qb, np.inf)
        r = row(C, "rule_replay", ri, ev_value(C["EV"], ri), a.field, a.seed, 0.0)
        r["listing_seed"] = ls; rows.append(r)
        msg = f"[s{a.seed}] listing {ls}: {rows[-2]['ratio']:.4f}"
        if a.baselines:
            for b in E.baselines(C, ann, a.field):
                b["listing_seed"] = ls; rows.append(b)
            rr = {x["rule"]: x["ratio"] for x in rows if x.get("listing_seed") == ls and x["seed"] == -1}
            msg += f" rank_ann {rr['rank_announced']:.4f} traj {rr['trajectory']:.4f}"
        print(msg, flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[s{a.seed}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
