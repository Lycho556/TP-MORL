# -*- coding: utf-8 -*-
"""p2_timing_space.py — where do the rules time units badly?

Fixed before running. Scenario field, M = 20, listing draws 7, 0, 1, 2, 3, 4.
For every schedule (persistence greedy, trajectory-informed greedy, rolling
horizon with announced / full information, reference) and every selected unit:
  best year  b_u = argmax_t EV[u, t]
  error      e_u = y_u - b_u          (negative: initiated too early)
  share      s_u = EV[u, y_u] / EV[u, b_u]
Reported per schedule and draw: mean |e|, share too early (e<0), share on time
(e=0), mean s, Moran's I of e among the 75 selected units (1.5 km band,
row-standardised, 999 permutations). Unit-level rows for draw 7 are written for
the map.

Output: results_p2/timing_space_summary.csv, results_p2/timing_space_units_ls7.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1_common as P                                                  # noqa: E402
import p1_spatial as SP                                                # noqa: E402
from p1_rolling_horizon import forecast_tables, run_rolling             # noqa: E402

OUT = os.path.join(P.REPO, "results_p2")
LS = [7, 0, 1, 2, 3, 4]


def main():
    os.makedirs(OUT, exist_ok=True)
    ctx = SP.build_context(20)                       # scenario field, sets globals
    G, EV, elig, oplan = ctx["G"], ctx["EV"], ctx["elig"], ctx["oplan"]
    T = EV.shape[1] if EV.shape[1] <= ctx["T"] else ctx["T"]
    EVh = EV[:, :ctx["T"]]
    best = EVh.argmax(1)
    W = SP._band_w(ctx["D"], 1500.0)
    e0 = P.env_factory(G, 20, 7)()
    ann = forecast_tables(e0, "announced", True)
    uids = e0.U["uid"].values
    _, _, EV2, EVm, _, _, _ = P.setup(20)           # same scenario; EVm for persistence greedy
    assert np.array_equal(EV2, EV)
    rows, units = [], []
    for ls in LS:
        mk = P.env_factory(G, 20, ls)
        sch = {}
        for tag, kind in (("persistence", "myopic"), ("trajectory", "forecast")):
            init, _ = G.run_policy(mk(), EV, EVm, kind, quota=3, rng=np.random.default_rng(0))
            sch[tag] = init
        sch["rh_announced"] = run_rolling(mk(), lambda t: ann[t], elig)
        sch["rh_full"] = run_rolling(mk(), lambda t: EV, elig)
        sch["reference"] = dict(oplan)
        for rule, init in sch.items():
            u = np.array(sorted(init), int)
            y = np.array([init[k] for k in u], int)
            e = y - best[u]
            s = EVh[u, y] / EVh[u, best[u]]
            mi = SP.morans_i(e, W[np.ix_(u, u)], nperm=999, rng=np.random.default_rng(0))
            rows.append(dict(rule=rule, listing_seed=ls, n=len(u), mean_abs_err=np.abs(e).mean(),
                             share_early=(e < 0).mean(), share_ontime=(e == 0).mean(),
                             share_late=(e > 0).mean(), mean_share=s.mean(),
                             I_err=mi["I"], p_err=mi["p_sim"]))
            if ls == 7:
                for k, yy, ee, ss in zip(u, y, e, s):
                    units.append(dict(rule=rule, unit=int(k), uid=uids[k], year=int(yy),
                                      best=int(best[k]), err=int(ee), share=float(ss),
                                      x=ctx["cent"][k, 0], y=ctx["cent"][k, 1]))
        print(f"ls{ls} " + " ".join(f"{r['rule']}:|e|={r['mean_abs_err']:.1f},I={r['I_err']:.3f}(p={r['p_err']:.3f})"
                                    for r in rows if r["listing_seed"] == ls), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "timing_space_summary.csv"), index=False)
    pd.DataFrame(units).to_csv(os.path.join(OUT, "timing_space_units_ls7.csv"), index=False)


if __name__ == "__main__":
    main()
