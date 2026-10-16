# -*- coding: utf-8 -*-
"""p3_timing_space_metro.py — where does the choice of year matter? (compute only, no training)

For every candidate unit, the timing stake is the value lost by starting now instead of
in the unit's own best year, on the true table EV (all four fields):
    S_i = max_y EV[i, y] - EV[i, 0],   relative stake r_i = S_i / max_y EV[i, y].
It belongs to the unit alone (no capacity competition). Units are grouped by distance to
the nearest of the 28 metro stations (Lines 6, 6 branch, 13 north; opened 2020-2026).
Observed metro field = test; scenario field (stations play no role) = placebo.
Output: results_p2/timing_stake_by_distance.csv, results_p2/timing_stake_units.csv
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1_common as P                                                    # noqa: E402

BANDS = [(0, 800), (800, 1500), (1500, 3000), (3000, np.inf)]
LAB = ["0-0.8 km", "0.8-1.5 km", "1.5-3 km", ">3 km"]
rng = np.random.default_rng(0)
ud = pd.read_csv(os.path.join(P.REPO, "data/processed/gm_dataset_v1/metro/unit_station_distance.csv")).set_index("uid")
units, rows = [], []
for infra in ("metro", "cluster"):
    G, env_fn, EV, EVm, elig, v_orc, _ = P.setup(20, infra_mode=infra)
    e0 = env_fn(); uid = e0.U["uid"].values
    best = EV.max(1); now = EV[:, 0]; S = best - now
    r = np.where(best > 0, S / np.where(best > 0, best, 1), 0.0)
    d = ud.loc[uid, "d_nearest"].to_numpy(float); op = ud.loc[uid, "nearest_open"].to_numpy(float)
    by = EV.argmax(1)
    units.append(pd.DataFrame(dict(field=infra, uid=uid, d_nearest=d, nearest_open=op, best_value=best,
                                   value_now=now, stake=S, rel_stake=r, best_year=by)))
    for (lo, hi), lab in zip(BANDS, LAB):
        m = (d > lo) & (d <= hi) if lo > 0 else (d <= hi)
        bs = [r[m][rng.integers(0, m.sum(), m.sum())].mean() for _ in range(2000)]
        rows.append(dict(field=infra, band=lab, n=int(m.sum()), share_units=m.mean(),
                         share_value=best[m].sum() / best.sum(), share_stake=S[m].sum() / S.sum(),
                         mean_rel_stake=r[m].mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                         median_best_year=float(np.median(by[m]))))
B = pd.DataFrame(rows); U = pd.concat(units)
B.to_csv(os.path.join(P.REPO, "results_p2/timing_stake_by_distance.csv"), index=False)
U.to_csv(os.path.join(P.REPO, "results_p2/timing_stake_units.csv"), index=False)
pd.set_option("display.width", 200); print(B.round(3).to_string(index=False))
