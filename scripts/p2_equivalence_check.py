# -*- coding: utf-8 -*-
"""p2_equivalence_check.py — unit-by-unit check that re-planning on a forecast in which
values only fall with delay gives the same schedule as greedy ranking on that forecast.

For the current and announced forecasts of p1_rolling_horizon, and every configuration
of p2_announced_greedy (42 = 7 configurations x 6 listing draws), compare the start
year of every unit under (a) greedy ranking on F_t[:, t] and (b) the rolling-horizon
assignment on F_t. Also measure how far each forecast is from the separable form
F_t(i, y) = F_t(i, t) * gamma**(y - t) (the condition of the proposition), and in which
years it departs from it (end-of-horizon truncation).

Output: results_p2/equivalence_check.csv, results_p2/separability_check.csv
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1_common as P                                                    # noqa: E402
from p1_rolling_horizon import forecast_tables, run_rolling, GAMMA       # noqa: E402
from p2_announced_greedy import CONFIGS, LS                              # noqa: E402

rows, sep = [], []
for infra, conf, M in CONFIGS:
    G, env_fn, EV, EVm, elig, v_orc, _ = P.setup(M, infra_mode=infra, conformity=conf)
    e0 = env_fn(); T = EV.shape[1]
    for mode in ("current", "announced"):
        tabs = forecast_tables(e0, mode, conf)
        col = np.stack([tabs[t][:, t] if t < len(tabs) else np.zeros(EV.shape[0]) for t in range(T)], 1)
        # separability: relative deviation of F_t(i,y) from F_t(i,t) * gamma^(y-t), by y - t and by year y
        for t in range(len(tabs)):
            F = tabs[t]; base = F[:, t]; ok = base > 0
            for y in range(t, F.shape[1]):
                dev = np.abs(F[ok, y] - base[ok] * GAMMA ** (y - t)) / base[ok]
                sep.append(dict(infra_mode=infra, conformity=conf, M=M, mode=mode, t=t, y=y,
                                max_rel_dev=float(dev.max()) if ok.any() else 0.0))
        for ls in LS:
            mk = P.env_factory(G, M, ls)
            ig, vg = G.run_policy(mk(), EV, col, "myopic", quota=3, rng=np.random.default_rng(0))
            ir = run_rolling(mk(), lambda t: tabs[t], elig)
            ig = {int(u): int(y) for u, y in ig.items()}; ir = {int(u): int(y) for u, y in ir.items()}
            units = set(ig) | set(ir)
            ndiff = sum(ig.get(u) != ir.get(u) for u in units)
            vr = sum(EV[u, y] for u, y in ir.items())
            rows.append(dict(infra_mode=infra, conformity=conf, M=M, mode=mode, listing_seed=ls,
                             n_greedy=len(ig), n_rolling=len(ir), units_differ=ndiff,
                             identical=(ig == ir), ratio_greedy=float(vg) / v_orc, ratio_rolling=vr / v_orc))
    print(infra, conf, M, "done", flush=True)
R = pd.DataFrame(rows); S = pd.DataFrame(sep)
out = os.path.join(P.REPO, "results_p2"); R.to_csv(os.path.join(out, "equivalence_check.csv"), index=False)
S.to_csv(os.path.join(out, "separability_check.csv"), index=False)
print(R.groupby("mode").identical.agg(["sum", "count"]).to_string())
print("max units differing:", R.groupby("mode").units_differ.max().to_dict())
