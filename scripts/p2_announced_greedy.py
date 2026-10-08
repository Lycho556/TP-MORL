# -*- coding: utf-8 -*-
"""p2_announced_greedy.py — greedy rule on announced information.

Persistence greedy ranks the listed units in year t by EVm[:, t], the value of
initiation if current field levels persist. The announced-information greedy
ranks them by F_t[:, t], the announced forecast of p1_rolling_horizon (current
levels, commissioned infrastructure known four years ahead). Like every rule it
is valued on the true table EV. It separates, at the announced-information
level, the value of information (announced greedy - persistence greedy) from
the value of coordination (announced rolling horizon - announced greedy).

Configurations: the same as the robustness table (scenario field M = 10, 20,
50, all; M = 20 without the conformity gate; metro field M = 20 and all),
listing draws 7, 0, 1, 2, 3, 4. A check re-runs persistence greedy through the
current forecast and compares with the stored results.

Output: results_p2/announced_greedy.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1_common as P                                                  # noqa: E402
from p1_rolling_horizon import forecast_tables                          # noqa: E402

LS = [7, 0, 1, 2, 3, 4]
CONFIGS = [("cluster", True, M) for M in (10, 20, 50, 0)] + \
          [("cluster", False, 20), ("metro", True, 20), ("metro", True, 0)]


def main():
    out = os.path.join(P.REPO, "results_p2"); os.makedirs(out, exist_ok=True)
    RH = pd.read_csv(os.path.join(P.REPO, "results_p1", "rolling", "rolling_results.csv"))
    rows = []
    for infra, conf, M in CONFIGS:
        G, env_fn, EV, EVm, elig, v_orc, _ = P.setup(M, infra_mode=infra, conformity=conf)
        e0 = env_fn()
        tabs = {m: forecast_tables(e0, m, conf) for m in ("current", "announced")}
        T = EV.shape[1]
        cols = {m: np.stack([tabs[m][t][:, t] if t < len(tabs[m]) else np.zeros(EV.shape[0])
                             for t in range(T)], 1) for m in tabs}
        for ls in LS:
            mk = P.env_factory(G, M, ls)
            for rule, tab in (("persistence_check", cols["current"]), ("announced_greedy", cols["announced"])):
                init, val = G.run_policy(mk(), EV, tab, "myopic", quota=3, rng=np.random.default_rng(0))
                dec = G.decompose(EV, init, 3, elig, v_orc)
                rows.append(dict(infra_mode=infra, conformity=conf, M=M, listing_seed=ls, rule=rule,
                                 ratio=float(val) / v_orc, loss_selection=dec["loss_selection"],
                                 loss_timing=dec["loss_timing"], n_init=len(init)))
            ref = RH[(RH.infra_mode == infra) & (RH.conformity == conf) & (RH.variant == "base")
                     & (RH.M == M) & (RH.listing_seed == ls) & (RH.rule == "persistence")].ratio
            chk = rows[-2]["ratio"]
            assert len(ref) == 1 and abs(ref.iloc[0] - chk) < 1e-9, (infra, conf, M, ls, ref.values, chk)
        r = [x for x in rows if (x["infra_mode"], x["conformity"], x["M"]) == (infra, conf, M)
             and x["rule"] == "announced_greedy"]
        print(f"{infra} conf={conf} M={M}: announced greedy "
              f"{np.mean([x['ratio'] for x in r]):.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "announced_greedy.csv"), index=False)


if __name__ == "__main__":
    main()
