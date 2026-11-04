# -*- coding: utf-8 -*-
"""p5_no_p5.py -- main comparison without the agricultural-conversion channel (P5),
Supplementary Table S17. Only channels P1, P2, P3 are actionable (schedule.ACTIONABLE_CHANNELS);
main field, M = 20, listing draws 7, 0, 1, 2, 3, 4. Rule-based rules only, no learning.
Output: results_p5_spatial/external/no_P5_rules.csv (one row per listing draw, ratios to the
reference schedule of the reduced inventory).
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from tpmorl.env import schedule as S                                    # noqa: E402
S.ACTIONABLE_CHANNELS = (1, 2, 3)
import p1_common as P                                                    # noqa: E402
import p1_rolling_horizon as RH                                          # noqa: E402
import p5_spatial_ext as X                                               # noqa: E402

OUT = os.path.join(HERE, "..", "results_p5_spatial", "external", "no_P5_rules.csv")


def main(out=OUT):
    G, env_fn, EV, EVm, elig, v_orc, _ = P.setup(prescreen=20)
    e0 = env_fn(); T = EV.shape[1]
    print("eligible units", int(elig.sum()), "channels", sorted(set(e0.U.ch_code[elig])))
    ann = RH.forecast_tables(e0, "announced", True)
    rows = []
    for ls in X.SEEDS:
        mk = P.env_factory(G, 20, ls)
        r = dict(seed=ls)
        r["persistence"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: EVm), T)[0]) / v_orc
        r["announced"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: ann[t]), T)[0]) / v_orc
        r["rh_announced"] = X.value(EV, X.run(mk(), X.rolling_chooser(lambda t: ann[t], T, elig), T)[0]) / v_orc
        r["trajectory"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: EV), T)[0]) / v_orc
        r["rh_full"] = X.value(EV, X.run(mk(), X.rolling_chooser(lambda t: EV, T, elig), T)[0]) / v_orc
        rows.append(r)
    d = pd.DataFrame(rows); os.makedirs(os.path.dirname(out), exist_ok=True); d.to_csv(out, index=False)
    print(d.mean(numeric_only=True).round(3).to_dict())
    return d


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT)
