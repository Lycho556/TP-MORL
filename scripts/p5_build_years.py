# -*- coding: utf-8 -*-
"""p5_build_years.py -- sensitivity of the main comparison to the construction duration b
(all units; b = 3, 5, 7 years). Main field, listing draws 7,0,1,2,3,4, M = 20.
Output: results_p5_spatial/build_years.csv"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p1_common as P, p1_rolling_horizon as RH, p5_spatial_ext as X
rows = []
for b in (3, 5, 7):
    G, env_fn, EV, EVm, elig, v_orc, _ = P.setup(prescreen=X.M, build_years=b)
    e0 = env_fn(); T = EV.shape[1]
    assert set(np.asarray(e0.env.build_years).tolist()) == {b}
    ann = RH.forecast_tables(e0, "announced", True); cur = RH.forecast_tables(e0, "current", True)
    for ls in X.SEEDS:
        mk = P.env_factory(G, X.M, ls)
        r = dict(build_years=b, seed=ls, T_eval=e0.T_eval, reference=v_orc)
        r["persistence"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: EVm), T)[0]) / v_orc
        r["rh_current"] = X.value(EV, X.run(mk(), X.rolling_chooser(lambda t: cur[t], T, elig), T)[0]) / v_orc
        r["announced"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: ann[t]), T)[0]) / v_orc
        r["rh_announced"] = X.value(EV, X.run(mk(), X.rolling_chooser(lambda t: ann[t], T, elig), T)[0]) / v_orc
        r["trajectory"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: EV), T)[0]) / v_orc
        r["rh_full"] = X.value(EV, X.run(mk(), X.rolling_chooser(lambda t: EV, T, elig), T)[0]) / v_orc
        rows.append(r)
    print("b", b, "done", flush=True)
d = pd.DataFrame(rows); d.to_csv(os.path.join(X.OUT, "build_years.csv"), index=False)
d["vinfo"] = d.trajectory - d.persistence; d["coord"] = d.rh_full - d.trajectory
d["ann_share"] = (d.announced - d.persistence) / d["vinfo"]; d["p1_ann"] = (d.rh_announced - d.announced).abs(); d["p1_cur"] = (d.rh_current - d.persistence).abs()
print(d.groupby("build_years").agg(T_eval=("T_eval","first"), persistence=("persistence","mean"), announced=("announced","mean"), trajectory=("trajectory","mean"), rh_full=("rh_full","mean"),
      info=("vinfo","mean"), info_min=("vinfo","min"), info_max=("vinfo","max"), coord=("coord","mean"), coord_min=("coord","min"), coord_max=("coord","max"), ann_share=("ann_share","mean"), p1_ann=("p1_ann","max"), p1_cur=("p1_cur","max")).round(4).T)
