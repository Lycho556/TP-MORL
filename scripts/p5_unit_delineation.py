# -*- coding: utf-8 -*-
"""p5_unit_delineation.py -- sensitivity of the main comparison to how candidate units are
delineated. Rebuilds zones_v0 with build_zones.segment_units at another target size
(SPLIT_ABOVE = 2.5 x target, as in the paper's 12 / 30), in a sibling dataset that links every
other folder of gm_dataset_v1. The scenario fields are regenerated on the new units with the
same generator and seed. Main field, listing draws 7,0,1,2,3,4, M = 20.
Output: results_p5_spatial/unit_delineation.csv"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p1_common as P, p1_rolling_horizon as RH, p5_spatial_ext as X
from tpmorl.data import build_zones as BZ

def make_dataset(target):
    src = P.DATASET; dst = src + f"_units{target}"
    os.makedirs(dst, exist_ok=True)
    for d in os.listdir(src):
        if d in ("zones_v0",) or d.startswith("."): continue
        p = os.path.join(dst, d)
        if not os.path.exists(p): os.symlink(os.path.join(src, d), p)
    out = os.path.join(dst, "zones_v0"); os.makedirs(out, exist_ok=True)
    G = os.path.join(src, "grid_100m")
    L0 = np.load(os.path.join(G, "L0_class.npy")); am = np.load(os.path.join(G, "action_mask.npy")); road = np.load(os.path.join(G, "road_100m.npy"))
    FUNC = BZ.compress_func(L0); CH, inside = BZ.build_channels(FUNC, am)
    lab, U = BZ.segment_units(CH, am, target=target, split_above=int(round(2.5 * target)))
    PRESS = BZ.renewal_pressure(FUNC, road, inside)
    U["压力"] = [float(np.nanmean(PRESS[lab == u])) for u in U["uid"]]
    U["距路m"] = [float(np.nanmean(road[lab == u])) for u in U["uid"]]
    np.save(os.path.join(out, "unit_id.npy"), lab); U.to_csv(os.path.join(out, "candidate_units.csv"), index=False, encoding="utf-8-sig")
    for f in ("channel_legend.csv", "func6_legend.csv", "func6.npy", "channel_v0.npy", "pressure_proxy.npy"):
        s = os.path.join(src, "zones_v0", f); t = os.path.join(out, f)
        if os.path.exists(s) and not os.path.exists(t): os.symlink(s, t)
    return dst, len(U), float(U["格数"].median())

rows = []
for target in (8, 12, 18):
    ds, n, med = (P.DATASET, None, None) if target == 12 else make_dataset(target)
    a = P.args(dataset=ds)
    G, env_fn, EV, EVm, elig, v_orc, _ = P.setup(prescreen=X.M, a=a)
    e0 = env_fn(); T = EV.shape[1]; n = len(e0.U); med = float(e0.U["格数"].median())
    ann = RH.forecast_tables(e0, "announced", True)
    for ls in X.SEEDS:
        mk = P.env_factory(G, X.M, ls, a)
        r = dict(target=target, n_units=n, median_cells=med, rho=n / (X.K * T), seed=ls, reference=v_orc)
        r["persistence"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: EVm), T)[0]) / v_orc
        r["announced"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: ann[t]), T)[0]) / v_orc
        r["rh_announced"] = X.value(EV, X.run(mk(), X.rolling_chooser(lambda t: ann[t], T, elig), T)[0]) / v_orc
        r["trajectory"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: EV), T)[0]) / v_orc
        r["rh_full"] = X.value(EV, X.run(mk(), X.rolling_chooser(lambda t: EV, T, elig), T)[0]) / v_orc
        rows.append(r)
    print("target", target, "units", n, "median", med, flush=True)
d = pd.DataFrame(rows); d.to_csv(os.path.join(X.OUT, "unit_delineation.csv"), index=False)
d["vinfo"] = d.trajectory - d.persistence; d["coord"] = d.rh_full - d.trajectory
d["ann_share"] = (d.announced - d.persistence) / d.vinfo; d["p1"] = (d.rh_announced - d.announced).abs()
print(d.groupby("target").agg(n=("n_units","first"), med=("median_cells","first"), rho=("rho","first"), persistence=("persistence","mean"), announced=("announced","mean"),
      trajectory=("trajectory","mean"), rh_full=("rh_full","mean"), info=("vinfo","mean"), info_min=("vinfo","min"), info_max=("vinfo","max"),
      coord=("coord","mean"), coord_min=("coord","min"), coord_max=("coord","max"), ann_share=("ann_share","mean"), p1=("p1","max")).round(4).T)
