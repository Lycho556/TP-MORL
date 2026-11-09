# -*- coding: utf-8 -*-
"""p8_cellfield_delineation.py -- unit delineation with the same cell-level future.
Pre-registered in results_p8_delineation/PREREGISTRATION_cellfield.md. Rule-based rules only.
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/p8_cellfield_delineation.py --target T --mode rep|mean
Output: results_p8_delineation/cell_{mode}_t{T}.csv"""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p1_common as P, p1_rolling_horizon as RH, p5_spatial_ext as X
from tpmorl.env import opportunity as OPP
OUT = os.path.join(HERE, "..", "results_p8_delineation")
FIELDS = [None] + list(range(1, 11))

def dataset(t):
    return P.DATASET if t == 12 else P.DATASET + f"_units{t}"

def cell_field(t, mode):
    base = np.load(os.path.join(P.DATASET, "zones_v0", "unit_id.npy"))
    rc = np.argwhere(base > 0)                         # row-major, identical for every delineation
    lab = np.load(os.path.join(dataset(t), "zones_v0", "unit_id.npy"))[rc[:, 0], rc[:, 1]]
    cells, rep = {}, {}
    for u in np.unique(lab):
        idx = np.where(lab == u)[0]; c = rc[idx].mean(0)
        d = ((rc[idx] - c) ** 2).sum(1); cells[int(u)] = idx; rep[int(u)] = int(idx[np.argmin(d)])
    return dict(cell_row=rc[:, 0].astype(float), cell_col=rc[:, 1].astype(float), rep=rep, cells=cells, mode=mode)

def main(t, mode, fields=FIELDS, out=None):
    a = P.args(dataset=dataset(t)); rows = []
    for fs in fields:
        OPP.CELL_FIELD = cell_field(t, mode)
        G, env_fn, EV, EVm, elig, v_orc, _ = P.setup(prescreen=X.M, a=a, field_seed=fs)
        e0 = env_fn(); T = EV.shape[1]; n = len(e0.U)
        assert e0.opp.n == n and e0.opp.plan.shape[0] == n
        ann = RH.forecast_tables(e0, "announced", True)
        for ls in X.SEEDS:
            mk = P.env_factory(G, X.M, ls, a)
            r = dict(mode=mode, target=t, field="main" if fs is None else int(fs), listing=ls, n_units=n, reference=v_orc)
            r["persistence"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t_: EVm), T)[0]) / v_orc
            r["announced"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t_: ann[t_]), T)[0]) / v_orc
            r["rh_announced"] = X.value(EV, X.run(mk(), X.rolling_chooser(lambda t_: ann[t_], T, elig), T)[0]) / v_orc
            r["trajectory"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t_: EV), T)[0]) / v_orc
            r["rh_full"] = X.value(EV, X.run(mk(), X.rolling_chooser(lambda t_: EV, T, elig), T)[0]) / v_orc
            rows.append(r)
        OPP.CELL_FIELD = None
        pd.DataFrame(rows).to_csv(out or os.path.join(OUT, f"cell_{mode}_t{t}.csv"), index=False)
        print("mode", mode, "target", t, "field", fs, "done", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=int, required=True); ap.add_argument("--mode", default="rep")
    a = ap.parse_args(); main(a.target, a.mode)
