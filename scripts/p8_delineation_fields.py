# -*- coding: utf-8 -*-
"""p8_delineation_fields.py -- unit delineation (8 / 12 / 18 cells) across field draws.
Pre-registered in results_p8_delineation/PREREGISTRATION.md. Rule-based rules only.
Usage: python scripts/p8_delineation_fields.py --target 18   (inventories from p5_unit_delineation.py)
Output: results_p8_delineation/rows_t{target}.csv, one row per (field, listing draw)."""
import argparse, os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p1_common as P, p1_rolling_horizon as RH, p5_spatial_ext as X
OUT = os.path.join(HERE, "..", "results_p8_delineation")
FIELDS = [None] + list(range(1, 11))

def main(target):
    ds = P.DATASET if target == 12 else P.DATASET + f"_units{target}"
    assert os.path.isdir(os.path.join(ds, "zones_v0")), "run p5_unit_delineation.py first"
    a = P.args(dataset=ds); rows = []
    for fs in FIELDS:
        G, env_fn, EV, EVm, elig, v_orc, _ = P.setup(prescreen=X.M, a=a, field_seed=fs)
        e0 = env_fn(); T = EV.shape[1]; n = len(e0.U)
        ann = RH.forecast_tables(e0, "announced", True)
        for ls in X.SEEDS:
            mk = P.env_factory(G, X.M, ls, a)
            r = dict(target=target, field="main" if fs is None else int(fs), listing=ls, n_units=n, rho=n / (X.K * T), reference=v_orc)
            r["persistence"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: EVm), T)[0]) / v_orc
            r["announced"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: ann[t]), T)[0]) / v_orc
            r["rh_announced"] = X.value(EV, X.run(mk(), X.rolling_chooser(lambda t: ann[t], T, elig), T)[0]) / v_orc
            r["trajectory"] = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: EV), T)[0]) / v_orc
            r["rh_full"] = X.value(EV, X.run(mk(), X.rolling_chooser(lambda t: EV, T, elig), T)[0]) / v_orc
            rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"rows_t{target}.csv"), index=False)
        print("target", target, "field", fs, "done", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=int, required=True); main(ap.parse_args().target)
