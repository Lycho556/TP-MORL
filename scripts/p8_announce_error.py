# -*- coding: utf-8 -*-
"""p8_announce_error.py -- ranking on announced plans with wrong announcement timetables.
Pre-registered in results_p8_annerr/PREREGISTRATION.md. Rule-based rules only.
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/p8_announce_error.py --field F   (F = main or 1..10)
Output: results_p8_annerr/rows_{F}.csv"""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p1_common as P, p1_rolling_horizon as RH, p5_spatial_ext as X
OUT = os.path.join(HERE, "..", "results_p8annerr".replace("p8annerr", "p8_annerr"))
CONDS = [("none", 0)] + [(k, e) for k in ("sym", "delay") for e in (1, 2, 3)]
N_ERR = 5

class BeliefInfo(RH._Info):
    """Announced forecast that believes the announced timetable: cluster c is read d_c years off."""
    def __init__(self, opp, t0, off, mode, d_unit):
        super().__init__(opp, t0, off, mode); self.d = d_unit
    def value_mult(self, u, s_init, s_done):
        td = int(s_done) + self.off; j = min(td, self.t0 + self.lead)
        if j > self.t0 and self.o.infra_onset[int(u)] > self.t0:
            j = j + int(self.d[int(u)])
        return self.o.value_mult(u, self.t0, j)

def belief_tables(env, d_unit):
    orig = RH._Info
    RH._Info = lambda opp, t0, off, mode: BeliefInfo(opp, t0, off, mode, d_unit)
    try:
        return RH.forecast_tables(env, "announced", True)
    finally:
        RH._Info = orig

def main(field):
    fs = None if field == "main" else int(field)
    G, env_fn, EV, EVm, elig, v_orc, _ = P.setup(prescreen=X.M, field_seed=fs)
    e0 = env_fn(); T = EV.shape[1]; cl = e0.opp.infra_cluster; ncl = int(cl.max()) + 1
    base = {}
    for ls in X.SEEDS:
        mk = P.env_factory(G, X.M, ls)
        base[ls] = dict(persistence=X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: EVm), T)[0]) / v_orc,
                        trajectory=X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: EV), T)[0]) / v_orc)
    ann0 = RH.forecast_tables(e0, "announced", True)
    rows = []
    for kind, e in CONDS:
        for rep in range(1 if e == 0 else N_ERR):
            if e == 0:
                ann = ann0; d = np.zeros(len(cl), int)
            else:
                rng = np.random.default_rng(100000 * (fs or 0) + 1000 * e + 100 * (kind == "delay") + rep)
                dc = rng.integers(-e if kind == "sym" else 0, e + 1, ncl); d = dc[cl]
                ann = belief_tables(e0, d)
            for ls in X.SEEDS:
                mk = P.env_factory(G, X.M, ls)
                r = X.value(EV, X.run(mk(), X.greedy_chooser(lambda t: ann[t]), T)[0]) / v_orc
                rows.append(dict(field=field, kind=kind, e=e, rep=rep, listing=ls, rank_announced=r, **base[ls],
                                 mean_abs_d=float(np.abs(d).mean())))
    os.makedirs(OUT, exist_ok=True); pd.DataFrame(rows).to_csv(os.path.join(OUT, f"rows_{field}.csv"), index=False)
    print("field", field, "done", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--field", required=True); main(ap.parse_args().field)
