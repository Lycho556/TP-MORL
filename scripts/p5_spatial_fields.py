# -*- coding: utf-8 -*-
"""p5_spatial_fields.py -- E1/E2 of p5_spatial_ext.py on other fields and settings.

  --field main            main scenario field, listing draws 7,0,1,2,3,4
  --field metro           observed metro infrastructure field, same listing draws
  --field test            the ten unseen test field draws 1-10, listing draw 7
  --delta D               spillover window in years (default 2)
  --restarts R            local-search restarts for the best-found schedule
                          (seeds 0..R-1 from the additive reference schedule);
                          the yardstick is the best of them, the spread is recorded
Output: results_p5_spatial/fields/<tag>_E1.csv, <tag>_E2.csv, <tag>_LS.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p1_common as P                                                  # noqa: E402
import p1_rolling_horizon as RH                                        # noqa: E402
import p5_spatial_ext as X                                             # noqa: E402

OUT = os.path.join(X.OUT, "fields")


def one_field(field_seed, infra, seeds, delta, betas, restarts, iters, do_e1, label):
    G, env_fn, EV, EVm, elig, v_orc, _ = P.setup(prescreen=X.M, infra_mode=infra,
                                                 field_seed=field_seed)
    e0 = env_fn(); T = EV.shape[1]
    sp = X.load_space(e0); s_of, edges, nbr = sp["s_of"], sp["edges"], sp["nbr"]
    quota = X.balanced_quota(s_of, X.K * T)
    ann = RH.forecast_tables(e0, "announced", True)
    cur = RH.forecast_tables(e0, "current", True)
    tab = {"persistence": lambda t: EVm, "announced": lambda t: ann[t], "trajectory": lambda t: EV}
    rtab = {"rh_current": lambda t: cur[t], "rh_announced": lambda t: ann[t], "rh_full": lambda t: EV}
    r1, r2, r3 = [], [], []
    if do_e1:
        refs = {"none": v_orc}
        for mode in ("cap1", "balanced"):
            refs[mode] = X.value(EV, X.solve_assignment(EV * elig[:, None], list(range(T)), s_of, mode,
                                                        quota if mode == "balanced" else None))
        for mode in ("none", "cap1", "balanced"):
            md = None if mode == "none" else mode
            for ls in seeds:
                mk = P.env_factory(G, X.M, ls)
                for name, tf in tab.items():
                    init, _ = X.run(mk(), X.greedy_chooser(tf, s_of, md, quota), T)
                    r1.append(dict(field=label, mode=mode, seed=ls, rule=name + "_greedy",
                                   ratio=X.value(EV, init) / refs[mode], ref_share=refs[mode] / v_orc))
                for name, tf in rtab.items():
                    init, _ = X.run(mk(), X.rolling_chooser(tf, T, elig, s_of, md, quota), T)
                    r1.append(dict(field=label, mode=mode, seed=ls, rule=name,
                                   ratio=X.value(EV, init) / refs[mode], ref_share=refs[mode] / v_orc))
    start = X.solve_assignment(EV * elig[:, None], list(range(T)))
    for beta in betas:
        if beta > 0:
            sols = [X.local_search(EV, start, edges, nbr, beta, elig, T, iters=iters, seed=k, delta=delta)
                    for k in range(restarts)]
            vals = [v for _, v in sols]
            plan, v_ref = sols[int(np.argmax(vals))]
        else:
            plan, v_ref, vals = start, X.value(EV, start), [X.value(EV, start)]
        v_add, _ = X.spill_value(EV, start, edges, beta, delta)
        adj = sum(1 for i, j in edges if i in plan and j in plan and abs(plan[i] - plan[j]) <= delta)
        r3.append(dict(field=label, beta=beta, delta=delta, best=v_ref, restart_min=min(vals),
                       restart_max=max(vals), additive_ref=v_add,
                       spatial_coordination=(v_ref - v_add) / v_ref, adjacent_best=adj))
        for ls in seeds:
            mk = P.env_factory(G, X.M, ls)
            runs = {}
            for name, tf in tab.items():
                runs[name + "_greedy"] = X.run(mk(), X.greedy_chooser(tf), T)[0]
                if beta > 0:
                    runs[name + "_greedy_aware"] = X.run(
                        mk(), X.greedy_chooser(tf, bonus=X.spill_bonus(nbr, beta, delta)), T)[0]
            runs["rh_announced"] = X.run(mk(), X.rolling_chooser(rtab["rh_announced"], T, elig), T)[0]
            runs["rh_full"] = X.run(mk(), X.rolling_chooser(rtab["rh_full"], T, elig), T)[0]
            for name, init in runs.items():
                v, b = X.spill_value(EV, init, edges, beta, delta)
                r2.append(dict(field=label, beta=beta, delta=delta, seed=ls, rule=name, ratio=v / v_ref,
                               bonus=b, adjacent=sum(1 for i, j in edges if i in init and j in init
                                                     and abs(init[i] - init[j]) <= delta)))
    return r1, r2, r3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", choices=["main", "metro", "test"], required=True)
    ap.add_argument("--delta", type=int, default=2)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--iters", type=int, default=80000)
    ap.add_argument("--betas", default="0,0.1,0.25,0.5")
    ap.add_argument("--no-e1", action="store_true")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    betas = [float(x) for x in a.betas.split(",")]
    tag = a.tag or f"{a.field}_d{a.delta}"
    os.makedirs(OUT, exist_ok=True)
    jobs = {"main": [(None, "cluster", X.SEEDS, "main")],
            "metro": [(None, "metro", X.SEEDS, "metro")],
            "test": [(k, "cluster", (7,), f"field{k}") for k in range(1, 11)]}[a.field]
    R1, R2, R3 = [], [], []
    t0 = time.time()
    for fs, infra, seeds, label in jobs:
        r1, r2, r3 = one_field(fs, infra, seeds, a.delta, betas, a.restarts, a.iters, not a.no_e1, label)
        R1 += r1; R2 += r2; R3 += r3
        print(f"[{tag}] {label} done ({time.time() - t0:.0f}s)", flush=True)
        if R1:
            pd.DataFrame(R1).to_csv(os.path.join(OUT, f"{tag}_E1.csv"), index=False)
        pd.DataFrame(R2).to_csv(os.path.join(OUT, f"{tag}_E2.csv"), index=False)
        pd.DataFrame(R3).to_csv(os.path.join(OUT, f"{tag}_LS.csv"), index=False)


if __name__ == "__main__":
    main()
