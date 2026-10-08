"""exp_field_sensitivity.py — sensitivity of the Guangming results to the
scenario-based temporal opportunity fields.

One shape property is changed at a time; N, K (prescreen 20), quota, horizon,
spatial value, estimator and behaviour data are held at the main configuration.
Code path is identical to exp_prescreen_ablation.run / baselines, so the
baseline row equals partB (K=20, double) seeds 0-2.

Configurations
    base         amplitudes (2.4, 1.8, 0.9, 1.0), window 3-7 yr, onset 0.15-0.60 T
    amp_low      amplitudes x 0.5
    amp_high     amplitudes x 1.5 (a_ready held at its upper bound 1.0)
    win_narrow   window 2-4 yr
    win_broad    window 6-10 yr
    onset_early  onset 0.05-0.35 T
    onset_late   onset 0.40-0.75 T

Usage
    python scripts/exp_field_sensitivity.py --config amp_low --seeds 0 1 2
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

from exp_fqi_local import make_trees                                   # noqa: E402
from tpmorl.rl import fqi                                              # noqa: E402

BASE_AMP = dict(a_plan=2.4, a_infra=1.8, a_age=0.9, a_ready=1.0)

CONFIGS = {
    "base":        dict(),
    "amp_low":     dict(amp=0.5),
    "amp_high":    dict(amp=1.5),
    "win_narrow":  dict(window_years=(2.0, 4.0)),
    "win_broad":   dict(window_years=(6.0, 10.0)),
    "onset_early": dict(onset_lo=0.05, onset_hi=0.35),
    "onset_late":  dict(onset_lo=0.40, onset_hi=0.75),
}


def setup(a, cfg):
    import exp_temporal_gate as G
    from tpmorl.rl import scenario as SC
    c = dict(CONFIGS[cfg])
    k = c.pop("amp", 1.0)
    # a_ready is bounded at 1 (hazard multiplier 1 + A(2r-1) must stay >= 0);
    # the main configuration already sits at the bound, so it is held there.
    amps = {n: (min(v * k, 1.0) if n == "a_ready" else v * k)
            for n, v in BASE_AMP.items()}
    SC.reset()
    SC.apply(horizon=a.horizon, horizon_eval="auto", opp_shape="window",
             foresight=0, quota=a.quota, budget=1e12, **amps, **c)
    env_fn = lambda: G.build_env(a.dataset, a.horizon, 0.5, 7, 1e12, "floor",
                                 True, False, a.prescreen, 0)
    e0 = env_fn()
    EV, EVm = G.ev_tables(e0, a.gamma)
    elig = np.asarray(e0.env.eligible, bool)
    v_orc, _ = G.oracle_plan(EV, a.quota, elig)
    return G, env_fn, EV, EVm, elig, v_orc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--out", default="results_field_sens")
    ap.add_argument("--prescreen", type=int, default=20)
    ap.add_argument("--ep-each", type=int, default=10)
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--quota", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.95)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"{a.config}_runs.csv")

    G, env_fn, EV, EVm, elig, v_orc = setup(a, a.config)
    rows = []
    for tag, kind in (("static_greedy", "myopic"), ("temporal_greedy", "forecast")):
        iv, vv = G.run_policy(env_fn(), EV, EVm, kind, quota=a.quota,
                              rng=np.random.default_rng(0))
        dec = G.decompose(EV, iv, a.quota, elig, v_orc)
        rows.append(dict(config=a.config, seed=-1, method=tag, ratio=vv / v_orc,
                         value=vv, oracle_value=v_orc,
                         where_loss=dec["loss_selection"],
                         when_loss=dec["loss_timing"], n_init=len(iv)))
    print(f"[{a.config}] oracle {v_orc:.2f}  static {rows[0]['ratio']:.4f}  "
          f"TG {rows[1]['ratio']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(fn, index=False)

    for s in a.seeds:
        t0 = time.time()
        data = fqi.collect(env_fn, lambda u, t: float(EV[u, min(t, EV.shape[1] - 1)]),
                           {"random": None, "myopic": EVm, "temporal_greedy": EV},
                           n_ep_each=a.ep_each, seed=s)
        qa, qb, _ = fqi.fit_double(data, a.gamma, make_trees(seed=s), verbose=False)
        init, _, _ = fqi.rollout(env_fn, qa, qb, None, collect_q=True)
        v = float(sum(EV[u, min(t, EV.shape[1] - 1)] for u, t in init.items()))
        dec = G.decompose(EV, init, a.quota, elig, v_orc)
        rows.append(dict(config=a.config, seed=s, method="double_fqi",
                         ratio=v / v_orc, value=v, oracle_value=v_orc,
                         where_loss=dec["loss_selection"],
                         when_loss=dec["loss_timing"], n_init=len(init),
                         n_samples=sum(len(d["r"]) for d in data),
                         seconds=round(time.time() - t0, 1)))
        print(f"  {a.config} seed={s} ratio {v / v_orc:.4f}  "
              f"{rows[-1]['seconds']}s", flush=True)
        pd.DataFrame(rows).to_csv(fn, index=False)


if __name__ == "__main__":
    main()
