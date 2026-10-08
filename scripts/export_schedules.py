"""export_schedules.py — unit-level initiation years for Figure 5.

Same code path as exp_prescreen_ablation (K=20, double FQI, ep-each 10):
static greedy, temporal greedy and the optimum are deterministic; the
value-based policy is re-trained for one seed (default 2, the median of the
five main seeds). Writes results_figures/schedules.json and unit_attrs.csv.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from exp_fqi_local import make_trees                       # noqa: E402
from exp_prescreen_ablation import setup                   # noqa: E402
from tpmorl.rl import fqi                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default="results_figures")
    ap.add_argument("--prescreen", type=int, default=20)
    ap.add_argument("--ep-each", type=int, default=10)
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--quota", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.95)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    G, env_fn, EV, EVm, elig, v_orc = setup(a, a.prescreen)
    _, oplan = G.oracle_plan(EV, a.quota, elig)
    out = {"oracle_value": v_orc}
    for tag, kind in (("static_greedy", "myopic"), ("temporal_greedy", "forecast")):
        iv, vv = G.run_policy(env_fn(), EV, EVm, kind, quota=a.quota,
                              rng=np.random.default_rng(0))
        out[tag] = {"ratio": vv / v_orc, "init": {int(u): int(t) for u, t in iv.items()}}
    out["optimum"] = {"ratio": 1.0, "init": {int(u): int(t) for u, t in dict(oplan).items()}}
    print({k: (v["ratio"], len(v["init"])) for k, v in out.items() if isinstance(v, dict)}, flush=True)
    data = fqi.collect(env_fn, lambda u, t: float(EV[u, min(t, EV.shape[1] - 1)]),
                       {"random": None, "myopic": EVm, "temporal_greedy": EV},
                       n_ep_each=a.ep_each, seed=a.seed)
    qa, qb, _ = fqi.fit_double(data, a.gamma, make_trees(seed=a.seed), verbose=False)
    init, _, _ = fqi.rollout(env_fn, qa, qb, None, collect_q=True)
    v = float(sum(EV[u, min(t, EV.shape[1] - 1)] for u, t in init.items()))
    out["value_based"] = {"ratio": v / v_orc, "seed": a.seed,
                          "init": {int(u): int(t) for u, t in init.items()}}
    print("value_based", v / v_orc, len(init), flush=True)
    json.dump(out, open(os.path.join(a.out, "schedules.json"), "w"), indent=1)
    e = env_fn()
    pd.DataFrame({"unit": np.arange(e.n), "uid": e.U["uid"].values, "channel": e.ch,
                  "ncell": e.ncell, "farcap": e.farcap,
                  "floor_area": e.farcap * e.ncell * 1e4}).to_csv(
        os.path.join(a.out, "unit_attrs.csv"), index=False)


if __name__ == "__main__":
    main()
