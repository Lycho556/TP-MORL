# -*- coding: utf-8 -*-
"""p2_generalize.py — does the value-based policy generalize to unseen fields?

Design (fixed before running; not tuned on the test fields):
  * Training: double FQI on pooled simulated experience from N_TRAIN scenario
    field draws (field seeds 101, 102, ...), EP_EACH episodes per behaviour rule
    (random, persistence greedy, trajectory-informed greedy) per field.
  * Test: field draws never used in training (default 1..10 and the main field),
    M = 20, listing draw 7. Each test field is valued on its own EV table and
    reference schedule.
  * Comparators on every test field: persistence greedy, rolling horizon with
    announced information, and the perfect-foresight bounds (trajectory-informed
    greedy, rolling horizon with full information, reference). Optionally the
    policy trained on the main field only (--with-single), i.e. the earlier
    out-of-sample test.

Output: <out>/gen_s<seed>.csv, one row per (rule, eval_field).

Usage (repo root):
  PYTHONPATH=src TPMORL_NJOBS=1 python scripts/p2_generalize.py --seed 0 --out results_p2
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1_common as P                                                  # noqa: E402
from p1_fqi_job import build, eval_fqi, greedy_rows, row, peak_rss_mb   # noqa: E402
from p1_rolling_horizon import forecast_tables, run_rolling             # noqa: E402
from exp_fqi_local import make_trees                                    # noqa: E402
from tpmorl.rl import fqi                                               # noqa: E402

MAIN_FIELD = "main"


def collect_field(field_seed, ep_each, seed):
    C = build("main", 20, field_seed=field_seed)          # sets global field
    EV = C["EV"]
    reward = (lambda E: (lambda u, t: float(E[u, min(t, E.shape[1] - 1)])))(EV)
    return fqi.collect(C["env_fn"], reward,
                       {"random": None, "myopic": C["EVm"], "temporal_greedy": EV},
                       n_ep_each=ep_each, seed=seed)


def rh_rows(C, field):
    e0 = C["env_fn"]()
    ann = forecast_tables(e0, "announced", True)
    EV, elig = C["EV"], C["elig"]
    out = []
    for rule, get in (("rh_announced", lambda t: ann[t]), ("rh_full", lambda t: EV)):
        t0 = time.time()
        init = run_rolling(C["env_fn"](), get, elig)
        val = float(sum(EV[u, y] for u, y in init.items()))
        out.append(row(C, rule, init, val, field, -1, time.time() - t0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--train-base", type=int, default=101)
    ap.add_argument("--ep-each", type=int, default=2)
    ap.add_argument("--test-fields", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--with-single", action="store_true",
                    help="also train on the main field only and evaluate on the test fields")
    ap.add_argument("--baselines", action="store_true",
                    help="also write greedy / rolling-horizon / reference rows")
    ap.add_argument("--history", type=int, default=0,
                    help="backward-looking trend features: level at t minus level at t-H (0 = off)")
    ap.add_argument("--out", default="results_p2")
    a = ap.parse_args()
    from tpmorl.env import opportunity as OPP
    OPP.HISTORY = int(a.history)
    os.makedirs(a.out, exist_ok=True)
    train = [a.train_base + i for i in range(a.n_train)]
    assert not set(train) & set(a.test_fields), "train/test fields overlap"
    path = os.path.join(a.out, f"gen_s{a.seed}" + (f"_h{a.history}" if a.history else "") + ".csv")
    rows = []

    t0 = time.time()
    data = []
    for k in train:
        data += collect_field(k, a.ep_each, seed=a.seed * 10007 + k)
    n_rows = sum(len(d["r"]) for d in data)
    t_col = time.time() - t0
    print(f"[s{a.seed}] collected {len(data)} transitions / {n_rows} rows "
          f"from {len(train)} fields in {t_col:.0f}s", flush=True)
    qa, qb, _ = fqi.fit_double(data, P.DEFAULTS["gamma"], make_trees(seed=a.seed),
                               verbose=False)
    t_fit = time.time() - t0 - t_col
    print(f"[s{a.seed}] fitted in {t_fit:.0f}s, rss {peak_rss_mb():.0f} MB", flush=True)
    del data

    single = None
    if a.with_single:
        s0 = time.time()
        d1 = collect_field(None, 10, seed=a.seed)       # same as the main-field job
        q1a, q1b, _ = fqi.fit_double(d1, P.DEFAULTS["gamma"], make_trees(seed=a.seed),
                                     verbose=False)
        single = (q1a, q1b)
        del d1
        print(f"[s{a.seed}] single-field policy fitted in {time.time() - s0:.0f}s", flush=True)

    for j in list(a.test_fields) + [MAIN_FIELD]:
        fs = None if j == MAIN_FIELD else j
        name = "main" if j == MAIN_FIELD else f"field{j}"
        C = build("main", 20, field_seed=fs)
        s0 = time.time()
        init, val, _ = eval_fqi(C, qa, qb)
        rows.append(row(C, "value_based_multi", init, val, name, a.seed, time.time() - s0,
                        n_train_fields=a.n_train, ep_each=a.ep_each, history=a.history))
        msg = f"[s{a.seed}] {name}: multi {val / C['v_orc']:.4f}"
        if single is not None:
            s0 = time.time()
            init, val, _ = eval_fqi(C, *single)
            rows.append(row(C, "value_based_single", init, val, name, a.seed,
                            time.time() - s0))
            msg += f" single {val / C['v_orc']:.4f}"
        if a.baselines:
            rows += greedy_rows(C, name, a.seed, with_random=False)
            rows += rh_rows(C, name)
            r = {x["rule"]: x["ratio"] for x in rows if x["eval_field"] == name}
            msg += f" ann {r['rh_announced']:.4f} pers {r['persistence']:.4f} traj {r['trajectory']:.4f}"
        print(msg, flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[s{a.seed}] done in {time.time() - t0:.0f}s, rss {peak_rss_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
