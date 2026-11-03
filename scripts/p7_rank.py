# -*- coding: utf-8 -*-
"""p7_rank.py — E3: learning to rank the year's candidates (EXPLORATORY, validation draws only).

Question: each year the decision is "pick the top K = 3 of about 20 prescreened candidates", but
the paper's learning correction (P3) is trained to minimise regression error. Does a ranking
objective aligned with the top-3 choice do better, and is it more stable across seeds?

Per training seed (one job):
  1. collect the P3 training experience unchanged (p7_common.collect_training);
  2. fit the P3 learning correction on it (regression control) and save it for E2 (p7_safe.py);
  3. on the same rows, fit LightGBM LambdaMART; one query group = one (field, rule, episode, year)
     candidate set; relevance label = within-group quintile of the one-step reward EV[u, t]
     (0..4, higher = better), the same target P3 regresses on:
       rank_scratch  ranker on the P3 features (f_now, f_best included as plain features);
       rank_resid    residual form: score = KAPPA * z(f_now within the year's candidates) + ranker,
                     the prior passed to LightGBM as init_score, i.e. start from ranking on
                     announced plans and learn the correction (KAPPA = 1);
  4. decision for every model: each year initiate the K highest-scoring candidates (fqi.rollout).
Evaluation: validation draws 201-210 (M = 20, listing draw 7). The P3 control is also evaluated on
the observed metro field (listing draws 7, 0, 1, 2, 3, 4) because E2 is compared against it there.
Seed 0 also writes the rule baselines and runs the leakage check (p3_formal_test.leak_check).

Usage (repo root):
  OMP_NUM_THREADS=1 PYTHONPATH=src TPMORL_NJOBS=1 python scripts/p7_rank.py --seed 0 --baselines --leak-check
"""
import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7_common as C7                                                    # noqa: E402
from p7_common import E, fqi, row, ev_value, F_NOW_COL                    # noqa: E402
from p1_fqi_job import peak_rss_mb                                        # noqa: E402

KAPPA = 1.0
LGB_PARAMS = dict(objective="lambdarank", n_estimators=400,   # no eval set: metric not needed
                  learning_rate=0.05, num_leaves=31, min_child_samples=20,
                  lambdarank_truncation_level=20, num_threads=1, deterministic=True,
                  force_row_wise=True, verbose=-1)


def zscore(f):
    s = f.std()
    return (f - f.mean()) / s if s > 1e-12 else np.zeros_like(f)


def groups(data):
    """Stack the collected years into LightGBM query groups."""
    X = np.vstack([d["X"] for d in data]).astype(np.float32)
    lab, prior, size = [], [], []
    for d in data:
        r = np.asarray(d["r"], float)
        pct = rankdata(r, method="average") / len(r)            # (0, 1], ties share a grade
        lab.append(np.minimum(4, np.floor(pct * 5 - 1e-9)).astype(int))
        prior.append(KAPPA * zscore(d["X"][:, F_NOW_COL].astype(float)))
        size.append(len(r))
    return X, np.concatenate(lab), np.concatenate(prior), np.asarray(size)


class Ranker:
    """Wraps a LightGBM ranker so fqi.rollout can call predict() on one year's candidate set."""

    def __init__(self, model, resid):
        self.model, self.resid = model, resid

    def predict(self, X):
        s = self.model.predict(np.asarray(X, np.float32), raw_score=True)
        if self.resid:
            s = s + KAPPA * zscore(np.asarray(X, float)[:, F_NOW_COL])
        return s


def fit_rankers(data, seed):
    import lightgbm as lgb
    X, y, prior, size = groups(data)
    out = {}
    for name, resid in (("rank_scratch", False), ("rank_resid", True)):
        m = lgb.LGBMRanker(random_state=seed, **LGB_PARAMS)
        m.fit(X, y, group=size, init_score=(prior if resid else None))
        out[name] = Ranker(m, resid)
    return out, dict(n_rows=len(y), n_groups=len(size), label_counts=np.bincount(y, minlength=5).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--leak-check", action="store_true")
    ap.add_argument("--n-train", type=int, default=None, help="default: frozen P3 value (20); smoke only")
    ap.add_argument("--val-fields", type=int, nargs="+", default=C7.VALIDATION)
    ap.add_argument("--metro-listing", type=int, nargs="*", default=C7.METRO_LISTING)
    ap.add_argument("--out", default="results_p7_rank")
    a = ap.parse_args()
    C7.configure()
    C7.guard_fields(a.val_fields)
    os.makedirs(a.out, exist_ok=True)
    tag = f"s{a.seed}"
    path = os.path.join(a.out, f"rank_{tag}.csv")
    t0 = time.time()

    if a.leak_check:
        from p3_formal_test import leak_check
        lc = pd.DataFrame(leak_check(a.val_fields[0], lead=int(C7.FROZEN["lead"])))
        lc.to_csv(os.path.join(a.out, "leak_check.csv"), index=False)
        main_rows = lc[~lc.year.astype(str).str.startswith("control")]
        ok = (main_rows.obs_max_abs_diff == 0).all() and (main_rows.forecast_max_abs_diff == 0).all()
        sens = lc[lc.year.astype(str).str.startswith("control")].forecast_max_abs_diff.iloc[0] > 0
        print(f"[{tag}] leak check on field{a.val_fields[0]}: {'PASS' if ok and sens else 'FAIL'}\n"
              f"{lc.to_string(index=False)}", flush=True)
        assert ok, "future information reaches the decision-time inputs"
        assert sens, "leak check is not sensitive"

    data = C7.collect_training(a.seed, a.n_train)
    print(f"[{tag}] collected {len(data)} transitions in {time.time() - t0:.0f}s", flush=True)
    qa, qb = C7.fit_p3(data, a.seed)
    os.makedirs(C7.MODEL_DIR, exist_ok=True)
    joblib.dump(dict(qa=qa, qb=qb, seed=a.seed, n_train=a.n_train or C7.FROZEN["n_train"]),
                C7.model_path(a.seed), compress=3)
    rankers, info = fit_rankers(data, a.seed)
    print(f"[{tag}] fitted P3 + rankers at {time.time() - t0:.0f}s, rss {peak_rss_mb():.0f} MB, "
          f"{info['n_groups']} groups / {info['n_rows']} rows, labels {info['label_counts']}", flush=True)
    del data

    models = dict(p3=(qa, qb), **{k: (v, v) for k, v in rankers.items()})
    rows = []
    for j in a.val_fields:
        name = f"field{j}"
        C, ann, efn = E.field_setup(j, 1)
        msg = f"[{tag}] {name}:"
        for rule, (ma, mb) in models.items():
            s0 = time.time()
            init, _, _ = fqi.rollout(efn, ma, mb, None)
            v = ev_value(C["EV"], init)
            rows.append(row(C, rule, init, v, name, a.seed, time.time() - s0, field_set="validation"))
            msg += f" {rule} {v / C['v_orc']:.4f}"
        ri, _ = E.conservative_rollout(efn, qa, qb, np.inf)          # fairness: must equal rank_announced
        rows.append(row(C, "rule_replay", ri, ev_value(C["EV"], ri), name, a.seed, 0.0, field_set="validation"))
        if a.baselines:
            for r in E.baselines(C, ann, name):
                r.update(field_set="validation"); rows.append(r)
            rr = {x["rule"]: x["ratio"] for x in rows if x["eval_field"] == name}
            msg += f" | rank_ann {rr['rank_announced']:.4f} replay {rr['rule_replay']:.4f}"
        print(msg, flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    for ls in a.metro_listing:                                       # P3 control on the metro field
        C, ann, efn = C7.metro_setup(ls)
        init, _, _ = fqi.rollout(efn, qa, qb, None)
        rows.append(row(C, "p3", init, ev_value(C["EV"], init), "metro", a.seed, 0.0,
                        listing_seed=ls, field_set="metro"))
        if a.baselines:
            for r in E.baselines(C, ann, "metro"):
                r.update(listing_seed=ls, field_set="metro"); rows.append(r)
        pd.DataFrame(rows).to_csv(path, index=False)

    D = pd.DataFrame(rows)
    ref = f"results_p3_explore/explore_L4_rf1_res_cons_it1_h1_s{a.seed}.csv"
    if os.path.exists(ref) and a.n_train is None:
        R = pd.read_csv(ref)
        new = D[D.rule == "p3"].set_index("eval_field").ratio
        print(f"[{tag}] " + C7.self_check(new, R[R.rule == "value_based"].set_index("eval_field").ratio,
                                          "P3 control vs results_p3_explore (validation)"), flush=True)
    ref = f"results_p3_metro/metro_s{a.seed}.csv"
    if os.path.exists(ref) and a.n_train is None and a.metro_listing:
        R = pd.read_csv(ref)
        new = D[(D.rule == "p3") & (D.field_set == "metro")].set_index("listing_seed").ratio
        print(f"[{tag}] " + C7.self_check(new, R[R.rule == "residual_one_step"].set_index("listing_seed").ratio,
                                          "P3 control vs results_p3_metro (metro)"), flush=True)
    if a.baselines:
        v = D[D.field_set == "validation"]
        rep = v[v.rule == "rule_replay"].set_index("eval_field").ratio
        ra = v[v.rule == "rank_announced"].set_index("eval_field").ratio
        mx = float((rep - ra).abs().max())
        print(f"[{tag}] fairness: rule_replay vs rank_announced max |diff| = {mx:.2e} -> "
              f"{'PASS' if mx < 1e-12 else 'FAIL'}", flush=True)
    print(f"[{tag}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
