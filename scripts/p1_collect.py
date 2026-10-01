# -*- coding: utf-8 -*-
"""p1_collect.py — merge the per-job CSVs of scripts/p1_fqi_job.py.

Writes results_p1/fqi/summary_by_config.csv (mean, sd, min, max over training
seeds at the default listing seed 7, and over all listing seeds) and
results_p1/fqi/summary_oos.csv; prints compact tables.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--fqi", default="results_p1/fqi")
a = ap.parse_args()
fs = [f for f in sorted(glob.glob(os.path.join(a.fqi, "*.csv"))) if not os.path.basename(f).startswith("summary")]
if not fs:
    raise SystemExit(f"no job csv in {a.fqi}")
D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
if "listing_seed" not in D:
    D["listing_seed"] = 7
D["listing_seed"] = D["listing_seed"].fillna(7).astype(int)
if "train_field" not in D:
    D["train_field"] = D["eval_field"]
D["train_field"] = D["train_field"].fillna(D["eval_field"])
D["config"] = np.where(D["mode"] == "oos", "oos", D["eval_field"].astype(str))
D.to_csv(os.path.join(a.fqi, "summary_all_rows.csv"), index=False)

home = D[(D["mode"] != "oos") | (D["eval_field"] == "main")]
# deterministic rules are repeated in every job; de-duplicate per (config, M, rule, listing seed, seed)
home = home.drop_duplicates(["eval_field", "prescreen", "rule", "listing_seed", "seed"])
g7 = home[home.listing_seed == 7].groupby(["eval_field", "prescreen", "rule"]).ratio
gA = home.groupby(["eval_field", "prescreen", "rule"]).ratio
S = pd.concat({"mean_ls7": g7.mean(), "sd_ls7": g7.std(), "min_ls7": g7.min(), "max_ls7": g7.max(),
               "n_ls7": g7.count(), "mean_allls": gA.mean(), "sd_allls": gA.std(),
               "n_allls": gA.count()}, axis=1).reset_index()
S.to_csv(os.path.join(a.fqi, "summary_by_config.csv"), index=False)
pd.set_option("display.width", 160)
for (f, M), s in S.groupby(["eval_field", "prescreen"]):
    print(f"\n== {f}  M={M}")
    print(s.drop(columns=["eval_field", "prescreen"]).round(4).to_string(index=False))
    m = s.set_index("rule")
    if {"value_based", "trajectory"} <= set(m.index):
        print(f"   margin value_based - trajectory: ls7 {m.loc['value_based','mean_ls7'] - m.loc['trajectory','mean_ls7']:+.4f}"
              f"   all listing seeds {m.loc['value_based','mean_allls'] - m.loc['trajectory','mean_allls']:+.4f}")

O = D[D["mode"] == "oos"].copy()
if len(O):
    O["kind"] = np.where(O["rule"] == "value_based",
                         np.where(O["train_field"] == O["eval_field"], "vb_insample", "vb_oos"), O["rule"])
    O.loc[(O["rule"] == "supervised"), "kind"] = np.where(
        O.loc[O["rule"] == "supervised", "train_field"] == O.loc[O["rule"] == "supervised", "eval_field"],
        "sup_insample", "sup_oos")
    O = O.drop_duplicates(["eval_field", "kind", "seed"])
    T = O.pivot_table(index="eval_field", columns="kind", values="ratio", aggfunc="mean")
    T.to_csv(os.path.join(a.fqi, "summary_oos.csv"))
    print("\n== out-of-sample (mean over training seeds)")
    print(T.round(4).to_string())
