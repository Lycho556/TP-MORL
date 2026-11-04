# -*- coding: utf-8 -*-
"""p6_summary.py -- summaries of the distribution-shift tests (Supplementary Section S13, Table S24).
Reads the per-run files written by p6_shift_diag.py (tags a1 and a1early), p6_early_infra.py and
p6_orig_on_early.py, and writes
  results_p6_shift/a1_compare_summary.csv  (A1: predicted correction vs true residual, by model and field)
  results_p6_shift/a2_summary.csv          (A2: early-ending training, learned minus ranking on announced plans)
Field groups: 'test*' -> test, 'early*' -> early, otherwise the field name (main, metro).
"""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "results_p6_shift")
grp = lambda s: "test" if str(s).startswith("test") else ("early" if str(s).startswith("early") else str(s))
COLS = ["ratio", "dq_sel", "R_sel", "dq_all", "R_all", "corr_all", "dq_swap_in", "R_swap_in", "n_swap_in"]


def a1(out_dir=D):
    parts = []
    for model, tag in (("original", "a1"), ("early_trained", "a1early")):
        df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, f"{tag}_rows_s*.csv")))])
        m = df.assign(grp=df.field.map(grp)).groupby("grp")[COLS].mean()
        m["over_sel"] = m.dq_sel - m.R_sel; m["over_all"] = m.dq_all - m.R_all
        parts.append(m.reset_index().assign(model=model).set_index("model"))
    s = pd.concat(parts).round(3); s.index.name = None
    s.to_csv(os.path.join(out_dir, "a1_compare_summary.csv")); return s


def a2(out_dir=D):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, "a2_rows_s*.csv")))])
    df = df.assign(grp=df["eval"].map(grp), diff=df.learned - df.rank_announced)
    g = df.groupby("grp")
    s = g[["learned", "rank_announced", "trajectory", "rh_full", "diff"]].mean()
    s["diff_min"] = g["diff"].min(); s["diff_max"] = g["diff"].max()
    s["runs_positive"] = g["diff"].apply(lambda x: int((x > 0).sum())); s["runs"] = g.size()
    s = s.round(4); s.to_csv(os.path.join(out_dir, "a2_summary.csv")); return s


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else D
    os.makedirs(out, exist_ok=True); a1(out); a2(out); print("written to", out)
