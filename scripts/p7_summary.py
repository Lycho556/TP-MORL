# -*- coding: utf-8 -*-
"""p7_summary.py — statistics for the P7 validation runs (E2 results_p7_safe, E3 results_p7_rank).

Rule 6 of the P7 plan: per draw, average the five seeds first (E2 is already a five-seed ensemble),
then subtract the benchmark; report the mean difference over the draws, its 95% t-interval, the
Wilcoxon signed-rank p-value and the number of draws won. "improves on" only if the whole interval
lies above zero. Benchmarks: ranking on announced plans and the paper's learning correction (P3,
retrained here as the control of p7_rank.py). For E3 also the seed spread, because the pass criterion
asks for a higher mean AND a smaller variance across seeds.

Usage (repo root):  python scripts/p7_summary.py [--rank-dir results_p7_rank] [--safe-dir results_p7_safe]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE_RULES = ["persistence", "rank_announced", "trajectory", "rh_announced", "rh_full"]


def compare(x, ref):
    d = (x - ref).dropna()
    n = len(d)
    if n < 2:
        return dict(n=n, mean=float(d.mean()) if n else np.nan, lo=np.nan, hi=np.nan, p_t=np.nan, p_w=np.nan,
                    wins=int((d > 1e-12).sum()))
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    try:
        pw = stats.wilcoxon(d).pvalue if (d != 0).any() else np.nan
    except ValueError:
        pw = np.nan
    return dict(n=n, mean=m, lo=m - h, hi=m + h, p_t=stats.ttest_1samp(d, 0).pvalue, p_w=pw,
                wins=int((d > 1e-12).sum()))                 # 1e-16 float noise is not a win


def wording(c):
    if not np.isfinite(c["lo"]):
        return "n/a"
    return "improves on" if c["lo"] > 0 else "falls below" if c["hi"] < 0 else "does not beat"


def fmt(c):
    return (f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | t p={c['p_t']:.3g} "
            f"Wilcoxon p={c['p_w']:.3g} | wins {c['wins']}/{c['n']} | {wording(c)}")


def load(d, pat):
    fs = sorted(glob.glob(os.path.join(d, pat)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def key(X):
    ls = X["listing_seed"] if "listing_seed" in X else pd.Series(np.nan, index=X.index)
    return X.eval_field + "_" + ls.fillna(-1).astype(int).astype(str)


def per_field(D, rule, fs):
    x = D[(D.rule == rule) & (D.field_set == fs)]
    return x.assign(k=key(x)).groupby("k").ratio.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-dir", default="results_p7_rank")
    ap.add_argument("--safe-dir", default="results_p7_safe")
    a = ap.parse_args()
    R = load(a.rank_dir, "rank_s*.csv")
    if R.empty:
        raise SystemExit(f"no rank_s*.csv in {a.rank_dir}")
    S = load(a.safe_dir, "safe.csv")
    seeds = sorted(int(s) for s in R[R.rule == "p3"].seed.unique())
    out = {"rank": [], "safe": []}

    for fs, title, n_label in (("validation", "Validation draws 201-210", "draws"),
                               ("metro", "Observed metro field (listing draws 7,0,1,2,3,4)", "listing draws")):
        base = {r: per_field(R, r, fs) for r in BASE_RULES}
        p3 = per_field(R, "p3", fs)
        if p3.empty or base["rank_announced"].empty:
            continue
        hdr = [f"## {title}", "",
               f"Seeds {seeds}; per {n_label[:-1]} the seed mean is taken first. "
               f"Differences: mean [95% t-interval] over {len(p3)} {n_label}.", "",
               "Benchmarks (mean ratio): " + ", ".join(f"{r} {base[r].mean():.4f}" for r in BASE_RULES
                                                      if not base[r].empty) + f", p3 {p3.mean():.4f}", ""]
        for kind, D, rules in (("rank", R, ["p3", "rank_scratch", "rank_resid"]),
                               ("safe", S, sorted(S.rule.unique()) if not S.empty else [])):
            if D.empty or (kind == "rank" and fs == "metro"):
                continue
            lines = list(hdr)
            lines += ["| policy | mean | vs rank_announced | vs p3 |", "|---|---|---|---|"]
            for r in rules:
                x = per_field(D, r, fs)
                if x.empty:
                    continue
                c1, c2 = compare(x, base["rank_announced"]), compare(x, p3)
                lines.append(f"| {r} | {x.mean():.4f} | {fmt(c1)} | " + ("—" if r == "p3" else fmt(c2)) + " |")
            if kind == "rank" and fs == "validation":
                lines += ["", "Seed stability (E3 pass criterion: smaller variance than p3):",
                          "| policy | sd over seeds of the 10-draw mean | mean over draws of the sd across seeds | range of seed means |",
                          "|---|---|---|---|"]
                for r in rules:
                    x = D[(D.rule == r) & (D.field_set == fs)]
                    sm = x.groupby("seed").ratio.mean()
                    within = x.groupby("eval_field").ratio.std(ddof=1).mean()
                    lines.append(f"| {r} | {sm.std(ddof=1):.4f} | {within:.4f} | {sm.min():.4f}–{sm.max():.4f} |")
            if kind == "safe":
                dv = D[D.field_set == fs].groupby("rule").n_deviate.mean()
                lines += ["", "Initiations that differ from the rule's choice, per run: " +
                          ", ".join(f"{r} {v:.1f}" for r, v in dv.items())]
            out[kind] += lines + [""]

    notes = {"rank": ["# P7 E3 — learning to rank (validation only)", "",
                      "Exploratory. Test draws 1-10 and the main field were not evaluated.", ""],
             "safe": ["# P7 E2 — conservative deviation (validation + observed metro field)", "",
                      "Exploratory. Variants chosen on validation draws only (see selection.txt); the metro field "
                      "had already been used by p3_metro_test.py and is reported, not used to choose.", ""]}
    for kind, d in (("rank", a.rank_dir), ("safe", a.safe_dir)):
        if out[kind] and os.path.isdir(d):
            txt = "\n".join(notes[kind] + out[kind])
            if kind == "safe" and os.path.exists(os.path.join(d, "selection.txt")):
                txt += "\n## Selection\n\n```\n" + open(os.path.join(d, "selection.txt"), encoding="utf-8").read() + "```\n"
            open(os.path.join(d, "SUMMARY.md"), "w", encoding="utf-8").write(txt)
            print(txt)


if __name__ == "__main__":
    main()
