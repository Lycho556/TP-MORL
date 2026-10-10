# -*- coding: utf-8 -*-
"""p2_formal_summary.py — summarize the P2 formal transfer test against ranking on announced plans.

Reads the CSVs written by scripts/p2_generalize.py. No training, no new method:
only averages and paired comparisons of numbers already on disk.

Two uses
  1. Summary:  --runs DIR [--baselines DIR] --fields 1 2 ... --out SUMMARY.md
     Value-based policy (rule value_based_multi, all seeds found in DIR) against the
     baselines (rows with seed == -1, taken from --baselines or from DIR itself).
  2. Selection of the history window H on validation fields (stage 1):
     --select-h DIR_H0 DIR_H1 DIR_H2 --fields 201 ... 210
     prints the chosen H on the last line of stdout. Rule (fixed in docs/P2正式测试说明.md):
     highest validation mean; among windows within 0.002 of the best, the smallest H.

Pre-registered wording for the formal test (docs/P2正式测试说明.md, section 2):
  d_f = (mean over seeds of the policy on field f) - (announced on field f), f = 1..10
  'outperforms'               mean(d) > 0 and the 95% t-interval of mean(d) over fields excludes 0
  'slightly above, not distinguishable'   mean(d) > 0 but the interval includes 0
  'does not beat'             mean(d) <= 0
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

POLICY = "value_based_multi"
BASE_RULES = ["persistence", "rh_announced", "trajectory", "rh_full"]


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "gen_s*.csv")))
    if not fs:
        sys.exit(f"no gen_s*.csv in {d}")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def baselines(df, fields):
    b = df[df.seed == -1].pivot_table(index="eval_field", columns="rule", values="ratio")
    missing = [f for f in fields if f not in b.index]
    if missing:
        sys.exit(f"baseline rows missing for {missing}")
    return b.loc[fields]


def policy(df, fields):
    p = df[df.rule == POLICY].pivot_table(index="eval_field", columns="seed", values="ratio")
    missing = [f for f in fields if f not in p.index]
    if missing:
        sys.exit(f"policy rows missing for {missing}")
    return p.loc[fields]


def compare(pol, base):
    ann = base["rh_announced"]
    d = pol.mean(axis=1) - ann                       # one paired difference per field
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    half = stats.t.ppf(0.975, n - 1) * se if n > 1 else np.nan
    lo, hi = d.mean() - half, d.mean() + half
    if d.mean() <= 0:
        verdict = "does not beat"
    elif lo > 0:
        verdict = "outperforms"
    else:
        verdict = "slightly above, not distinguishable"
    return dict(policy_mean=float(pol.values.mean()), announced_mean=float(ann.mean()),
                diff=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi),
                pair_wins=int((pol.values > ann.values[:, None]).sum()), pairs=int(pol.size),
                field_wins=int((d > 0).sum()), fields=n, seeds=int(pol.shape[1]),
                verdict=verdict, per_field=d)


def fields_arg(xs):
    return [x if str(x).startswith("field") or x == "main" else f"field{x}" for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs")
    ap.add_argument("--baselines", default=None)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--select-h", nargs="+", default=None,
                    help="directories of the H validation runs, in the order H=0, H=1, H=2")
    ap.add_argument("--h-values", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    fields = fields_arg(a.fields)

    if a.select_h:
        means = {}
        for h, d in zip(a.h_values, a.select_h):
            df = load(d)
            means[h] = float(policy(df, fields).values.mean())
            print(f"H={h}: validation mean {means[h]:.4f}  ({d})")
        best = max(means.values())
        chosen = min(h for h, m in means.items() if m >= best - 0.002)
        print(f"chosen H = {chosen} (best {best:.4f}; smallest H within 0.002)")
        print(chosen)
        return

    runs = load(a.runs)
    base = baselines(load(a.baselines) if a.baselines else runs, fields)
    pol = policy(runs, fields)
    r = compare(pol, base)
    lines = [
        "# P2 formal transfer test — summary",
        "",
        f"Runs: `{a.runs}`  |  fields: {', '.join(fields)}  |  seeds: {sorted(pol.columns.tolist())}",
        "",
        "| Rule | Mean ratio |",
        "|---|---|",
    ]
    for k in BASE_RULES:
        if k in base.columns:
            lines.append(f"| {k} | {base[k].mean():.4f} |")
    lines += [
        f"| **value-based policy** | **{r['policy_mean']:.4f}** |",
        "",
        f"- Difference to ranking on announced plans (mean over fields of seed-mean minus announced): "
        f"**{r['diff']:+.4f}**, 95% t-interval over {r['fields']} fields [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]",
        f"- Field--seed pairs above announced: {r['pair_wins']}/{r['pairs']}; "
        f"fields whose seed-mean is above announced: {r['field_wins']}/{r['fields']}",
        f"- Pre-registered wording: **{r['verdict']}**",
        "",
        "| Field | Policy (seed mean) | Announced | Difference | Trajectory-informed |",
        "|---|---|---|---|---|",
    ]
    for f in fields:
        lines.append(f"| {f} | {pol.loc[f].mean():.4f} | {base.loc[f, 'rh_announced']:.4f} | "
                     f"{r['per_field'][f]:+.4f} | {base.loc[f, 'trajectory']:.4f} |")
    txt = "\n".join(lines) + "\n"
    print(txt)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt)


if __name__ == "__main__":
    main()
