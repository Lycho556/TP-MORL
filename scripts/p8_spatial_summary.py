# -*- coding: utf-8 -*-
"""p8_spatial_summary.py -- summary of p8_spatial_holdout.py by the pre-registered criteria.
Output: results_p8_spatial/summary.json, corr_summary.csv."""
import glob, json, os, numpy as np, pandas as pd
from scipy import stats
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_p8_spatial")
r = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, "rows_s*.csv")))]); r["diff"] = r.learned - r.rank_announced
te = r[r["eval"].str.startswith("test")].groupby("eval")[["learned", "rank_announced", "diff"]].mean()
x = te["diff"].values; h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); m = x.mean()
lo, hi = m - h, m + h
verdict = ("strongly supported" if (m >= 0.020 and lo > 0) else "supported" if (m >= 0.0125 and lo > 0)
           else "not supported" if (lo <= 0 <= hi or m <= 0) else "partly supported")
me = r[r["eval"] == "metro"]; mx = me.groupby("listing")["diff"].mean().values; mh = stats.t.ppf(0.975, len(mx) - 1) * mx.std(ddof=1) / np.sqrt(len(mx))
tr = r[r["eval"].str.startswith("test")]
S = dict(test_learned=float(te.learned.mean()), test_rank=float(te.rank_announced.mean()), test_diff=float(m), test_ci=[float(lo), float(hi)],
         test_draws_positive=int((x > 0).sum()), test_runs_positive=int((tr["diff"] > 0).sum()), test_runs=len(tr),
         seed_means=tr.groupby("seed")["diff"].mean().round(4).tolist(), wilcoxon_p=float(stats.wilcoxon(x).pvalue),
         metro_learned=float(me.learned.mean()), metro_rank=float(me.rank_announced.mean()), metro_diff=float(mx.mean()),
         metro_ci=[float(mx.mean() - mh), float(mx.mean() + mh)], metro_runs_positive=int((me["diff"] > 0).sum()), metro_runs=len(me), verdict=verdict)
c = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, "corr_s*.csv")))])
cs = c.groupby(["grp", "fold"])[["corr", "dq", "R", "n"]].mean().reset_index(); cs.to_csv(os.path.join(D, "corr_summary.csv"), index=False)
json.dump(S, open(os.path.join(D, "summary.json"), "w"), indent=1); print(json.dumps(S, indent=1)); print(cs.round(3).to_string(index=False))
