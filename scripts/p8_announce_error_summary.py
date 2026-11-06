# -*- coding: utf-8 -*-
"""p8_announce_error_summary.py -- summary of p8_announce_error.py by the pre-registered criterion.
Output: results_p8_annerr/{per_field.csv, summary.csv}."""
import glob, os, numpy as np, pandas as pd
from scipy import stats
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_p8_annerr")
r = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, "rows_*.csv")))])
r["field"] = r["field"].astype(str)
pf = r.groupby(["field", "kind", "e"])[["persistence", "rank_announced", "trajectory", "mean_abs_d"]].mean().reset_index()
pf["share"] = (pf.rank_announced - pf.persistence) / (pf.trajectory - pf.persistence); pf.to_csv(os.path.join(D, "per_field.csv"), index=False)
out = []
for (k, e), g in pf.groupby(["kind", "e"]):
    x = g.share.values; h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    m = g[g.field == "main"].iloc[0]
    out.append(dict(kind=k, e=e, fields=len(x), share_mean=x.mean(), share_lo=x.mean() - h, share_hi=x.mean() + h, share_min=x.min(),
                    rank_mean=g.rank_announced.mean(), main_rank=m.rank_announced, main_share=m.share))
S = pd.DataFrame(out).sort_values(["kind", "e"]); S.to_csv(os.path.join(D, "summary.csv"), index=False)
sym2 = S[(S.kind == "sym") & (S.e == 2)].share_mean.iloc[0]
print(S.round(3).to_string(index=False)); print("criterion (sym e=2 mean share >= 0.50):", "robust" if sym2 >= 0.5 else "depends on accuracy", round(sym2, 3))
