# -*- coding: utf-8 -*-
"""p8_delineation_summary.py -- summary of p8_delineation_fields.py by the pre-registered criteria
(results_p8_delineation/PREREGISTRATION.md). Output: results_p8_delineation/{per_field.csv, summary.csv}."""
import os, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "results_p8_delineation")
r = pd.concat([pd.read_csv(os.path.join(D, f"rows_t{t}.csv")) for t in (8, 12, 18)])
r["info"] = r.trajectory - r.persistence; r["coord"] = r.rh_full - r.trajectory; r["p1"] = (r.rh_announced - r.announced).abs()
pf = r.groupby(["target", "field"])[["persistence", "announced", "trajectory", "rh_full", "info", "coord"]].mean().reset_index()
pf["coord_minus_info"] = pf["coord"] - pf["info"]; pf.to_csv(os.path.join(D, "per_field.csv"), index=False)
rows = []
for t, g in pf.groupby("target"):
    x = g.coord_minus_info.values; m = x.mean(); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    n_pos = int((x > 0).sum()); lo, hi = m - h, m + h
    verdict = "robust reversal" if (n_pos >= 9 and lo > 0) else ("not robust" if (n_pos <= 5 or lo <= 0 <= hi) else "partly robust")
    rows.append(dict(target=t, fields=len(x), info_mean=g["info"].mean(), info_min=g["info"].min(), info_max=g["info"].max(),
                     coord_mean=g.coord.mean(), coord_min=g.coord.min(), coord_max=g.coord.max(), coord_gt_info=n_pos,
                     diff_mean=m, diff_lo=lo, diff_hi=hi, persistence=g.persistence.mean(), p1_max=r[r.target == t].p1.max(), verdict=verdict))
S = pd.DataFrame(rows); S.to_csv(os.path.join(D, "summary.csv"), index=False)
print(S.round(4).T.to_string()); print(pf[pf.field == "main"].round(3).to_string(index=False))
