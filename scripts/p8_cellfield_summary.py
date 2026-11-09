# -*- coding: utf-8 -*-
"""p8_cellfield_summary.py -- summary of p8_cellfield_delineation.py by the pre-registered criteria.
Output: results_p8_delineation/{cell_per_field.csv, cell_summary.csv, cell_paired.csv}."""
import glob, os, numpy as np, pandas as pd
from scipy import stats
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_p8_delineation")
r = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, "cell_*_t*.csv")))]); r["field"] = r["field"].astype(str)
r["info"] = r.trajectory - r.persistence; r["coord"] = r.rh_full - r.trajectory; r["p1"] = (r.rh_announced - r.announced).abs()
pf = r.groupby(["mode", "target", "field"])[["persistence", "announced", "trajectory", "rh_full", "info", "coord"]].mean().reset_index()
pf["cmi"] = pf["coord"] - pf["info"]; pf.to_csv(os.path.join(D, "cell_per_field.csv"), index=False)
ci = lambda x: (x.mean(), *(x.mean() + np.array([-1, 1]) * stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))))
S, Pd = [], []
for (m, t), g in pf.groupby(["mode", "target"]):
    mu, lo, hi = ci(g.cmi.values)
    S.append(dict(mode=m, target=t, persistence=g.persistence.mean(), info=g["info"].mean(), coord=g["coord"].mean(),
                  coord_gt_info=int((g.cmi > 0).sum()), fields=len(g), cmi=mu, cmi_lo=lo, cmi_hi=hi, p1_max=r[(r["mode"] == m) & (r.target == t)].p1.max()))
for m, g in pf.groupby("mode"):
    w = g.pivot(index="field", columns="target", values=["cmi", "info", "coord", "persistence"])
    for t in (8, 18):
        for q in ("cmi", "info", "coord", "persistence"):
            mu, lo, hi = ci((w[q][t] - w[q][12]).values)
            Pd.append(dict(mode=m, contrast=f"{t} - 12", quantity=q, mean=mu, lo=lo, hi=hi))
    Pd.append(dict(mode=m, contrast="corr info 12 vs 18", quantity="info", mean=np.corrcoef(w["info"][12], w["info"][18])[0, 1], lo=np.nan, hi=np.nan))
S = pd.DataFrame(S); Pd = pd.DataFrame(Pd)
S.to_csv(os.path.join(D, "cell_summary.csv"), index=False); Pd.to_csv(os.path.join(D, "cell_paired.csv"), index=False)
pd.set_option("display.width", 200); print(S.round(3).to_string(index=False)); print(Pd.round(3).to_string(index=False))
