"""Summarize results_p3_explore (exploratory, validation fields 201-210)."""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_p3_explore")
d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, "explore_*.csv"))], ignore_index=True)
if "residual" not in d: d["residual"] = 0
d["residual"] = d["residual"].fillna(0).astype(int)
out = []
for L in sorted(d.lead.dropna().unique()):
    b = d[(d.lead == L) & (d.rule != "value_based")].pivot_table(index="eval_field", columns="rule", values="ratio")
    for rf, res in ((0, 0), (1, 0), (1, 1)):
        v = d[(d.lead == L) & (d.rule == "value_based") & (d.rulefeat == rf) & (d.residual == res)]
        if v.empty: continue
        pv = v.pivot_table(index="eval_field", columns="seed", values="ratio")
        m = pv.mean(axis=1).reindex(b.index)
        rec = dict(lead=L, rulefeat=rf, residual=res, n_seeds=pv.shape[1], vb=m.mean(),
                   seed_means=" ".join(f"{x:.3f}" for x in pv.mean(axis=0)))
        for ref in ("rank_announced", "rh_announced", "trajectory"):
            dd = m - b[ref]; h = stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
            rec[f"d_{ref}"] = dd.mean(); rec[f"ci_{ref}"] = f"{dd.mean()-h:+.4f} to {dd.mean()+h:+.4f}"
            rec[f"fields_above_{ref}"] = int((dd > 0).sum())
            rec[f"pairs_above_{ref}"] = int((pv.reindex(b.index).sub(b[ref], axis=0) > 0).values.sum())
        rec["sd_seed_means"] = float(pv.mean(axis=0).std(ddof=1)) if pv.shape[1] > 1 else np.nan
        if pv.shape[1] > 2:   # same comparison restricted to seeds 0-1, comparable with the other rows
            m01 = pv[[c for c in pv.columns if c in (0, 1)]].mean(axis=1).reindex(b.index)
            rec["vb_seeds01"] = m01.mean(); rec["d_rank_seeds01"] = (m01 - b["rank_announced"]).mean()
        out.append(rec)
    for r in ("persistence", "rank_announced", "rh_announced", "trajectory", "rh_full"):
        print(f"lead {L:g} {r:15s} {b[r].mean():.4f}")
s = pd.DataFrame(out); pd.set_option("display.width", 250); print(s.to_string(index=False))
s.to_csv(os.path.join(R, "explore_summary.csv"), index=False)
