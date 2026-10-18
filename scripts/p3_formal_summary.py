"""Registered analysis of results_p3_formal (see docs/P3/registration_p3_formal.md)."""
import glob, os
import numpy as np, pandas as pd
from scipy import stats
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_p3_formal")
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "test_s*.csv")))], ignore_index=True)
test = [f"field{i}" for i in range(1, 11)]
B = d[d.seed == -1].pivot_table(index="eval_field", columns="rule", values="ratio")
P = d[d.rule == "residual_one_step"].pivot_table(index="eval_field", columns="seed", values="ratio")
RP = d[d.rule == "rule_replay"].pivot_table(index="eval_field", columns="seed", values="ratio")
out = {}
m = P.loc[test].mean(axis=1); ann = B.loc[test, "rank_announced"]; dd = m - ann
h = stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
out.update(n_seeds=P.shape[1], policy=m.mean(), announced=ann.mean(),
           persistence=B.loc[test, "persistence"].mean(), trajectory=B.loc[test, "trajectory"].mean(),
           rh_full=B.loc[test, "rh_full"].mean(), diff=dd.mean(), ci_lo=dd.mean() - h, ci_hi=dd.mean() + h,
           t=stats.ttest_1samp(dd, 0).statistic, p_t=stats.ttest_1samp(dd, 0).pvalue,
           p_wilcoxon=stats.wilcoxon(dd).pvalue, draws_above=int((dd > 0).sum()),
           pairs_above=int((P.loc[test].sub(ann, axis=0) > 0).values.sum()),
           seed_mean_lo=P.loc[test].mean(axis=0).min(), seed_mean_hi=P.loc[test].mean(axis=0).max(),
           policy_minus_trajectory=(m - B.loc[test, "trajectory"]).mean(),
           main_policy=P.loc["main"].mean() if "main" in P.index else np.nan,
           main_announced=B.loc["main", "rank_announced"] if "main" in B.index else np.nan,
           rule_replay_max_abs_dev=float(np.abs(RP.sub(B["rank_announced"], axis=0)).values.max()),
           rank_equals_rh_announced=float(np.abs(B["rank_announced"] - B["rh_announced"]).max()))
s = pd.Series(out); print(s.to_string())
per = pd.DataFrame({"announced": ann, "policy_mean": m, "diff": dd, "seed_min": P.loc[test].min(axis=1),
                    "seed_max": P.loc[test].max(axis=1), "trajectory": B.loc[test, "trajectory"]})
print(per.round(4).to_string())
s.to_frame("value").to_csv(os.path.join(R, "formal_summary.csv")); per.to_csv(os.path.join(R, "formal_by_field.csv"))
