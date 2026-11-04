# Figure 7 — reinforcement learning (this study) against ranking on announced plans in the ten unseen
# test field draws (registered test, results_p3_formal; listing draw 7, M = 20, five training seeds).
# (a) ratio to the reference schedule per draw; (b) paired difference to ranking on announced plans.
import glob, os
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from scipy import stats
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")) + "/"
OUT = _ROOT + "paper_ijgis/figures/"
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6, "xtick.major.size": 3,
    "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
C = dict(announced="#E08A3C", rl="#2E7EC1", trajectory="#555555", persistence="#E6B800")
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(_ROOT + "results_p3_formal/test_s*.csv"))])
test = [f"field{i}" for i in range(1, 11)]
B = d[d.seed == -1].pivot_table(index="eval_field", columns="rule", values="ratio").loc[test]
P = d[d.rule == "residual_one_step"].pivot_table(index="eval_field", columns="seed", values="ratio").loc[test]
order = B["rank_announced"].sort_values().index
x = np.arange(len(order))
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[2.2, 1], wspace=0.32))
axa.plot(x, B.loc[order, "trajectory"], ls="none", marker="_", ms=11, mew=1.6, color=C["trajectory"],
         label="Trajectory-informed greedy (knows the future)")
axa.plot(x, B.loc[order, "rank_announced"], ls="none", marker="o", ms=5, color=C["announced"],
         label="Ranking on announced plans")
pm = P.loc[order].mean(axis=1)
axa.errorbar(x, pm, yerr=[pm - P.loc[order].min(axis=1), P.loc[order].max(axis=1) - pm], fmt="o", ms=5,
             color=C["rl"], ecolor=C["rl"], elinewidth=0.8, capsize=1.5,
             label="Learned value correction")
for i, k in enumerate(order):
    axa.plot([i, i], [B.loc[k, "rank_announced"], pm[k]], color="#bbbbbb", lw=0.7, zorder=0)
axa.set_xticks(x); axa.set_xticklabels([k.replace("field", "") for k in order])
axa.set_xlabel("Unseen field draw (ordered by ranking on announced plans)")
axa.set_ylabel("Ratio to the reference schedule")
axa.set_ylim(0.82, 0.965)
axa.legend(loc="upper left", fontsize=7.2, handletextpad=0.3, borderaxespad=0.1)
axa.text(-0.12, 1.04, "a", transform=axa.transAxes, fontsize=10, fontweight="bold")
dd = (P.mean(axis=1) - B["rank_announced"]).values
gap = (B["trajectory"] - B["rank_announced"]).values
h = stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
rng = np.random.default_rng(1)
axb.axhline(0, color="#999999", lw=0.7)
axb.scatter(rng.uniform(-0.12, 0.12, len(dd)), dd, s=14, color=C["rl"], zorder=3)
axb.errorbar([0.32], [dd.mean()], yerr=[[h], [h]], fmt="D", ms=5, color="black", capsize=3, lw=1)
axb.text(0.5, dd.mean() - h - 0.0035, f"mean of 10 draws\n{dd.mean():+.3f}; 95% CI\n{dd.mean()-h:+.3f} to {dd.mean()+h:+.3f}",
         va="top", fontsize=7.2)
axb.axhline(gap.mean(), color=C["trajectory"], lw=0.9, ls="--")
axb.text(1.28, gap.mean() + 0.0045, f"gap to knowing\nthe future, {gap.mean():.3f}", fontsize=7, color=C["trajectory"], ha="right", va="bottom")
axb.set_xlim(-0.35, 1.3); axb.set_xticks([])
axb.set_ylim(-0.012, 0.062)
axb.set_ylabel("Learned value correction\nminus ranking on announced plans")
axb.text(-0.3, 1.04, "b", transform=axb.transAxes, fontsize=10, fontweight="bold")
fig.savefig(OUT + "fig6_learning.pdf"); fig.savefig(OUT + "fig6_learning.png", dpi=300)
