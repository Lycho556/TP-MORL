# Figure 5 — information ladder (where the value of timing comes from).
# (a) ratio to the reference schedule per information level, six listing draws (M = 20);
# (b) selection- and timing-related loss of the same rules (Equation decomp).
# Data: results_p1/rolling/rolling_results.csv, results_p1/tables/main_M20.csv.
# Run inside a kernel where apply_figure_style() (figure-style skill) is defined.
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt

import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")) + "/"  # repository root
R = _ROOT + ""
OUT = R + "paper_ijgis/figures/"
# Publication rcParams (explicit, so the script runs standalone)
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.grid": False, "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.titleweight": "normal", "axes.titlelocation": "left",
    "axes.labelweight": "normal", "lines.linewidth": 1.2, "patch.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
RH = pd.read_csv(R + "results_p1/rolling/rolling_results.csv")
rh = RH[(RH.infra_mode == "cluster") & (RH.conformity) & (RH.variant == "base") & (RH.M == 20)] \
    .pivot_table(index="listing_seed", columns="rule", values="ratio")
T3 = pd.read_csv(R + "results_p1/tables/main_M20.csv", index_col=0)
assert np.allclose(rh.rh_current, rh.persistence)

LEV = [  # key, label, what the rule knows, colour
    ("persistence", "Persistence greedy", "current conditions only", "#E69F00"),
    ("rh_announced", "Ranking on announced plans", "infrastructure 4 years ahead", "#F0A875"),
    ("trajectory", "Trajectory-informed greedy", "true trajectories", "#0072B2"),
    ("rh_full", "Rolling horizon, full", "true trajectories + coordination", "#D55E00"),
]
fig = plt.figure(figsize=(6.9, 3.1))
axa = fig.add_axes([0.235, 0.17, 0.40, 0.70])
axb = fig.add_axes([0.665, 0.17, 0.32, 0.70])
ys = np.arange(len(LEV) + 1)
for y, (k, lab, info, c) in enumerate(LEV):
    v = rh[k].values
    axa.plot([v.min(), v.max()], [y, y], color=c, lw=2.2, solid_capstyle="round", zorder=2)
    axa.plot(v.mean(), y, "o", ms=6, color=c, mec="black", mew=0.5, zorder=3)
    axa.text(v.max() + 0.006, y, f"{v.mean():.3f}", ha="left", va="center", fontsize=6.5)
    axa.text(-0.02, y + 0.12, lab, transform=mpl.transforms.blended_transform_factory(axa.transAxes, axa.transData),
             ha="right", va="center", fontsize=7)
    axa.text(-0.02, y - 0.2, info, transform=mpl.transforms.blended_transform_factory(axa.transAxes, axa.transData),
             ha="right", va="center", fontsize=6, color="#555555")
yr = len(LEV)
axa.plot(1.0, yr, "D", ms=5, color="black", zorder=3)
axa.text(0.992, yr, "1.000", ha="right", va="center", fontsize=6.5)
axa.text(-0.02, yr + 0.12, "Reference schedule", transform=mpl.transforms.blended_transform_factory(axa.transAxes, axa.transData),
         ha="right", va="center", fontsize=7)
axa.text(-0.02, yr - 0.2, "exact maximum of the table", transform=mpl.transforms.blended_transform_factory(axa.transAxes, axa.transData),
         ha="right", va="center", fontsize=6, color="#555555")
m = {k: rh[k].mean() for k, *_ in LEV}
steps = [(0, 1, m["rh_announced"] - m["persistence"], "announced\ninformation"),
         (1, 2, m["trajectory"] - m["rh_announced"], "knowing the full\ntrajectories"),
         (2, 3, m["rh_full"] - m["trajectory"], "coordination\nacross years")]
for y0, y1, d, txt in steps:
    xm = 0.712
    axa.annotate("", xy=(xm, y1 - 0.12), xytext=(xm, y0 + 0.12),
                 arrowprops=dict(arrowstyle="-|>", color="#444444", lw=0.8))
    axa.text(xm + 0.006, (y0 + y1) / 2, f"+{d:.3f} {txt.replace(chr(10), ' ')}", ha="left", va="center", fontsize=6,
             color="#333333")
axa.set_xlim(0.70, 1.02)
axa.set_ylim(-0.6, len(LEV) + 0.7)
axa.set_yticks([])
for s in ("left", "right", "top"):
    axa.spines[s].set_visible(False)
axa.set_xlabel("Ratio to the reference schedule (higher = better)")
axa.text(-0.62, 1.04, "a", transform=axa.transAxes, fontsize=9, fontweight="bold", va="bottom")
axa.text(-0.56, 1.04, "Value of timing rises with what the rule knows", transform=axa.transAxes, fontsize=7.5, va="bottom")

TIM, SEL = "#34495e", "#c9d3dd"
for y, (k, lab, info, c) in enumerate(LEV):
    tim, sel = T3.loc[k, "l_tim"], T3.loc[k, "l_sel"]
    axb.barh(y, tim, color=TIM, height=0.5)
    axb.barh(y, sel, left=tim, color=SEL, height=0.5)
    axb.plot(-40, y, "s", ms=5, color=c, clip_on=False)
axb.barh(yr, 0, height=0.5)
axb.text(15, yr, "no loss", va="center", fontsize=6, color="#555555")
axb.text(T3.loc["persistence", "l_tim"] / 2, 0, "timing", color="white", ha="center", va="center", fontsize=6)
axb.text(T3.loc["persistence", "l_tim"] + T3.loc["persistence", "l_sel"] / 2, 0, "selection", color="#2c3e50",
         ha="center", va="center", fontsize=6)
axb.set_ylim(axa.get_ylim())
axb.set_yticks([])
axb.set_xlim(0, 1700)
for s in ("left", "right", "top"):
    axb.spines[s].set_visible(False)
axb.set_xlabel("Value lost to the reference (objective units)")
axb.text(-0.04, 1.04, "b", transform=axb.transAxes, fontsize=9, fontweight="bold", va="bottom")
axb.text(0.03, 1.04, "Most of the loss is mistiming", transform=axb.transAxes, fontsize=7.5, va="bottom")
fig.savefig(OUT + "fig5_ladder.pdf")
fig.savefig(OUT + "fig5_ladder.png", dpi=300)
