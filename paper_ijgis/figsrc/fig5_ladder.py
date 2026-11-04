# Figure 5 — where the value of timing comes from.
# (a) ratio to the reference schedule per information level, six listing draws (M = 20);
# (b) selection- and timing-related loss of the same rules (Equation decomp);
# (b) initiation year relative to each unit's best year, listing draw 7 (as in Figure 4).
# Data: results_p1/rolling/rolling_results.csv, results_p1/tables/main_M20.csv,
#       results_p1/rolling/schedules_cluster_M20.json, results_figures/{schedules.json, ev_table_main.npy}.
# Drawn at the 34 pc text width (5.65 in): font sizes are final printed sizes.
import json
import os as _os
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")) + "/"
R = _ROOT
OUT = R + "paper_ijgis/figures/"
FS, FSS = 7, 6.5
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "legend.fontsize": FSS, "xtick.labelsize": FSS, "ytick.labelsize": FSS,
    "axes.linewidth": 0.7, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})
RH = pd.read_csv(R + "results_p1/rolling/rolling_results.csv")
rh = RH[(RH.infra_mode == "cluster") & (RH.conformity) & (RH.variant == "base") & (RH.M == 20)] \
    .pivot_table(index="listing_seed", columns="rule", values="ratio")
T3 = pd.read_csv(R + "results_p1/tables/main_M20.csv", index_col=0)
assert np.allclose(rh.rh_current, rh.persistence)
# reinforcement learning (this study): trained on twenty other scenario draws, five seeds x six listing draws
import glob as _glob
_RL = pd.concat([pd.read_csv(f) for f in sorted(_glob.glob(R + "results_p3_metro/main_s*.csv"))])
_RL = _RL[_RL.rule == "residual_one_step"]
rh["rl"] = _RL.groupby("listing_seed").ratio.mean().reindex(rh.index)
T3.loc["rl", "l_tim"] = _RL.loss_timing.mean(); T3.loc["rl", "l_sel"] = _RL.loss_selection.mean()
assert rh["rl"].notna().all()

LEV = [  # key, label, what the rule knows, colour
    ("persistence", "Persistence greedy", "current conditions only", "#E69F00"),
    ("rh_announced", "Ranking on announced plans", "infrastructure 4 years ahead", "#F0A875"),
    ("rl", "Learned value correction", "announced plans + learned correction", "#2E7EC1"),
    ("trajectory", "Trajectory-informed greedy", "true trajectories", "#0072B2"),
    ("rh_full", "Rolling horizon, full", "true trajectories + coordination", "#D55E00"),
]
fig = plt.figure(figsize=(5.65, 5.2))
axa = fig.add_axes([0.335, 0.575, 0.63, 0.37])
axc = fig.add_axes([0.335, 0.075, 0.63, 0.265])
blend = lambda ax: mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
for y, (k, lab, info, c) in enumerate(LEV):
    v = rh[k].values
    axa.plot([v.min(), v.max()], [y, y], color=c, lw=2.4, solid_capstyle="round", zorder=2)
    axa.plot(v.mean(), y, "o", ms=6, color=c, mec="black", mew=0.5, zorder=3)
    axa.text(v.max() + 0.006, y, f"{v.mean():.3f}", ha="left", va="center", fontsize=FSS)
    axa.text(-0.03, y + 0.14, lab, transform=blend(axa), ha="right", va="center", fontsize=FS)
    axa.text(-0.03, y - 0.24, info, transform=blend(axa), ha="right", va="center", fontsize=FSS, color="#555555")
yr = len(LEV)
axa.plot(1.0, yr, "D", ms=5, color="black", zorder=3)
axa.text(0.990, yr, "1.000", ha="right", va="center", fontsize=FSS)
axa.text(-0.03, yr + 0.14, "Reference schedule", transform=blend(axa), ha="right", va="center", fontsize=FS)
axa.text(-0.03, yr - 0.24, "exact maximum of the table", transform=blend(axa), ha="right", va="center",
         fontsize=FSS, color="#555555")
m = {k: rh[k].mean() for k, *_ in LEV}
steps = [(0, 1, m["rh_announced"] - m["persistence"], "announced"),
         (1, 3, m["trajectory"] - m["rh_announced"], "full trajectories\n(ranking to trajectory-informed)"),
         (3, 4, m["rh_full"] - m["trajectory"], "coordination")]
xm = 0.708
for y0, y1, d, txt in steps:
    axa.annotate("", xy=(xm, y1 - 0.14), xytext=(xm, y0 + 0.14),
                 arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.2))
    axa.text(xm + 0.005, (y0 + y1) / 2 + (0.5 if y1 - y0 > 1 else 0), f"+{d:.3f} {txt}", ha="left", va="center", fontsize=FS + 0.5, color="#222222", linespacing=1.0,
             fontweight="bold")
_yrl = [k for k, *_ in LEV].index("rl")
axa.text(0.775, _yrl - 0.42, f"+{m['rl'] - m['rh_announced']:.3f} over announced plans",
         ha="left", va="center", fontsize=FSS, color="#2E7EC1", fontweight="bold")
axa.set_xlim(0.70, 1.02); axa.set_ylim(-0.6, len(LEV) + 0.6)
axa.set_xticks([0.7, 0.8, 0.9, 1.0]); axa.set_yticks([])
axa.spines["left"].set_visible(False)
axa.set_xlabel("Ratio to the reference schedule")
fig.text(0.005, 0.975, "a", fontsize=9, fontweight="bold", va="center")
fig.text(0.04, 0.975, "Value of timing rises with what the rule knows", fontsize=FS, fontweight="bold", va="center")

# (former panel b, loss decomposition, removed: the values are in Table 3)

# (b) initiation year relative to each unit's best year, listing draw 7
EVT = np.load(R + "results_figures/ev_table_main.npy"); bidx = EVT.argmax(1)
RS = json.load(open(R + "results_p1/rolling/schedules_cluster_M20.json"))["schedules"]
SJ = json.load(open(R + "results_figures/schedules.json"))
SCH = {k: {int(u): int(t) for u, t in RS[k].items()} for k, *_ in LEV if k != "rl"}
SCH["rl"] = {int(u): int(t) for u, t in json.load(open(R + "results_p3_metro/rl_schedule_main_ls7_s0.json"))["init"].items()}
SCH["optimum"] = {int(u): int(t) for u, t in SJ["optimum"]["init"].items()}
ROWS = [(k, lab) for k, lab, *_ in LEV] + [("optimum", "Reference schedule")]
CAT = [("early", lambda e: e < 0, "#A9B8C9"), ("best year", lambda e: e == 0, "#1B365D"),
       ("1\u20132 yr late", lambda e: (e >= 1) & (e <= 2), "#F3C79A"),
       ("3\u20135 yr late", lambda e: (e >= 3) & (e <= 5), "#E58C4A"),
       ("> 5 yr late", lambda e: e > 5, "#C8735C")]
shares = {}
for i, (k, lab) in enumerate(ROWS):
    u = np.array(list(SCH[k].keys())); t = np.array(list(SCH[k].values()))
    e = t - bidx[u]; assert len(e) == 75
    y = len(ROWS) - 1 - i; left = 0.0
    shares[k] = [float(f(e).mean()) for _, f, _ in CAT]
    for (name, f, col), sh in zip(CAT, shares[k]):
        axc.barh(y, sh * 100, left=left, color=col, height=0.62, ec="white", lw=0.6)
        if sh >= 0.035:
            axc.text(left + sh * 50, y, f"{sh * 100:.0f}", ha="center", va="center", fontsize=(FSS if sh >= 0.08 else FSS - 1.2),
                     color=("white" if col in ("#1B365D", "#C8735C", "#E58C4A") else "#1A1A1A"))
        left += sh * 100
    axc.text(-0.03, y, lab, transform=blend(axc), ha="right", va="center", fontsize=FS)
axc.set_xlim(0, 100); axc.set_ylim(-0.6, len(ROWS) - 0.4)
axc.set_yticks([]); axc.spines["left"].set_visible(False)
axc.set_xticks([0, 25, 50, 75, 100]); axc.set_xlabel("Share of the 75 selected units (%)")
for j, (name, _, col) in enumerate(CAT):
    xx = 0.0 + j * 0.205
    axc.add_patch(mpl.patches.Rectangle((xx, 1.06), 0.022, 0.07, transform=axc.transAxes, fc=col, ec="none",
                                        clip_on=False))
    axc.text(xx + 0.03, 1.095, name, transform=axc.transAxes, fontsize=FSS, va="center", ha="left")
fig.text(0.005, 0.425, "b", fontsize=9, fontweight="bold", va="center")
fig.text(0.04, 0.425, "Initiation year relative to each unit's best year (listing draw 7)", fontsize=FS,
         fontweight="bold", va="center")
json.dump({k: dict(zip([c[0] for c in CAT], v)) for k, v in shares.items()},
          open(R + "results_figures/fig5_start_offsets_ls7.json", "w"), indent=1)
fig.savefig(OUT + "fig5_ladder.pdf")
fig.savefig(OUT + "fig5_ladder.png", dpi=300)
