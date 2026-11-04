"""Figure 2 — time-varying renewal opportunity (compact layout, drawn at the 34 pc text width).

(a) scenario-based trajectories of the four temporal fields, one small panel each; the arrow line
    under each title names the pipeline stage the field acts on (cf. Figure 1b);
(b) the combined effect: expected value of initiation by initiation year for two case-study units.
Inputs: results_figures/gm_field_curves.json, results_figures/fig2_example_units.json.
Typography: figure is 5.65 in wide (= \\linewidth), so sizes below are final printed sizes.
"""
import json
import os as _os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")) + "/"
FD, OUT = _ROOT + "results_figures/", _ROOT + "paper_ijgis/figures/"
FS, FS_HEAD, FS_LET = 7, 8, 9            # one body size, one heading size, one panel-letter size
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "mathtext.fontset": "dejavusans",
})
DK, LT, GR, INK, SUB = "#1B365D", "#2F6DB5", "#8A9097", "#1A1A1A", "#4A4A4A"
LW = 1.9
STY = {DK: dict(color=DK, lw=LW, ls="-"), LT: dict(color=LT, lw=LW, ls=(0, (3.6, 1.5))),
       GR: dict(color=GR, lw=LW, ls="-")}
F = json.load(open(FD + "gm_field_curves.json"))
U = json.load(open(FD + "fig2_example_units.json"))
T = 25; yrs = np.arange(1, T + 1)

fig = plt.figure(figsize=(5.65, 3.55))
# ---------------- (a) four field panels
panels = [("plan", "Planning support $P$", "listing, conformity", [DK, LT, GR]),
          ("infra", "Infrastructure $I$", "delivered value", [DK, LT]),
          ("obsolescence", "Obsolescence $B$", "approval", [DK, LT]),
          ("readiness", "Readiness $C$", "approval", [DK])]
LABELS = {"plan": [("high", DK, 2.0, 1.13, "left"), ("window", LT, 16.6, 0.62, "left"),
                   ("persistently low", GR, 2.0, -0.04, "left")],
          "infra": [("early", DK, 8.4, 1.13, "center"), ("late", LT, 17.0, 1.13, "center")],
          "obsolescence": [("older", DK, 2.0, 0.84, "left"), ("newer", LT, 2.0, 0.40, "left")]}
x0, w, gap, ya, ha = 0.105, 0.2, 0.024, 0.585, 0.255
for j, (key, title, stage, cols) in enumerate(panels):
    ax = fig.add_axes([x0 + j * (w + gap), ya, w, ha])
    for c, y in zip(cols, list(F[key].values())):
        ax.plot(yrs, np.array(y[:T]), **STY[c], solid_capstyle="round")
    ax.set_xlim(1, T); ax.set_ylim(-0.14, 1.24)
    ax.set_xticks([1, 10, 20]); ax.set_yticks([0, 1])
    if j: ax.set_yticklabels([])
    ax.text(0.0, 1.20, title, transform=ax.transAxes, fontsize=FS, fontweight="bold", color=INK, va="bottom")
    ax.text(0.0, 1.05, "\u2192 " + stage, transform=ax.transAxes, fontsize=FS, color=SUB, va="bottom")
    for (txt_, c, x_, y_, ha_) in LABELS.get(key, []):
        ax.text(x_, y_, txt_, color=c, fontsize=FS, ha=ha_, va="center")
fig.text(0.035, ya + ha / 2, "Field level", rotation=90, va="center", ha="center", fontsize=FS)
fig.text(0.0, 0.975, "a", fontsize=FS_LET, fontweight="bold", va="center")
fig.text(0.035, 0.975, "Scenario-based field trajectories (illustrative)", fontsize=FS_HEAD, fontweight="bold", va="center")

# ---------------- (b) combined effect for two case-study units
axv = fig.add_axes([0.105, 0.115, 0.872, 0.285])
v1 = np.array(U["331"]["ev_rel"]); v2 = np.array(U["249"]["ev_rel"]); lst = np.array(U["249"]["admit"]) > 0
axv.plot(yrs, v1, **STY[DK])
axv.plot(yrs[lst], v2[lst], **STY[LT])
axv.plot(yrs, np.where(lst, np.nan, 0.0), color=GR, lw=1.2, ls=(0, (1, 1.6)))
for yy in (3, 8, 13):
    axv.plot(yy, v1[yy - 1], "o", ms=4.2, color=DK, mec="white", mew=0.7, zorder=4)
    axv.text(yy, v1[yy - 1] + 0.08, f"{v1[yy - 1]:.2f}", fontsize=FS, ha="center", va="bottom", color=DK)
axv.text(18.6, 0.40, "Unit A: listable every year", fontsize=FS, color=DK, ha="left", va="bottom")
axv.text(14.8, 0.86, "Unit B: listable only in years 10\u201317", fontsize=FS, color=LT, ha="left", va="bottom")
axv.text(18.2, 0.04, "not listable", fontsize=FS, color=GR, ha="left", va="bottom")
axv.set_xlim(1, T); axv.set_ylim(-0.08, 1.22)
axv.set_xticks([1, 5, 10, 15, 20, 25]); axv.set_yticks([0, 0.5, 1])
axv.set_xlabel("Initiation year", labelpad=2)
axv.set_ylabel(""); fig.text(0.035, 0.115 + 0.285 / 2, "Share of\nbest-year value", rotation=90, va="center", ha="center", fontsize=FS, linespacing=1.1)
fig.text(0.0, 0.465, "b", fontsize=FS_LET, fontweight="bold", va="center")
fig.text(0.035, 0.465, "Expected value by initiation year", fontsize=FS_HEAD,
         fontweight="bold", va="center")

fig.savefig(OUT + "fig2_temporal_opportunity.pdf")
fig.savefig(OUT + "fig2_temporal_opportunity.png", dpi=300)
