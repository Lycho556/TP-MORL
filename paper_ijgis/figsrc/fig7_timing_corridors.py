# Figure 7 — where starting now is the best year (observed metro field vs scenario placebo).
# Data: results_p2/timing_stake_units.csv (scripts/p3_timing_space_metro.py), metro/stations.csv.
import os
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")) + "/"
G = R + "data/processed/gm_dataset_v1/"; OUT = R + "paper_ijgis/figures/"
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
NOW, LATER, BG, CAND = "#D9822B", "#3C5A7D", "#F2F4F7", "#D9DDE3"
U = pd.read_csv(R + "results_p2/timing_stake_units.csv")
uidr = np.load(G + "zones_v0/unit_id.npy"); L0 = np.load(G + "grid_100m/L0_class.npy")
XY = np.load(G + "grid_100m/cell_centre_xy.npy"); X00, Y00 = XY[0, 0, 0], XY[0, 0, 1]
ST = pd.read_csv(G + "metro/stations.csv"); ST = ST[ST.in_extent]
inside = L0 != 255
M = U[U.field == "metro"].set_index("uid")
img = np.ones(uidr.shape + (4,)); img[inside] = mpl.colors.to_rgba(BG)
now_ids = set(M.index[M.best_year == 0]); later_ids = set(M.index[M.best_year > 0])
cand = np.isin(uidr, list(M.index)); img[cand] = mpl.colors.to_rgba(CAND)
img[np.isin(uidr, list(now_ids))] = mpl.colors.to_rgba(NOW)
img[np.isin(uidr, list(later_ids))] = mpl.colors.to_rgba(LATER, 0.55)

fig = plt.figure(figsize=(5.65, 3.05))
axa = fig.add_axes([0.0, 0.20, 0.50, 0.72]); axb = fig.add_axes([0.62, 0.27, 0.37, 0.60])
axa.imshow(img, interpolation="nearest")
axa.contour(inside.astype(float), levels=[0.5], colors="#666666", linewidths=0.4)
for _, s in ST.iterrows():
    c, r = (s.x - X00) / 100.0, (Y00 - s.y) / 100.0
    axa.add_patch(Circle((c, r), 15, fill=False, ls=(0, (2, 2)), lw=0.45, ec="#555555"))
    axa.plot(c, r, "o", ms=3.2, mfc="white", mec="black", mew=0.7)
axa.set_xticks([]); axa.set_yticks([]); [sp.set_visible(False) for sp in axa.spines.values()]
axa.legend(handles=[Patch(fc=NOW, label="best year: initiate now"), Patch(fc=mpl.colors.to_rgba(LATER, .55), label="best year: later"),
                    Line2D([], [], marker="o", ls="", mfc="white", mec="black", ms=4, label="metro station"),
                    Line2D([], [], ls=(0, (2, 2)), color="#555555", lw=0.6, label="1.5 km from station")],
           loc="upper left", bbox_to_anchor=(-0.02, -0.01), ncol=2, handlelength=1.3, columnspacing=0.8, borderaxespad=0.1)
_H, _W = uidr.shape
axa.plot([6, 26], [_H - 6, _H - 6], color="black", lw=1.6, solid_capstyle="butt")
axa.text(16, _H - 9, "2 km", ha="center", va="bottom", fontsize=6.5)
axa.annotate("", xy=(_W - 10, _H - 40), xytext=(_W - 10, _H - 22),
             arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black", mutation_scale=8))
axa.text(_W - 10, _H - 42, "N", ha="center", va="bottom", fontsize=7, fontweight="bold")
axa.text(0.0, 1.04, "a", transform=axa.transAxes, fontsize=9, fontweight="bold")
axa.text(0.06, 1.04, "Best initiation year, observed metro field", transform=axa.transAxes, fontsize=8)

bands = [(0, 800), (800, 1500), (1500, 3000), (3000, np.inf)]; labs = ["0–0.8", "0.8–1.5", "1.5–3", ">3"]
rng = np.random.default_rng(0)
for k, (f, col, off, fill) in enumerate((("metro", LATER, -0.18, True), ("cluster", "#8A8F98", 0.18, False))):
    x = U[U.field == f]; z = (x.best_year == 0).values; d = x.d_nearest.values
    for j, (lo, hi) in enumerate(bands):
        m = (d > lo) & (d <= hi) if lo > 0 else (d <= hi)
        bs = [rng.choice(z[m], m.sum()).mean() for _ in range(2000)]
        axb.bar(j + off, z[m].mean(), width=0.34, color=col if fill else "white", edgecolor=col, lw=0.9)
        axb.errorbar(j + off, z[m].mean(), yerr=[[z[m].mean() - np.percentile(bs, 2.5)], [np.percentile(bs, 97.5) - z[m].mean()]],
                     color="black", lw=0.7, capsize=1.5)
axb.set_xticks(range(4)); axb.set_xticklabels(labs); axb.set_xlabel("Distance to nearest station (km)")
axb.set_ylabel("Share of units best initiated now"); axb.set_ylim(0, 0.75)
axb.legend(handles=[Patch(fc=LATER, ec=LATER, label="observed metro field"), Patch(fc="white", ec="#8A8F98", label="scenario field (placebo)")],
           loc="upper right", bbox_to_anchor=(1.02, 1.04))
axb.text(-0.30, 1.13, "b", transform=axb.transAxes, fontsize=9, fontweight="bold")
axb.text(-0.20, 1.13, "Initiating now, by distance", transform=axb.transAxes, fontsize=8)
fig.savefig(OUT + "fig7_timing_corridors.pdf"); fig.savefig(OUT + "fig7_timing_corridors.png", dpi=300)
