"""Figure 7: observed metro infrastructure and the value ladder under both fields.

(a) Guangming metro stations (OpenStreetMap; opening dates from official notices)
    and candidate units by the opening year of the nearest station within 1.5 km.
(b) Ratio to the reference schedule, M = 20, for the scenario field and the
    observed-metro field; dots are means over six listing draws (learned rule:
    x five training seeds), bars span min-max.
Inputs: data/processed/gm_dataset_v1/metro/, results_p1/tables/main_M20.csv, metro.csv
"""
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

R = "/Users/user/Desktop/TP-MORL/"
G = R + "data/processed/gm_dataset_v1/"
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 6, "axes.titlesize": 7,
                     "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"persistence": "#E69F00", "rh_announced": "#F0A875", "trajectory": "#0072B2",
       "value_based": "#56B4E9", "rh_full": "#D55E00"}
NAME = {"persistence": "Persistence greedy", "rh_announced": "Rolling horizon, announced",
        "trajectory": "Trajectory-informed greedy", "value_based": "Value-based policy",
        "rh_full": "Rolling horizon, full trajectories"}
LINE_COL = {"L6": "#6a3d9a", "L6B": "#9e7cc1", "L13": "#b15928"}

uid = np.load(G + "zones_v0/unit_id.npy")
L0 = np.load(G + "grid_100m/L0_class.npy")
xy = np.load(G + "grid_100m/cell_centre_xy.npy")
x0, y0 = xy[0, 0, 0], xy[0, 0, 1]
st = pd.read_csv(G + "metro/stations.csv")
ud = pd.read_csv(G + "metro/unit_station_distance.csv").set_index("uid")
st["col"] = (st.x - x0) / 100.0
st["row"] = (y0 - st.y) / 100.0
inside = L0 != 255
H, W = L0.shape

# nearest station within 1.5 km -> its opening year
D = ud[[f"d_{s}" for s in st.station]].to_numpy()
D = np.where(D <= 1500, D, np.inf)
k = D.argmin(1)
open_year = np.where(np.isfinite(D.min(1)), np.floor(st.open.to_numpy()[k]), np.nan)
oy = dict(zip(ud.index, open_year))

years = [2020, 2022, 2025, 2026]
ycol = dict(zip(years, ["#08306b", "#2171b5", "#6baed6", "#c6dbef"]))
img = np.ones((H, W, 4))
img[inside] = (0.96, 0.96, 0.96, 1)
for u, y in oy.items():
    m = uid == u
    img[m] = (0.80, 0.80, 0.80, 1) if np.isnan(y) else (*mpl.colors.to_rgb(ycol[int(y)]), 1)

# line geometry from OSM route relations (member order)
rt = json.load(open(G + "metro/osm_routes_raw.json"))
nd = {e["id"]: e for e in rt["elements"] if e["type"] == "node"}
pos = st.set_index("station")[["col", "row"]]
seqs = {}
for rel in [e for e in rt["elements"] if e["type"] == "relation"]:
    ref = rel["tags"]["ref"]
    seq = []
    for m in rel["members"]:
        if m["type"] == "node" and m["ref"] in nd:
            nm = nd[m["ref"]]["tags"].get("name")
            if nm in pos.index and (not seq or seq[-1] != nm):
                seq.append(nm)
    seqs[{"6": "L6", "6B": "L6B", "13": "L13"}[ref]] = seq

fig = plt.figure(figsize=(6.9, 3.5))
axa = fig.add_axes([0.0, 0.17, 0.40, 0.74])
axa.imshow(img, interpolation="nearest")
axa.contour(inside.astype(float), levels=[0.5], colors="#555555", linewidths=0.4)
for ln, seq in seqs.items():
    p = pos.loc[seq]
    axa.plot(p.col, p.row, color=LINE_COL[ln], lw=1.0, zorder=3)
axa.scatter(st.col, st.row, s=7, facecolor="white", edgecolor="#222222", lw=0.5, zorder=4)
axa.set_xlim(-0.5, W - 0.5); axa.set_ylim(H - 0.5, -0.5); axa.axis("off")
fig.text(0.005, 0.955, "a", fontsize=9, fontweight="bold", va="bottom")
fig.text(0.03, 0.955, "Units within 1.5 km of a station, by opening year", fontsize=7, va="bottom")
hs = [mpl.patches.Patch(color=ycol[y], label=str(y)) for y in years]
hs += [mpl.patches.Patch(color=(0.80, 0.80, 0.80), label="beyond 1.5 km")]
hs += [mpl.lines.Line2D([], [], color=LINE_COL[l], lw=1.2, label=n) for l, n in
       [("L6", "Line 6"), ("L6B", "Line 6 branch"), ("L13", "Line 13 north")]]
fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.01, 0.0), frameon=False, fontsize=6,
           handlelength=1.2, ncol=4, columnspacing=0.8, borderaxespad=0.2)

T = {"Scenario field": pd.read_csv(R + "results_p1/tables/main_M20.csv", index_col=0),
     "Observed metro field": pd.read_csv(R + "results_p1/tables/metro.csv", index_col=0)}
T["Observed metro field"] = T["Observed metro field"][T["Observed metro field"].M == 20]
order = ["persistence", "rh_announced", "trajectory", "value_based", "rh_full"]
axb = fig.add_axes([0.68, 0.15, 0.30, 0.76])
mk = {"Scenario field": ("o", "white"), "Observed metro field": ("D", None)}
for i, r in enumerate(order):
    y = len(order) - 1 - i
    for j, (fld, t) in enumerate(T.items()):
        dy = 0.15 if j == 0 else -0.15
        m, lo, hi = t.loc[r, "ratio"], t.loc[r, "lo"], t.loc[r, "hi"]
        axb.plot([lo, hi], [y + dy, y + dy], color=COL[r], lw=1.0)
        marker, face = mk[fld]
        axb.plot(m, y + dy, marker=marker, ms=4, mec=COL[r], mfc=face or COL[r], mew=1.0, ls="none")
    axb.text(0.733, y, NAME[r], ha="right", va="center", fontsize=7,
             transform=mpl.transforms.blended_transform_factory(axb.transData, axb.transData))
axb.set_xlim(0.74, 1.0)
axb.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
axb.set_ylim(-0.6, len(order) - 0.4)
axb.set_yticks([])
axb.spines["left"].set_visible(False)
axb.set_xlabel("Ratio to the reference schedule")
fig.text(0.44, 0.955, "b", fontsize=9, fontweight="bold", va="bottom")
fig.text(0.465, 0.955, "Information and coordination raise value under both fields", fontsize=7, va="bottom")
lh = [mpl.lines.Line2D([], [], marker="o", mfc="white", mec="#444444", ls="none", ms=4, label="scenario field"),
      mpl.lines.Line2D([], [], marker="D", mfc="#444444", mec="#444444", ls="none", ms=4, label="observed metro field")]
axb.legend(handles=lh, loc="lower left", frameon=False, fontsize=6, borderaxespad=0.1)
fig.savefig("fig7_metro.png", dpi=300)
fig.savefig("fig7_metro.pdf")
print("units within 1.5 km:", int(np.isfinite(D.min(1)).sum()))
