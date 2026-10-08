# skill:figure-style kernel.py (auto-injected on skill load)
META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")

    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(dx, dy, s, transform=ax.transAxes,
            fontweight="bold", fontsize=fontsize, va="bottom", ha="left")


import json, os, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

R = "/Users/user/Desktop/TP-MORL/"
FD = R + "results_figures/"
COL = {"random": "#8a8a8a", "static": "#E69F00", "supervised": "#CC79A7",
       "temporal": "#0072B2", "value": "#56B4E9", "rolling": "#D55E00", "optimum": "black"}
TIM, SEL = "#34495e", "#c9d3dd"
CH_COL = {1: "#8c6bb1", 2: "#e6550d", 3: "#3182bd", 5: "#31a354"}
CH_NAME = {1: "P1 industrial redevelopment", 2: "P2 old residential",
           3: "P3 commercial-service", 5: "P5 agricultural conversion"}


def overlaps(fig):
    r = fig.canvas.get_renderer()
    t = [(x, x.get_window_extent(r)) for x in fig.findobj(mpl.text.Text)
         if x.get_text().strip() and x.get_visible()]
    return [(a.get_text()[:24], b.get_text()[:24])
            for i, (a, ba) in enumerate(t) for b, bb in t[i + 1:] if ba.overlaps(bb)]


apply_figure_style(sizes=(8, 7, 6))

G = R + "data/processed/gm_dataset_v1/"
uid = np.load(G + "zones_v0/unit_id.npy")
ch = np.load(G + "zones_v0/channel_v0.npy")
L0 = np.load(G + "grid_100m/L0_class.npy")
vf = np.load(G + "grid_100m/valid_frac.npy")
am = np.load(G + "grid_100m/action_mask.npy")
cu = pd.read_csv(G + "zones_v0/candidate_units.csv")
leg = pd.read_csv(G + "tables/class_legend.csv")
chl = pd.read_csv(G + "zones_v0/channel_legend.csv")

UA = pd.read_csv(FD + "unit_attrs.csv")

SCH = json.load(open(FD + "schedules.json"))
# Direction C: information and coordination ladder (listing seed 7, M = 20)
_RS = json.load(open(R + "results_p1/rolling/schedules_cluster_M20.json"))["schedules"]["rh_full"]
_RR = pd.read_csv(R + "results_p1/rolling/rolling_results.csv")
_rr = _RR[(_RR.infra_mode == "cluster") & (_RR.conformity) & (_RR.variant == "base")
          & (_RR.M == 20) & (_RR.listing_seed == 7) & (_RR.rule == "rh_full")].ratio.iloc[0]
SCH["rolling_full"] = dict(ratio=float(_rr), init={str(k): int(v) for k, v in _RS.items()})
rules = [("static_greedy", "Persistence greedy", "static"),
         ("temporal_greedy", "Trajectory-informed greedy", "temporal"),
         ("rolling_full", "Rolling horizon, full", "rolling"),
         ("optimum", "Reference schedule", "optimum")]

LUG = {0: ("Restricted", "#e3e3e3"), 1: ("Residential", "#fbf2c4"),
       2: ("Residential", "#fbf2c4"), 4: ("Commercial", "#f6d5cf"),
       7: ("Industrial", "#e3d9ee"), 8: ("Industrial", "#e3d9ee"),
       9: ("Agricultural", "#eef3d2"), 10: ("Ecological and green", "#d8ecd6"),
       11: ("Ecological and green", "#d8ecd6"), 12: ("Public utilities", "#dde6ee")}

H, W = L0.shape
rgb = np.ones((H, W, 3))
for k, (n, c) in LUG.items():
    rgb[L0 == k] = mpl.colors.to_rgb(c)

chan_of = dict(zip(UA.uid, UA.channel))
urgb = np.zeros((H, W, 4))
for u, c in chan_of.items():
    m = uid == u
    urgb[m] = (*mpl.colors.to_rgb(CH_COL[c]), 0.95)

inside = (L0 != 255)
cand = uid > 0
idx2uid = dict(zip(UA.unit, UA.uid))

union = sorted(set().union(*[set(map(int, SCH[k]["init"])) for k, _, _ in rules]))
U2 = UA.set_index("unit").loc[union].reset_index().sort_values(
    ["channel", "floor_area"], ascending=[True, False]).reset_index(drop=True)
rowpos = {u: i for i, u in enumerate(U2.unit)}


def draw_units(ax, lw=0.3):
    segs = []
    for i in range(H):
        for j in range(W):
            u = uid[i, j]
            if u <= 0:
                continue
            if i == 0 or uid[i - 1, j] != u:
                segs.append([(j - 0.5, i - 0.5), (j + 0.5, i - 0.5)])
            if i == H - 1 or uid[i + 1, j] != u:
                segs.append([(j - 0.5, i + 0.5), (j + 0.5, i + 0.5)])
            if j == 0 or uid[i, j - 1] != u:
                segs.append([(j - 0.5, i - 0.5), (j - 0.5, i + 0.5)])
            if j == W - 1 or uid[i, j + 1] != u:
                segs.append([(j + 0.5, i - 0.5), (j + 0.5, i + 0.5)])
    ax.add_collection(mpl.collections.LineCollection(segs, colors="#222222", linewidths=lw))



# ---- zoom window (panel c): around Zhongda and Zhenmei stations, Line 6 branch
XY = np.load(G + "grid_100m/cell_centre_xy.npy")
X00, Y00 = XY[0, 0, 0], XY[0, 0, 1]
ZX, ZY, ZW = 492900.0, 2521200.0, 3000.0
zc0, zc1 = (ZX - X00) / 100.0, (ZX + ZW - X00) / 100.0
zr0, zr1 = (Y00 - (ZY + ZW)) / 100.0, (Y00 - ZY) / 100.0

cmap = plt.get_cmap("viridis")
norm = mpl.colors.Normalize(1, 25)

plt.close("all")
EVT = np.load(FD + "ev_table_main.npy")
BEST = EVT.max(1)
SHOW_B = False   # unit-by-year matrices (former panel b) dropped for readability
fig = plt.figure(figsize=(6.9, 6.6))
mpl.rcParams["savefig.bbox"] = "standard"
gtop = fig.add_gridspec(1, 4, left=0.003, right=0.997, top=0.935, bottom=0.675, wspace=0.0)
gbot = fig.add_gridspec(1, 4, left=0.075, right=0.99, top=0.555, bottom=0.335, wspace=0.07)
axc = fig.add_axes([0.215, 0.43, 0.705, 0.13])

eff = {}
for j, (k, name, ck) in enumerate(rules):
    init = {int(u): int(t) for u, t in SCH[k]["init"].items()}
    eff[k] = np.array([EVT[u, t] / BEST[u] for u, t in init.items()])
    img = np.ones((H, W, 4))
    img[inside] = (0.97, 0.97, 0.97, 1)
    img[cand] = (0.83, 0.83, 0.83, 1)
    for u, t in init.items():
        img[uid == idx2uid[u]] = cmap(norm(t + 1))
    ax = fig.add_subplot(gtop[0, j])
    ax.imshow(img, interpolation="nearest")
    ax.contour(inside.astype(float), levels=[0.5], colors="#555555", linewidths=0.4)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.axis("off")
    ax.add_patch(mpl.patches.Rectangle((zc0, zr0), zc1 - zc0, zr1 - zr0, fill=False, ec="#d62728", lw=0.8))
    ax.text(0.02, 1.0, f"{name}\n{SCH[k]['ratio']:.3f} of reference", transform=ax.transAxes,
            fontsize=7, ha="left", va="bottom")
    if j == 0:
        ax.text(0.0, 1.13, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    if SHOW_B:
        axm = fig.add_subplot(gbot[0, j])
        axm.scatter([t + 1 for t in init.values()], [rowpos[u] for u in init],
                    s=5, color=COL[ck], lw=0)
        for c in [1, 2, 3, 5]:
            rr = U2.index[U2.channel == c]
            if len(rr):
                axm.axhspan(rr.min() - 0.5, rr.max() + 0.5, color=CH_COL[c], alpha=0.08, lw=0)
        axm.set_xlim(0, 26)
        axm.set_ylim(len(U2) - 0.5, -0.5)
        axm.set_xticks([1, 5, 10, 15, 20, 25])
        axm.set_yticks([])
        axm.set_xlabel("Initiation year")
        axm.set_title(name, fontsize=7, loc="left")
        if j == 0:
            axm.text(-0.3, 1.13, "b", transform=axm.transAxes, fontsize=9, fontweight="bold", va="bottom")
            axm.text(-0.3, 1.13, f"      Only units selected by at least one rule are shown ({len(U2)} of {len(UA)} candidates)",
                     transform=axm.transAxes, fontsize=7, va="bottom")
            axm.set_ylabel("Selected units, by channel", labelpad=11)
            for c in [1, 2, 3, 5]:
                rr = U2.index[U2.channel == c]
                axm.text(-0.8, (rr.min() + rr.max()) / 2,
                         CH_NAME[c].split(" ")[0], ha="right", va="center",
                         fontsize=6, color=CH_COL[c])

# c: timing efficiency, value in the chosen year relative to the unit's best year
rng = np.random.default_rng(0)
for i, (k, name, ck) in enumerate(rules):
    y = len(rules) - 1 - i
    e = eff[k]
    axc.scatter(e, y + rng.uniform(-0.22, 0.22, len(e)), s=4, color=COL[ck], alpha=0.55, lw=0)
    m = e.mean()
    axc.plot([m, m], [y - 0.34, y + 0.34], color=COL[ck], lw=1.6)
    axc.text(1.012, y, f"mean {m:.2f}", va="center", ha="left", fontsize=6, color="#333333", transform=mpl.transforms.blended_transform_factory(axc.transAxes, axc.transData))
    axc.text(-0.012, y, name, va="center", ha="right", fontsize=7, transform=mpl.transforms.blended_transform_factory(axc.transAxes, axc.transData))
axc.set_xlim(0.2, 1.0)
axc.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
axc.set_ylim(-0.6, len(rules) - 0.4)
axc.set_yticks([])
axc.spines["left"].set_visible(False)
axc.set_xlabel("Value in the chosen initiation year / value in the unit's best year")
axc.text(-0.19, 1.12, "b", transform=axc.transAxes, fontsize=9, fontweight="bold", va="bottom")
axc.text(-0.19, 1.12, "      Timing of each selected unit relative to its best year", transform=axc.transAxes,
         fontsize=7, va="bottom")

cax = fig.add_axes([0.36, 0.652, 0.3, 0.012])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
cb.set_ticks([1, 5, 10, 15, 20, 25])
cb.ax.tick_params(labelsize=6)
fig.text(0.35, 0.658, "Initiation year", ha="right", va="center", fontsize=7)
fig.text(0.67, 0.658, "grey: candidate not initiated", ha="left", va="center", fontsize=6, color="#555555")


# c: zoom on one window, persistence greedy vs rolling horizon with full information
ST = pd.read_csv(G + "metro/stations.csv")
PINYIN = {"中大": "Zhongda", "圳美": "Zhenmei", "深理工": "Shenligong", "光明": "Guangming",
          "科学公园": "Kexue Gongyuan", "楼村": "Loucun"}
smap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.2, 1.0), cmap=plt.get_cmap("YlOrRd_r"))
zoom_rules = [("static_greedy", "Persistence greedy"), ("rolling_full", "Rolling horizon, full")]
for j, (k, name) in enumerate(zoom_rules):
    axz = fig.add_axes([0.13 + j * 0.30, 0.035, 0.268, 0.28])
    init = {int(u): int(t) for u, t in SCH[k]["init"].items()}
    img = np.ones((H, W, 4))
    img[inside] = (0.97, 0.97, 0.97, 1)
    img[cand] = (0.86, 0.86, 0.86, 1)
    lab = []
    for u, t in init.items():
        m = uid == idx2uid[u]
        img[m] = smap.to_rgba(EVT[u, t] / BEST[u])
        rr, cc = np.where(m)
        if zc0 <= cc.mean() <= zc1 and zr0 <= rr.mean() <= zr1:
            lab.append((cc.mean(), rr.mean(), t + 1))
    axz.imshow(img, interpolation="nearest")
    draw_units(axz, lw=0.25)
    for x_, y_, t_ in lab:
        axz.text(x_, y_, str(t_), ha="center", va="center", fontsize=5.5, color="black",
                 fontweight="bold")
    for _, r_ in ST.iterrows():
        sc, sr = (r_.x - X00) / 100.0, (Y00 - r_.y) / 100.0
        if zc0 <= sc <= zc1 and zr0 <= sr <= zr1:
            axz.plot(sc, sr, marker="o", ms=4, mfc="white", mec="black", mew=0.7)
            axz.text(sc + 0.6, sr - 0.6, PINYIN.get(r_.station, r_.station), fontsize=5.5,
                     ha="left", va="bottom", color="#222222")
    axz.set_xlim(zc0, zc1); axz.set_ylim(zr1, zr0)
    axz.set_xticks([]); axz.set_yticks([])
    for sp in axz.spines.values():
        sp.set_edgecolor("#d62728"); sp.set_linewidth(0.8)
    axz.set_title(f"{name}", fontsize=7, loc="left")
    if j == 1:
        frac = 10.0 / (zc1 - zc0)                      # 10 cells = 1 km
        axz.plot([0.0, frac], [-0.05, -0.05], transform=axz.transAxes, color="black", lw=1.2,
                 solid_capstyle="butt", clip_on=False)
        axz.text(frac / 2, -0.07, "1 km", transform=axz.transAxes, ha="center", va="top", fontsize=5.5)
    if j == 0:
        axz.text(-0.30, 1.10, "c", transform=axz.transAxes, fontsize=9, fontweight="bold", va="bottom")
        axz.text(-0.30, 1.10, "      Window marked in (a): numbers are initiation years", transform=axz.transAxes,
                 fontsize=7, va="bottom")
caxz = fig.add_axes([0.75, 0.06, 0.012, 0.22])
cbz = fig.colorbar(smap, cax=caxz)
cbz.ax.tick_params(labelsize=6)
cbz.set_label("Share of best-year value", fontsize=6.5)

json.dump({k: dict(mean=float(v.mean()), median=float(np.median(v)), share_ge_0p9=float((v >= 0.9).mean()), n=int(len(v)))
           for k, v in eff.items()}, open(FD + "fig5_timing_efficiency.json", "w"), indent=1)
print("overlaps:", overlaps(fig))
fig.savefig("fig5_schedules.png", dpi=300)
fig.savefig("fig5_schedules.pdf")
