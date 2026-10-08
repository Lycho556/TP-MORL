import json, os, sys
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MPoly, Rectangle
from matplotlib.lines import Line2D
import requests

# ── global state ──────────────────────────────────────────────────────────────
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


def overlaps(fig):
    r = fig.canvas.get_renderer()
    t = [(x, x.get_window_extent(r)) for x in fig.findobj(mpl.text.Text)
         if x.get_text().strip() and x.get_visible()]
    return [(a.get_text()[:24], b.get_text()[:24])
            for i, (a, ba) in enumerate(t) for b, bb in t[i + 1:] if ba.overlaps(bb)]


# ── constants ─────────────────────────────────────────────────────────────────
R = "/Users/user/Desktop/TP-MORL/"
FD = R + "results_figures/"
COL = {"random": "#8a8a8a", "static": "#E69F00", "supervised": "#CC79A7",
       "temporal": "#0072B2", "value": "#56B4E9", "optimum": "black"}
TIM, SEL = "#34495e", "#c9d3dd"
CH_COL = {1: "#8c6bb1", 2: "#e6550d", 3: "#3182bd", 5: "#31a354"}
CH_NAME = {1: "P1 industrial redevelopment", 2: "P2 old residential",
           3: "P3 commercial-service", 5: "P5 agricultural conversion"}

apply_figure_style(sizes=(8, 7, 6))

# ── load data ─────────────────────────────────────────────────────────────────
os.makedirs(FD, exist_ok=True)

G = R + "data/processed/gm_dataset_v1/"
uid = np.load(G + "zones_v0/unit_id.npy")
ch  = np.load(G + "zones_v0/channel_v0.npy")
L0  = np.load(G + "grid_100m/L0_class.npy")
vf  = np.load(G + "grid_100m/valid_frac.npy")
am  = np.load(G + "grid_100m/action_mask.npy")
cu  = pd.read_csv(G + "zones_v0/candidate_units.csv")
leg = pd.read_csv(G + "tables/class_legend.csv")
chl = pd.read_csv(G + "zones_v0/channel_legend.csv")

UA = pd.read_csv(FD + "unit_attrs.csv")
chan_of = dict(zip(UA.uid, UA.channel))

# Shenzhen district GeoJSON
gj_path = FD + "shenzhen_districts.geojson"
if not os.path.exists(gj_path):
    r = requests.get("https://geo.datav.aliyun.com/areas_v3/bound/440300_full.json", timeout=30)
    with open(gj_path, "w") as f:
        f.write(r.text)
gj = json.load(open(gj_path))

# ── build colour arrays ───────────────────────────────────────────────────────
LUG = {0: ("Restricted", "#e3e3e3"), 1: ("Residential", "#fbf2c4"),
       2: ("Residential", "#fbf2c4"), 4: ("Commercial", "#f6d5cf"),
       7: ("Industrial", "#e3d9ee"), 8: ("Industrial", "#e3d9ee"),
       9: ("Agricultural", "#eef3d2"), 10: ("Ecological and green", "#d8ecd6"),
       11: ("Ecological and green", "#d8ecd6"), 12: ("Public utilities", "#dde6ee")}

H, W = L0.shape
rgb = np.ones((H, W, 3))
for k, (n, c) in LUG.items():
    rgb[L0 == k] = mpl.colors.to_rgb(c)

urgb = np.zeros((H, W, 4))
for u, c in chan_of.items():
    m = uid == u
    urgb[m] = (*mpl.colors.to_rgb(CH_COL[c]), 0.95)

inside = (L0 != 255)


def edges(lab):
    e = np.zeros_like(lab, bool)
    e[:-1, :] |= lab[:-1, :] != lab[1:, :]
    e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    return e


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


# ── find zoom window ──────────────────────────────────────────────────────────
S = 20
best = None
for i in range(0, H - S, 2):
    for j in range(0, W - S, 2):
        w = uid[i:i + S, j:j + S]
        us = np.unique(w[w > 0])
        if len(us) < 8:
            continue
        chs = set(chan_of[u] for u in us)
        cov = (w > 0).mean()
        sc = len(chs) * 10 + len(us) * 0.3 + cov * 5
        if best is None or sc > best[0]:
            best = (sc, i, j, len(chs), len(us), cov)
_, r0, c0, _, _, _ = best

# ── draw figure ───────────────────────────────────────────────────────────────
plt.close("all")
fig = plt.figure(figsize=(6.9, 3.6))
axa = fig.add_axes([0.01, 0.45, 0.24, 0.45])
axb = fig.add_axes([0.26, 0.06, 0.42, 0.84])
axc = fig.add_axes([0.71, 0.36, 0.27, 0.52])

# (a) location map
for f in gj["features"]:
    gm = f["properties"]["adcode"] == 440311
    geoms = (f["geometry"]["coordinates"]
             if f["geometry"]["type"] == "MultiPolygon"
             else [f["geometry"]["coordinates"]])
    for poly in geoms:
        axa.add_patch(MPoly(np.array(poly[0]), closed=True,
                            fc="#E69F00" if gm else "#f0f0f0", ec="#666666", lw=0.4))
    if gm:
        c = np.array(geoms[0][0]).mean(0)
        axa.annotate("Guangming", c,
                     xytext=(c[0] + 0.05, c[1] + 0.13), fontsize=7, ha="center",
                     arrowprops=dict(arrowstyle="-", lw=0.5))
axa.set_xlim(113.72, 114.65)
axa.set_ylim(22.42, 22.92)
axa.set_aspect(1 / np.cos(np.deg2rad(22.6)))
axa.axis("off")
axa.set_title("Location in Shenzhen", loc="left")

# (b) district map
axb.imshow(rgb, interpolation="nearest")
axb.imshow(urgb, interpolation="nearest")
axb.contour(inside.astype(float), levels=[0.5], colors="#444444", linewidths=0.6)
draw_units(axb, lw=0.15)
axb.set_xlim(-1, W)
axb.set_ylim(H, -1)
axb.axis("off")
axb.set_title("Guangming: 717 candidate renewal units", loc="left")

x0, y0 = 4, H - 6
axb.plot([x0, x0 + 20], [y0, y0], color="black", lw=1.5)
axb.text(x0 + 10, y0 - 2, "2 km", ha="center", va="bottom", fontsize=6)
axb.annotate("", xy=(W - 6, 10), xytext=(W - 6, 22),
             arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black"))
axb.text(W - 6, 8, "N", ha="center", va="bottom", fontsize=7)
axb.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), S, S, fill=False, ec="black", lw=0.9))

# (c) zoom
axc.imshow(rgb, interpolation="nearest")
axc.imshow(urgb, interpolation="nearest")
for k in range(S + 1):
    axc.axhline(r0 - 0.5 + k, color="white", lw=0.4)
    axc.axvline(c0 - 0.5 + k, color="white", lw=0.4)
draw_units(axc, lw=1.0)
axc.set_xlim(c0 - 0.5, c0 + S - 0.5)
axc.set_ylim(r0 + S - 0.5, r0 - 0.5)
axc.set_xticks([])
axc.set_yticks([])
for s in axc.spines.values():
    s.set_visible(True)
    s.set_color("black")
    s.set_linewidth(0.9)
axc.set_title("Decision units on the 100 m grid", loc="left")
axc.legend(handles=[Line2D([], [], color="#222222", lw=1.2, label="renewal unit (decision unit)"),
                    Line2D([], [], color="#bbbbbb", lw=0.8, label="100 m grid cell")],
           loc="upper left", bbox_to_anchor=(0, -0.02), frameon=False, fontsize=6)

# legends
hch = [mpatches.Patch(color=CH_COL[c], label=CH_NAME[c]) for c in [1, 2, 3, 5]]
hlu = [mpatches.Patch(color=c, label=n)
       for n, c in dict((v[0], v[1]) for v in LUG.values()).items()]
fig.legend(handles=hch, loc="upper left", bbox_to_anchor=(0.01, 0.42),
           frameon=False, fontsize=6, title="Institutional channel",
           title_fontsize=6, alignment="left")
fig.legend(handles=hlu, loc="upper left", bbox_to_anchor=(0.71, 0.25),
           frameon=False, fontsize=5.5, ncol=2,
           title="Current land use (background)", title_fontsize=6, alignment="left")

for a_, l in [(axa, "a"), (axb, "b"), (axc, "c")]:
    panel_letter(a_, l)

fig.savefig("fig4_study_area.png", dpi=300)
fig.savefig("fig4_study_area.pdf")
print(overlaps(fig))