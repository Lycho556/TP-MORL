import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Common setup
META_GREY = "#95a5a6"
COL = {"random": META_GREY, "static": "#E69F00", "supervised": "#CC79A7",
       "temporal": "#0072B2", "value": "#56B4E9", "optimum": "black"}
CH_COL = {1: "#E69F00", 2: "#56B4E9", 3: "#009E73", 5: "#CC79A7"}
CH_NAME = {1: "P1 industrial redevelopment", 2: "P2 old residential",
           3: "P3 commercial-service", 5: "P5 agricultural conversion"}

def apply_figure_style(sizes=(8, 7, 6)):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "font.size": sizes[0],
        "axes.labelsize": sizes[1],
        "xtick.labelsize": sizes[2],
        "ytick.labelsize": sizes[2],
    })

def panel_letter(ax, letter):
    ax.text(-0.08, 1.05, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="right")

def overlaps(fig):
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    fb = fig.bbox
    ov = [(x.get_text()[:24], y.get_text()[:24])
          for i, (x, bx) in enumerate(texts)
          for y, by in texts[i+1:] if bx.overlaps(by)]
    return ov

apply_figure_style(sizes=(8, 7, 6))

FD = "/Users/user/Desktop/TP-MORL/results_figures/"
F = json.load(open(FD + "gm_field_curves.json"))
T = 25
yrs = np.arange(1, T + 1)

rows2 = [("Statutory\nplanning support", "when a unit\ncan be listed", "listing and\nconformity", "plan"),
         ("Infrastructure\nmaturity", "when completed floor\narea realizes its value", "delivered\nvalue", "infra"),
         ("Building\nobsolescence", "urgency that grows\nwith building age", "approval", "obsolescence"),
         ("Implementation\nreadiness", "whether an application\ncan be carried through", "approval", "readiness")]
DK, LT, GR = "#1b4f72", "#5dade2", "#95a5a6"
lab_spec = {
    "plan": [("high throughout", DK, 2, 1.0, "bottom"),
             ("rise and decline", LT, None, None, "peak"),
             ("persistently low", GR, 19, 0.06, "top")],
    "infra": [("earlier cluster", DK, None, None, "peak"),
              ("later cluster", LT, None, None, "peak")],
    "obsolescence": [("older building", DK, 2, 0.98, "top"),
                     ("newer building", LT, 16, None, "above")],
    "readiness": [("example unit", DK, 17, None, "above")]
}

top, bot = 0.92, 0.30
h = (top - bot) / 4

fig = plt.figure(figsize=(6.6, 5.9))
for j, (x, hd) in enumerate([(0.02, "Factor"), (0.19, "Planning meaning"),
                               (0.385, "Acts on"), (0.53, "Scenario-based trajectories (not observed data)")]):
    fig.text(x, top + 0.02, hd, fontsize=8, fontweight="bold", va="bottom")

for i, (fac, mean, stage, key) in enumerate(rows2):
    yc = top - (i + 0.5) * h
    fig.text(0.02, yc, fac, fontsize=8, va="center")
    fig.text(0.19, yc, mean, fontsize=7, va="center")
    fig.text(0.385, yc, stage, fontsize=7, va="center")
    ax = fig.add_axes([0.54, top - (i + 1) * h + 0.03, 0.42, h - 0.055])
    for (nm, c, xl, yl, mode), y in zip(lab_spec[key], list(F[key].values())):
        y = np.array(y[:T])
        ax.plot(yrs, y, color=c, lw=1.4)
        if mode == "peak":
            k = int(np.argmax(y))
            ax.text(yrs[k], y[k] + 0.05, nm, color=c, fontsize=6, ha="center", va="bottom")
        elif mode == "above":
            ax.text(xl, y[xl - 1] + 0.07, nm, color=c, fontsize=6, ha="center", va="bottom")
        else:
            ax.text(xl, y[xl - 1] + (0.04 if mode == "bottom" else -0.04),
                    nm, color=c, fontsize=6, ha="left", va=mode)
    ax.set_ylim(-0.3 if key == "plan" else -0.08, 1.25)
    ax.set_xlim(1, T)
    ax.set_yticks([0, 1])
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_xticklabels([])

fig.text(0.505, (top + bot) / 2, "Field level (0–1)", rotation=90,
         va="center", ha="center", fontsize=7)

# Fix "newer building" label position
ax3 = fig.axes[2]
for t in ax3.texts:
    if t.get_text() == "newer building":
        x, y = t.get_position()
        t.set_position((x - 2, y + 0.1))

# ---- bottom panel: the combined effect on the value of initiating a unit
U = json.load(open(FD + "fig2_example_units.json"))
yb = bot - 0.2
fig.text(0.02, yb + 0.1, "Combined effect\n(case study)", fontsize=8, va="center")
fig.text(0.19, yb + 0.1, "value of initiating\nthe unit in each year\n(Equation 11)", fontsize=7, va="center")
fig.text(0.385, yb + 0.1, "the timing\ndecision", fontsize=7, va="center")
axv = fig.add_axes([0.54, yb, 0.42, 0.2])
v1 = np.array(U["331"]["ev_rel"]); v2 = np.array(U["249"]["ev_rel"]); a2 = np.array(U["249"]["admit"])
axv.plot(yrs, v1, color=DK, lw=1.4)
lst = a2 > 0
axv.plot(yrs[lst], v2[lst], color=LT, lw=1.4)
axv.plot(yrs, np.where(lst, np.nan, 0.0), color=LT, lw=1.0, ls=":")
for yy in (3, 8, 13):
    axv.plot(yy, v1[yy - 1], "o", ms=3.5, color=DK, zorder=4)
    axv.text(yy, v1[yy - 1] + 0.07, f"{v1[yy - 1]:.2f}", fontsize=6, ha="center", va="bottom", color=DK)
axv.text(19.3, 0.42, "unit A: listed\nevery year", fontsize=6, color=DK, ha="left", va="bottom")
axv.text(14.7, 0.80, "unit B: listed only\nin years 10\u201317", fontsize=6, color=LT, ha="left", va="bottom")
axv.set_ylim(-0.08, 1.25); axv.set_xlim(1, T)
axv.set_yticks([0, 1]); axv.set_xticks([1, 5, 10, 15, 20, 25])
axv.set_xlabel("Decision (initiation) year")
axv.set_ylabel("Share of best\nyear's value", fontsize=7)
fig.text(0.505, (top + bot) / 2, "", fontsize=1)
print("overlaps:", overlaps(fig))
fig.savefig("fig2_temporal_opportunity.png", dpi=300)
fig.savefig("fig2_temporal_opportunity.pdf")