import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Setup from figs_common.py
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")) + "/"  # repository root
FD = _ROOT + "results_figures/"
META_GREY = "#888888"
COL = {"random": META_GREY, "static": "#E69F00", "supervised": "#CC79A7",
       "temporal": "#0072B2", "value": "#56B4E9", "optimum": "black"}
CH_COL = {1: "#E69F00", 2: "#56B4E9", 3: "#CC79A7", 5: "#009E73"}
CH_NAME = {1: "P1 industrial redevelopment", 2: "P2 old residential",
           3: "P3 commercial-service", 5: "P5 agricultural conversion"}

def apply_figure_style(sizes=(8, 7, 6)):
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': sizes[2],
        'axes.titlesize': sizes[1],
        'axes.labelsize': sizes[1],
        'xtick.labelsize': sizes[2],
        'ytick.labelsize': sizes[2],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
    })

def panel_letter(ax, letter):
    ax.text(-0.08, 1.05, letter, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='right')

def overlaps(fig):
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    return [(x.get_text()[:22], y.get_text()[:22])
            for i, (x, bx) in enumerate(texts)
            for y, by in texts[i+1:]
            if bx.overlaps(by)]

apply_figure_style(sizes=(8, 7, 6))

import pandas as pd
# P1: means over six listing draws (learned rules: x five training seeds), M = 20.
# Built by scripts/p1_build_tables.py from results_p1/rolling and results_p1/fqi.
T = pd.read_csv(_ROOT + "results_p1/tables/main_M20.csv", index_col=0)
COL.update({"rh_announced": "#F0A875", "rh_full": "#D55E00"})
name = {"random": "Random", "persistence": "Persistence greedy", "supervised": "Supervised scorer",
        "rh_announced": "Rolling horizon, announced",
        "trajectory": "Trajectory-informed greedy", "value_based": "Value-based policy",
        "rh_full": "Rolling horizon, full"}
ckey = {"random": "random", "persistence": "static", "supervised": "supervised",
        "rh_announced": "rh_announced", "trajectory": "temporal", "value_based": "value",
        "rh_full": "rh_full"}
order = ["random", "persistence", "supervised", "rh_announced", "trajectory", "value_based", "rh_full"]
rows = [(name[k], ckey[k], T.loc[k, "ratio"], T.loc[k, "l_sel"], T.loc[k, "l_tim"]) for k in order]
TIM, SEL = "#34495e", "#c9d3dd"
n = len(rows)
fig, ax = plt.subplots(figsize=(6.3, 3.0))
for i, (nm, key, ratio, sel, tim) in enumerate(rows):
    y = n - 1 - i
    ax.barh(y, tim, color=TIM, height=0.6, edgecolor="none")
    ax.barh(y, sel, left=tim, color=SEL, height=0.6, edgecolor="none")
    ax.text(tim + sel + 25, y, f"{ratio:.3f}", va="center", ha="left", fontsize=7)
    ax.text(-60, y, nm, va="center", ha="right", color="#222222", fontsize=7)
    ax.plot([-30], [y], marker="s", ms=6, color=COL[key], clip_on=False)
for k in ("trajectory", "rh_full"):
    i = order.index(k); y = n - 1 - i
    _, _, ratio, sel, tim = rows[i]
    ax.text(tim + sel + 230, y, f"timing = {tim / (tim + sel) * 100:.0f}% of this gap",
            va="center", ha="left", fontsize=7, color=TIM)
ax.text(rows[0][4] / 2, n - 1, "timing-related loss", color="white", va="center", ha="center", fontsize=7)
ax.text(rows[0][4] + rows[0][3] / 2, n - 1, "selection-related", color="#2c3e50", va="center",
        ha="center", fontsize=7)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 1900)
ax.set_ylim(-0.6, n - 0.4)
ax.set_xlabel("Value gap to the reference schedule (objective units)")
ax.text(1900, n - 0.45, "ratio to reference", ha="right", va="bottom", fontsize=7)
fig.subplots_adjust(left=0.36, right=0.97, bottom=0.16, top=0.92)
fig.savefig("fig6_decomposition.png", dpi=300)
fig.savefig("fig6_decomposition.pdf")