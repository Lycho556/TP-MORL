import sys, os, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")) + "/"  # repository root
R = _ROOT + ""
FD = R + "results_figures/"

sys.path.insert(0, R + "scripts")
sys.path.insert(0, R + "src")

# figure style setup
def apply_figure_style(sizes=(8, 7, 6)):
    mpl.rcParams.update({
        'font.size': sizes[2],
        'axes.titlesize': sizes[1],
        'axes.labelsize': sizes[2],
        'xtick.labelsize': sizes[2],
        'ytick.labelsize': sizes[2],
        'legend.fontsize': sizes[2],
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def panel_letter(ax, letter):
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top', ha='right')

def overlaps(fig):
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    return [(x.get_text()[:22], y.get_text()[:22])
            for i, (x, bx) in enumerate(texts)
            for y, by in texts[i+1:] if bx.overlaps(by)]

META_GREY = "#777777"
COL = {
    "random": META_GREY,
    "static": "#E69F00",
    "supervised": "#CC79A7",
    "temporal": "#0072B2",
    "value": "#56B4E9",
    "optimum": "black"
}

apply_figure_style(sizes=(8, 7, 6))

cur = json.load(open(FD + "synth_opp_curves.json"))
syn = pd.read_csv(R + "results_fqi/fqi_synth_summary.csv")
m = {"随机": "random", "纯空间排序": "static", "时间感知贪心": "temporal",
     "Double-FQI": "value", "oracle": "optimum"}
syn["key"] = syn.method.map(m)

shade = ["#9aa5b1", "#5d6d7e", "#2c3e50"]
lab = {"early": "early peak", "late": "middle peak", "verylate": "late peak"}
order = [("random", "Random"), ("static", "Spatial ranking"),
         ("temporal", "Trajectory-informed greedy"), ("value", "Value-based")]

pos_fix = {"early": (1.05, 0.975, "left"), "late": (5.5, 0.975, "center"),
           "verylate": (9.0, 1.03, "center")}

def fig3():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.6),
                                gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.34})
    for (k, v), c in zip(cur.items(), shade):
        y = np.array(v["curve"][:10])
        x = np.arange(1, 11)
        a.plot(x, y, color=c, lw=1.6)
        a.fill_between(x, 0, y, color=c, alpha=0.12, lw=0)

    a.set_xlim(0.6, 10.4)
    a.set_ylim(0, 1.16)
    a.set_xticks([1, 4, 7, 10])
    a.set_xlabel("Decision year")
    a.set_ylabel("Time-varying opportunity")
    a.set_title("Parcel types peak at different times", loc="left")

    for key, name in order:
        d = syn[syn.key == key].sort_values("alpha")
        xs = d.alpha.values
        ys = d["相对oracle"].values
        if key == "value":
            sd = np.nan_to_num(d["标准差"].values)
            b.errorbar(xs, ys, yerr=sd, fmt="o", mfc="white", mec=COL[key],
                       mew=1.4, ms=6.5, color=COL[key], lw=0, elinewidth=1,
                       capsize=2, zorder=4)
        else:
            b.plot(xs, ys, "-o", color=COL[key], lw=1.5, ms=4, zorder=3)

    b.axhline(1.0, color="black", ls="--", lw=0.9)
    b.text(0.22, 1.012, "reference schedule", va="bottom", ha="left", fontsize=7)

    lbl = {
        "random": ("Random", 0),
        "static": ("Spatial ranking", 0),
        "temporal": ("Trajectory-informed greedy", 0.0),
        "value": ("Value-based (open circles)", -0.065)
    }
    tg1 = float(syn[(syn.key == "temporal") & (syn.alpha == 1.0)]["相对oracle"].iloc[0])
    for key, (txt, dy) in lbl.items():
        yv = syn[(syn.key == key) & (syn.alpha == 1.0)]["相对oracle"]
        yv = float(yv.iloc[0]) if len(yv) else tg1
        b.text(1.03, yv + dy, txt, color=COL[key], fontsize=7, va="center", ha="left")

    b.set_xticks([0.25, 0.5, 0.75, 1.0])
    b.set_xlim(0.2, 1.02)
    b.set_ylim(0.3, 1.07)
    b.set_xlabel("Weight on time-varying opportunity (α)")
    b.set_ylabel("Ratio to reference schedule")
    b.set_title("Performance relative to the reference schedule", loc="left")

    panel_letter(a, "a")
    panel_letter(b, "b")
    fig.subplots_adjust(left=0.075, right=0.79, bottom=0.17, top=0.87)
    return fig


def fig3b():
    f = fig3()
    a = f.axes[0]
    for t in list(a.texts):
        t.remove()
    for (k, v), c in zip(cur.items(), shade):
        px, py, ha = pos_fix[k]
        a.text(px, py, lab[k], ha=ha, va="bottom", fontsize=7, color=c)
    panel_letter(a, "a")
    return f


f3 = fig3b()
f3.savefig("fig3_controlled_experiment.png", dpi=300)
f3.savefig("fig3_controlled_experiment.pdf")