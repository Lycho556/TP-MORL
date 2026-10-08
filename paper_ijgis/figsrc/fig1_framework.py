import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
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


apply_figure_style(sizes=(8, 7, 6))

plt.close("all")


def box(ax, x, y, w, h, txt, fc="#f7f7f7", ec="#555555", fs=7, bold=False, ha="center", lw=0.8, color="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    ax.text(x + (w / 2 if ha == "center" else 1.2), y + h / 2, txt, ha=ha, va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=color, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, c="#444444", lw=1.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, color=c, lw=lw))


fig = plt.figure(figsize=(6.9, 5.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 72)
ax.axis("off")

for y, t in [(69, "a  The decision"), (44.5, "b  Planning information"), (15.5, "c  Model")]:
    ax.text(1, y, t, fontsize=8, fontweight="bold", va="center")

box(ax, 2, 50, 27, 15, "", fc="white")
ax.text(15.5, 62.5, "Static spatial prioritization", ha="center", fontsize=7.5, fontweight="bold")
ax.text(15.5, 59.3, "Which units should be renewed?", ha="center", fontsize=6.5, style="italic")
for i, (xx, yy) in enumerate([(5, 54), (9.8, 56.5), (14.6, 53.5), (19.4, 56), (24.2, 54)]):
    ax.add_patch(Rectangle((xx, yy), 3.2, 2.4, fc="#E69F00", ec="#444444", lw=0.5))
    ax.text(xx + 1.6, yy + 1.2, str(i + 1), ha="center", va="center", fontsize=6)
ax.text(15.5, 51.3, "units ranked once", ha="center", fontsize=6, color="#555555")

arrow(ax, 30, 57.5, 41, 57.5, lw=1.4)
ax.text(35.5, 56.4, "decision object:\nunit → (unit, year)", ha="center", va="top", fontsize=5.5)

box(ax, 42, 50, 56, 15, "", fc="white")
ax.text(70, 63.2, "Spatio-temporal scheduling", ha="center", fontsize=7.5, fontweight="bold")
ax.text(70, 60.3, "Which units use this year's capacity, and which wait?", ha="center", fontsize=6.5, style="italic")

yrs_x = [47, 58.5, 70, 81.5, 93]
ax.plot([44, 97], [52.2, 52.2], color="#888888", lw=0.8)
for k, xx in enumerate(yrs_x):
    ax.plot([xx, xx], [51.7, 52.7], color="#888888", lw=0.8)
    ax.text(xx, 50.6, f"year {k + 1}" if k < 4 else "…", ha="center", fontsize=5.5, color="#555555")
    ax.add_patch(Rectangle((xx - 4.5, 53.3), 9, 4.4, fc="none", ec="#0072B2", lw=0.7, ls="--"))
ax.text(97, 58.3, "dashed box: capacity of K units per year", ha="right", fontsize=5.5, color="#0072B2")

for dx in [-3, 0, 3]:
    ax.add_patch(Rectangle((47 + dx - 1.1, 54.3), 2.2, 2.2, fc="#0072B2", ec="none"))
for tx in [58.5, 70]:
    ax.add_patch(Rectangle((tx - 1.1, 54.3), 2.2, 2.2, fc="#9ecae1", ec="none"))

arrow(ax, 50.5, 55.4, 57.2, 55.4, c="#9ecae1", lw=0.9)
arrow(ax, 61.8, 55.4, 68.7, 55.4, c="#9ecae1", lw=0.9)

ax.text(47, 53.1, "ACT NOW", ha="center", va="top", fontsize=5.5, color="#0072B2", fontweight="bold")
ax.text(64.2, 53.1, "WAIT", ha="center", va="top", fontsize=5.5, color="#3182bd", fontweight="bold")

# Fix label positions
for t in ax.texts:
    s = t.get_text()
    if s == "ACT NOW":
        t.set_position((47, 58.3))
        t.set_va("bottom")
    if s == "WAIT":
        t.set_position((58.5, 58.3))
        t.set_va("bottom")

facs = [
    ("Statutory planning", "can the unit\nbe proposed?"),
    ("Infrastructure", "is completion\nworth more later?"),
    ("Building obsolescence", "will the plan\nbe approved?"),
    ("Implementation\nreadiness", "will the plan\nbe approved?"),
]
for i, (f, q) in enumerate(facs):
    x = 2 + i * 16.5
    ax.add_patch(FancyBboxPatch((x, 32), 15, 9, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc="#f4f6f8", ec="#555555", lw=0.8))
    ax.text(x + 7.5, 39.3, f, ha="center", va="top", fontsize=6, fontweight="bold", linespacing=1.1)
    ax.text(x + 7.5, 34.8 if "\n" not in f else 34.3, q, ha="center", va="center",
            fontsize=6, style="italic", linespacing=1.15)
    arrow(ax, x + 7.5, 31.3, 52, 27.8, c="#888888", lw=0.7)

box(ax, 40, 22.5, 24, 5, "time-varying renewal opportunity", fc="#e8f1f8", ec="#0072B2", fs=7)
arrow(ax, 52, 22, 52, 19.8, lw=1.0)

box(ax, 70, 24, 28, 17, "", fc="white", ec="#0072B2")
ax.text(84, 38.6, "When does timing matter?", ha="center", fontsize=7, fontweight="bold", color="#0072B2")
ax.text(84, 31.2,
        "units differ in how opportunity changes\n+ annual capacity binds\n↓\nwaiting has an opportunity cost\n↓\nthe implementation year becomes\na decision variable",
        ha="center", va="center", fontsize=6, linespacing=1.25)

box(ax, 2, 6, 20, 7, "Spatio-temporal\nscheduling model", fc="#f4f6f8", fs=7)
box(ax, 28, 10.5, 18, 5, "myopic temporal rule", fc="#dbe9f6", ec="#0072B2", fs=6.5)
box(ax, 28, 3.5, 18, 5, "value-based policy", fc="#e7f4fb", ec="#56B4E9", fs=6.5)
box(ax, 53, 6, 18, 7, "annual renewal\nschedule", fc="#f4f6f8", fs=7)
box(ax, 78, 6, 20, 7, "compared with the\nfull-information\nreference schedule", fc="white", fs=6.5)

arrow(ax, 22.8, 9.5, 27.2, 13)
arrow(ax, 22.8, 9.5, 27.2, 6)
arrow(ax, 46.8, 13, 52.2, 10.3)
arrow(ax, 46.8, 6, 52.2, 8.7)
arrow(ax, 71.8, 9.5, 77.2, 9.5)

fig.savefig("fig1_framework.png", dpi=300)
fig.savefig("fig1_framework.pdf")