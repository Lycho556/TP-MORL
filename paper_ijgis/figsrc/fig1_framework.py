"""Figure 1 — from spatial prioritization to spatio-temporal scheduling.

(a) change of decision object; (b) where each time-varying field enters the renewal
pipeline and how the expected-value table feeds the scheduling rules.
Drawn at the 34 pc text width (5.65 in), so font sizes are final printed sizes (>= 6.5 pt).
"""
import os as _os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")) + "/"
OUT = _ROOT + "paper_ijgis/figures/"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "savefig.bbox": "tight", "mathtext.fontset": "dejavusans",
})
GREEN, DGREEN, CREAM, ORANGE, INK, GREY = "#3C5A7D", "#24405F", "#F2F4F7", "#D9822B", "#222222", "#5E5E5E"
F_PANEL, F_HEAD, F_KEY, F_SUB = 8.0, 7.5, 7.0, 6.5      # panel titles, box headers, key labels, secondary labels

fig = plt.figure(figsize=(5.65, 4.42))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 78); ax.axis("off")


def rbox(x, y, w, h, fc="white", ec=GREEN, lw=1.0, r=1.2, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def header_box(x, y, w, h, title, bullets, hh=5.2, tag=None, tag_c=GREEN, fs=F_KEY, step=3.3, sub_step=2.8):
    rbox(x, y, w, h, fc="white", ec=GREEN, lw=1.1)
    ax.add_patch(FancyBboxPatch((x, y + h - hh), w, hh, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=GREEN, ec=GREEN, lw=1.1, zorder=3))
    ax.add_patch(Rectangle((x, y + h - hh), w, hh / 2, fc=GREEN, ec="none", zorder=3))
    ax.text(x + w / 2, y + h - hh / 2, title, ha="center", va="center", color="white",
            fontsize=F_HEAD, fontweight="bold", zorder=4)
    yy = y + h - hh - 2.8
    for b in bullets:
        if b.startswith("  "):
            ax.text(x + 4.4, yy + 0.6, b.strip(), ha="left", va="center", fontsize=F_SUB, color=GREY, zorder=4)
            yy -= sub_step
        else:
            ax.text(x + 1.8, yy, "–  " + b, ha="left", va="center", fontsize=fs, color=INK, zorder=4)
            yy -= step
    if tag:
        ax.text(x + w / 2, y + 2.6, tag, ha="center", va="center", fontsize=F_HEAD, fontweight="bold",
                color=tag_c, zorder=4)


def chevron(x, y, w, h, txt, first=False, fs=F_KEY):
    d = 1.8
    pts = [(x, y), (x + w - d, y), (x + w, y + h / 2), (x + w - d, y + h), (x, y + h)]
    if not first:
        pts.append((x + d, y + h / 2))
    ax.add_patch(Polygon(pts, closed=True, fc=GREEN, ec="none", zorder=2))
    ax.text(x + w / 2 + (0 if first else d / 2), y + h / 2, txt, ha="center", va="center",
            color="white", fontsize=fs, zorder=3, linespacing=1.05)


def arrow(x0, y0, x1, y1, c=INK, lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=c,
                                 lw=lw, zorder=5, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ panel a
rbox(1.5, 49.0, 97, 25.0, fc=CREAM, ec="none", r=1.8, z=0)
ax.text(0.5, 76.2, "a", fontsize=9, fontweight="bold", va="center")
ax.text(50, 76.2, "Change of decision object", fontsize=F_PANEL, fontweight="bold", ha="center", va="center")
header_box(3.0, 51.0, 27, 21.0, "Spatial prioritization",
           ["decision: which units", "one planning moment", "output: ranked list"], tag="WHERE")
arrow(31.2, 61.5, 35.3, 61.5, lw=1.2)
header_box(36.5, 51.0, 35.5, 21.0, "Spatio-temporal scheduling",
           ["decision: unit and year $y_i$", "annual capacity $K$ couples years", "output: annual schedule"],
           tag="WHERE + WHEN", tag_c=ORANGE)
# capacity inset: K = 3 slots per year; years 1-3 full, year 4 free; u takes the last year-2 slot, so v waits
FREE_FC, FREE_EC = "#EEF1F4", "#A7AFB8"
cw, chh, gp, vg = 4.2, 2.7, 2.0, 0.4
ncol = 4; gw = ncol * cw + (ncol - 1) * gp
gx = 73.5 + (24.5 - gw) / 2; gy = 59.9
taken = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)}
def cell(c, r):
    return gx + c * (cw + gp), gy + r * (chh + vg)
for c in range(ncol):
    for r in range(3):
        x, y = cell(c, r)
        if (c, r) in ((1, 2), (2, 2)): fc, ec = ORANGE, ORANGE
        elif (c, r) in taken:           fc, ec = GREEN, GREEN
        else:                           fc, ec = FREE_FC, FREE_EC
        ax.add_patch(Rectangle((x, y), cw, chh, fc=fc, ec=ec, lw=0.8, zorder=3))
    ax.text(cell(c, 0)[0] + cw / 2, gy - 1.5, f"Year {c + 1}", ha="center", va="center", fontsize=F_SUB, color=INK)
ux, uy = cell(1, 2); vx, vy = cell(2, 2)
for (x, y, lab) in ((ux, uy, "$u$"), (vx, vy, "$v$")):
    ax.text(x + cw / 2, y + chh / 2, lab, ha="center", va="center", fontsize=F_KEY, color="white",
            fontweight="bold", zorder=4)
ax.add_patch(FancyArrowPatch((ux + cw / 2, uy + chh + 0.15), (vx + cw / 2, vy + chh + 0.15),
                             arrowstyle="-|>", mutation_scale=11, color=ORANGE, lw=2.1,
                             connectionstyle="arc3,rad=-0.6", zorder=5, shrinkA=0, shrinkB=0))
gmid = gx + gw / 2
ax.text((ux + vx + cw) / 2, 72.7, "capacity forces delay", ha="center", va="center", fontsize=F_SUB,
        color=ORANGE, fontweight="bold")
# legend directly under the year labels
ls_ = 1.6; lx = gmid - 8.6; ly = 56.3
ax.add_patch(Rectangle((lx, ly - ls_ / 2), ls_, ls_, fc=GREEN, ec=GREEN, lw=0.8, zorder=3))
ax.text(lx + ls_ + 0.7, ly, "taken", ha="left", va="center", fontsize=F_SUB, color=GREY)
ax.add_patch(Rectangle((lx + 9.6, ly - ls_ / 2), ls_, ls_, fc=FREE_FC, ec=FREE_EC, lw=0.8, zorder=3))
ax.text(lx + 9.6 + ls_ + 0.7, ly, "free", ha="left", va="center", fontsize=F_SUB, color=GREY)
# short text block
ax.text(gmid, 54.1, "$K = 3$ slots / year", ha="center", va="center", fontsize=F_KEY, color=INK, fontweight="bold")
ax.text(gmid, 52.1, "$u$ fills year 2", ha="center", va="center", fontsize=F_SUB, color=INK)
ax.text(gmid, 50.3, "\u2192 $v$ waits until year 3", ha="center", va="center", fontsize=F_SUB, color=INK)

# ------------------------------------------------------------------ panel b
ax.text(0.5, 46.2, "b", fontsize=9, fontweight="bold", va="center")
ax.text(50, 46.2, "Where planning and implementation information enters", fontsize=F_PANEL, fontweight="bold",
        ha="center", va="center")
stages = [("Listing", "gate $a_{i,t}$"), ("Conformity\nreview", "factor $a_{i,y}$"),
          ("Approval\nwindow", "hazard $h_{i,j}$"), ("Construction", "$b_i$ years"), ("Delivery", "value at $t_c$")]
x0, cwid, cy, chg = 1.5, 14.6, 35.6, 6.4
cx = []
for k, (name, sub) in enumerate(stages):
    x = x0 + k * cwid; cx.append(x + cwid / 2 + (0 if k == 0 else 0.9))
    chevron(x, cy, cwid, chg, name, first=(k == 0))
    ax.text(cx[k], cy - 2.3, sub, ha="center", va="center", fontsize=F_SUB, color=GREY)
chevron(76.0, cy, 22.5, chg, "Scheduling")

fy, fh = 21.5, 8.2
def field(xa, xb, title, sub):
    rbox(xa, fy, xb - xa, fh, fc=CREAM, ec=GREEN, lw=1.0, r=1.0, z=3)
    ax.text((xa + xb) / 2, fy + fh * 0.70, title, ha="center", va="center", fontsize=F_KEY, fontweight="bold",
            color=DGREEN, zorder=4)
    ax.text((xa + xb) / 2, fy + fh * 0.30, sub, ha="center", va="center", fontsize=F_SUB, color=INK, zorder=4,
            linespacing=1.05)
pad = 0.8
field(x0 + pad, x0 + 2 * cwid - pad, "Planning support $P_{i,t}$", "admission probability")
field(x0 + 2 * cwid + pad, x0 + 3 * cwid - pad, "$B_{i,t}$, $C_{i,t}$", "obsolescence,\nreadiness")
field(x0 + 4 * cwid + pad, x0 + 5 * cwid - pad, "$I_{i,t}$", "infrastructure\nat $t_c$")
for k in (0, 1, 2, 4):
    arrow(cx[k], fy + fh, cx[k], cy - 3.9, c=DGREEN, lw=1.0)

# expected value table (key element)
ex0, ex1 = x0 + pad, x0 + 5 * cwid - pad
rbox(ex0, 3.5, ex1 - ex0, 9.6, fc="white", ec=ORANGE, lw=1.5, r=1.0, z=3)
ax.text((ex0 + ex1) / 2, 10.0, "Expected value $\\mathrm{EV}_{i,y}$ of initiating unit $i$ in year $y$",
        ha="center", va="center", fontsize=F_HEAD, fontweight="bold", color=ORANGE, zorder=4)
ax.text((ex0 + ex1) / 2, 6.2, "the rules differ in how they forecast this table",
        ha="center", va="center", fontsize=F_SUB, color=INK, zorder=4)
mid_bc = x0 + 2.5 * cwid
arrow(mid_bc, fy, mid_bc, 13.1, c=INK, lw=1.0)

# scheduling rules and output
header_box(76.5, 10.8, 22.0, 23.0, "Rules under $K$",
           ["greedy ranking", "rolling horizon", "  current, announced,", "  or full forecast", "learned policy"],
           hh=4.8, step=3.5)
ax.plot([ex1, 74.5, 74.5], [8.3, 8.3, 24.0], color=INK, lw=1.0, zorder=4, solid_capstyle="butt")
arrow(74.5, 24.0, 76.5, 24.0, c=INK, lw=1.0)
arrow(87.5, 10.8, 87.5, 8.8, c=INK, lw=1.0)
rbox(76.5, 3.5, 22.0, 5.3, fc=GREEN, ec=GREEN, lw=1.0, r=1.0, z=3)
ax.text(87.5, 6.15, "Annual schedule", ha="center", va="center", fontsize=F_HEAD, color="white",
        fontweight="bold", zorder=4)

fig.savefig(OUT + "fig1_framework.pdf")
fig.savefig(OUT + "fig1_framework.png", dpi=300)
