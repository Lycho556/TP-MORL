
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 7, "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "savefig.dpi": 300})
DK, FILL, GATE, GREY, VAL = "#1b4f72", "#eaf2fa", "#fdf1e6", "#eeeeee", "#e6f4ea"
fig = plt.figure(figsize=(6.9, 3.9)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 65); ax.axis("off")
def box(x, y, w, h, title, body="", fc=FILL, ec=DK, lw=0.9, tcol=DK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7, fontweight="bold", color=tcol)
    if body:
        ax.text(x + w / 2, y + h - 5.0, body, ha="center", va="top", fontsize=6.3, color="#222222", linespacing=1.35)
def arr(x0, y0, x1, y1, col=DK, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=col, ls=ls,
                                 shrinkA=0, shrinkB=0))
# ---- row 1: planning, listing, initiation, conformity, approval window
Y1, H1 = 40, 14
box(1, Y1, 14, H1, "Planning support", "$P_{i,t}\\in[0,1]$\nscenario-based\nfield")
box(19, Y1, 17, H1, "Listing gate", "listed in year $t$ with\nprobability $a_{i,t}$ (Eq. 7),\nredrawn every year", fc=GATE)
box(40, Y1, 14, H1, "Initiation", "year $y$, chosen by\nthe rule; at most\n$K$ units per year")
box(58, Y1, 15, H1, "Conformity gate", "value scaled by\n$a_{i,y}$ (Eq. 11)", fc=GATE)
box(77, Y1, 22, H1, "Approval window", "$j = 1,\\dots,\\tau_{\\max}=5$ years\n$h_{i,j}(y)=\\min\\{1,\\ h_j\\,m_{i,y+j-1}\\}$\n(Eq. 8)")
for a, b in [(15.3, 19), (36.3, 40), (54.3, 58), (73.3, 77)]:
    arr(a, Y1 + H1 / 2, b - 0.3, Y1 + H1 / 2)
# side input: obsolescence and readiness
box(66, 57, 33, 7.5, "Obsolescence $B_{i,t}$ and readiness $C_{i,t}$", "", fc="white")
ax.text(82.5, 58.3, "$m_{i,t}=[1+A_B(2B_{i,t}-1)]\\,[1+A_C(2C_{i,t}-1)]$", ha="center", fontsize=6.0, color="#222222")
arr(88, 56.7, 88, Y1 + H1 + 0.3)
# ---- row 2 (right to left): outcomes, construction, completion, value
Y2, H2 = 19, 13
box(61, Y2, 17, H2, "Approved", "in window year $k$\n$t_a = y + k$  (Eq. 9)\n$\\Pr_i(k\\mid y)$ (Eq. 10)")
box(83, Y2, 16, H2, "Lapsed", "not approved within\nthe window; baseline\nprobability 0.294", fc=GREY, ec="#777777", tcol="#555555")
arr(81, Y1 - 0.3, 72, Y2 + H2 + 0.3); arr(91, Y1 - 0.3, 91, Y2 + H2 + 0.3, col="#777777")
box(40, Y2, 18, H2, "Construction", "$b_i$ = 3\u20135 years\nby institutional channel\n(Appendix B)")
box(19, Y2, 17, H2, "Completion", "$t_c = t_a + 1 + b_i$\n(Eq. 9)")
box(1, Y2, 14, H2, "Delivered value", "$\\mathrm{FL}(u_i)\\,(1+A_I I_{i,t_c})$\n$\\times\\,\\gamma^{t_c}$ (Eqs 4, 11)", fc=VAL, ec="#2e7d32", tcol="#1b5e20")
arr(60.7, Y2 + H2 / 2, 58.3, Y2 + H2 / 2)
arr(39.7, Y2 + H2 / 2, 36.3, Y2 + H2 / 2); arr(18.7, Y2 + H2 / 2, 15.3, Y2 + H2 / 2)
# side input: infrastructure
box(15.5, 8.6, 21, 6.2, "Infrastructure $I_{i,t}$", "", fc="white")
ax.text(26, 9.6, "read at the completion year", ha="center", fontsize=5.8, color="#222222")
ax.add_patch(FancyArrowPatch((15.5, 11.7), (8, Y2 - 0.3), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=DK,
                             connectionstyle="arc3,rad=-0.25", shrinkA=0, shrinkB=0))
# ---- timeline of an example unit
T0, T1, yl = 53, 98, 8.0
sc = lambda yr: T0 + (yr - 7) * (T1 - T0) / 11.0
ax.plot([T0, T1], [yl, yl], color="#555555", lw=0.8)
for yr, lab in [(8, "$y=8$\ninitiation"), (11, "$t_a=11$\napproval"), (17, "$t_c=17$\ncompletion")]:
    ax.plot([sc(yr)] * 2, [yl - 0.8, yl + 0.8], color=DK, lw=1.0)
    ax.text(sc(yr), yl + 1.3, lab, ha="center", va="bottom", fontsize=5.8, color=DK, linespacing=1.15)
for a0, a1, lab in [(8, 11, "window year $k=3$"), (12, 17, "construction $b_i=5$ (from year 12)")]:
    ax.annotate("", xy=(sc(a1), yl - 2.0), xytext=(sc(a0), yl - 2.0), arrowprops=dict(arrowstyle="<->", lw=0.7, color="#555555"))
    ax.text((sc(a0) + sc(a1)) / 2, yl - 2.7, lab, ha="center", va="top", fontsize=5.8, color="#555555")
ax.text(T0 - 1.5, yl, "Example timeline\n(one approval path)", ha="right", va="center", fontsize=6.3, color="#222222")
fig.savefig("figA1_construction.png", dpi=300); fig.savefig("figA1_construction.pdf")
