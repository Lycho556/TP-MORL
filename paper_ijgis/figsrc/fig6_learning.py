# Figure 6 — can learning recover the value of timing, and does it transfer?
# (a) scenario the policy was trained on: six listing draws (learned rules: x five seeds), M = 20;
# (b) ten unseen scenario draws, listing draw 7: one point per draw (learned rules: mean of five seeds);
# (c) look-back of 0-5 years, same ten unseen draws: mean of 50 draw-seed pairs, bar = range of seed means.
# Data: results_p1/rolling, results_p1/fqi/main_M20_s*.csv, results_p2/gen_s*.csv.
# Run inside a kernel where apply_figure_style() (figure-style skill) is defined.
import glob
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt

R = "/Users/user/Desktop/TP-MORL/"
OUT = R + "paper_ijgis/figures/"
apply_figure_style(sizes=(8, 7, 6))
C = dict(persistence="#E69F00", announced="#F0A875", supervised="#CC79A7",
         trajectory="#0072B2", value="#56B4E9")
rng = np.random.default_rng(1)

RH = pd.read_csv(R + "results_p1/rolling/rolling_results.csv")
rh = RH[(RH.infra_mode == "cluster") & (RH.conformity) & (RH.variant == "base") & (RH.M == 20)] \
    .pivot_table(index="listing_seed", columns="rule", values="ratio")
F = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + "results_p1/fqi/main_M20_s*.csv"))])
vb = F[F.rule == "value_based"].ratio.values; sup = F[F.rule == "supervised"].ratio.values
assert len(vb) == 30 and len(sup) == 30

G0 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + "results_p2/gen_s?.csv"))])
base = G0[G0.seed == -1].pivot_table(index="eval_field", columns="rule", values="ratio")
test = [f"field{i}" for i in range(1, 11)]
one = G0[G0.rule == "value_based_single"].pivot_table(index="eval_field", columns="seed", values="ratio").loc[test]
twenty = {0: G0[G0.rule == "value_based_multi"].pivot_table(index="eval_field", columns="seed", values="ratio").loc[test]}
for H in (1, 2, 3, 5):
    X = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + f"results_p2/gen_s?_h{H}.csv"))])
    twenty[H] = X.pivot_table(index="eval_field", columns="seed", values="ratio").loc[test]
    assert twenty[H].shape == (10, 5)

fig = plt.figure(figsize=(6.9, 2.9))
axa = fig.add_axes([0.075, 0.23, 0.27, 0.59])
axb = fig.add_axes([0.40, 0.23, 0.27, 0.59])
axc = fig.add_axes([0.73, 0.23, 0.26, 0.59])


def strip(ax, x, vals, col, filled=True, ms=3.2):
    j = rng.uniform(-0.16, 0.16, len(vals))
    ax.scatter(np.full(len(vals), x) + j, vals, s=ms ** 2, color=col if filled else "white",
               edgecolor=col, lw=0.7, zorder=2, alpha=0.9)
    ax.plot([x - 0.28, x + 0.28], [np.mean(vals)] * 2, color="black", lw=1.4, zorder=3)


# (a)
cats = [("Persis-\ntence", rh.persistence.values, C["persistence"]),
        ("Super-\nvised", sup, C["supervised"]),
        ("Trajec-\ntory", rh.trajectory.values, C["trajectory"]),
        ("Value-\nbased", vb, C["value"])]
for i, (lab, v, c) in enumerate(cats):
    strip(axa, i, v, c)
axa.text(3, 0.922, f"{np.mean(vb):.3f}", ha="center", va="bottom", fontsize=6)
axa.text(2, 0.922, f"{rh.trajectory.mean():.3f}", ha="center", va="bottom", fontsize=6)
axa.set_xticks(range(len(cats))); axa.set_xticklabels([c[0] for c in cats], fontsize=6)
axa.set_ylim(0.74, 0.94); axa.set_xlim(-0.6, 3.6)
axa.set_ylabel("Ratio to the reference schedule")
axa.text(-0.27, 1.06, "a", transform=axa.transAxes, fontsize=9, fontweight="bold", va="bottom")
axa.text(-0.17, 1.06, "Scenario it was trained on", transform=axa.transAxes, fontsize=7.5, va="bottom")

# (b)
cb = [("Announ-\nced", base.loc[test, "rh_announced"].values, C["announced"], True),
      ("Trajec-\ntory", base.loc[test, "trajectory"].values, C["trajectory"], True),
      ("Value-\nbased,\n1 field", one.mean(1).values, C["value"], False),
      ("Value-\nbased,\n20 fields", twenty[0].mean(1).values, C["value"], True)]
for i, (lab, v, c, f) in enumerate(cb):
    strip(axb, i, v, c, filled=f)
for i in (0, 3):
    axb.text(i, 0.94, f"{cb[i][1].mean():.3f}", ha="center", va="bottom", fontsize=6)
axb.set_xticks(range(len(cb))); axb.set_xticklabels([c[0] for c in cb], fontsize=6)
axb.set_ylim(0.74, 0.96); axb.set_xlim(-0.6, 3.6)
axb.text(-0.12, 1.06, "b", transform=axb.transAxes, fontsize=9, fontweight="bold", va="bottom")
axb.text(-0.02, 1.06, "Ten scenarios it had not seen", transform=axb.transAxes, fontsize=7.5, va="bottom")

# (c)
Hs = [0, 1, 2, 3, 5]
mean = [twenty[h].values.mean() for h in Hs]
lo = [twenty[h].mean(0).min() for h in Hs]; hi = [twenty[h].mean(0).max() for h in Hs]
annm = base.loc[test, "rh_announced"].mean(); perm = base.loc[test, "persistence"].mean()
trm = base.loc[test, "trajectory"].mean()
for yv, c, lab, ls in ((trm, C["trajectory"], "trajectory-informed", "--"),
                       (annm, C["announced"], "announced plans", "-"),
                       (perm, C["persistence"], "persistence", "-")):
    axc.axhline(yv, color=c, lw=1.1, ls=ls, zorder=1)
    axc.text(-0.3, yv + 0.002, lab, color=c, fontsize=6, va="bottom", ha="left")
axc.vlines(Hs, lo, hi, color=C["value"], lw=1.6, zorder=2)
axc.plot(Hs, mean, "o-", color=C["value"], mec="black", mew=0.5, ms=4.5, lw=1.0, zorder=3)
axc.set_xticks(Hs); axc.set_xlim(-0.4, 5.4); axc.set_ylim(0.74, 0.92)
axc.set_xlabel("Years of history in the state")
axc.text(-0.14, 1.06, "c", transform=axc.transAxes, fontsize=9, fontweight="bold", va="bottom")
axc.text(-0.04, 1.06, "History helps, but not enough", transform=axc.transAxes, fontsize=7.5, va="bottom")
for ax in (axa, axb, axc):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig(OUT + "fig6_learning.pdf")
fig.savefig(OUT + "fig6_learning.png", dpi=300)
