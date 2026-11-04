# -*- coding: utf-8 -*-
"""p5_make_supp.py -- writes paper_ijgis/supp_S8.tex (Supplementary Section S8) from
results_p5_spatial/{fields,learned,E3_choice_set_benchmark.csv,meta.json}."""
import glob
import json
import os

import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_p5_spatial")
F = os.path.join(R, "fields")
OUT = os.path.join(R, "..", "paper_ijgis", "supp_S8.tex")
NAMES = {"Fenghuang Sub-District": "Fenghuang", "Gongming Sub-district": "Gongming",
         "Guangming Sub-district": "Guangming", "Matian Sub-district": "Matian",
         "Xinhu Sub-district": "Xinhu", "Yutang Sub-District": "Yutang"}


def sg(v):
    return "$0$" if abs(v) < 5e-5 else f"${v:+.4f}$"


def tab(label, caption, head, rows, cols):
    L = [r"\begin{table}[H]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\footnotesize", rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule", head + r" \\", r"\midrule"]
    L += [" & ".join(r) + r" \\" for r in rows]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def main():
    m = json.load(open(os.path.join(R, "meta.json")))
    sub = ", ".join(f"{NAMES[n]} {c} (quota {q})" for n, c, q in
                    zip(m["names"], m["n_per_subdistrict"], m["balanced_quota"]))
    lsm = pd.read_csv(os.path.join(F, "main_d2_LS.csv")).set_index("beta")
    L = [r"""\section{Spatial constraints, spillovers, and the choice-set benchmark}
\label{sec:spatialext}
\emph{Subdistricts.} Each candidate unit was assigned to one of the six
subdistricts of Guangming by the position of its centroid in the
boundaries of OpenStreetMap (administrative level 8, relations
9692940--9692945, map data: OpenStreetMap contributors, ODbL). The 41
centroids that fall just outside the polygons, where the OpenStreetMap
boundary and the district raster differ, were assigned to the nearest
subdistrict. Units and balanced quotas (largest-remainder shares of the
75 slots): """ + sub + r""". The reference schedule and every
rolling-horizon year were solved as integer programs with HiGHS
(scipy.optimize.milp, relative gap 0); greedy rules and the learned value
correction rank the listed units and skip a unit whose subdistrict limit
is reached.

\emph{Spillover.} Adjacency is a shared cell edge on the 100\,m grid
between two candidate units (1{,}038 pairs; 673 units have at least one
neighbour). Two adjacent units that are both initiated, in years at most
$\Delta$ apart ($\Delta = 2$ unless stated), add
$\beta\,\min(\mathrm{EV}_{i,y_i}, \mathrm{EV}_{j,y_j})$ to the objective.
The best-found schedule is the best of several local-search runs from the
additive reference schedule (five in the main field, three in the metro
field and with $\Delta = 1, 3$, two on each test draw), each of 80{,}000
moves (move a unit to a year with a free slot, swap the years of two
units, or replace a selected unit by an unselected neighbour or
candidate), accepting improvements only. The spillover-aware rules add to
a unit's score $\beta \min(s_u, s_j)$ for each neighbour $j$ initiated
within $\Delta$ years, where $s$ is the rule's own score (for the
neighbour, the score it had when it was initiated), so no rule uses
information it would not otherwise have.

\emph{Learned value correction.} The five ensembles were refitted with
the frozen protocol of the paper (training draws 101--120, seeds 0--4);
without spatial terms they reach $0.899$ on the main field, as in
Table~\ref{tab:main}.

\emph{Choice-set benchmark.} Listing and approval draws share one random
stream in the simulator, so the pools a rule faces depend on its earlier
decisions. The choice-set-matched benchmark is therefore the
full-information rolling horizon run on the same listing draws and the
same prescreen as the practical rules.
"""]
    # S11: E1 main
    e1 = pd.read_csv(os.path.join(F, "main_d2_E1.csv"))
    le = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "learned", "learned_s*.csv")))])
    w = e1.groupby(["mode", "rule"]).ratio.agg(["mean", "std"])
    order = [("persistence_greedy", "Persistence greedy"), ("rh_current", "Rolling horizon, current"),
             ("announced_greedy", "Ranking on announced plans"), ("rh_announced", "Rolling horizon, announced"),
             ("trajectory_greedy", "Trajectory-informed greedy"), ("rh_full", "Rolling horizon, full")]
    rows = [[lab] + [f"${w.loc[(md, k), 'mean']:.3f} \\pm {w.loc[(md, k), 'std']:.3f}$" for md in ("none", "cap1", "balanced")]
            for k, lab in order]
    lw = le[le.exp == "E1"].groupby("mode").ratio.agg(["mean", "std"])
    rows.append(["Learned value correction ($n = 30$)"] + [f"${lw.loc[md, 'mean']:.3f} \\pm {lw.loc[md, 'std']:.3f}$" for md in ("none", "cap1", "balanced")])
    rs = e1.groupby("mode").ref_share.mean()
    L.append(tab("tab:E1", f"Subdistrict limits, main field: ratio to the reference schedule under the same limits, mean $\\pm$ standard deviation over six listing draws (learned value correction: five training seeds $\\times$ six listing draws), $M = 20$. The reference schedule under the per-year cap and the balanced quotas is {rs['cap1']:.4f} and {rs['balanced']:.4f} of the unconstrained one.",
                 "Rule & None & Per-year cap & Balanced quotas", rows, "lccc"))
    # S12: E2 main
    e2 = pd.read_csv(os.path.join(F, "main_d2_E2.csv"))
    g = e2.groupby(["beta", "rule"]).agg(r=("ratio", "mean"), adj=("adjacent", "mean"))
    lg = le[le.exp == "E2"].groupby(["beta", "rule"]).agg(r=("ratio", "mean"), adj=("adjacent", "mean"))
    rules2 = [("persistence_greedy", "Persistence greedy"), ("persistence_greedy_aware", r"\quad spillover-aware"),
              ("announced_greedy", "Ranking on announced plans"), ("announced_greedy_aware", r"\quad spillover-aware"),
              ("trajectory_greedy", "Trajectory-informed greedy"), ("trajectory_greedy_aware", r"\quad spillover-aware"),
              ("rh_full", "Rolling horizon, full (additive)")]
    B = [0.0, 0.1, 0.25, 0.5]
    rows = []
    for k, lab in rules2:
        rows.append([lab] + ["---" if (b, k) not in g.index else f"{g.loc[(b, k), 'r']:.3f} ({g.loc[(b, k), 'adj']:.1f})" for b in B])
    for k, lab in (("learned", "Learned value correction"), ("learned_aware", r"\quad spillover-aware")):
        rows.append([lab, "---"] + [f"{lg.loc[(b, k), 'r']:.3f} ({lg.loc[(b, k), 'adj']:.1f})" for b in B[1:]])
    rows.append(["Best found"] + [f"{lsm.loc[b, 'best']:.0f} ({lsm.loc[b, 'adjacent_best']:.0f})" for b in B])
    rows.append(["Additive reference schedule"] + [f"{lsm.loc[b, 'additive_ref']:.0f}" for b in B])
    L.append(tab("tab:E2", r"Neighbour spillover, main field, $\Delta = 2$: ratio to the best schedule found under the spillover objective, means over six listing draws (learned value correction: 30 runs), $M = 20$; in parentheses, adjacent pairs initiated within two years. Last two rows: objective value of the best-found and of the additive reference schedule.",
                 r"Rule & $\beta = 0$ & $\beta = 0.1$ & $\beta = 0.25$ & $\beta = 0.5$", rows, "lcccc"))
    # S13: E3
    e3 = pd.read_csv(os.path.join(R, "E3_choice_set_benchmark.csv"))
    t = e3.groupby("M")[["rh_full_vs_ref", "persistence_greedy_vs_matched", "rh_announced_vs_matched",
                         "trajectory_greedy_vs_matched"]].mean().reindex(["10", "20", "50", "all"])
    labs = [("rh_full_vs_ref", "Rolling horizon, full / reference"), ("persistence_greedy_vs_matched", "Persistence greedy / benchmark"),
            ("rh_announced_vs_matched", "Ranking on announced plans / benchmark"), ("trajectory_greedy_vs_matched", "Trajectory-informed greedy / benchmark")]
    L.append(tab("tab:E3", "Choice-set-matched benchmark: the full-information rolling horizon on the same listing draws and prescreen, as a share of the reference schedule, and the practical rules as a share of that benchmark; means over six listing draws.",
                 r" & $M = 10$ & $M = 20$ & $M = 50$ & All", [[lab] + [f"{v:.3f}" for v in t[c]] for c, lab in labs], "lcccc"))
    # S14: E1 across fields
    rows = []
    for tag, lab in (("main_d2", "Main field (6 draws)"), ("metro_d2", "Metro field (6 draws)"), ("test_d2", "Test draws 1--10")):
        e = pd.read_csv(os.path.join(F, f"{tag}_E1.csv"))
        w = e.pivot_table(index=["field", "mode", "seed"], columns="rule", values="ratio")
        for md, mlab in (("none", "none"), ("cap1", "per-year cap"), ("balanced", "balanced quotas")):
            x = w.xs(md, level="mode")
            info = (x.trajectory_greedy - x.persistence_greedy).mean()
            coord = (x.rh_full - x.trajectory_greedy).mean()
            d = x.rh_announced - x.announced_greedy
            rows.append([lab if md == "none" else "", mlab, f"{info:.3f}", f"{coord:.3f}",
                         sg(d.mean()), sg(d.min()) + " to " + sg(d.max()), f"{int((d > 1e-9).sum())} of {len(d)}"])
    L.append(tab("tab:E1fields", "Subdistrict limits across fields: value of information (trajectory-informed minus persistence greedy) and of coordination (full-information rolling horizon minus trajectory-informed greedy), and re-planning minus ranking on announced plans (mean, range, and number of draws in which it is positive).",
                 r"Field & Limit & Information & Coordination & Re-plan $-$ rank & Range & Positive", rows, "llccccc"))
    # S15: spatial coordination across fields and windows
    rows = []
    for tag, lab in (("main_d2", r"Main, $\Delta = 2$"), ("main_d1", r"Main, $\Delta = 1$"), ("main_d3", r"Main, $\Delta = 3$"),
                     ("metro_d2", r"Metro, $\Delta = 2$"), ("test_d2", r"Test draws, $\Delta = 2$")):
        ls = pd.read_csv(os.path.join(F, f"{tag}_LS.csv"))
        ls["spread"] = (ls.restart_max - ls.restart_min) / ls.restart_max
        gg = ls.groupby("beta").agg(sc=("spatial_coordination", "mean"), lo=("spatial_coordination", "min"),
                                    hi=("spatial_coordination", "max"), sp=("spread", "max"))
        cells = []
        for b in (0.1, 0.25, 0.5):
            r = gg.loc[b]
            rng = f" ({100 * r.lo:.1f}--{100 * r.hi:.1f})" if tag == "test_d2" else ""
            cells.append(f"{100 * r.sc:.1f}{rng}")
        rows.append([lab] + cells + [f"{100 * gg.loc[0.5, 'sp']:.1f}"])
    L.append(tab("tab:E2fields", r"Value of spatial coordination (best found minus additive reference schedule, \% of best found) across fields and spillover windows; test draws: mean (range) over ten draws. Last column: largest spread between local-search restarts at $\beta = 0.5$ (\% of best found).",
                 r" & $\beta = 0.1$ & $\beta = 0.25$ & $\beta = 0.5$ & Restart spread", rows, "lcccc"))
    open(OUT, "w", encoding="utf8").write("\n".join(L))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
