"""Audit: recompute every tabulated number of the manuscript from result files
and check that it appears in the tex (whitespace-insensitive).

Direction-C version: Tables 3-4 and D3-D4 come from results_p1/tables (built by
scripts/p1_build_tables.py from results_p1/rolling and the server batch in
results_p1/fqi); Table C1 from the local deterministic runs (results_det);
Tables D1-D2 from the simulation results.
"""
import glob, json, math, re
import numpy as np, pandas as pd

import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")) + "/"  # repository root
R = _ROOT + ""
tex = open(R + "paper_ijgis/IJGIS_manuscript_EN.tex", encoding="utf-8").read()
TEXN = " ".join(tex.split())
checks = []
def chk(name, s): checks.append((name, s, " ".join(s.split()) in TEXN))
def chkre(name, pat, show): checks.append((name, show, re.search(pat, TEXN) is not None))
f3 = lambda x: f"{x:.3f}"
pm = lambda m, s: f"${m:.3f} \\pm {s:.3f}$"
sg = lambda x: f"{x:+.3f}"
TB = R + "results_p1/tables/"

# ---- Table 3 (main, M=20), ladder order
NM = {"random": "Random", "persistence": "Persistence greedy", "supervised": "Supervised scorer",
      "rh_announced": "Ranking on announced plans", "trajectory": "Trajectory-informed greedy",
      "value_based": "Value-based policy", "rh_full": "Rolling horizon, full"}
T3 = pd.read_csv(TB + "main_M20.csv", index_col=0)
for k, n in NM.items():
    r = T3.loc[k]
    tail = f"{pm(r.ratio, r.sd)} & {r.lo:.3f}--{r.hi:.3f} & {r.l_sel:.1f} & {r.l_tim:.1f} & {int(r.n)} \\\\"
    chkre(f"T3 {k}", re.escape(n) + r" & [^&]+ & [^&]+ & " + re.escape(tail), f"{n} & ... & ... & {tail}")
g = T3.ratio
RH = pd.read_csv(R + "results_p1/rolling/rolling_results.csv")
def per_ls(infra="cluster", conf=True, var="base", M=20):
    x = RH[(RH.infra_mode == infra) & (RH.conformity == conf) & (RH.variant == var) & (RH.M == M)]
    return x.pivot_table(index="listing_seed", columns="rule", values="ratio")
P = per_ls(); gi = P.trajectory - P.persistence; gc = P.rh_full - P.trajectory
assert (P.rh_current - P.persistence).abs().max() == 0
chk("info", f"a gain of {g.trajectory - g.persistence:.3f} ({gi.min():.3f} to {gi.max():.3f} across listing draws)")
chk("announced", f"recovers {g.rh_announced - g.persistence:.3f} of the {g.trajectory - g.persistence:.3f} ({g.rh_announced:.3f})")
chk("traj step", f"Knowing the full trajectories adds a further {g.trajectory - g.rh_announced:.3f}")
chk("coord", f"years adds {g.rh_full - g.trajectory:.3f} ({gc.min():.3f} to {gc.max():.3f})")
MGn = pd.read_csv(TB + "margins.csv"); mg = MGn[(MGn.config == "main") & (MGn.M == 20)].iloc[0]
chk("q3 vb-traj", f"({g.value_based - g.trajectory:.3f} higher, and above it in {int(round(mg.share_pos * 30))} of 30 runs")
chk("q3 vb-sup", f"reaches {g.supervised:.3f}; learning the value of the years that follow as well adds {g.value_based - g.supervised:.3f}")
chk("q3 vb-rh", f"remains {g.rh_full - g.value_based:.3f} below the rolling horizon")
for k in ["trajectory", "rh_full"]:
    chk(f"share {k}", f"{100 * T3.loc[k, 'timing_share']:.0f}\\%")
chk("vb", f"reaches {pm(g.value_based, T3.loc['value_based', 'sd'])}")
# ---- Table 4: prescreens, no gate, metro
PS = pd.read_csv(TB + "prescreen.csv", index_col=0)
inf, coo = [], []
for M in [10, 20, 50, 0]:
    x = PS[PS.M == M].ratio; lab = "All" if M == 0 else str(M)
    inf.append(x.trajectory - x.persistence); coo.append(x.rh_full - x.trajectory)
    chk(f"T4 M={lab}", f"{lab} & {x.persistence:.3f} & {x.rh_announced:.3f} & {x.trajectory:.3f} & {x.rh_full:.3f} & {inf[-1]:.3f} & {coo[-1]:.3f} \\\\")
chk("T4 ranges", f"ranges from {min(inf):.3f} to {max(inf):.3f} and that of coordination from {min(coo):.3f} to {max(coo):.3f}")
for M in [10, 20, 50, 0]:
    q = per_ls(M=M); assert ((q.trajectory - q.persistence) > 0).all() and ((q.rh_full - q.trajectory) > 0).all()
NC = pd.read_csv(TB + "noconf.csv", index_col=0).ratio
chk("T4 noconf", f"20, no gate & {NC.persistence:.3f} & {NC.rh_announced:.3f} & {NC.trajectory:.3f} & {NC.rh_full:.3f} & {NC.trajectory - NC.persistence:.3f} & {NC.rh_full - NC.trajectory:.3f}")
chk("noconf text", f"({NC.trajectory - NC.persistence:.3f} and {NC.rh_full - NC.trajectory:.3f})")
MT = pd.read_csv(TB + "metro.csv", index_col=0); m = MT[MT.M == 20].ratio
chk("T4 metro", f"20, metro & {m.persistence:.3f} & {m.rh_announced:.3f} & {m.trajectory:.3f} & {m.rh_full:.3f} & {m.trajectory - m.persistence:.3f} & {m.rh_full - m.trajectory:.3f}")
chk("metro text", f"recovers {m.rh_announced - m.persistence:.3f} of the {m.trajectory - m.persistence:.3f}")
for M in [20, 0]:
    x = MT[MT.M == M]; r_ = x.ratio; lab = "All" if M == 0 else str(M)
    chk(f"D3 metro {lab}", f"{lab} & {r_.persistence:.3f} & {r_.rh_announced:.3f} & {r_.trajectory:.3f} & {pm(r_.value_based, x.loc['value_based', 'sd'])} & {r_.rh_full:.3f} & {r_.trajectory - r_.persistence:.3f} & {r_.rh_full - r_.trajectory:.3f}")
# ---- Table D4 fields
FV = pd.read_csv(TB + "fields.csv", index_col=0)
VN = {"base": "Main configuration", "amp_low": "Amplitude $\\times 0.5$", "amp_high": "Amplitude $\\times 1.5$",
      "win_narrow": "Window 2--4 yr", "win_broad": "Window 6--10 yr", "onset_early": "Onset 0.05--0.35",
      "onset_late": "Onset 0.40--0.75"}
fi, fc = [], []
for v, n in VN.items():
    q = FV[FV.variant == v]; r_ = q.ratio
    chk(f"D4 {v}", f"{n} & {r_.persistence:.3f} & {r_.rh_announced:.3f} & {r_.trajectory:.3f} & {pm(r_.value_based, q.loc['value_based', 'sd'])} & {r_.rh_full:.3f} & {r_.trajectory - r_.persistence:.3f} & {r_.rh_full - r_.trajectory:.3f} \\\\")
    if v != "base": fi.append(r_.trajectory - r_.persistence); fc.append(r_.rh_full - r_.trajectory)
chk("D4 text", f"ranges from {min(fi):.3f} to {max(fi):.3f}, following the amplitude of temporal variation, and that of coordination from {min(fc):.3f} to {max(fc):.3f}")
# ---- Table D5 transfer to unseen fields (results_p2)
Gp = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + "results_p2/gen_s*.csv")) if re.search(r"gen_s\d+\.csv$", f)])
assert sorted(Gp[Gp.rule == "value_based_multi"].seed.unique()) == [0, 1, 2, 3, 4]
B = Gp[Gp.seed == -1].pivot_table(index="eval_field", columns="rule", values="ratio")
Lp = Gp[Gp.rule.str.startswith("value_based")].pivot_table(index=["eval_field", "seed"], columns="rule", values="ratio")
Mn = Lp.groupby(level=0).mean()
test = [f"field{i}" for i in range(1, 11)]
for f in test:
    chk(f"D5 {f}", f"{f[5:]} & {B.loc[f, 'persistence']:.3f} & {B.loc[f, 'rh_announced']:.3f} & {Mn.loc[f, 'value_based_single']:.3f} & {Mn.loc[f, 'value_based_multi']:.3f} & {B.loc[f, 'trajectory']:.3f} & {B.loc[f, 'rh_full']:.3f} \\\\")
mb = B.loc[test].mean(); ml = Mn.loc[test].mean()
chk("D5 mean", f"Mean & {mb.persistence:.3f} & {mb.rh_announced:.3f} & {ml.value_based_single:.3f} & {ml.value_based_multi:.3f} & {mb.trajectory:.3f} & {mb.rh_full:.3f}")
Lt = Lp.loc[test]; annv = B.loc[[f for f, _ in Lt.index], "rh_announced"].values
below = int((Lt.value_based_multi.values < annv).sum())
assert (Lt.value_based_multi.values > B.loc[[f for f, _ in Lt.index], "persistence"].values).all()
chk("D5 text", f"reaches {ml.value_based_multi:.3f} on average, above persistence greedy ({mb.persistence:.3f}) but below ranking on announced plans ({mb.rh_announced:.3f}) in {below} of 50")
chk("D5 single", f"Trained on the main field alone it reaches {ml.value_based_single:.3f}")
chk("ls7 benchmark", f"ranking on announced plans scores {B.loc['main', 'rh_announced']:.3f} on the main field and {mb.rh_announced:.3f} on average")
# ---- Table D7 history in the state (results_p2/gen_s*_h{H}.csv); H=0 is the twenty-field policy above
annt = B.loc[test, "rh_announced"]; HT = {}
for H in [0, 1, 2, 3, 5]:
    if H == 0: X = Lp["value_based_multi"].rename("ratio").reset_index()
    else:
        X = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + f"results_p2/gen_s[0-9]_h{H}.csv"))])
        X = X[X.rule == "value_based_multi"]
    assert sorted(X.seed.unique()) == [0, 1, 2, 3, 4], H
    Xt = X[X.eval_field.isin(test)]; sm = Xt.groupby("seed").ratio.mean()
    wins = int((Xt.ratio.values > annt.loc[Xt.eval_field].values).sum())
    HT[H] = dict(mean=Xt.ratio.mean(), lo=sm.min(), hi=sm.max(), d=Xt.ratio.mean() - mb.rh_announced, wins=wins, main=X[X.eval_field == "main"].ratio.mean())
    assert HT[H]["mean"] > mb.persistence
    h = HT[H]; chk(f"D7 H={H}", f"{H} & {h['mean']:.3f} & {h['lo']:.3f}--{h['hi']:.3f} & ${h['d']:.3f}$ & {h['wins']} & {h['main']:.3f} \\\\")
pos = [H for H in HT if H > 0]
chk("hist text 2", f"rises from {HT[0]['mean']:.3f} to {HT[2]['mean']:.3f} at $H = 2$")
best = max(pos, key=lambda H: HT[H]["mean"]); assert best == 1
chk("hist best", f"one year performs best ({HT[1]['mean']:.3f})")
chk("hist gap", f"by {-max(HT[H]['d'] for H in pos):.3f} to {-min(HT[H]['d'] for H in pos):.3f}")
chk("hist wins", f"none beats it in more than {max(HT[H]['wins'] for H in pos)} of 50 pairs")
chk("D5 appx", f"in {below} of 50 field--seed pairs")
chk("D5 main", f"reaches {Mn.loc['main', 'value_based_multi']:.3f}, against {Mn.loc['main', 'value_based_single']:.3f}")
# ---- Table C1 (local deterministic runs, listing draw 7)
T = pd.read_csv(R + "results_figures/table_main.csv").set_index("key")
for k in ["value", "single"]:
    r = T.loc[k]; chk(f"C1 {k}", f"{pm(r.ratio, r.sd)} & {r.lo:.3f}--{r.hi:.3f} & {r.l_sel:.1f} & {r.l_tim:.1f}")
D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + "results_det/s[0-4]/partB_runs.csv"))])
d = D[D.method == "double_fqi"].sort_values("seed").ratio.values - D[D.method == "single_fqi"].sort_values("seed").ratio.values
m, s = d.mean(), d.std(ddof=1); h = 2.776445105 * s / math.sqrt(5)
chk("C1 paired", f"paired difference of {m:.3f} (SD {s:.3f}, range {d.min():.3f}--{d.max():.3f}, 95\\% confidence interval {m-h:.3f} to {m+h:.3f}")
# ---- Table D1 synthetic, D2 rho
S = pd.read_csv(R + "results_fqi/fqi_synth_summary.csv")
for a in sorted(S.alpha.unique()):
    q = S[S.alpha == a].set_index("method")
    chk(f"D1 alpha={a}", f"{a:.2f} & {f3(q.loc['随机','相对oracle'])} & {f3(q.loc['纯空间排序','相对oracle'])} & "
        f"{f3(q.loc['时间感知贪心','相对oracle'])} & {pm(q.loc['Double-FQI','相对oracle'], np.nan_to_num(q.loc['Double-FQI','标准差']))}")
for f in ["results_rho/rho_summary.csv", "results_rho_tighten/rho_summary.csv"]:
    for _, r in pd.read_csv(R + f).iterrows():
        chk(f"D2 rho={r.rho:.2f}", f"{r.temporal_greedy:.4f} & {r.double_fqi:.4f}")
# ---- timing errors in space (results_p2/timing_space_summary.csv)
TSp = pd.read_csv(R + "results_p2/timing_space_summary.csv")
chk("TS I range", f"lies between ${TSp.I_err.min():.2f}$ and ${TSp.I_err.max():.2f}$")
chk("TS p", f"($p \\ge {TSp.p_err.min():.2f}$")
assert TSp.p_err.min() >= 0.05 and len(TSp) == 30
on = TSp.groupby("rule").share_ontime.mean()
chk("TS ontime", f"starts {100*on.persistence:.0f}\\% of its units in their best year, trajectory-informed greedy {100*on.trajectory:.0f}\\%, the coordinated schedule {100*on.rh_full:.0f}\\%, and the reference schedule {100*on.reference:.0f}\\%")
# ---- Fig 5c zoom window (results_p2/fig5_zoom_window.json)
ZW = json.load(open(R + "results_p2/fig5_zoom_window.json"))
zp = pd.DataFrame(ZW["persistence"]).set_index("unit"); zf = pd.DataFrame(ZW["rh_full"]).set_index("unit")
assert (zp.loc[[454, 471], "best"] == 10).all() and list(zp.loc[[471, 454], "year"]) == [17, 20]
assert list(zf.loc[[454, 471], "year"]) == [11, 12] and list(zf.loc[[499, 541], "year"]) == [1, 1]
assert zp.loc[45, "year"] - zp.loc[45, "best"] == 14 and set(zf.loc[[238, 503], "best"]) == {17}
chk("zoom shares", f"years 17 and 20, when they deliver {100*zp.loc[471,'share']:.0f}\\% and {100*zp.loc[454,'share']:.0f}\\%")
chk("zoom rh", f"in years 11 and 12, at {100*zf.loc[454,'share']:.0f}\\% and {100*zf.loc[471,'share']:.0f}\\%")
chk("zoom mean", f"{zp.share.mean():.2f} of best-year value per unit and the coordinated schedule {zf.share.mean():.2f}")
# ---- announced-information greedy == announced rolling horizon (results_p2/announced_greedy.csv)
AGr = pd.read_csv(R + "results_p2/announced_greedy.csv"); AGr = AGr[AGr.rule == "announced_greedy"]
RHb = RH[RH.variant == "base"]
for (inf, conf, M), g_ in AGr.groupby(["infra_mode", "conformity", "M"]):
    h_ = RHb[(RHb.infra_mode == inf) & (RHb.conformity == conf) & (RHb.M == M) & (RHb.rule == "rh_announced")].set_index("listing_seed").ratio
    assert np.allclose(g_.set_index("listing_seed").ratio.loc[h_.index].values, h_.values, atol=1e-9), (inf, conf, M)
assert len(AGr) == 42
chk("ann greedy text", "all of the 0.090 is the value of information")
# ---- Figure 5 timing efficiency
E = json.load(open(R + "results_figures/fig5_timing_efficiency.json"))
chk("F5 eff", f"{E['static_greedy']['mean']:.2f} for persistence greedy, {E['temporal_greedy']['mean']:.2f} for trajectory-informed greedy, {E['rolling_full']['mean']:.2f} for the rolling horizon with full information, and {E['optimum']['mean']:.2f} for the reference schedule")
bad = [c for c in checks if not c[2]]
print(f"{len(checks)} checks, {len(bad)} not found")
for n, s, _ in bad: print("  MISSING:", n, "->", s[:170])
