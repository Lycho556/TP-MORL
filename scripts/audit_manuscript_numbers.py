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
tex += "\n" + open(R + "paper_ijgis/IJGIS_supplementary.tex", encoding="utf-8").read()  # supplementary material
TEXN = " ".join(tex.split())
checks = []
# checks retired in the P3 revision: the text they verified (multi-step policy in the main text,
# old Tables 3-4, Table B1) was removed or replaced; the multi-step results remain in the supplement.
RETIRED = set(['T3 supervised', 'T3 value_based', 'q3 vb-traj', 'q3 paired', 'q3 vb-sup', 'q3 vb-rh', 'vb', 'D3 metro 20', 'D3 metro All', 'D5 text', 'D5 single', 'ls7 benchmark', 'hist text 2', 'C1 value', 'C1 single', 'C1 paired', 'P2F text', 'P2F fields', 'P2F main', 'P2F ci', 'P2F traj gap', 'infolearn cur', 'infolearn ann', 'infolearn traj', 'infolearn full', 'infolearn text'])
def chk(name, s):
    if name in RETIRED: return
    checks.append((name, s, " ".join(s.split()) in TEXN))
def chkre(name, pat, show):
    if name in RETIRED: return
    checks.append((name, show, re.search(pat, TEXN) is not None))
f3 = lambda x: f"{x:.3f}"
pm = lambda m, s: f"${m:.3f} \\pm {s:.3f}$"
sg = lambda x: f"{x:+.3f}"
TB = R + "results_p1/tables/"
# registered transfer test: v2 (corrected FQI backup) once complete, else the first run
PF = R + ("results_p2_formal_v2/" if _os.path.exists(R + "results_p2_formal_v2/SUMMARY.md") else "results_p2_formal/")
_HF = int(open(PF + "stage1_selection.txt").read().split()[-1]); _SFX = "" if _HF == 0 else f"_h{_HF}"

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
chk("q3 vb-traj", f"it is {g.value_based - g.trajectory:.3f} higher on average (standard deviation of the run differences 0.012) and above the rule in {int(round(mg.share_pos * 30))} of 30 runs")
_Fq = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + "results_p1/fqi/main_M20_s*.csv"))]); _Fq = _Fq[_Fq.rule == "value_based"]
_dq = _Fq.ratio.values - _Fq.listing_seed.map(P.trajectory).values
chk("q3 paired", f"0.002 higher on average (standard deviation of the run differences {np.std(_dq, ddof=1):.3f}) and above the rule in {int((_dq > 0).sum())} of 30 runs")
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
    chk(f"D4 {v}", f"{n} & {r_.persistence:.3f} & {r_.rh_announced:.3f} & {r_.trajectory:.3f} & {r_.rh_full:.3f} & {r_.trajectory - r_.persistence:.3f} & {r_.rh_full - r_.trajectory:.3f} \\\\")
    if v != "base": fi.append(r_.trajectory - r_.persistence); fc.append(r_.rh_full - r_.trajectory)
chk("D4 text", f"ranges from {min(fi):.3f} to {max(fi):.3f}, following the amplitude of temporal variation, and that of coordination from {min(fc):.3f} to {max(fc):.3f}")
# ---- Table D5 transfer to unseen fields (results_p2)
Gp = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + "results_p2/gen_s*.csv")) if re.search(r"gen_s\d+\.csv$", f)])
assert sorted(Gp[Gp.rule == "value_based_multi"].seed.unique()) == [0, 1, 2, 3, 4]
B = Gp[Gp.seed == -1].pivot_table(index="eval_field", columns="rule", values="ratio")
Lp = Gp[Gp.rule.str.startswith("value_based")].pivot_table(index=["eval_field", "seed"], columns="rule", values="ratio")
Mn = Lp.groupby(level=0).mean()
test = [f"field{i}" for i in range(1, 11)]
_XF = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(PF + "test/gen_s?" + _SFX + ".csv"))])
_FM = _XF[_XF.rule == "value_based_multi"].pivot_table(index="eval_field", columns="seed", values="ratio").mean(axis=1)
for f in test:
    chk(f"D5 {f}", f"{f[5:]} & {B.loc[f, 'persistence']:.3f} & {B.loc[f, 'rh_announced']:.3f} & {Mn.loc[f, 'value_based_single']:.3f} & {Mn.loc[f, 'value_based_multi']:.3f} & {_FM.loc[f]:.3f} & {B.loc[f, 'trajectory']:.3f} & {B.loc[f, 'rh_full']:.3f} \\\\")
mb = B.loc[test].mean(); ml = Mn.loc[test].mean()
chk("D5 mean", f"Mean & {mb.persistence:.3f} & {mb.rh_announced:.3f} & {ml.value_based_single:.3f} & {ml.value_based_multi:.3f} & {_FM.loc[test].mean():.3f} & {mb.trajectory:.3f} & {mb.rh_full:.3f}")
Lt = Lp.loc[test]; annv = B.loc[[f for f, _ in Lt.index], "rh_announced"].values
below = int((Lt.value_based_multi.values < annv).sum())
assert (Lt.value_based_multi.values > B.loc[[f for f, _ in Lt.index], "persistence"].values).all()
chk("D5 text", f"trained on twenty draws {ml.value_based_multi:.3f}, below ranking on announced plans in {below} of 50 pairs")
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
chk("hist text 2", f"raise the mean to between {min(HT[H]['mean'] for H in pos):.3f} and {max(HT[H]['mean'] for H in pos):.3f} for $H = 1$, 2, 3, and 5")
best = max(pos, key=lambda H: HT[H]["mean"]); assert best == 1
assert all(HT[H]['d'] < 0 for H in pos)


chk("D5 appx", f"in {below}\nof 50 draw--seed combinations")
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
        f"{f3(q.loc['时间感知贪心','相对oracle'])} & 1.000 \\\\")
# Table D2 (rho sweep of the multi-step policy) removed from the paper in revision 25
# ---- timing errors in space (results_p2/timing_space_summary.csv)
TSp = pd.read_csv(R + "results_p2/timing_space_summary.csv")
chk("TS I range", f"lies between ${TSp.I_err.min():.2f}$ and ${TSp.I_err.max():.2f}$")
chk("TS p", f"($p \\ge {TSp.p_err.min():.2f}$")
assert TSp.p_err.min() >= 0.05 and len(TSp) == 30
on = TSp.groupby("rule").share_ontime.mean()
chk("TS ontime", f"persistence greedy initiates {100*on.persistence:.0f}\\% of its units in their best year, trajectory-informed greedy {100*on.trajectory:.0f}\\%, the coordinated schedule {100*on.rh_full:.0f}\\%, and the reference schedule {100*on.reference:.0f}\\%")
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
assert f"{E['rl']['mean']:.2f}" == f"{E['temporal_greedy']['mean']:.2f}"
_rl7 = json.load(open(R + "results_p3_metro/rl_schedule_main_ls7_s0.json"))["init"]; _tg7 = json.load(open(R + "results_figures/schedules.json"))["temporal_greedy"]["init"]
chk("F4 rl overlap", f"selects {len(set(_rl7) & set(_tg7))} of the {len(_tg7)} units that trajectory-informed greedy\nselects, without seeing the trajectories, and initiates {sum(_rl7[u] == _tg7[u] for u in _rl7 if u in _tg7)} of them in")
chk("F4 rl ratio", f"{json.load(open(R + 'results_p3_metro/rl_schedule_main_ls7_s0.json'))['ratio']:.4f}"[:5])
chk("F5 eff", f"{E['static_greedy']['mean']:.2f} for persistence greedy, {E['temporal_greedy']['mean']:.2f} for both\nreinforcement learning and trajectory-informed greedy, {E['rolling_full']['mean']:.2f} for the\nrolling horizon with full information, and {E['optimum']['mean']:.2f} for the reference\nschedule")

# ---- registered transfer test with 100 training draws (results_p2_formal, server)
from scipy import stats as _st
_tf = [f"field{i}" for i in range(1, 11)]; _vf = [f"field{i}" for i in range(201, 211)]
_T = pd.concat([pd.read_csv(f) for f in glob.glob(PF + "test/gen_s?" + _SFX + ".csv")])
_b = _T[_T.seed == -1].pivot_table(index="eval_field", columns="rule", values="ratio")
_p = _T[_T.rule == "value_based_multi"].pivot_table(index="eval_field", columns="seed", values="ratio")
assert _p.loc[_tf].shape == (10, 5)
_ann = _b.loc[_tf, "rh_announced"]; _d = _p.loc[_tf].mean(axis=1) - _ann
_hw = _st.t.ppf(0.975, 9) * _d.std(ddof=1) / np.sqrt(10)
_pairs = int((_p.loc[_tf].values > _ann.values[:, None]).sum())
_sm = _p.loc[_tf].mean(axis=0)
chk("P2F row", f"1, 100 draws & {f3(_p.loc[_tf].values.mean())} & {_sm.min():.3f}--{_sm.max():.3f} & ${_d.mean():+.3f}$ & {_pairs} & {f3(_p.loc['main'].mean())} \\\\")
chk("P2F text", f"It reaches {f3(_p.loc[_tf].values.mean())}, against {f3(_ann.mean())} for ranking on announced plans and {f3(_b.loc[_tf,'persistence'].mean())} for persistence greedy")
chk("P2F fields", f"and on {int((_d > 0).sum())} of 10 draws")
_V1 = pd.concat([pd.read_csv(f) for f in glob.glob(R + "results_p2_formal/test/gen_s?_h1.csv")]); _q1 = _V1[_V1.rule == "value_based_multi"].pivot_table(index="eval_field", columns="seed", values="ratio")
if PF.endswith("_v2/"):
    chk("P2F v1", f"reached {f3(_q1.loc[_tf].values.mean())}")
    chk("P2F v1-v2", f"which is {(_q1.loc[_tf] - _p.loc[_tf]).values.mean():.3f} lower on average")
chk("P2F main", f"on the main field it reaches {f3(_p.loc['main'].mean())} against {f3(_b.loc['main','rh_announced'])}")
_rng = (_p.loc[_tf].max(axis=1) - _p.loc[_tf].min(axis=1)).median()
chk("P2F seed range", f"varies by a median of {_rng:.3f} per draw")
if PF.endswith("results_p2_formal/"):
  _Vs = pd.read_csv(PF + "hval_h1/gen_s0_h1.csv"); _Vp = pd.read_csv(R + "results_p2_val/D/gen_s0_h1.csv")
  _vs = _Vs[_Vs.rule == "value_based_multi"].set_index("eval_field").ratio.loc[_vf]; _vp = _Vp[_Vp.rule == "value_based_multi"].set_index("eval_field").ratio.loc[_vf]
  assert abs(_vs.mean() - _vp.mean()) < 1e-3 and f3(_vs.mean()) == f3(_vp.mean())
  chk("P2F pilot", f"(0.877, difference below 0.001) but per-draw differences of up to {(_vs - _vp).abs().max():.3f}")
# ---- information ablation and announcement lead (results_p2_diag, scripts/diag_*.py)
_D = R + "results_p2_diag/"
_wv = pd.read_csv(_D + "waiting_value.csv"); _ld = pd.read_csv(_D + "announce_lead.csv")
_bs = _wv[(_wv["info"] == "announced (rule in paper)") & (_wv.use == "rank")].set_index("field").ratio
assert np.allclose(_bs.values, _b.loc[[f"field{i}" for i in _bs.index], "rh_announced"].values, atol=1e-9)
_chk_all = _wv[_wv["info"] == "true P/B/C and true infra"]
assert np.allclose(_chk_all[_chk_all.use == "rank"].set_index("field").ratio.values, _b.loc[[f"field{i}" for i in _bs.index], "trajectory"].values, atol=1e-9)
assert np.allclose(_chk_all[_chk_all.use == "rolling"].set_index("field").ratio.values, _b.loc[[f"field{i}" for i in _bs.index], "rh_full"].values, atol=1e-9)
def _cell(g):
    g = g.set_index("field").ratio; df = g - _bs
    dd = "---" if abs(df.mean()) < 1e-12 else f"${df.mean():+.3f}$ ({int((df > 1e-9).sum())})"
    return f"{g.mean():.3f} ({g.std():.3f}) & {dd}"
def _row(lab, sub):
    chk(f"abl {lab}", f"{lab} & {_cell(sub[sub.use == 'rank'])} & {_cell(sub[sub.use == 'rolling'])} \\\\")
_row("+ true $P$, $B$, $C$", _wv[_wv["info"] == "+ true P/B/C, announced infra"])
_row("+ true infrastructure", _wv[_wv["info"] == "frozen P/B/C, + true infra"])
_row("All trajectories true", _chk_all)
for _L in (8, 12):
    _row(f"Announced, {_L}-year lead", _ld[(_ld.lead == _L) & (~_ld.pbc_true)])
_m = lambda i, u: _wv[(_wv["info"] == i) & (_wv.use == u)].ratio.mean()
chk("abl text pbc", f"leaves ranking almost unchanged ({f3(_bs.mean())} to {f3(_m('+ true P/B/C, announced infra','rank'))}) and raises the coordinated schedule to {f3(_m('+ true P/B/C, announced infra','rolling'))}")
chk("abl text inf", f"raises ranking to {f3(_m('frozen P/B/C, + true infra','rank'))} and the coordinated schedule to {f3(_m('frozen P/B/C, + true infra','rolling'))}, against {f3(_m('true P/B/C and true infra','rolling'))}")
chk("abl lead8", f"reaches {f3(_ld[(_ld.lead == 8) & (~_ld.pbc_true) & (_ld.use == 'rank')].ratio.mean())} on the same draws, against {f3(_bs.mean())}")
_sg = lambda x: f"${x:.3f}$" if x < 0 else f"{x:.3f}"
chk("P2F ci", f"a difference of {_sg(_d.mean())} (95\\% interval over draws {_sg(_d.mean()-_hw)} to {_sg(_d.mean()+_hw)}), with the policy above the rule in {_pairs} of 50")
chk("P2F traj gap", f"remains {_b.loc[_tf,'trajectory'].mean()-_p.loc[_tf].values.mean():.3f} below trajectory-informed")
chk("P2F ann main", f"0.871 ({f3(_b.loc['main','rh_announced'])} on the main field)")
_vm = {}
for _h in (0, 1, 2):
    _V = pd.read_csv(PF + f"hval_h{_h}/gen_s0" + ("" if _h == 0 else f"_h{_h}") + ".csv")
    _vm[_h] = _V[_V.rule == "value_based_multi"].set_index("eval_field").ratio.loc[_vf].mean()
    if _h == 0:
        _vann = _V[_V.seed == -1].pivot_table(index="eval_field", columns="rule", values="ratio").loc[_vf, "rh_announced"].mean()
chk("P2F val", f"score {f3(_vm[0])}, {f3(_vm[1])}, and {f3(_vm[2])}, against {f3(_vann)} for ranking on announced plans")
chk("P2F diff app", f"is {_sg(_d.mean())} with a 95\\% $t$-interval of {_sg(_d.mean()-_hw)} to {_sg(_d.mean()+_hw)}")

# ---- Proposition 1 check and timing corridors (scripts/p2_equivalence_check.py, p3_timing_space_metro.py)
_EQ = pd.read_csv(R + "results_p2/equivalence_check.csv"); _SP = pd.read_csv(R + "results_p2/separability_check.csv")
assert _SP.max_rel_dev.max() < 1e-12 and _EQ.identical.all() and (_EQ.groupby("mode").size() == 42).all()
chk("prop 42", "(42 of 42 configurations for each forecast;")
_SN = pd.read_csv(R + "results_p2/start_now_by_distance.csv").set_index("field")
_m, _c = _SN.loc["metro"], _SN.loc["cluster"]
chk("corr metro", f"{100*_m.share_now_near:.0f}\\% of the units are best\ninitiated now, against {100*_m.share_now_far:.0f}\\% farther away (difference {_m['diff']:.2f}, 95\\% bootstrap interval {_m.lo:.2f} to {_m.hi:.2f}")
assert _m.perm_p < 0.001
chk("corr placebo", f"the shares are {100*_c.share_now_near:.0f}\\% and {100*_c.share_now_far:.0f}\\% ($p = {_c.perm_p:.2f}$)")
_TB = pd.read_csv(R + "results_p2/timing_stake_by_distance.csv"); _tm = _TB[_TB.field == "metro"].mean_rel_stake
chk("corr stake", f"is {100*_tm.min():.0f}--{100*_tm.max():.0f}\\% of the best-year value in every distance band")

# ---- start years in the controlled simulation (results_fqi/fqi_synth_runs.csv) and Table infolearn
_SR = pd.read_csv(R + "results_fqi/fqi_synth_runs.csv").groupby(["alpha", "method"])[["when_mae", "when_hit1"]].mean()
for _a in sorted(_SR.index.get_level_values(0).unique()):
    _sp, _tg = (_SR.loc[(_a, k)] for k in ("纯空间排序", "时间感知贪心"))
    chk(f"synthyears {_a}", f"{_a:.2f} & {_sp.when_mae:.2f} & {_tg.when_mae:.2f} & {100*_sp.when_hit1:.0f} & {100*_tg.when_hit1:.0f} \\\\")
_spm = _SR.xs("纯空间排序", level=1).when_mae; _tgm = _SR.xs("时间感知贪心", level=1).when_mae
chk("synthyears text", f"spatial ranking initiates a parcel {_spm.mean():.1f} years from its optimal year on average, against {_tgm.min():.2f} to {_tgm.max():.1f} years")
_U = lambda r: f3(_b.loc[_tf, r].mean())
chk("infolearn cur", f"Current conditions & {f3(g.persistence)} & --- & {_U('persistence')} & --- \\\\")
chk("infolearn ann", f"Announced plans & {f3(g.rh_announced)} & ${f3(g.value_based)}^{{a}}$ & {f3(_ann.mean())} & ${f3(_p.loc[_tf].values.mean())}^{{b}}$ \\\\")
chk("infolearn traj", f"True trajectories & {f3(g.trajectory)} & --- & {_U('trajectory')} & --- \\\\")
chk("infolearn full", f"True trajectories and re-planning & {f3(g.rh_full)} & --- & {_U('rh_full')} & --- \\\\")
chk("infolearn text", f"({f3(g.value_based)} against {f3(g.trajectory)}), but stays at the level of the announced-information rule in unseen scenarios ({f3(_p.loc[_tf].values.mean())} against {f3(_ann.mean())})")

# ---- P3: reinforcement learning (registered test results_p3_formal; main/metro results_p3_metro; selection results_p3_explore)
import glob as _g
from scipy import stats as _st
_fo = pd.concat([pd.read_csv(f) for f in sorted(_g.glob(R + "results_p3_formal/test_s*.csv"))])
_tt = [f"field{i}" for i in range(1, 11)]
_FB = _fo[_fo.seed == -1].pivot_table(index="eval_field", columns="rule", values="ratio").loc[_tt]
_FP = _fo[_fo.rule == "residual_one_step"].pivot_table(index="eval_field", columns="seed", values="ratio").loc[_tt]
_fm = _FP.mean(axis=1); _fd = _fm - _FB.rank_announced; _fh = _st.t.ppf(0.975, 9) * _fd.std(ddof=1) / np.sqrt(10)
chk("P3 test", f"On the ten test draws the policy reaches {f3(_fm.mean())}, against {f3(_FB.rank_announced.mean())} for ranking on announced plans and {f3(_FB.trajectory.mean())} for trajectory-informed greedy")
_tp = _st.ttest_1samp(_fd, 0).pvalue; _wp = _st.wilcoxon(_fd).pvalue
chk("P3 diff", f"it is above the rule on all ten, by {f3(_fd.mean())} on\naverage (95\\% $t$-interval {f3(_fd.mean()-_fh)} to {f3(_fd.mean()+_fh)}; paired $t$-test\n$p = {_tp:.1e}".replace("e-0", " \\times 10^{-").replace("10^{-4", "10^{-4}") + f"$, Wilcoxon signed-rank $p = {_wp:.3f}$)")
assert (_fd > 0).all() and (_FP.sub(_FB.rank_announced, axis=0) > 0).values.all()
chk("P3 seeds", f"seed means lie\nbetween {f3(_FP.mean(axis=0).min())} and {f3(_FP.mean(axis=0).max())}")
chk("P3 gap", f"Of the {f3((_FB.trajectory - _FB.rank_announced).mean())} that separates the announced-plan rule from the rule told the true trajectories, the policy recovers {f3(_fd.mean())}")
chk("P3 rhfull", f"remains {f3(_FB.rh_full.mean() - _fm.mean())} higher")
_mt = {}
for _k in ("main", "metro"):
    _d = pd.concat([pd.read_csv(f) for f in sorted(_g.glob(R + f"results_p3_metro/{_k}_s*.csv"))])
    _mt[_k] = (_d[_d.seed == -1].pivot_table(index="listing_seed", columns="rule", values="ratio"), _d[_d.rule == "residual_one_step"])
_mb, _mr = _mt["main"]
chk("P3 T3 row", f"\\textbf{{Reinforcement learning}}$^{{\\dagger}}$ & announced plans; correction learned in simulation & no & $\\mathbf{{{f3(_mr.ratio.mean())} \\pm {f3(_mr.ratio.std(ddof=1))}}}$ & {f3(_mr.ratio.min())}--{f3(_mr.ratio.max())} & {_mr.loss_selection.mean():.1f} & {_mr.loss_timing.mean():.1f} & {len(_mr)}")
chk("P3 main text", f"On the main field it reaches {f3(_mr.ratio.mean())} over 30 runs, against {f3(_mb.rank_announced.mean())}")
_xb, _xr = _mt["metro"]
chk("P3 metro text", f"ranking on announced plans reaches {f3(_xb.rank_announced.mean())}, against {f3(_xb.trajectory.mean())} for trajectory-informed greedy. There the policy, trained on the scenario fields, reaches {f3(_xr.ratio.mean())} and falls {f3(_xb.rank_announced.mean() - _xr.ratio.mean())} below the rule")
chk("P3 S2 row", f"20 & {f3(_xb.persistence.mean())} & {f3(_xb.rank_announced.mean())} & ${f3(_xr.ratio.mean())} \\pm {f3(_xr.ratio.std(ddof=1))}$ & {f3(_xb.trajectory.mean())} & {f3(_xb.rh_full.mean())}")
_sh = pd.concat([pd.read_csv(f) for f in sorted(_g.glob(R + "results_p3_shift/shift_s*.csv"))])
chk("P3 shift", f"it reaches {f3(_sh[_sh.set == 'metro'].ratio.mean())}, and {f3(_sh[_sh.set == 'test'].ratio.mean())} on the")
chk("P3 T4 ann", f"Ranking on announced plans & announced plans & {f3(_FB.rank_announced.mean())} & {f3(_mb.rank_announced.mean())} \\\\")
chk("P3 T4 rl", f"\\textbf{{Reinforcement learning}}$^{{\\dagger}}$ & announced plans; learned correction & \\textbf{{{f3(_fm.mean())}}} & \\textbf{{{f3(_mr.ratio.mean())}}} \\\\")
chk("P3 T4 pers", f"Persistence greedy & current conditions & {f3(_FB.persistence.mean())} & {f3(_mb.persistence.mean())} \\\\")
chk("P3 T4 traj", f"Trajectory-informed greedy & true trajectories & {f3(_FB.trajectory.mean())} & {f3(_mb.trajectory.mean())} \\\\")
chk("P3 T4 rh", f"Rolling horizon, full & true trajectories; re-planning & {f3(_FB.rh_full.mean())} & {f3(_mb.rh_full.mean())} \\\\")
# Table 4 summary rows and Table 3 footnote: RL minus ranking on announced plans
_md = _mr.groupby("listing_seed").ratio.mean() - _mb.rank_announced; _mh = _st.t.ppf(0.975, len(_md) - 1) * _md.std(ddof=1) / np.sqrt(len(_md))
_mabove = int((_mr.ratio.values > _mb.rank_announced.reindex(_mr.listing_seed).values).sum())
_fabove = int((_FP.sub(_FB.rank_announced, axis=0) > 0).values.sum())
chk("P3 T4 diff", f"Reinforcement learning minus ranking on announced plans}} & ${_fd.mean():+.3f}$ & ${_md.mean():+.3f}$ \\\\")
chk("P3 T4 ci", f"& {f3(_fd.mean()-_fh)}--{f3(_fd.mean()+_fh)} & {f3(_md.mean()-_mh)}--{f3(_md.mean()+_mh)} \\\\")
chk("P3 T4 above", f"Runs above ranking on announced plans}} & {_fabove} of {_FP.size} & {_mabove} of {len(_mr)} \\\\")
chk("P3 T3 foot", "Its difference to ranking on announced plans is given in Table~\\ref{tab:infolearn}")
assert _mabove == len(_mr)
_sel = pd.read_csv(R + "results_p3_explore/residual_steps_summary.csv")
_one = _sel[_sel.method == "5 residual, one-step target"].iloc[0]
chk("P3 S7 selected", f"One-step, correction to the announced forecast (selected) & 5 & {f3(_one['mean'])} & ${_one.diff_vs_rule:+.3f}$")

# metro "All" row (rule-based columns; RL not run at M = All) and every row of Table S7
_x0 = MT[MT.M == 0].ratio
chk("P3 S2 All", f"All & {f3(_x0.persistence)} & {f3(_x0.rh_announced)} & --- & {f3(_x0.trajectory)} & {f3(_x0.rh_full)} & {f3(_x0.trajectory - _x0.persistence)} & {f3(_x0.rh_full - _x0.trajectory)} \\\\")
for _i, _r in _sel.iterrows():
    _lo, _hi = [float(x) for x in _r.ci95.replace(" to ", " ").split()]
    chk(f"P3 S7 row {_i}", f"& {int(_r.seeds)} & {f3(_r['mean'])} & ${_r.diff_vs_rule:+.3f}$ & ${_lo:+.3f}$ to ${_hi:+.3f}$ & {int(_r.fields_above)} \\\\")

bad = [c for c in checks if not c[2]]
print(f"{len(checks)} checks, {len(bad)} not found")
for n, s, _ in bad: print("  MISSING:", n, "->", s[:170])
