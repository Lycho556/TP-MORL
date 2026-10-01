# -*- coding: utf-8 -*-
"""p1_spatial.py — spatial statistics and the seven inherited spatial objectives
of renewal schedules (P1, spatial track).

Usage (from repo root):
    PYTHONPATH=src TPMORL_NJOBS=1 python scripts/p1_spatial.py
        [--schedules results_figures/schedules.json results_p1/rolling/schedules_*.json]
        [--out results_p1/spatial] [--infra_mode cluster] [--no_fig]

Library use:
    import p1_spatial as SP
    ctx = SP.build_context()                       # main config, M=20
    sch = SP.load_schedules("results_p1/rolling/schedules_x.json")
    st  = SP.spatial_stats(sch, ctx)               # DataFrame
    ob  = SP.spatial_objectives(sch, ctx)          # DataFrame

Schedule files: {rule: {unit: year}} or {rule: {'init': {unit: year}, ...}};
non-dict top-level entries (e.g. 'oracle_value') are ignored. Years are 0-based
decision years (0..T-1). Unit keys are env unit indices (0..716).

Methodological choices (stated in the outputs and in the report):
  * Unit location = mean of the EPSG:2383 cell centres of the unit's 100 m cells.
  * Distance-band weights d_ij <= band (i != j), row-standardised; units with no
    neighbour inside the band (islands) keep a zero row and are counted in S0
    only through non-zero rows (PySAL convention). Moran's I pseudo p-value is
    PySAL's folded p_sim with 999 permutations (fixed seed); BB join count uses
    the binary (non-standardised) band weights, one-sided upper-tail p.
  * Metro distance = unit_station_distance.csv d_nearest = minimum over the
    unit's 100 m cell centres of the distance to any station (all 28 stations of
    L6, L6 branch, L13 north; full network; same table the metro field uses).
    This is a nearest-edge distance, on median ~155 m shorter than the
    centroid distance (checked in build_context, geo_err). A second variant uses only stations
    announced by the unit's initiation year (announce - model_year0).
  * Expected land use follows tpmorl.eval.ev.ev_matrix exactly: admission
    a_{i,y}; approval in valid year k with h=min(1,HAZARD[k-1]*m_{i,y+k-1});
    completion year t_c=y+k+1+b_i; completions with t_c > T_eval-1 never happen.
    LU in evaluation year s includes completions with t_c <= s, matching the
    env, where the S3->S4 conversion is written into LU inside step s.
  * Target land use = the fixed (cheapest) target per unit kept by
    env.fix_one_target_per_unit (the configuration used for all published rules).
  * Objectives reported as level differences vs LU0 (no renewal): at the end of
    the evaluation horizon (s=T_eval-1), at the end of the decision period
    (s=T-1), and as a discounted sum sum_s gamma^s [O(LU_s) - O(LU0)],
    s=0..T_eval-1. E2r is a mismatch measure: lower is better (SIGN=-1).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)
DS = os.path.join(REPO, "data", "processed", "gm_dataset_v1")
METRO = os.path.join(DS, "metro")

LABELS = {"static_greedy": "persistence", "myopic": "persistence",
          "temporal_greedy": "trajectory_informed", "forecast": "trajectory_informed",
          "value_based": "value_based_s0", "optimum": "reference",
          "oracle": "reference"}
OBJ = ("Gdp", "Eco", "Res", "Emp", "Aec", "E2r", "Cpt")
PERIODS = (("early_y1_8", 0, 7), ("mid_y9_16", 8, 15), ("late_y17_25", 16, 24))
BANDS = (1500.0, 3000.0)
NPERM = 999


# --------------------------------------------------------------- schedules
def _norm_init(v):
    if isinstance(v, dict) and "init" in v:
        v = v["init"]
    if not isinstance(v, dict):
        return None
    try:
        return {int(k): int(y) for k, y in v.items()}
    except (TypeError, ValueError):
        return None


def load_schedules(path, prefix="", relabel=True, with_meta=False):
    """Read a schedule json -> {name: {unit: year}}.

    Accepted structures: {rule: {unit: year}}, {rule: {'init': {...}}}, and the
    wrapped form {'schedules': {rule: ...}, 'infra_mode': ..., ...} written by
    the rolling-horizon track. Metadata entries are ignored for the schedules;
    with_meta=True also returns (schedules, meta) where meta holds infra_mode.
    """
    d = json.load(open(path))
    meta = {}
    if isinstance(d, dict) and isinstance(d.get("schedules"), dict):
        meta = {k: v for k, v in d.items() if k != "schedules" and not isinstance(v, dict)}
        d = d["schedules"]
    out = {}
    for k, v in d.items():
        init = _norm_init(v)
        if init is None:
            continue
        name = LABELS.get(k, k) if relabel else k
        out[prefix + name] = init
    return (out, meta) if with_meta else out


# --------------------------------------------------------------- context
def build_context(prescreen=20, infra_mode="cluster", gamma=0.95, **kw):
    import p1_common as P
    from tpmorl.env import schedule as S
    G, env_fn, EV, EVm, elig, v_orc, oplan = P.setup(prescreen, infra_mode=infra_mode, **kw)
    env = env_fn()
    n = env.n
    xy = np.load(os.path.join(DS, "grid_100m", "cell_centre_xy.npy"))
    cent = np.array([xy[m].mean(0) for m in env.cells])
    # fixed target per unit (env.fix_one_target_per_unit was applied in build_env)
    pu, pt = np.asarray(env._pu_all), np.asarray(env._pt_all)
    assert len(np.unique(pu)) == len(pu) == n, "expected one target per unit"
    tgt = np.empty(n, int); tgt[pu] = pt
    by = np.array([S.BUILD_YEARS_BY_CHANNEL[int(c)] for c in env.ch], int)
    tau = int(S.TAU_VALID + S.TAU_EXT)
    T, Te = int(env.T), int(env.T_eval)
    adm = np.array([np.broadcast_to(np.asarray(env.opp.admit_prob(t), float), (n,))
                    for t in range(T + tau + 1)])
    hm = np.array([np.broadcast_to(np.asarray(env.opp.hazard_mult(t), float), (n,))
                   for t in range(T + tau + 1)])
    # metro
    st = pd.read_csv(os.path.join(METRO, "stations.csv"))
    meta = json.load(open(os.path.join(METRO, "README.json")))
    y0 = int(meta.get("model_year0", 2019))
    usd = pd.read_csv(os.path.join(METRO, "unit_station_distance.csv")).set_index("uid")
    uids = env.U["uid"].values
    d_near = usd.loc[uids, "d_nearest"].values.astype(float)
    dcols = ["d_" + s for s in st["station"]]
    Dst = usd.loc[uids, dcols].values.astype(float)           # (n, n_station)
    ann_year = st["announce"].values.astype(float) - y0         # model year
    # cross-check: csv d_nearest equals min over unit cells (verified 6e-11 m);
    # geo_err reports the max gap to the centroid distance for information
    Dgeo = np.hypot(cent[:, None, 0] - st["x"].values[None],
                    cent[:, None, 1] - st["y"].values[None])
    geo_err = float(np.nanmax(np.abs(Dgeo.min(1) - d_near)))
    D = np.hypot(cent[:, None, 0] - cent[None, :, 0], cent[:, None, 1] - cent[None, :, 1])
    return dict(env=env, G=G, EV=EV, elig=elig, v_orc=v_orc, oplan=oplan, n=n,
                cent=cent, D=D, tgt=tgt, by=by, tau=tau, T=T, T_eval=Te,
                adm=adm, hm=hm, hazard=np.asarray(S.HAZARD, float), gamma=gamma,
                stations=st, d_near=d_near, Dst=Dst, ann_year=ann_year,
                geo_err=geo_err, infra_mode=infra_mode)


# --------------------------------------------------------------- spatial stats
def _band_w(D, band):
    W = ((D <= band) & (D > 0)).astype(float)
    np.fill_diagonal(W, 0.0)
    return W


def morans_i(x, W, nperm=NPERM, rng=None, row_std=True):
    """Global Moran's I with permutation inference (PySAL-style folded p_sim)."""
    x = np.asarray(x, float)
    rs = W.sum(1)
    if row_std:
        W = np.divide(W, rs[:, None], out=np.zeros_like(W), where=rs[:, None] > 0)
    n, S0 = len(x), W.sum()
    z = x - x.mean()
    zz = (z * z).sum()
    if S0 == 0 or zz == 0:
        return dict(I=np.nan, EI=-1.0 / (n - 1), z_sim=np.nan, p_sim=np.nan,
                    n=n, islands=int((rs == 0).sum()), mean_nb=float(rs.mean()))
    I = n / S0 * (z @ W @ z) / zz
    rng = rng or np.random.default_rng(0)
    Z = np.array([rng.permutation(z) for _ in range(nperm)])
    Ip = n / S0 * np.einsum("pi,ij,pj->p", Z, W, Z) / zz
    larger = int((Ip >= I).sum())
    if nperm - larger < larger:
        larger = nperm - larger
    return dict(I=float(I), EI=-1.0 / (n - 1), z_sim=float((I - Ip.mean()) / Ip.std()),
                p_sim=(larger + 1.0) / (nperm + 1.0), n=n,
                islands=int((rs == 0).sum()), mean_nb=float(rs.mean()))


def join_count_bb(x, Wb, nperm=NPERM, rng=None):
    """BB join count on binary symmetric band weights; upper-tail permutation p."""
    x = np.asarray(x, float)
    bb = 0.5 * x @ Wb @ x
    rng = rng or np.random.default_rng(1)
    P = np.array([rng.permutation(x) for _ in range(nperm)])
    bbp = 0.5 * np.einsum("pi,ij,pj->p", P, Wb, P)
    return dict(BB=float(bb), BB_perm_mean=float(bbp.mean()),
                BB_z=float((bb - bbp.mean()) / bbp.std()),
                BB_p_upper=(int((bbp >= bb).sum()) + 1.0) / (nperm + 1.0))


def spatial_stats(schedules, ctx, bands=BANDS, nperm=NPERM, seed=0):
    """One row per schedule. Returns a DataFrame."""
    n, D = ctx["n"], ctx["D"]
    rows = []
    Wfull = {b: _band_w(D, b) for b in bands}
    for name, init in schedules.items():
        u = np.array(sorted(init), int)
        y = np.array([init[i] for i in u], float)
        r = dict(schedule=name, n_selected=len(u), mean_init_year1=float((y + 1).mean()))
        sel = np.zeros(n); sel[u] = 1
        for b in bands:
            tag = f"{b/1000:g}km"
            mi = morans_i(y, Wfull[b][np.ix_(u, u)], nperm,
                          np.random.default_rng(seed))
            r.update({f"I_year_{tag}": mi["I"], f"p_year_{tag}": mi["p_sim"],
                      f"z_year_{tag}": mi["z_sim"], f"islands_year_{tag}": mi["islands"],
                      f"meannb_year_{tag}": mi["mean_nb"]})
            ms = morans_i(sel, Wfull[b], nperm, np.random.default_rng(seed + 1))
            r.update({f"I_sel_{tag}": ms["I"], f"p_sel_{tag}": ms["p_sim"],
                      f"z_sel_{tag}": ms["z_sim"]})
            jc = join_count_bb(sel, Wfull[b], nperm, np.random.default_rng(seed + 2))
            r.update({f"BB_{tag}": jc["BB"], f"BBexp_{tag}": jc["BB_perm_mean"],
                      f"pBB_{tag}": jc["BB_p_upper"]})
        d = ctx["d_near"][u]
        r.update(dist_mean=float(d.mean()), dist_median=float(np.median(d)),
                 share_800m=float((d <= 800).mean()))
        for lab, a, b_ in PERIODS:
            m = (y >= a) & (y <= b_)
            r[f"n_{lab}"] = int(m.sum())
            r[f"dist_mean_{lab}"] = float(d[m].mean()) if m.any() else np.nan
            r[f"dist_median_{lab}"] = float(np.median(d[m])) if m.any() else np.nan
            r[f"share_800m_{lab}"] = float((d[m] <= 800).mean()) if m.any() else np.nan
        # nearest station already announced at the initiation year
        ok = ctx["ann_year"][None, :] <= y[:, None]
        da = np.where(ok, ctx["Dst"][u], np.inf).min(1)
        da = np.where(np.isfinite(da), da, np.nan)
        r.update(dist_announced_mean=float(np.nanmean(da)),
                 share_800m_announced=float(np.nanmean(da <= 800)))
        rows.append(r)
    # reference pools for the metro columns
    for pname, mask in (("pool_all_717", np.ones(n, bool)),
                        ("pool_eligible", np.asarray(ctx["elig"], bool))):
        if pname == "pool_eligible" and mask.all():
            continue                       # identical to pool_all_717
        d = ctx["d_near"][mask]
        rows.append(dict(schedule=pname, n_selected=int(mask.sum()),
                         dist_mean=float(d.mean()), dist_median=float(np.median(d)),
                         share_800m=float((d <= 800).mean())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- objectives
def completion_paths(ctx, i, y):
    """[(t_c, prob)] for unit i initiated in year y (ev.ev_matrix pathway)."""
    T, tau, H = ctx["T"], ctx["tau"], ctx["hazard"]
    alive, out = 1.0, []
    for k in range(1, tau + 1):
        h = min(float(H[min(k - 1, len(H) - 1)])
                * float(ctx["hm"][min(y + k - 1, T + tau)][i]), 1.0)
        pk = alive * h
        alive *= 1.0 - h
        if pk <= 0:
            continue
        tc = y + k + 1 + int(ctx["by"][i])
        if tc > ctx["T_eval"] - 1:
            continue
        out.append((tc, float(ctx["adm"][y][i]) * pk))
    return out


def completion_matrix(init, ctx):
    """P[i, s] = P(unit i completed by evaluation year s)."""
    P = np.zeros((ctx["n"], ctx["T_eval"]))
    for i, y in init.items():
        for tc, p in completion_paths(ctx, i, y):
            P[i, tc:] += p
    return P


def check_pipeline(ctx, init, tol=1e-9):
    """Recompute EV[i,y] from our pathway probabilities and compare to the table."""
    env, g = ctx["env"], ctx["gamma"]
    base = np.asarray(env.farcap, float) * np.asarray(env.ncell, float)
    err = 0.0
    for i, y in init.items():
        v = sum(p * base[i] * env.opp.value_mult(i, y, tc) * g ** tc
                for tc, p in completion_paths(ctx, i, y))
        err = max(err, abs(v - ctx["EV"][i, y]) / max(abs(ctx["EV"][i, y]), 1e-12))
    return err


def expected_lu(P_s, ctx):
    env = ctx["env"]
    LU = env.LU0.copy()
    for i in np.nonzero(P_s > 0)[0]:
        m = env.cells[i]; p = P_s[i]
        LU[m] *= (1.0 - p)
        LU[..., ctx["tgt"][i]][m] += p
    return LU


def spatial_objectives(schedules, ctx):
    """Long table: schedule x objective with end-of-horizon and discounted deltas."""
    env, g, Te, T = ctx["env"], ctx["gamma"], ctx["T_eval"], ctx["T"]
    from tpmorl.objectives.reward import SIGN
    base = env.R.spatial(env.LU0)
    rows = []
    for name, init in schedules.items():
        P = completion_matrix(init, ctx)
        traj, cache = [], {}
        for s in range(Te):
            key = P[:, s].tobytes()
            if key not in cache:
                cache[key] = env.R.spatial(expected_lu(P[:, s], ctx))
            traj.append(cache[key])
        for k in OBJ:
            dl = np.array([o[k] - base[k] for o in traj])
            disc = float((g ** np.arange(Te) * dl).sum())
            rows.append(dict(schedule=name, objective=k, sign=SIGN[k],
                             baseline=base[k],
                             delta_end_eval=float(dl[-1]),
                             pct_end_eval=100 * float(dl[-1]) / base[k],
                             delta_end_decision=float(dl[T - 1]),
                             delta_discounted=disc,
                             pct_discounted=100 * disc / (base[k] * (g ** np.arange(Te)).sum()),
                             improvement_discounted=SIGN[k] * disc,
                             exp_completed_end=float(P[:, -1].sum()),
                             exp_cells_converted_end=float((P[:, -1] * env.ncell).sum())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- figure
def draft_figure(schedules, ctx, ob, path, show=("trajectory_informed", "value_based_s0", "reference")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False})
    env = ctx["env"]
    xy = np.load(os.path.join(DS, "grid_100m", "cell_centre_xy.npy"))
    x0, x1 = xy[..., 0].min() - 50, xy[..., 0].max() + 50
    y0, y1 = xy[..., 1].min() - 50, xy[..., 1].max() + 50
    flip = xy[0, 0, 1] > xy[-1, 0, 1]     # row 0 at the top?
    ext = (x0, x1, y0, y1)
    st = ctx["stations"]
    show = [s for s in show if s in schedules]
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.62], hspace=0.32, wspace=0.05)
    cmap = plt.get_cmap("viridis")
    pool = np.zeros(env.uid.shape, bool)
    for m in env.cells:
        pool |= m
    for j, name in enumerate(show):
        ax = fig.add_subplot(gs[0, j])
        bg = np.full(env.uid.shape, np.nan)
        bg[env.inside] = 0.0; bg[pool] = 1.0
        ax.imshow(bg if flip else bg[::-1], extent=ext, origin="upper",
                  cmap=ListedColormap(["#efefef", "#c9c9c9"]), vmin=0, vmax=1,
                  interpolation="nearest")
        yr = np.full(env.uid.shape, np.nan)
        for i, y in schedules[name].items():
            yr[env.cells[i]] = y + 1
        im = ax.imshow(yr if flip else yr[::-1], extent=ext, origin="upper", cmap=cmap,
                       vmin=1, vmax=ctx["T"], interpolation="nearest")
        inx = st["in_extent"].astype(bool).values if "in_extent" in st else np.ones(len(st), bool)
        ax.scatter(st["x"][inx], st["y"][inx], s=9, marker="^", facecolor="white",
                   edgecolor="#b2182b", lw=0.8, zorder=5, label="metro station")
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(name.replace("_", " ").replace("s0", "(seed 0)"), loc="left")
        ax.text(-0.02, 1.06, "abc"[j], transform=ax.transAxes, fontweight="bold",
                fontsize=9, va="bottom", ha="right")
        if j == 0:
            from matplotlib.patches import Patch
            h, l = ax.get_legend_handles_labels()
            h.append(Patch(color="#c9c9c9")); l.append("unselected candidate unit")
            ax.legend(h, l, loc="upper left", bbox_to_anchor=(0.0, 0.02), frameon=False,
                      handletextpad=0.3, borderaxespad=0.0, ncol=2)
    cax = fig.add_axes([0.92, 0.47, 0.012, 0.36])
    cb = fig.colorbar(im, cax=cax); cb.set_label("initiation year")
    cb.set_ticks([1, 8, 16, 25])
    # objective panel
    ax = fig.add_subplot(gs[1, :])
    order = [s for s in ("persistence", "trajectory_informed", "value_based_s0", "reference")
             if s in schedules]
    cols = {"persistence": "#9e9e9e", "trajectory_informed": "#4393c3",
            "value_based_s0": "#2166ac", "reference": "#222222"}
    w = 0.8 / len(order)
    xs = np.arange(len(OBJ))
    for k, s in enumerate(order):
        sub = ob[ob.schedule == s].set_index("objective").loc[list(OBJ)]
        v = (sub["sign"] * sub["pct_discounted"]).values
        ax.bar(xs + (k - (len(order) - 1) / 2) * w, v, w, color=cols.get(s, "C%d" % k),
               label=s.replace("_", " ").replace("s0", "(seed 0)"))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([o + (" (sign flipped)" if o == "E2r" else "") for o in OBJ])
    ax.set_ylabel("discounted change vs\nno renewal (%)")
    ax.set_title("Spatial objectives, discounted over the evaluation horizon "
                 "(positive = improvement)", loc="left")
    ax.text(-0.04, 1.06, "d", transform=ax.transAxes, fontweight="bold", fontsize=9,
            va="bottom", ha="right")
    ax.legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.18))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedules", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(REPO, "results_p1", "spatial"))
    ap.add_argument("--infra_mode", default="cluster")
    ap.add_argument("--prescreen", type=int, default=20)
    ap.add_argument("--no_fig", action="store_true")
    a = ap.parse_args()
    files = a.schedules
    if files is None:
        files = [os.path.join(REPO, "results_figures", "schedules.json")]
        files += sorted(glob.glob(os.path.join(REPO, "results_p1", "rolling",
                                               "schedules_*.json")))
    groups = {}                       # infra_mode -> {name: init}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        pre = "" if stem == "schedules" else "rolling_" + stem.replace("schedules_", "") + ":"
        sc, meta = load_schedules(f, prefix=pre, with_meta=True)
        groups.setdefault(meta.get("infra_mode", a.infra_mode), {}).update(sc)
    os.makedirs(a.out, exist_ok=True)
    ST, OB, msg = [], [], []
    for mode, sch in groups.items():
        ctx = build_context(a.prescreen, infra_mode=mode)
        for nm, init in sch.items():
            bad = [y for y in init.values() if not 0 <= y < ctx["T"]]
            ids = [u for u in init if not 0 <= u < ctx["n"]]
            assert not bad and not ids, f"{nm}: years/units out of range"
        pipe_err = max(check_pipeline(ctx, s_) for s_ in sch.values())
        st = spatial_stats(sch, ctx); st.insert(1, "infra_mode", mode)
        ob = spatial_objectives(sch, ctx); ob.insert(1, "infra_mode", mode)
        ST.append(st); OB.append(ob)
        if not a.no_fig and mode == "cluster":
            main_s = {k: v for k, v in sch.items() if ":" not in k}
            draft_figure(main_s, ctx, ob[ob.schedule.isin(list(main_s))],
                         os.path.join(a.out, "fig_spatial_draft.png"))
        msg.append(f"{mode}: n={len(sch)} T_eval={ctx['T_eval']} ev_relerr={pipe_err:.1e}")
    pd.concat(ST, ignore_index=True).to_csv(os.path.join(a.out, "spatial_stats.csv"), index=False)
    pd.concat(OB, ignore_index=True).to_csv(os.path.join(a.out, "spatial_objectives.csv"), index=False)
    print("; ".join(msg))


if __name__ == "__main__":
    main()
