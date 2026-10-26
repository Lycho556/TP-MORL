# -*- coding: utf-8 -*-
"""p4_robustness.py — rule-only robustness checks (revision 30). No learning.

  --part quota    annual capacity K in {1, 2, 3, 5, 8}, gamma 0.95, M = 20, six listing draws
  --part gamma    gamma in {0.90, 0.98}, K = 3, M = 20, six listing draws
  --part metro    observed metro field: catchment (800/1500 m base, 500/1000, 1100/2000) and
                  premium after opening (decline as in the paper, or held at its peak);
                  share of units whose best initiation year is now, near vs far
  --part spatial  Moran's I of timing errors at several distance bands (1, 1.5, 2, 3, 5 km),
                  neighbours per unit and isolates; overlap of the selected sets between rules
Rules: persistence greedy, trajectory-informed greedy, rolling horizon (announced, full).
K = 3 and gamma = 0.95 reproduce results_p1/rolling/rolling_results.csv (asserted).
Output: results_p4/*.csv
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p1_common as P                                                    # noqa: E402
import p1_rolling_horizon as RH                                          # noqa: E402
OUT = os.path.join(P.REPO, "results_p4"); LS = [7, 0, 1, 2, 3, 4]


def run_rules(quota=3, gamma=0.95, M=20, infra_mode="cluster", seeds=LS):
    RH.GAMMA = gamma
    a = P.args(quota=quota, gamma=gamma)
    G, env_fn, EV, EVm, elig, v_orc, oplan = P.setup(M, infra_mode=infra_mode, a=a)
    e0 = env_fn(); ann = RH.forecast_tables(e0, "announced", True)
    rows, sched = [], {}
    for ls in seeds:
        mk = P.env_factory(G, M, ls, a)
        res = {}
        for tag, kind in (("persistence", "myopic"), ("trajectory", "forecast")):
            init, _ = G.run_policy(mk(), EV, EVm, kind, quota=quota, rng=np.random.default_rng(0))
            res[tag] = init
        res["rh_announced"] = RH.run_rolling(mk(), lambda t: ann[t], elig, quota=quota)
        res["rh_full"] = RH.run_rolling(mk(), lambda t: EV, elig, quota=quota)
        for rule, init in res.items():
            val = float(sum(EV[u, y] for u, y in init.items()))
            dec = G.decompose(EV, init, quota, elig, v_orc)
            rows.append(dict(quota=quota, gamma=gamma, M=M, infra_mode=infra_mode, listing_seed=ls,
                             rule=rule, ratio=val / v_orc, n_init=len(init),
                             loss_selection=dec["loss_selection"], loss_timing=dec["loss_timing"]))
        sched[ls] = res
    return pd.DataFrame(rows), sched, dict(EV=EV, oplan=oplan, elig=elig, v_orc=v_orc, G=G)


def summarise(df, keys):
    p = df.pivot_table(index=keys + ["listing_seed"], columns="rule", values="ratio").reset_index()
    p["info"] = p.trajectory - p.persistence; p["coord"] = p.rh_full - p.trajectory
    p["ann_gain"] = p.rh_announced - p.persistence
    g = p.groupby(keys)
    s = g[["persistence", "rh_announced", "trajectory", "rh_full", "info", "coord", "ann_gain"]].mean()
    s["info_min"] = g["info"].min(); s["coord_min"] = g["coord"].min()
    s["ann_share_of_info"] = s.ann_gain / s["info"]
    return s.reset_index()


def check_base(df):
    ref = pd.read_csv(os.path.join(P.REPO, "results_p1/rolling/rolling_results.csv"))
    ref = ref[(ref.infra_mode == "cluster") & ref.conformity & (ref.variant == "base") & (ref.M == 20)]
    for _, r in df[(df.quota == 3) & (df.gamma == 0.95)].iterrows():
        q = ref[(ref.rule == r.rule) & (ref.listing_seed == r.listing_seed)].ratio
        assert len(q) == 1 and abs(q.iloc[0] - r.ratio) < 1e-9, (r.rule, r.listing_seed, r.ratio, q.values)


def part_quota():
    df = pd.concat([run_rules(quota=k)[0] for k in (1, 2, 3, 5, 8)]); check_base(df)
    df.to_csv(os.path.join(OUT, "quota_runs.csv"), index=False)
    s = summarise(df, ["quota"]); s["rho"] = 717 / (25 * s.quota)
    s.to_csv(os.path.join(OUT, "quota_summary.csv"), index=False); print(s.round(3).to_string(index=False))


def part_gamma():
    df = pd.concat([run_rules(gamma=g)[0] for g in (0.90, 0.95, 0.98)]); check_base(df)
    df.to_csv(os.path.join(OUT, "gamma_runs.csv"), index=False)
    s = summarise(df, ["gamma"]); s.to_csv(os.path.join(OUT, "gamma_summary.csv"), index=False)
    print(s.round(3).to_string(index=False))


def _plateau_metro(self, uid):
    """As OpportunityField._apply_metro, but the premium is held at its peak after opening."""
    from tpmorl.env import opportunity as O
    d = O.METRO_DIR or os.path.join(P.DATASET, "metro")
    st = pd.read_csv(os.path.join(d, "stations.csv"))
    ud = pd.read_csv(os.path.join(d, "unit_station_distance.csv")).set_index("uid")
    D = ud.loc[np.asarray(uid), [f"d_{x}" for x in st["station"]]].to_numpy(float)
    w = np.clip((O.METRO_R_ZERO - D) / (O.METRO_R_ZERO - O.METRO_R_FULL), 0.0, 1.0)
    t_open = st["open"].to_numpy(float) - O.METRO_YEAR0
    t_ann = st["announce"].to_numpy(float) - O.METRO_YEAR0
    width = np.maximum(self.window * 0.75, 1e-6)[:, None, None]

    def layer(t_s):
        x = (self.years[None, None, :] - t_s[None, :, None]) / width
        g = np.where(x < 0, np.exp(-x ** 2), 1.0)
        return np.clip(O.LEVEL_LOW + (O.LEVEL_HIGH - O.LEVEL_LOW) * (w[:, :, None] * g).max(1), 0.0, 1.0)
    self.infra = layer(t_open); self.infra_plan = layer(t_ann); self.metro_w = w.max(1)


def start_now(rf, rz, plateau, rng):
    from tpmorl.rl import scenario as SC
    from tpmorl.env import opportunity as O
    orig_reset, orig_apply = SC.reset, O.OpportunityField._apply_metro
    def reset():
        orig_reset(); O.METRO_R_FULL, O.METRO_R_ZERO = float(rf), float(rz)
    SC.reset = reset
    if plateau:
        O.OpportunityField._apply_metro = _plateau_metro
    try:
        G, env_fn, EV, EVm, elig, v_orc, _ = P.setup(20, infra_mode="metro")
        assert (O.METRO_R_FULL, O.METRO_R_ZERO) == (rf, rz)
    finally:
        SC.reset, O.OpportunityField._apply_metro = orig_reset, orig_apply
    uid = env_fn().U["uid"].values
    ud = pd.read_csv(os.path.join(P.DATASET, "metro/unit_station_distance.csv")).set_index("uid")
    d = ud.loc[uid, "d_nearest"].to_numpy(float); now = (EV.argmax(1) == 0).astype(float)
    out = dict(r_full=rf, r_zero=rz, premium="held after opening" if plateau else "declines after opening")
    for tag, thr in (("catch", rz), ("fixed1500", 1500.0)):
        near = d <= thr; a, b = now[near], now[~near]; diff = a.mean() - b.mean()
        bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(2000)]
        perm = np.array([(lambda z: z[:len(a)].mean() - z[len(a):].mean())(rng.permutation(now)) for _ in range(9999)])
        out.update({f"{tag}_n_near": int(near.sum()), f"{tag}_near": a.mean(), f"{tag}_far": b.mean(),
                    f"{tag}_diff": diff, f"{tag}_lo": np.percentile(bs, 2.5), f"{tag}_hi": np.percentile(bs, 97.5),
                    f"{tag}_p": (np.sum(np.abs(perm) >= abs(diff)) + 1) / (len(perm) + 1)})
    return out


def part_metro():
    rng = np.random.default_rng(0); rows = []
    for rf, rz in ((800.0, 1500.0), (500.0, 1000.0), (1100.0, 2000.0)):
        for plateau in (False, True):
            t0 = time.time(); rows.append(start_now(rf, rz, plateau, rng))
            print(rows[-1]["r_zero"], rows[-1]["premium"], round(rows[-1]["catch_near"], 3),
                  round(rows[-1]["catch_far"], 3), f"{time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); base = pd.read_csv(os.path.join(P.REPO, "results_p2/start_now_by_distance.csv")).set_index("field").loc["metro"]
    b = df.iloc[0]; assert abs(b.catch_near - base.share_now_near) < 1e-12 and abs(b.catch_far - base.share_now_far) < 1e-12
    df.to_csv(os.path.join(OUT, "metro_start_now.csv"), index=False); print(df.round(3).to_string(index=False))


def part_spatial():
    import p1_spatial as SP
    df, sched, ctx = run_rules()
    EV = ctx["EV"]; T = 25; EVh = EV[:, :T]; best = EVh.argmax(1)
    C = SP.build_context(20); D = C["D"]
    rows, ov = [], []
    rl = {int(u): int(t) for u, t in json.load(open(os.path.join(P.REPO, "results_p3_metro/rl_schedule_main_ls7_s0.json")))["init"].items()}
    for ls, res in sched.items():
        res = dict(res); res["reference"] = dict(ctx["oplan"])
        if ls == 7:
            res["rl_seed0"] = rl
        for rule, init in res.items():
            u = np.array(sorted(init), int); e = np.array([init[k] for k in u]) - best[u]
            for band in (1000.0, 1500.0, 2000.0, 3000.0, 5000.0):
                W = SP._band_w(D, band)[np.ix_(u, u)]; nb = W.sum(1)
                mi = SP.morans_i(e, W, nperm=999, rng=np.random.default_rng(0))
                rows.append(dict(rule=rule, listing_seed=ls, band_km=band / 1000, n=len(u), I=mi["I"],
                                 p=mi["p_sim"], mean_neighbours=nb.mean(), isolates=int((nb == 0).sum())))
        names = list(res)
        for i, r1 in enumerate(names):
            for r2 in names[i + 1:]:
                ov.append(dict(listing_seed=ls, a=r1, b=r2, common=len(set(res[r1]) & set(res[r2])),
                               same_year=sum(res[r1][k] == res[r2][k] for k in set(res[r1]) & set(res[r2]))))
    R = pd.DataFrame(rows); O_ = pd.DataFrame(ov)
    R.to_csv(os.path.join(OUT, "correlogram_runs.csv"), index=False); O_.to_csv(os.path.join(OUT, "overlap_runs.csv"), index=False)
    S = R[R.rule != "rl_seed0"].groupby("band_km").agg(I_min=("I", "min"), I_max=("I", "max"), p_min=("p", "min"),
                                                       nb=("mean_neighbours", "mean"), iso=("isolates", "mean"), runs=("I", "size"))
    S.to_csv(os.path.join(OUT, "correlogram_summary.csv")); print(S.round(3).to_string())
    print(O_.groupby(["a", "b"])[["common", "same_year"]].mean().round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True, choices=["quota", "gamma", "metro", "spatial"])
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    {"quota": part_quota, "gamma": part_gamma, "metro": part_metro, "spatial": part_spatial}[a.part]()
