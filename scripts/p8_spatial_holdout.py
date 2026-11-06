# -*- coding: utf-8 -*-
"""p8_spatial_holdout.py -- spatial hold-out (two-fold cross-fitting by former subdistrict) of the
learned value correction. Pre-registered in results_p8_spatial/PREREGISTRATION.md.
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src TPMORL_NJOBS=1 python scripts/p8_spatial_holdout.py --seed S [--sanity]
Output: results_p8_spatial/rows_s{S}.csv (one row per evaluation draw), corr_s{S}.csv."""
import argparse, os, sys, time
import joblib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p1_common as P
from exp_fqi_local import make_trees
from tpmorl.rl import fqi
import p3_explore_rulefeat as E
from p3_metro_test import metro_setup
from p1_fqi_job import ev_value
import p5_spatial_ext as X
from tpmorl.env import opportunity as OPP
OUT = os.path.join(HERE, "..", "results_p8_spatial")
FOLD_G = {"Guangming", "Fenghuang", "Xinhu"}
LISTING = [7, 0, 1, 2, 3, 4]

def fold_of(env):
    sp = X.load_space(env); names = [n.split(" ")[0] for n in sp["names"]]
    return np.array([0 if names[s] in FOLD_G else 1 for s in sp["s_of"]])   # 0 = G, 1 = M

def collect(seed):
    data = []
    for k in range(101, 121):
        C, ann, efn = E.field_setup(k, 1); EV = C["EV"]
        reward = (lambda Q: (lambda u, t: float(Q[u, min(t, Q.shape[1] - 1)])))(EV)
        data += fqi.collect(efn, reward, {"random": None, "myopic": C["EVm"], "temporal_greedy": EV},
                            n_ep_each=2, seed=seed * 10007 + k)
    return data

def restrict(data, keep_unit):
    out = []
    for d in data:
        m = keep_unit[d["units"]]
        if m.any():
            e = dict(d); e["X"], e["units"], e["r"] = d["X"][m], d["units"][m], d["r"][m]; out.append(e)
    return out

def fit(data, seed):
    qa, qb, _ = fqi.fit_double(data, P.BACKUP_GAMMA, lambda: E.ResidualModel(make_trees(seed=seed)()), verbose=False, n_iter=1)
    return qa, qb

def rollout(env_fn, models, fold, EV):
    """fqi.rollout with per-unit model choice: unit in fold f is scored by models[f]."""
    env = env_fn(); q = int(env.quota); init = {}; diag = []
    for t in range(env.T):
        Xr, meta, cost, units = env.pairs()
        keep = [i for i, m in enumerate(meta) if m[0] >= 0]
        if not keep:
            if env.step([])[2]: break
            continue
        us = np.array([meta[i][0] for i in keep], int)
        Xt = fqi._append_global(np.asarray(Xr, np.float32)[keep], t, env, q)
        qv = np.empty(len(us))
        for f in (0, 1):
            m = fold[us] == f
            if m.any():
                qa, qb = models[f]; qv[m] = 0.5 * (qa.predict(Xt[m]) + qb.predict(Xt[m]))
        fnow = Xt[:, E.ResidualModel.F_NOW_COL]
        diag.append(pd.DataFrame(dict(t=t, u=us, fold=fold[us], dq=qv - fnow, R=EV[us, min(t, EV.shape[1] - 1)] - fnow)))
        act, used = [], set()
        for j in np.argsort(-qv):
            if len(act) >= q: break
            u = int(us[j])
            if u in used: continue
            act.append(meta[keep[j]]); used.add(u); init.setdefault(u, t)
        if env.step(act)[2]: break
    return init, pd.concat(diag)

def evaluate(name, C, ann, efn, models, fold, ls):
    init, dg = rollout(efn, models, fold, C["EV"])
    b = {x["rule"]: x["ratio"] for x in E.baselines(C, ann, name)}
    return dict(eval=name, listing=ls, learned=ev_value(C["EV"], init) / C["v_orc"], rank_announced=b["rank_announced"],
                trajectory=b.get("trajectory", np.nan), rh_full=b["rh_full"]), dg.assign(eval=name, listing=ls)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--sanity", action="store_true")
    a = ap.parse_args(); OPP.HISTORY = 1; OPP.INFRA_ANNOUNCE_LEAD = 4.0; os.makedirs(OUT, exist_ok=True); t0 = time.time()
    C0, _, _ = E.field_setup(None, 1); fold = fold_of(C0["env_fn"]())
    if a.sanity:   # full model in both slots must reproduce the published learned ratio on test draw 1
        full = joblib.load(os.path.join(X.OUT, "models", f"s{a.seed}.joblib"))
        C, ann, efn = E.field_setup(1, 1); r, _ = evaluate("test1", C, ann, efn, (full, full), fold, 7)
        ref, _, _ = fqi.rollout(efn, full[0], full[1], None)
        print("sanity: crossfit-rollout", r["learned"], "fqi.rollout", ev_value(C["EV"], ref) / C["v_orc"], "fold sizes", np.bincount(fold)); return
    data = collect(a.seed)
    models = {f: fit(restrict(data, fold != f), a.seed) for f in (0, 1)}   # models[f] never saw fold f
    print(f"[s{a.seed}] trained {time.time()-t0:.0f}s, rows used: G-free {sum(len(d['r']) for d in restrict(data, fold != 0))}, M-free {sum(len(d['r']) for d in restrict(data, fold != 1))}", flush=True)
    rows, diags = [], []
    for k in range(1, 11):
        C, ann, efn = E.field_setup(k, 1); r, dg = evaluate(f"test{k}", C, ann, efn, models, fold, 7); rows.append(r); diags.append(dg)
    for ls in LISTING:
        C, ann, efn = metro_setup(ls, "metro"); r, dg = evaluate("metro", C, ann, efn, models, fold, ls); rows.append(r); diags.append(dg)
    pd.DataFrame(rows).assign(seed=a.seed).to_csv(os.path.join(OUT, f"rows_s{a.seed}.csv"), index=False)
    D = pd.concat(diags); D["grp"] = np.where(D["eval"].str.startswith("test"), "test", "metro")
    cr = D.groupby(["grp", "fold"]).apply(lambda g: pd.Series(dict(corr=np.corrcoef(g.dq, g.R)[0, 1], n=len(g), dq=g.dq.mean(), R=g.R.mean()))).reset_index()
    cr.assign(seed=a.seed).to_csv(os.path.join(OUT, f"corr_s{a.seed}.csv"), index=False)
    print(f"[s{a.seed}] done {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
