# -*- coding: utf-8 -*-
"""p6_shift_diag.py -- A1: does the learned correction add value the metro field does not contain?
Saved models (results_p5_spatial/models/s<seed>.joblib, frozen protocol). For every prescreened
candidate in every year of the policy's own rollout: predicted correction dQ = Q - F_now and
true residual R = EV[u,t] - F_now (EV = true expected value of initiating now).
Fields: observed metro (6 listing draws), main scenario field (6 listing draws), ten test draws
(listing draw 7). Output: results_p6_shift/a1_rows.csv (per field x seed x listing) .
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/p6_shift_diag.py --seed S"""
import argparse, os, sys
import joblib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p1_common as P
from tpmorl.rl import fqi
import p3_explore_rulefeat as E
from p3_metro_test import metro_setup
from p1_fqi_job import ev_value
LISTING = [7, 0, 1, 2, 3, 4]
OUT = os.path.join(HERE, "..", "results_p6_shift")

def diag_rollout(efn, EV, qa, qb):
    env = efn(); q = int(env.quota); init = {}; rec = []
    for t in range(env.T):
        X, meta, cost, units = env.pairs()
        keep = [i for i, m in enumerate(meta) if m[0] >= 0]
        if not keep:
            _, _, done, _ = env.step([])
            if done: break
            continue
        us = np.array([meta[i][0] for i in keep], int)
        Xt = fqi._append_global(np.asarray(X, np.float32)[keep], t, env, q)
        qv = 0.5 * (qa.predict(Xt) + qb.predict(Xt))
        fnow = Xt[:, E.ResidualModel.F_NOW_COL].astype(float)
        true = EV[us, min(t, EV.shape[1] - 1)]
        order = np.argsort(-qv); act, used, chosen = [], set(), []
        for j in order:
            if len(act) >= q: break
            u = int(us[j])
            if u in used: continue
            act.append(meta[keep[j]]); used.add(u); chosen.append(j); init.setdefault(u, t)
        ann_top = set(np.argsort(-fnow)[:q].tolist())
        for j in range(len(us)):
            rec.append(dict(t=t, u=int(us[j]), fnow=fnow[j], dq=qv[j] - fnow[j], R=true[j] - fnow[j],
                            sel=j in chosen, ann_sel=j in ann_top))
        _, _, done, _ = env.step(act)
        if done: break
    return init, pd.DataFrame(rec)

def summarise(df, field, seed, ls, ratio):
    out = dict(field=field, seed=seed, listing=ls, ratio=ratio)
    for k, m in (("all", np.ones(len(df), bool)), ("sel", df.sel.values),
                 ("swap_in", (df.sel & ~df.ann_sel).values), ("swap_out", (~df.sel & df.ann_sel).values)):
        d = df[m]
        out[f"n_{k}"] = len(d)
        out[f"dq_{k}"] = d.dq.mean() if len(d) else np.nan          # predicted correction
        out[f"R_{k}"] = d.R.mean() if len(d) else np.nan            # true residual
        out[f"fnow_{k}"] = d.fnow.mean() if len(d) else np.nan
    out["corr_all"] = np.corrcoef(df.dq, df.R)[0, 1] if len(df) > 2 else np.nan
    out["mae_all"] = (df.dq - df.R).abs().mean()
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--models", default=os.path.join(HERE, "..", "results_p5_spatial", "models"))
    ap.add_argument("--tag", default="a1")
    a = ap.parse_args()
    from tpmorl.env import opportunity as OPP
    OPP.HISTORY = 1; OPP.INFRA_ANNOUNCE_LEAD = 4.0
    qa, qb = joblib.load(os.path.join(a.models, f"s{a.seed}.joblib"))
    os.makedirs(OUT, exist_ok=True); rows = []
    for ls in LISTING:                                        # observed metro field
        C, ann, efn = metro_setup(ls, "metro")
        init, df = diag_rollout(efn, C["EV"], qa, qb)
        rows.append(summarise(df, "metro", a.seed, ls, ev_value(C["EV"], init) / C["v_orc"]))
        df.assign(field="metro", seed=a.seed, listing=ls).to_csv(os.path.join(OUT, f"{a.tag}_cand_metro_s{a.seed}_l{ls}.csv.gz"), index=False)
    for ls in LISTING:                                        # main scenario field
        C, ann, efn = metro_setup(ls, "cluster")
        init, df = diag_rollout(efn, C["EV"], qa, qb)
        rows.append(summarise(df, "main", a.seed, ls, ev_value(C["EV"], init) / C["v_orc"]))
    for k in range(1, 11):                                    # ten test draws, listing 7
        C, ann, efn = E.field_setup(k, 1)
        init, df = diag_rollout(efn, C["EV"], qa, qb)
        rows.append(summarise(df, f"test{k}", a.seed, 7, ev_value(C["EV"], init) / C["v_orc"]))
        print(f"[s{a.seed}] test{k} ratio {rows[-1]['ratio']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{a.tag}_rows_s{a.seed}.csv"), index=False)
    print(f"[s{a.seed}] done", flush=True)

if __name__ == "__main__":
    main()
