# -*- coding: utf-8 -*-
"""p5_learned_spatial.py -- the learned value correction under the spatial extensions.

Training is exactly the frozen protocol of scripts/p3_formal_test.py (field draws 101-120,
two episodes per behaviour rule, one-step target, lead 4, M = 20); the fitted ensembles are
saved to results_p5_spatial/models/s<seed>.joblib this time.
Evaluation on the main field, listing draws 7,0,1,2,3,4:
  * unconstrained (must reproduce the paper's 0.899 on average),
  * E1 subdistrict limits (per-year cap, balanced quotas): rank by the learned score and skip
    units whose subdistrict limit is reached,
  * E2 neighbour spillover (beta 0.1, 0.25, 0.5; Delta = 2): plain learned ranking, and a
    spillover-aware version that adds beta * min(Q_u, Q_j at j's initiation) for neighbours
    initiated within Delta years (the policy's own scores only).
Usage:  python scripts/p5_learned_spatial.py --seed S        (train + evaluate one seed)
"""
import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p1_common as P                                                    # noqa: E402
from exp_fqi_local import make_trees                                      # noqa: E402
from tpmorl.rl import fqi                                                 # noqa: E402
import p3_explore_rulefeat as E                                           # noqa: E402
import p5_spatial_ext as X                                                # noqa: E402

FROZEN = dict(lead=4.0, history=1, n_train=20, train_base=101, ep_each=2, n_iter=1, prescreen=20)
OUTM = os.path.join(X.OUT, "models")


def train(seed):
    f = FROZEN
    path = os.path.join(OUTM, f"s{seed}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    data = []
    for k in [f["train_base"] + i for i in range(f["n_train"])]:
        C, ann, efn = E.field_setup(k, 1)
        EV = C["EV"]
        reward = (lambda Q: (lambda u, t: float(Q[u, min(t, Q.shape[1] - 1)])))(EV)
        data += fqi.collect(efn, reward, {"random": None, "myopic": C["EVm"], "temporal_greedy": EV},
                            n_ep_each=f["ep_each"], seed=seed * 10007 + k)
    mk = make_trees(seed=seed)
    qa, qb, _ = fqi.fit_double(data, P.BACKUP_GAMMA, lambda: E.ResidualModel(mk()),
                               verbose=False, n_iter=f["n_iter"])
    os.makedirs(OUTM, exist_ok=True)
    joblib.dump((qa, qb), path)
    return qa, qb


def rollout(env, qa, qb, T, s_of=None, mode=None, quota=None, nbr=None, beta=0.0, delta=2):
    init, seen = {}, {}
    for t in range(T):
        Xr, meta, cost, units = env.pairs()
        keep = [i for i, m in enumerate(meta) if m[0] >= 0 and int(m[0]) not in init]
        if not keep:
            env.step([]); continue
        us = np.array([meta[i][0] for i in keep], int)
        Xt = fqi._append_global(np.asarray(Xr, np.float32)[keep], t, env, X.K)
        qv = 0.5 * (qa.predict(Xt) + qb.predict(Xt))
        if beta > 0 and nbr is not None:
            qv = qv + np.array([sum(beta * min(q, seen.get(j, 0.0)) for j in nbr[u]
                                    if j in init and abs(init[j] - t) <= delta)
                                for u, q in zip(us, qv)])
        used_s, act = {}, []
        tot = np.bincount([s_of[u] for u in init], minlength=len(quota)) if mode == "balanced" else None
        for j in np.argsort(-qv):
            if len(act) >= X.K:
                break
            u = int(us[j]); s = s_of[u] if s_of is not None else None
            if mode == "cap1" and used_s.get(s, 0) >= 1:
                continue
            if mode == "balanced" and tot[s] + used_s.get(s, 0) >= quota[s]:
                continue
            act.append(meta[keep[j]]); init[u] = t; seen[u] = float(qv[j])
            used_s[s] = used_s.get(s, 0) + 1
        env.step(act)
    return init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    from tpmorl.env import opportunity as OPP
    OPP.HISTORY = int(FROZEN["history"]); OPP.INFRA_ANNOUNCE_LEAD = float(FROZEN["lead"])
    t0 = time.time()
    qa, qb = train(a.seed)
    print(f"[s{a.seed}] trained/loaded in {time.time() - t0:.0f}s", flush=True)
    C, ann, _ = E.field_setup(None, 1)
    EV, elig, v_orc, G = C["EV"], C["elig"], C["v_orc"], C["G"]
    T = EV.shape[1]
    sp = X.load_space(C["env_fn"]()); s_of, edges, nbr = sp["s_of"], sp["edges"], sp["nbr"]
    quota = X.balanced_quota(s_of, X.K * T)
    refs = {"none": v_orc}
    for mode in ("cap1", "balanced"):
        refs[mode] = X.value(EV, X.solve_assignment(EV * elig[:, None], list(range(T)), s_of, mode,
                                                    quota if mode == "balanced" else None))
    lsf = pd.read_csv(os.path.join(X.OUT, "fields", "main_d2_LS.csv")).set_index("beta")
    rows = []
    for ls in X.SEEDS:
        mk = E.wrap_env_fn(P.env_factory(G, X.M, ls), ann)
        for mode in ("none", "cap1", "balanced"):
            init = rollout(mk(), qa, qb, T, s_of, None if mode == "none" else mode, quota)
            rows.append(dict(exp="E1", mode=mode, beta=np.nan, seed=ls, train_seed=a.seed,
                             rule="learned", ratio=X.value(EV, init) / refs[mode]))
        for beta in (0.1, 0.25, 0.5):
            v_best = float(lsf.loc[beta, "best"])
            for aware in (False, True):
                init = rollout(mk(), qa, qb, T, nbr=nbr if aware else None, beta=beta if aware else 0.0)
                v, b = X.spill_value(EV, init, edges, beta)
                rows.append(dict(exp="E2", mode="none", beta=beta, seed=ls, train_seed=a.seed,
                                 rule="learned_aware" if aware else "learned", ratio=v / v_best,
                                 adjacent=sum(1 for i, j in edges if i in init and j in init
                                              and abs(init[i] - init[j]) <= 2)))
        print(f"[s{a.seed}] listing {ls} done ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.join(X.OUT, "learned"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(X.OUT, "learned", f"learned_s{a.seed}.csv"), index=False)


if __name__ == "__main__":
    main()
