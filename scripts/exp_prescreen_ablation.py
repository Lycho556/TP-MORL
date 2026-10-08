# -*- coding: utf-8 -*-
"""exp_prescreen_ablation.py —— 光明区两项审稿敏感性。

A. 年度初筛 K 的敏感性  K ∈ {10, 20, 50, 0=全部}
   初筛作用在 env.pairs() 上，因此**所有规则**（静态贪心、时间感知贪心、FQI）
   看到的都是同一批前 K 个候选；只有 oracle 在全部合规单元上求解。
   所以每个 K 都要同时报 FQI 与时间感知贪心，读的是二者之差，
   而不是 FQI 单独的绝对值 —— 否则分不清成绩来自初筛还是来自学习器。

B. Single-FQI vs Double-FQI（K=20，主表配置）
   Single：一个估计器，目标 r + γ max_{u'≠u} Q(s', u')，全体转移拟合。
   Double：沿用 fqi.fit_double（按年整组二分、交叉目标）。
   Double 的 seed 0–2 应逐位复现主表 0.9073 / 0.9064 / 0.8969 —— 这是本脚本
   的自检。

口径与主表一致：unit-only、只押 Floor、25 年、每年 3 个、γ=0.95、预算放开、
每种行为策略 --ep-each 个回合（主表实测 n_samples=15000 ⇒ 10 回合）。

用法：
    PYTHONPATH=src python scripts/exp_prescreen_ablation.py --part B --seeds 0 1 2 3 4
    PYTHONPATH=src python scripts/exp_prescreen_ablation.py --part A --ks 10 20 50 0 --seeds 0 1 2
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_fqi_local import make_trees                                   # noqa: E402

from tpmorl.rl import fqi                                              # noqa: E402


def fit_single(data, gamma, make_model, n_iter=None):
    """单估计器有限期 FQI。返回 (q, q)，使 rollout 的 ½(qa+qb) 退化为 q。

    与 fit_double 唯一的区别是：动作选择与评估用同一个估计器，且不分半数据。
    后继池同样是「下一年候选去掉 u」，与 _targets 的近似完全一致，
    故两者只差 Double 这一件事。
    """
    T_max = max(d["t"] for d in data) + 1
    n_iter = int(T_max if n_iter is None else n_iter)
    X = np.vstack([d["X"] for d in data])
    r0 = np.concatenate([d["r"] for d in data])
    q = None
    for _ in range(n_iter):
        if q is None:
            y = r0.copy()
        else:
            ys = []
            for d in data:
                r = d["r"]
                if d["done"] or len(d["nunits"]) == 0:
                    ys.append(r.copy()); continue
                qn = q.predict(d["nX"])
                o = np.argsort(-qn)[:2]
                nu = d["nunits"]
                best = []
                for u in d["units"]:
                    i = o[0] if nu[o[0]] != u or len(o) == 1 else o[1]
                    best.append(qn[i])
                ys.append(r + gamma * np.asarray(best))
            y = np.concatenate(ys)
        q = make_model()
        q.fit(X, y)
    return q, q


def setup(a, prescreen):
    import exp_temporal_gate as G
    from tpmorl.rl import scenario as SC
    SC.reset()
    SC.apply(horizon=a.horizon, horizon_eval="auto", opp_shape="window",
             a_plan=2.4, a_infra=1.8, a_age=0.9, a_ready=1.0,
             foresight=0, quota=a.quota, budget=1e12)
    env_fn = lambda: G.build_env(a.dataset, a.horizon, 0.5, 7, 1e12, "floor",
                                 True, False, prescreen, 0)
    e0 = env_fn()
    EV, EVm = G.ev_tables(e0, a.gamma)
    elig = np.asarray(e0.env.eligible, bool)
    v_orc, oplan = G.oracle_plan(EV, a.quota, elig)
    return G, env_fn, EV, EVm, elig, v_orc


def run(a, prescreen, seed, estimator):
    G, env_fn, EV, EVm, elig, v_orc = setup(a, prescreen)
    t0 = time.time()
    data = fqi.collect(env_fn, lambda u, t: float(EV[u, min(t, EV.shape[1] - 1)]),
                       {"random": None, "myopic": EVm, "temporal_greedy": EV},
                       n_ep_each=a.ep_each, seed=seed)
    if estimator == "double":
        qa, qb, _ = fqi.fit_double(data, a.gamma, make_trees(seed=seed),
                                   verbose=False)
    else:
        qa, qb = fit_single(data, a.gamma, make_trees(seed=seed))
    init, R, _ = fqi.rollout(env_fn, qa, qb, None, collect_q=True)
    v = float(sum(EV[u, min(t, EV.shape[1] - 1)] for u, t in init.items()))
    dec = G.decompose(EV, init, a.quota, elig, v_orc)
    row = dict(prescreen=prescreen, seed=seed, method=f"{estimator}_fqi",
               ratio=v / v_orc, value=v, oracle_value=v_orc,
               where_loss=dec["loss_selection"], when_loss=dec["loss_timing"],
               n_samples=sum(len(d["r"]) for d in data), n_init=len(init),
               q_vs_value_rho=fqi.q_vs_value_corr(data, qa, qb, None),
               seconds=round(time.time() - t0, 1))
    return row


def baselines(a, prescreen):
    G, env_fn, EV, EVm, elig, v_orc = setup(a, prescreen)
    out = []
    for tag, kind in (("static_greedy", "myopic"), ("temporal_greedy", "forecast")):
        iv, vv = G.run_policy(env_fn(), EV, EVm, kind, quota=a.quota,
                              rng=np.random.default_rng(0))
        dec = G.decompose(EV, iv, a.quota, elig, v_orc)
        out.append(dict(prescreen=prescreen, seed=-1, method=tag,
                        ratio=vv / v_orc, value=vv, oracle_value=v_orc,
                        where_loss=dec["loss_selection"],
                        when_loss=dec["loss_timing"], n_init=len(iv)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B"], required=True)
    ap.add_argument("--out", default="results_ablation")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ep-each", type=int, default=10)
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--quota", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.95)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"part{a.part}_runs.csv")
    rows = []

    if a.part == "A":
        jobs = [(k, "double") for k in a.ks]
    else:
        jobs = [(20, "single"), (20, "double")]

    for k, est in jobs:
        if a.part == "A" or est == "double":
            rows += baselines(a, k)
            b = {r["method"]: r["ratio"] for r in rows if r["prescreen"] == k}
            print(f"[K={k or 'all'}] static {b['static_greedy']:.4f}  "
                  f"TG {b['temporal_greedy']:.4f}", flush=True)
        for s in a.seeds:
            r = run(a, k, s, est)
            rows.append(r)
            print(f"  K={k or 'all'} {est:6s} seed={s}  ratio {r['ratio']:.4f}  "
                  f"where {r['where_loss']:.1f}  when {r['when_loss']:.1f}  "
                  f"n={r['n_samples']}  {r['seconds']}s", flush=True)
            pd.DataFrame(rows).to_csv(fn, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(fn, index=False)
    s = (df.groupby(["prescreen", "method"])
           .agg(ratio=("ratio", "mean"), sd=("ratio", "std"), n=("ratio", "size"),
                where=("where_loss", "mean"), when=("when_loss", "mean"))
           .reset_index())
    s.to_csv(os.path.join(a.out, f"part{a.part}_summary.csv"), index=False)
    print("\n" + s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
