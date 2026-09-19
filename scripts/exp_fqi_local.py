# -*- coding: utf-8 -*-
"""exp_fqi_local.py —— Double-FQI 的本地测试（合成世界 → 光明区）。

按指南的本地顺序：

    L0  环境一致性：FQI 的 r 与环境实付奖励、与 oracle 计价必须同源（0 差异）
    L1  合成 1 种子 α=0.5：只看能不能跑、Q 目标是否收敛
    L2  合成 3 种子 × α ∈ {0.25, 0.5, 0.75, 1.0}：与三条基线同台
    L3  光明区 1 种子：FQI 是否 > BC 的 0.846
    L4  光明区 3 种子：出最终表

三档判据（第 13 节）：
    Gate A  FQI > 纯空间排序
    Gate B  FQI ≈ 时间感知贪心
    Gate C  FQI > 时间感知贪心      ← 只有这一档才是强结果

评价沿用既有指标（Where 召回 / When MAE / 相对 oracle），不另立一套。

用法：
    PYTHONPATH=src python scripts/exp_fqi_local.py --stage L1
    PYTHONPATH=src python scripts/exp_fqi_local.py --stage L2 --seeds 0 1 2
    PYTHONPATH=src python scripts/exp_fqi_local.py --stage L3 --seeds 0
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_timing_world import TimingWorld                      # noqa: E402
from exp_where_when import (value_matrix, spatial_matrix,      # noqa: E402
                            oracle as synth_oracle, run_rank,
                            where_when_metrics)

from tpmorl.rl import fqi                                      # noqa: E402

SYNTH = dict(n_per_kind=8, T=10, quota=2, suit=[1.0, 0.8, 0.6, 0.45],
             width=1.8, value_at="init")


def make_trees(n_estimators=200, seed=0):
    """ExtraTrees 先行（第 19 节）：快、稳、对特征尺度不敏感。

    若连它都学不出来，说明问题不在网络容量，不该急着换神经网络。

    **每次调用给不同的 random_state。** 第一版实测两个估计器的相关恒为
    1.0000 —— 因为 Q_A 与 Q_B 用了同一个种子，目标接近时两棵森林几乎逐位相同，
    Double-Q 的去相关等于没生效（它要的正是两个**独立**的估计器，
    一个选动作、另一个评估，才能抑制 max 的高估）。
    """
    box = {"k": 0}

    def f():
        box["k"] += 1
        return ExtraTreesRegressor(
            n_estimators=n_estimators, min_samples_leaf=2, n_jobs=-1,
            random_state=seed * 1000 + box["k"])
    return f


# ------------------------------------------------------------ 合成世界

def synth_setup(alpha):
    w0 = TimingWorld(alpha=alpha, **SYNTH); w0.reset(seed=0)
    V = value_matrix(w0)                 # EV[u, t]：环境自己的价值函数
    S = spatial_matrix(w0)
    v_orc, oplan = synth_oracle(V, w0.quota)
    Vm = np.repeat(V[:, :1], V.shape[1], axis=1)   # 近视：只按第 0 年的条件排
    return w0, V, S, Vm, v_orc, oplan


def l0_check(alpha=0.5):
    """L0：FQI 的即时回报与环境实付的折现奖励流必须同源。"""
    w0, V, S, Vm, v_orc, oplan = synth_setup(alpha)
    d = w0.validate()
    # 沿一个固定计划走一遍，比较 Σ EV[u, t_init] 与环境实付折现奖励
    w = TimingWorld(alpha=alpha, **SYNTH); w.reset(seed=0)
    ev, R = 0.0, 0.0
    for t in range(w.T):
        X, meta, cost, units = w.pairs()
        real = [(i, m[0]) for i, m in enumerate(meta) if m[0] >= 0]
        act, used = [], set()
        for i, u in sorted(real, key=lambda p: -V[p[1], t]):
            if len(act) >= w.quota:
                break
            if u in used:
                continue
            act.append(meta[i]); used.add(u); ev += float(V[u, t])
        _, rr, done, _ = w.step(act)
        R += (w.gamma ** t) * float(np.sum(rr))
        if done:
            break
    print(f"L0  奖励=计价机械自证 {d['reward_matches_value']}  |  "
          f"同一计划：Σ EV {ev:.4f}  环境实付折现奖励 {R:.4f}  "
          f"差 {abs(ev - R):.2e}")
    assert d["reward_matches_value"], "环境奖励与 dvalue 不一致，先修口径"
    assert abs(ev - R) < 1e-6 * max(abs(ev), 1.0), \
        f"Σ EV 与环境实付奖励不等（{ev} vs {R}）——口径打架，不要继续训练"
    return True


def run_synth(alpha, seed, n_ep_each, n_iter, verbose=True):
    w0, V, S, Vm, v_orc, oplan = synth_setup(alpha)
    env_fn = lambda: _mk(TimingWorld(alpha=alpha, **SYNTH))
    data = fqi.collect(env_fn, lambda u, t: float(V[u, min(t, V.shape[1] - 1)]),
                       {"random": None, "myopic": Vm, "temporal_greedy": V},
                       n_ep_each=n_ep_each, seed=seed)
    qa, qb, log = fqi.fit_double(data, w0.gamma, make_trees(seed=seed),
                                 n_iter=n_iter, verbose=verbose)
    init, R, diag = fqi.rollout(env_fn, qa, qb, None, collect_q=True)
    v = float(sum(V[u, t] for u, t in init.items()))
    m = where_when_metrics(init, oplan, V)
    rho = fqi.q_vs_value_corr(data, qa, qb, None)
    return dict(alpha=alpha, seed=seed, method="Double-FQI", value=v,
                ratio=v / v_orc, oracle_value=v_orc, actual_reward=R,
                q_vs_value_rho=rho, n_samples=sum(len(d["r"]) for d in data),
                q_gap=float(np.nanmean([x["q_gap"] for x in diag])), **m)


def _mk(w):
    w.reset(seed=0)
    return w


def baselines_synth(alpha, seeds):
    w0, V, S, Vm, v_orc, oplan = synth_setup(alpha)
    out = []

    def rec(tag, val, init, seed=-1):
        m = where_when_metrics(init, oplan, V)
        out.append(dict(alpha=alpha, seed=seed, method=tag, value=val,
                        ratio=val / v_orc, oracle_value=v_orc, **m))
    for s in seeds:
        v, iv = run_rank(TimingWorld(alpha=alpha, **SYNTH), S, V,
                         rng=np.random.default_rng(s))
        rec("随机", v, iv, s)
    v, iv = run_rank(TimingWorld(alpha=alpha, **SYNTH), S, V)
    rec("纯空间排序", v, iv)
    v, iv = run_rank(TimingWorld(alpha=alpha, **SYNTH), V, V)
    rec("时间感知贪心", v, iv)
    rec("oracle", v_orc, oplan)
    return out


# ------------------------------------------------------------ 光明区

def run_real(a, seed):
    """L3/L4：717 单元、unit-only、只押 Floor，沿用既有闸的口径与基线。"""
    import exp_temporal_gate as G
    from tpmorl.rl import scenario as SC

    SC.reset()
    SC.apply(horizon=a.horizon, horizon_eval="auto", opp_shape="window",
             a_plan=2.4, a_infra=1.8, a_age=0.9, a_ready=1.0,
             foresight=a.foresight, quota=a.quota, budget=1e12)
    env_fn = lambda: G.build_env(a.dataset, a.horizon, 0.5, 7, 1e12, "floor",
                                 True, False, a.prescreen, a.foresight)
    e0 = env_fn()
    EV, EVm = G.ev_tables(e0, a.gamma)
    elig = np.asarray(e0.env.eligible, bool)
    v_orc, oplan = G.oracle_plan(EV, a.quota, elig)

    data = fqi.collect(env_fn, lambda u, t: float(EV[u, min(t, EV.shape[1] - 1)]),
                       {"random": None, "myopic": EVm, "temporal_greedy": EV},
                       n_ep_each=a.ep_each, seed=seed)
    qa, qb, log = fqi.fit_double(data, a.gamma, make_trees(seed=seed),
                                 n_iter=a.n_iter)
    if getattr(a, "dump_model", False):
        # 落盘 Q 与训练数据的特征/即时回报，供 S3-B 的 Q 诊断复用 ——
        # 那一步不需要重训，重训反而会引入与主结果不同的模型。
        import joblib
        joblib.dump(dict(qa=qa, qb=qb,
                         X=np.vstack([d["X"] for d in data]),
                         r=np.concatenate([d["r"] for d in data]),
                         year=np.concatenate([np.full(len(d["r"]), d["t"])
                                              for d in data]),
                         gamma=a.gamma, seed=seed),
                    os.path.join(a.out, f"fqi_model_seed{seed}.joblib"),
                    compress=3)
    init, R, diag = fqi.rollout(env_fn, qa, qb, None, collect_q=True)
    v = float(sum(EV[u, min(t, EV.shape[1] - 1)] for u, t in init.items()))
    dec = G.decompose(EV, init, a.quota, elig, v_orc)
    rows = [dict(seed=seed, method="Double-FQI", value=v, ratio=v / v_orc,
                 oracle_value=v_orc, actual_reward=R,
                 where_wrong=dec["loss_selection"], when_error=dec["loss_timing"],
                 q_vs_value_rho=fqi.q_vs_value_corr(data, qa, qb, None),
                 q_gap=float(np.nanmean([x["q_gap"] for x in diag])),
                 n_samples=sum(len(d["r"]) for d in data), n_init=len(init))]
    if getattr(a, "with_bc", False):
        # BC（v19 的监督时空打分器）：论文主表的 M3 行。
        # 教师用真实 EV（"按已公布的规划与设施计划前瞻"这一信息档），
        # 与 v19 的 --bc-init forecast --bc-only 完全同配置，故数值可比。
        init_sd, bcm = G.bc_actor(env_fn, EV, None, seed=seed)
        bcnet = G.TP.Pointer(); bcnet.load_state_dict(init_sd, strict=False)
        e_bc = env_fn()
        iv_bc, v_bc = G.run_policy(e_bc, EV, EVm, "ppo", net=bcnet,
                                   quota=a.quota)
        d_bc = G.decompose(EV, iv_bc, a.quota, elig, v_orc)
        rows.append(dict(seed=seed, method="BC", value=v_bc, ratio=v_bc / v_orc,
                         oracle_value=v_orc, actual_reward=np.nan,
                         where_wrong=d_bc["loss_selection"],
                         when_error=d_bc["loss_timing"],
                         q_vs_value_rho=np.nan, q_gap=np.nan,
                         n_samples=bcm["bc_n"], n_init=len(iv_bc)))

    # 三条既有基线：同一 EV 计价、同一掩码，故可直接相减
    for tag, kind, rk in (("随机", "random", None), ("静态/近视贪心", "myopic", None),
                          ("时间感知贪心", "forecast", None)):
        e = env_fn()
        iv, vv = G.run_policy(e, EV, EVm, kind, quota=a.quota,
                              rng=np.random.default_rng(seed))
        d2 = G.decompose(EV, iv, a.quota, elig, v_orc)
        rows.append(dict(seed=seed, method=tag, value=vv, ratio=vv / v_orc,
                         oracle_value=v_orc, actual_reward=np.nan,
                         where_wrong=d2["loss_selection"],
                         when_error=d2["loss_timing"], q_vs_value_rho=np.nan,
                         q_gap=np.nan, n_samples=np.nan, n_init=len(iv)))
    rows.append(dict(seed=seed, method="oracle", value=v_orc, ratio=1.0,
                     oracle_value=v_orc, actual_reward=np.nan, where_wrong=0.0,
                     when_error=0.0, q_vs_value_rho=np.nan, q_gap=np.nan,
                     n_samples=np.nan, n_init=len(oplan)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="L1", choices=["L0", "L1", "L2", "L3"])
    ap.add_argument("--out", default="results_fqi")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--ep-each", type=int, default=20,
                    help="每种行为策略的回合数（随机/近视/时间感知贪心各这么多）")
    ap.add_argument("--n-iter", type=int, default=None,
                    help="FQI 迭代次数，默认 = 决策期长度")
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--quota", type=int, default=3)
    ap.add_argument("--prescreen", type=int, default=20)
    ap.add_argument("--foresight", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--with-bc", action="store_true",
                    help="主表加入 BC 行（v19 监督时空打分器，教师=真实 EV，"
                         "与 v19 的 --bc-init forecast --bc-only 同配置）")
    ap.add_argument("--dump-model", action="store_true",
                    help="落盘 Q 模型与训练特征，供 S3-B 的 Q 诊断复用，不必重训")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.stage == "L0":
        l0_check(); return

    if a.stage == "L1":
        l0_check(0.5)
        r = run_synth(0.5, a.seeds[0], a.ep_each, a.n_iter)
        print(f"\nL1  α=0.5 seed={a.seeds[0]}  相对oracle {r['ratio']:.3f}  "
              f"Where 召回 {r['where_recall']:.3f}  When MAE {r['when_mae']:.2f}  "
              f"Q与即时价值秩相关 {r['q_vs_value_rho']:.3f}  "
              f"样本 {r['n_samples']}")
        json.dump(r, open(os.path.join(a.out, "l1.json"), "w"),
                  ensure_ascii=False, indent=1, default=float)
        return

    if a.stage == "L2":
        l0_check(0.5)
        rows = []
        for al in a.alphas:
            rows += baselines_synth(al, a.seeds)
            for s in a.seeds:
                rows.append(run_synth(al, s, a.ep_each, a.n_iter, verbose=False))
                print(f"  α={al:g} seed={s} FQI 相对oracle "
                      f"{rows[-1]['ratio']:.3f}")
        D = pd.DataFrame(rows)
        D.to_csv(os.path.join(a.out, "fqi_synth_runs.csv"), index=False,
                 encoding="utf-8-sig")
        Sm = (D.groupby(["alpha", "method"], sort=False)
              .agg(n=("ratio", "size"), 相对oracle=("ratio", "mean"),
                   标准差=("ratio", "std"), Where召回=("where_recall", "mean"),
                   When_MAE=("when_mae", "mean"),
                   Q与价值秩相关=("q_vs_value_rho", "mean")).reset_index())
        Sm.to_csv(os.path.join(a.out, "fqi_synth_summary.csv"), index=False,
                  encoding="utf-8-sig")
        print("\n" + Sm.round(3).to_string(index=False))
        print("\n判据：Gate A FQI > 纯空间排序；Gate B ≈ 时间感知贪心；"
              "Gate C > 时间感知贪心（强结果）")
        return

    rows = []
    for s in a.seeds:
        rows += run_real(a, s)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "fqi_real_runs.csv"), index=False,
             encoding="utf-8-sig")
    Sm = (D.groupby("method", sort=False)
          .agg(n=("ratio", "size"), 相对oracle=("ratio", "mean"),
               标准差=("ratio", "std"), 挑错单元=("where_wrong", "mean"),
               放错年份=("when_error", "mean"),
               Q与价值秩相关=("q_vs_value_rho", "mean")).reset_index())
    Sm.to_csv(os.path.join(a.out, "fqi_real_summary.csv"), index=False,
              encoding="utf-8-sig")
    print("\n" + Sm.round(3).to_string(index=False))
    print("\n判据：FQI ≤ 0.846（BC）= 未通过；0.846~0.894 = 有意义；"
          "> 0.894（时间感知贪心）= 强结果")


if __name__ == "__main__":
    main()
