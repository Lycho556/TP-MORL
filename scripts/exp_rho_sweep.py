# -*- coding: utf-8 -*-
"""exp_rho_sweep.py —— 候选/容量比 ρ 的机制扫描。

## 这个实验回答什么

v20 留下的唯一硬缺口：合成世界里 Double-FQI 与时间感知贪心**逐位相同**
（α≥0.5，三种子标准差 0.000），而光明区 FQI 0.904 > 贪心 0.894。两个方向相反的
结果只有一个候选解释：

    ρ = N / (K · T_eff)          候选数 / 累计名额

    合成   24 / 18 = 1.33        名额宽裕 → 大多数候选最终都能排上
                                 → 「今年选谁」几乎不改变后续可选集
                                 → 延续项在年内近似常数 → 不改变 top-K
    光明区 717 / 75 = 9.56       名额紧张 → 今年的选择改变明年的竞争

但两个点只能支撑「容量压力区分了两种情形」，**定位不了分界**。本脚本把 ρ 做成
自变量，其余结构全部固定，看 margin = ratio(FQI) − ratio(时间感知贪心)
随 ρ 如何变化。

## 累计名额的口径（这里最容易算错）

`TimingWorld.validate()` 第 454 行：

    slots = max(T - lead, 0) * quota

默认 `lead=1`，故 T=10 / quota=2 时累计名额是 **(10−1)×2 = 18**，不是 20。
`results_fqi/l1.json` 的 `n_init=18` 与之逐位吻合。所以

    ρ = N / (K · (T − lead)) = 3·n_per_kind / 18 = n_per_kind / 6

取 n_per_kind ∈ {8, 12, 24, 48, 60} 得到 ρ ∈ {1.33, 2, 4, 8, 10}，**精确落点**。

## 只动一个量

变的只有 `n_per_kind`（即 N）。T、quota、lead、alpha、suit 档位、机会窗宽度、
三种峰值型的构成全部不变 —— 于是 ρ 的变化不掺入「机会结构也变了」这个混淆。
suit 档位按 `i % len(suit)` 循环分配，放大 n_per_kind 时各档比例保持不变。

用法：
    PYTHONPATH=src python scripts/exp_rho_sweep.py --out results_rho \
        --seeds 0 1 2 3 4 --alpha 0.5 --ep-each 20
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_timing_world import TimingWorld                              # noqa: E402
from exp_where_when import (value_matrix, spatial_matrix,              # noqa: E402
                            oracle as synth_oracle, run_rank,
                            where_when_metrics)
from exp_fqi_local import make_trees                                   # noqa: E402

from tpmorl.rl import fqi                                              # noqa: E402

# 与 exp_fqi_local.SYNTH 完全一致，n_per_kind 由 ρ 决定故不写在这里
BASE = dict(T=10, quota=2, suit=[1.0, 0.8, 0.6, 0.45], width=1.8,
            value_at="init")
N_KINDS = 3          # exp_timing_world.KINDS = (early, late, verylate)


def rho_to_nper(rho, T=10, quota=2, lead=1):
    """ρ → n_per_kind。返回整数与其**实际**达到的 ρ（不假装等于目标值）。"""
    slots = max(T - lead, 0) * quota
    nper = int(round(rho * slots / N_KINDS))
    return nper, (N_KINDS * nper) / slots


# ---------------------------------------------------------------- 两条臂
#
# 第一版只做了 inflate 臂（放大 N、固定名额），实测 margin 在每一档都是 0，
# 且 ρ≥4 起 temporal greedy 直接打到 1.000。诊断结果是**这条臂被混淆了**：
#
#     TimingWorld 的地块只有 3 种峰值型 × 4 档 suit = **12 个原型**，
#     n_per_kind=60 时每个原型有 15 份**完全相同**的拷贝
#     （实测 distinct value rows 恒为 12，与 N 无关）。
#
# 于是放大 N 不是加剧竞争，而是给每个原型加替补：ρ=10 时有 15 个地块在同一年
# 同时达到峰值 91.5，而名额只有 18 个，贪心可以把每个名额都填上处于峰值的
# 地块，因此恰好达到最优。**"候选变多"在这个世界里让问题变简单了。**
#
# 真正隔离"容量稀缺"的做法是反过来：**地块总体固定，只收紧名额**。
#     tighten 臂：N=24、T=10 固定，quota ∈ {3,2,1} → slots ∈ {27,18,9}
#                 → ρ ∈ {0.89, 1.33, 2.67}
# 地块构成、峰值分布、机会窗宽度一字不动，唯一变化的是每年能推几个。
def quota_to_rho(quota, nper=8, T=10, lead=1):
    slots = max(T - lead, 0) * quota
    return (N_KINDS * nper) / slots, slots


def setup(nper, alpha, quota=None):
    kw = dict(BASE)
    if quota is not None:
        kw["quota"] = quota
    w0 = TimingWorld(n_per_kind=nper, alpha=alpha, **kw)
    w0.reset(seed=0)
    V = value_matrix(w0)
    S = spatial_matrix(w0)
    v_orc, oplan = synth_oracle(V, w0.quota)
    Vm = np.repeat(V[:, :1], V.shape[1], axis=1)
    return w0, V, S, Vm, v_orc, oplan, kw


def _mk(w):
    w.reset(seed=0)
    return w


def run_one(rho_target, alpha, seeds, ep_each, n_iter, nper=None, quota=None):
    """inflate 臂给 rho_target（换算 n_per_kind）；tighten 臂直接给 nper+quota。"""
    if quota is None:
        nper, rho = rho_to_nper(rho_target)
    else:
        rho, _ = quota_to_rho(quota, nper)
    w0, V, S, Vm, v_orc, oplan, kw = setup(nper, alpha, quota)
    diag = w0.validate()
    n_units, slots = int(diag["n_units"]), int(diag["slots"])
    assert abs(n_units / slots - rho) < 1e-9, "ρ 口径与环境自证不一致"

    rows = []
    arche = len({(w0.kind[u], round(float(w0.base[u]), 6)) for u in range(w0.n)})

    def rec(tag, val, init, seed=-1, **extra):
        m = where_when_metrics(init, oplan, V)
        rows.append(dict(rho_target=rho_target, rho=rho, n_units=n_units,
                         slots=slots, quota=int(w0.quota), archetypes=arche,
                         alpha=alpha, seed=seed, method=tag,
                         value=val, ratio=val / v_orc, oracle_value=v_orc,
                         **m, **extra))

    # 三条基线（手写规则，不吃种子的只跑一次）
    for s in seeds:
        v, iv = run_rank(TimingWorld(n_per_kind=nper, alpha=alpha, **kw),
                         S, V, rng=np.random.default_rng(s))
        rec("random", v, iv, s)
    v_sp, iv = run_rank(TimingWorld(n_per_kind=nper, alpha=alpha, **kw), S, V)
    rec("spatial_ranking", v_sp, iv)
    v_tg, iv = run_rank(TimingWorld(n_per_kind=nper, alpha=alpha, **kw), V, V)
    rec("temporal_greedy", v_tg, iv)
    rec("oracle", v_orc, oplan)

    # Double-FQI
    env_fn = lambda: _mk(TimingWorld(n_per_kind=nper, alpha=alpha, **kw))
    for s in seeds:
        data = fqi.collect(env_fn,
                           lambda u, t: float(V[u, min(t, V.shape[1] - 1)]),
                           {"random": None, "myopic": Vm,
                            "temporal_greedy": V},
                           n_ep_each=ep_each, seed=s)
        qa, qb, _ = fqi.fit_double(data, w0.gamma, make_trees(seed=s),
                                   n_iter=n_iter, verbose=False)
        init, R, dg = fqi.rollout(env_fn, qa, qb, None, collect_q=True)
        v = float(sum(V[u, t] for u, t in init.items()))
        rec("double_fqi", v, init, s,
            q_vs_value_rho=fqi.q_vs_value_corr(data, qa, qb, None),
            n_samples=sum(len(d["r"]) for d in data))
        print(f"  rho={rho:5.2f} (N={n_units}, slots={slots}) seed={s}  "
              f"FQI {v / v_orc:.4f}  TG {v_tg / v_orc:.4f}  "
              f"margin {(v - v_tg) / v_orc:+.4f}", flush=True)
    return rows, v_tg / v_orc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_rho")
    ap.add_argument("--rhos", type=float, nargs="+",
                    default=[1.33, 2.0, 4.0, 8.0, 10.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="机会结构固定在这一档；0.5 是合成世界里 FQI 与贪心"
                         "逐位相同的那一档，故最能看出 margin 何时出现")
    ap.add_argument("--ep-each", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=None)
    ap.add_argument("--mode", default="inflate", choices=["inflate", "tighten"],
                    help="inflate=放大 N 固定名额（已知被原型冗余混淆）；"
                         "tighten=地块总体固定、只收紧 quota（干净的容量工具）")
    ap.add_argument("--quotas", type=int, nargs="+", default=[3, 2, 1],
                    help="tighten 臂的每年名额")
    ap.add_argument("--nper", type=int, default=8,
                    help="tighten 臂固定的 n_per_kind")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    all_rows = []
    if a.mode == "inflate":
        jobs = [(rt, None, None) for rt in a.rhos]
    else:
        jobs = [(quota_to_rho(q, a.nper)[0], a.nper, q) for q in a.quotas]
    for rt, nper, quota in jobs:
        if quota is None:
            nper_, rho = rho_to_nper(rt)
            print(f"[inflate ρ={rt}] n_per_kind={nper_} → 实际 ρ={rho:.4f}", flush=True)
        else:
            rho, slots = quota_to_rho(quota, nper)
            print(f"[tighten quota={quota}] N={N_KINDS*nper} slots={slots} "
                  f"→ ρ={rho:.4f}", flush=True)
        rows, _ = run_one(rt, a.alpha, a.seeds, a.ep_each, a.n_iter,
                          nper=nper, quota=quota)
        all_rows += rows
        pd.DataFrame(all_rows).to_csv(
            os.path.join(a.out, "rho_runs.csv"), index=False)

    df = pd.DataFrame(all_rows)
    g = (df.groupby(["rho", "n_units", "slots", "quota", "method"])
           .agg(ratio=("ratio", "mean"), sd=("ratio", "std"),
                n=("ratio", "size"))
           .reset_index())
    piv = g.pivot_table(index=["rho", "n_units", "slots", "quota"],
                        columns="method", values="ratio").reset_index()
    piv["margin"] = piv["double_fqi"] - piv["temporal_greedy"]
    sd = (df[df.method == "double_fqi"].groupby("rho")["ratio"].std()
          .rename("fqi_sd").reset_index())
    piv = piv.merge(sd, on="rho", how="left")
    piv["mode"] = a.mode
    g.to_csv(os.path.join(a.out, "rho_summary_long.csv"), index=False)
    piv.to_csv(os.path.join(a.out, "rho_summary.csv"), index=False)
    json.dump(dict(mode=a.mode, alpha=a.alpha, seeds=a.seeds,
                   ep_each=a.ep_each, base=BASE, n_kinds=N_KINDS,
                   slots_formula="(T - lead) * quota, lead=1"),
              open(os.path.join(a.out, "rho_config.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n" + piv.to_string(index=False))


if __name__ == "__main__":
    main()
