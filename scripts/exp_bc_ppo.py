# -*- coding: utf-8 -*-
"""exp_bc_ppo.py —— 「先学会选哪里、再让 RL 学什么时候」的最小验证（本机小世界）。

## 这一轮只回答一个问题

前一轮已经定位到：瓶颈不在环境、动作空间、特征或网络容量，而在 RL 的信用分配
（同一批 50 维特征、同样大小的网络，改用**监督**目标训练，选择问题完美解决；
同一网络在 PPO 下只到随机水平）。所以这一轮问：

> **把空间选择能力先用监督方式装进 actor，PPO 是否就能把学习能力集中到
> "什么时候动手"上？**

不问"BC 能不能让 PPO 更强"这种笼统问题，也不追求逼近 oracle。

## 严格的单变量控制

**唯一改动是 actor 的初始化。** 环境、奖励、动作空间、特征、掩码、配额、
GAE、γ、学习率、迭代数——全部与随机初始化的对照组逐字相同。
`stop_context` 与 `advantage=mc` 这两个诊断开关这一轮**都不开**，
否则一次变三样，结果无法归因。

## 教师信号：只用当年可见的信息

教师是**近视分数** `dvalue(u, t)`（此刻立项的期望折现交付价值）。它不含
任何未来信息——不读 oracle 指派年、不读未来机会曲线、不读未来回报。
否则 BC→PPO 就失去意义（等于把答案直接写进初始化）。

监督目标用**打分回归**而非"前 3 名二分类"：前一轮实测，595 个候选里
"前 3 名重叠率"只有 0.05 而价值比是 1.00（顶端近似平手），
命中率型指标会把一个已经最优的打分器判成失败。

## BC 不学 STOP

候选矩阵末行是「今年到此为止」。BC 只训练真实候选行，**不把 STOP 作为
监督目标**——STOP 是时序决策，正是要留给 PPO 去学的东西。若 BC 一开始就教
"停或不停"，"选哪里"与"什么时候"就又混在一起了。

## 只训练 enc + score

`Pointer` 的 actor 是 `enc + score`，critic 是 `enc + val`。BC 只碰前者；
加载进 PPO 时 critic 保持随机初始化（critic 的目标依赖策略本身，
用监督表示初始化它没有意义，还会把两件事混在一起）。

用法：
    PYTHONPATH=src python scripts/exp_bc_ppo.py --out results_bcppo \\
        --iters 200 --seeds 0
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_timing_world import TimingWorld              # noqa: E402

from tpmorl.rl import train_ppo as TP                 # noqa: E402


def ev_table(w):
    """小世界的 EV 表：EV[u, t] = 此刻立项的期望折现交付价值（教师与计价同源）。"""
    return np.array([[w.dvalue(u, t) for t in range(w.T)] for u in range(w.n)])


def assign_best(EV, quota, rows=None):
    """配额下的最优时间分配（二分图最大权指派，精确解）。rows=None 表示全体。"""
    n, T = EV.shape
    rows = np.arange(n) if rows is None else np.asarray(sorted(rows))
    years = [y for y in range(T) for _ in range(quota)]
    if not len(rows) or not years:
        return 0.0, {}
    M = EV[np.ix_(rows, years)]
    ri, ci = linear_sum_assignment(-M)
    plan, tot = {}, 0.0
    for r, c in zip(ri, ci):
        if M[r, c] <= 0:
            continue
        plan[int(rows[r])] = years[c]
        tot += float(M[r, c])
    return tot, plan


def decompose(EV, init, quota, v_oracle):
    """把缺口拆成"挑错单元"与"放错年份"（与 exp_temporal_gate 同口径）。

    对**同一批被选中的单元**重解一次配额约束下的最优时序，而不是直接换成
    oracle 年份——后者会让多个单元挤到同一年、突破名额，给出不可行的上界。
    反事实分解，非严格可加的因果分解。
    """
    if not init:
        return dict(v_actual=0.0, v_best_timing=0.0,
                    loss_timing=np.nan, loss_unit=np.nan)
    v_best, _ = assign_best(EV, quota, rows=list(init))
    v_act = float(sum(EV[u, t] for u, t in init.items()))
    return dict(v_actual=v_act, v_best_timing=float(v_best),
                loss_timing=float(v_best - v_act),
                loss_unit=float(v_oracle - v_best))


# --------------------------------------------------------------------------
# Phase A：BC（只训练 enc + score，只用当年可见信息，不学 STOP）
# --------------------------------------------------------------------------
def collect_bc_data(w, EV, n_ep=40, seed=0):
    """沿随机策略走，收集 (候选行特征, 教师分数)。**排除 STOP 行。**

    用随机策略而非近视策略来采样状态：近视策略只会访问"它自己走出来的"状态，
    而 PPO 训练初期处在近乎随机的状态分布上。状态覆盖不匹配，BC 的权重装进
    PPO 后立刻失效——这是 BC→RL 最常见的失败方式。
    """
    rng = np.random.default_rng(seed)
    Xs, ys, yr = [], [], []
    for ep in range(n_ep):
        w.reset(seed=ep)
        for t in range(w.T):
            X, meta, cost, units = w.pairs()
            real = [(i, m[0]) for i, m in enumerate(meta) if m[0] >= 0]
            for i, u in real:
                Xs.append(X[i]); ys.append(EV[u, t]); yr.append(t)
            act = []
            if real and rng.random() < 0.8:        # 20% 的年份留空，覆盖"停"后的状态
                i, u = real[int(rng.integers(len(real)))]
                act = [meta[i]]
            if w.step(act)[2]:
                break
    return (np.asarray(Xs, np.float32), np.asarray(ys, np.float32),
            np.asarray(yr, int))


def train_bc(X, y, yr, nf, iters=600, lr=3e-3, seed=0, holdout_from=None):
    """监督打分回归。返回 (state_dict 仅含 enc+score, 指标)。

    网络结构与 `Pointer` 的 actor **逐层相同**，故权重可以直接对号入座地加载。
    """
    torch.manual_seed(seed)
    net = TP.Pointer()                       # 用同一个类，保证结构与命名一致
    cut = holdout_from if holdout_from is not None else int(yr.max() * 0.7) + 1
    tr, te = yr < cut, yr >= cut
    if te.sum() < 10:                        # 小世界年份少时退化为随机切分
        rng = np.random.default_rng(seed)
        m = rng.random(len(y)) < 0.75
        tr, te = m, ~m
    ym, ysd = y[tr].mean(), y[tr].std() + 1e-8
    # 只优化 actor 的参数：enc + score。val（critic）不参与。
    params = list(net.enc.parameters()) + list(net.score.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    Xt = torch.as_tensor(X[tr])
    yt = torch.as_tensor((y[tr] - ym) / ysd)
    for _ in range(iters):
        idx = torch.randint(0, len(Xt), (min(1024, len(Xt)),))
        z = net.enc(Xt[idx])
        pred = net.score(z).squeeze(-1)
        loss = ((pred - yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        pr = net.score(net.enc(torch.as_tensor(X[te]))).squeeze(-1).numpy()
    yte, yrte = y[te], yr[te]
    rho = float(spearmanr(pr, yte).statistic)
    # 逐年"前 quota 名的价值比"——比命中率有意义（顶端常近似平手）
    vr, rr = [], []
    rng = np.random.default_rng(seed)
    for t in np.unique(yrte):
        m = yrte == t
        if m.sum() < 3:
            continue
        p, tv = pr[m], yte[m]
        best = np.sort(tv)[-1:].sum()
        if best <= 0:
            continue
        vr.append(tv[np.argsort(-p)[:1]].sum() / best)
        rr.append(tv[rng.permutation(int(m.sum()))[:1]].sum() / best)
    sd = {k: v for k, v in net.state_dict().items()
          if k.startswith("enc.") or k.startswith("score.")}
    return sd, dict(bc_rho=rho, bc_value_ratio=float(np.mean(vr)) if vr else np.nan,
                    bc_value_ratio_random=float(np.mean(rr)) if rr else np.nan,
                    n_train=int(tr.sum()), n_test=int(te.sum()))


# --------------------------------------------------------------------------
# 评估：贪心走一遍，记录立项方案与"停"的频率
# --------------------------------------------------------------------------
def rollout(w, net=None, kind="ppo", EV=None, rng=None):
    init, n_stop, n_year = {}, 0, 0
    w.reset(seed=0)
    for t in range(w.T):
        X, meta, cost, units = w.pairs()
        real = [(i, m[0]) for i, m in enumerate(meta) if m[0] >= 0]
        if not real:
            w.step([]); continue
        n_year += 1
        if kind == "random":
            i = int(rng.integers(len(meta)))
            pick = None if meta[i][0] < 0 else meta[i]
        elif kind == "myopic":
            i, u = max(real, key=lambda p: EV[p[1], t])
            pick = meta[i]
        else:
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(X))
            i = int(torch.argmax(logits))
            pick = None if meta[i][0] < 0 else meta[i]
        if pick is None:
            n_stop += 1
            w.step([])
        else:
            init.setdefault(int(pick[0]), t)
            w.step([pick])
    return init, (n_stop / n_year if n_year else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_bcppo")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--eps", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--g-target", type=float, default=0.20)
    ap.add_argument("--n-per-kind", type=int, default=2)
    ap.add_argument("--bc-episodes", type=int, default=40)
    ap.add_argument("--bc-iters", type=int, default=600)
    ap.add_argument("--remedies", action="store_true",
                    help="加跑指南对「情况 A」的两个处方：actor 学习率 ÷10、"
                         "先冻结 actor 若干迭代")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    w = TimingWorld(g_target=a.g_target, n_per_kind=a.n_per_kind)
    diag = w.validate()
    EV = ev_table(w)
    v_orc, oplan = assign_best(EV, w.quota)
    print(f"小世界：{w.n} 地块 / T={w.T} / 每年 {w.quota} 个名额 / lead={w.lead}")
    print(f"oracle {v_orc:.1f}，其计划 {dict(sorted(oplan.items()))}")

    # ---- Phase A：BC ----
    Xb, yb, yrb = collect_bc_data(w, EV, n_ep=a.bc_episodes, seed=0)
    sd, bcm = train_bc(Xb, yb, yrb, nf=Xb.shape[1], iters=a.bc_iters, seed=0)
    torch.save(dict(state=sd, **bcm), os.path.join(a.out, "bc_actor.pt"))
    print(f"\n[A] BC：样本 {len(Xb)}（特征 {Xb.shape[1]} 维，已排除 STOP 行）  "
          f"秩相关 rho={bcm['bc_rho']:.3f}  "
          f"当年首选价值比 {bcm['bc_value_ratio']:.3f}"
          f"（随机 {bcm['bc_value_ratio_random']:.3f}）")

    # ---- Phase B：权重接入自证（不训练） ----
    bc_net = TP.Pointer(); bc_net.load_state_dict(sd, strict=False)
    ld_net = TP.Pointer()
    miss = ld_net.load_state_dict(sd, strict=False)
    w.reset(seed=0)
    X0, meta0, _, _ = w.pairs()
    with torch.no_grad():
        s_bc = bc_net(torch.as_tensor(X0))[0].numpy()
        s_ld = ld_net(torch.as_tensor(X0))[0].numpy()
    same = bool(np.allclose(s_bc, s_ld, atol=1e-6))
    crit_random = not any(k.startswith("val.") for k in sd)
    print(f"[B] 接入自证：加载后打分逐位一致 {same}；critic 保持随机 {crit_random}；"
          f"缺失键仅 critic {all(k.startswith('val.') for k in miss.missing_keys)}")
    if not (same and crit_random):
        raise SystemExit("权重接入自证失败 —— 按指南要求，此时不得继续训练")

    # ---- Phase C：随机初始化 PPO vs BC→PPO ----
    rows = []
    for seed in a.seeds:
        arms = [("PPO(随机初始化)", None, 1.0, 0), ("BC→PPO", sd, 1.0, 0)]
        if a.remedies:
            # 指南对"情况 A（BC 好但被 PPO 迅速破坏）"给的两个处方
            arms += [("BC→PPO(actor 学习率 ÷10)", sd, 0.1, 0),
                     ("BC→PPO(先冻结 actor 50 迭代)", sd, 1.0, 50)]
        for tag, init, als, frz in arms:
            ww = TimingWorld(g_target=a.g_target, n_per_kind=a.n_per_kind)
            net, hist = TP.train(ww, iters=a.iters, eps_per_iter=a.eps, seed=seed,
                                 init_actor=init, actor_lr_scale=als,
                                 freeze_actor_iters=frz)
            iv, sf = rollout(ww, net=net, kind="ppo")
            d = decompose(EV, iv, ww.quota, v_orc)
            rows.append(dict(seed=seed, method=tag, stop_freq=sf,
                             ratio=d["v_actual"] / v_orc, n_init=len(iv),
                             init_mean=float(np.mean(list(iv.values()))) if iv else np.nan,
                             years=str(dict(sorted(iv.items()))), **d, **bcm))
            print(f"  seed={seed} {tag}: 价值 {d['v_actual']:.1f}"
                  f"（oracle 的 {d['v_actual']/v_orc:.3f}）  "
                  f"挑错单元损失 {d['loss_unit']:.0f}  放错年份损失 {d['loss_timing']:.0f}  "
                  f"停止频率 {sf:.2f}  立项年 {dict(sorted(iv.items()))}")

    # 参照：BC 自己（不经 PPO）与近视贪心、随机
    for tag, kind, net in (("BC(不经 PPO)", "ppo", bc_net),
                           ("近视贪心", "myopic", None),
                           ("随机", "random", None)):
        ww = TimingWorld(g_target=a.g_target, n_per_kind=a.n_per_kind)
        iv, sf = rollout(ww, net=net, kind=kind, EV=EV,
                         rng=np.random.default_rng(0))
        d = decompose(EV, iv, ww.quota, v_orc)
        rows.append(dict(seed=-1, method=tag, stop_freq=sf,
                         ratio=d["v_actual"] / v_orc, n_init=len(iv),
                         init_mean=float(np.mean(list(iv.values()))) if iv else np.nan,
                         years=str(dict(sorted(iv.items()))), **d))
        print(f"  {tag}: 价值 {d['v_actual']:.1f}（{d['v_actual']/v_orc:.3f}）  "
              f"挑错 {d['loss_unit']:.0f}  放错年份 {d['loss_timing']:.0f}  "
              f"停止频率 {sf:.2f}  立项年 {dict(sorted(iv.items()))}")

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "bcppo_runs.csv"), index=False,
             encoding="utf-8-sig")
    keep = ["method", "ratio", "v_actual", "v_best_timing", "loss_unit",
            "loss_timing", "stop_freq", "init_mean", "n_init"]
    Sm = D.groupby("method", sort=False)[keep[1:]].mean().reset_index()
    Sm.to_csv(os.path.join(a.out, "bcppo_summary.csv"), index=False,
              encoding="utf-8-sig")
    print("\n" + Sm.round(3).to_string(index=False))
    json.dump(dict(vars(a), oracle=v_orc, oracle_plan={str(k): v for k, v in
                                                       oplan.items()},
                   world=diag, bc=bcm),
              open(os.path.join(a.out, "bcppo_config.json"), "w"),
              ensure_ascii=False, indent=1, default=str)
    print("\n及格线（按指南）：① BC 秩相关明显高于随机；"
          "② BC→PPO 的挑错单元损失显著低于随机初始化的 PPO；"
          "③ 立项年不再全挤在最早年份、出现非零的等待。")


if __name__ == "__main__":
    main()
