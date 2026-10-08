# -*- coding: utf-8 -*-
"""exp_counterfactual_credit.py —— Gate 4：信用分配尸检（**不训练**）。

## 只回答一个问题

> **在当前奖励定义下，"今年不做"的真实长期回报，到底有没有高于"现在就做"？**

前面几轮反复出现"策略不等待"，但一直没有直接测过**等待在这个环境里到底值不值**。
若不值，那就不是模型学不会，而是我们以为该等的地方其实不该等——
必须先把这一条钉死，否则后面改算法都是在解一个不存在的问题。

## 做法：逐状态的三支反事实，不训练、不估值

沿一个固定参照策略走到第 t 年，然后把环境状态**复制三份**：

    Case 1  step([])              —— 今年不做（WAIT）
    Case 2  step([最优当期候选])   —— 现在就做（NOW，按参照策略的排序）
    Case 3  step([候选 i])        —— 现在就做某个指定地块

三支都从第 t+1 年起用**同一个固定参照策略**走到终点，于是

    Q(s, WAIT) 与 Q(s, a)

都是**实测的真实折现回报**，不含任何值函数估计——这正是本诊断的意义：
它不受 critic 好坏影响。判据

    Δ_wait(s) = Q(s, WAIT) − max_a Q(s, a)

    Δ_wait > 0  ⇒ 环境里等待确实更值，模型学不会是算法问题
    Δ_wait < 0  ⇒ 当前奖励定义下等待本就不占优，该先修环境/口径

## 参照策略用"冻结场近视"，而不是 oracle

参照策略若用 oracle，Q 值衡量的是"在一个完美后续下等不等更好"，
与训练时策略实际面对的后续无关。用**冻结场近视**（假设今天的条件保持不变，
只按当期排序选，名额用满）：它是"没有前瞻的能干策略"，
正是我们希望 RL 超越的那条基线。

用法：
    PYTHONPATH=src python scripts/exp_counterfactual_credit.py \\
        --out results_cfcredit --n-per-kind 6 --lead 2 --g-target 0.40
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_timing_world import TimingWorld              # noqa: E402
from exp_bc_ppo import assign_best                    # noqa: E402


def tables(w):
    """真实 EV 与冻结场 EV。前者给 oracle 与计价，后者给参照策略与 BC 教师。"""
    EV = np.array([[w.dvalue(u, t) for t in range(w.T)] for u in range(w.n)])
    EVf = np.array([[w.dvalue_frozen(u, t) for t in range(w.T)]
                    for u in range(w.n)])
    return EV, EVf


def ref_rollout(w, EVf, EV):
    """固定参照策略：冻结场近视（按当期冻结估计排序，名额用满）。

    返回从当前状态走到终点所累积的**真实**折现价值。注意价值用真实 EV 计，
    排序用冻结 EVf —— 这正是"决策时只看得到当期条件，但后果按真实机制结算"。
    """
    tot = 0.0
    for t in range(w.t, w.T):
        X, meta, cost, units = w.pairs()
        real = [(i, m[0]) for i, m in enumerate(meta) if m[0] >= 0]
        if not real:
            if w.step([])[2]:
                break
            continue
        real.sort(key=lambda p: -EVf[p[1], t])
        act = []
        for i, u in real[:w.quota]:
            if EVf[u, t] <= 0:
                continue
            act.append(meta[i])
            tot += float(EV[u, t])
        if w.step(act)[2]:
            break
    return tot


def real_env(dataset, horizon=25, quota=3, budget=1e12):
    """真实 717 单元环境 + 其冻结场 EV 表（与 exp_temporal_gate 同口径）。

    为什么必须在真实环境上重测一遍：小世界给出的阈值是
    "候选数 ≥ 名额数 ⇒ 等待被支配"。真实环境是 717 个单元抢 75 个名额
    （约 9.6:1），按这个阈值等待应当处处被支配——但**小世界的结论不能直接
    外推**（上一轮已经吃过一次亏：无决策步的修正对 717 单元是空操作）。
    这里直接测。
    """
    import numpy as np
    from tpmorl.rl import scenario as SC
    from tpmorl.rl.env_gym import RenewalEnv
    from tpmorl.objectives.reward import OBJ_NAMES
    from tpmorl.eval import ev as EVM
    from tpmorl.env import schedule as S

    SC.reset()
    SC.apply(horizon=horizon, horizon_eval="auto", opp_shape="window",
             a_plan=2.4, a_infra=1.8, a_age=0.9, a_ready=1.0, foresight=0,
             quota=quota, budget=budget)
    w = np.zeros(len(OBJ_NAMES)); w[OBJ_NAMES.index("Floor")] = 1.0
    env = RenewalEnv(dataset, T=horizon, T_eval=SC.horizon_eval(),
                     weights=w, scale=np.ones(len(OBJ_NAMES)))
    env.reset(seed=7); env.budget = budget
    env.fix_one_target_per_unit()
    base = np.asarray(env.farcap, float) * np.asarray(env.ncell, float)
    by = np.array([S.BUILD_YEARS_BY_CHANNEL[int(c)] for c in env.ch], int)
    EV = EVM.ev_matrix(env.opp, base, env.T, env.T_eval, by, S.HAZARD,
                       int(S.TAU_VALID + S.TAU_EXT), gamma=0.95)
    # 冻结场：把机会场冻结在当年，逐年重算一张表 —— 决策时"假设条件不变"
    return env, EV


def ref_rollout_real(env, EV, budget):
    """真实环境的参照策略：按当期 EV 排序、名额用满。返回真实折现价值增量。"""
    import numpy as np
    tot = 0.0
    for t in range(env.t, env.T):
        X, meta, cost, units = env.pairs()
        real = [(i, m[0]) for i, m in enumerate(meta) if m[0] >= 0]
        if not real:
            if env.step([])[2]:
                break
            continue
        real.sort(key=lambda p: -EV[p[1], min(t, EV.shape[1] - 1)])
        act, used = [], set()
        for i, u in real:
            if len(act) >= env.quota:
                break
            if u in used:
                continue
            act.append(meta[i]); used.add(u)
            tot += float(EV[u, min(t, EV.shape[1] - 1)])
        if env.step(act)[2]:
            break
    return tot


def run_real(a):
    """真实环境上的 Δ等 诊断。"""
    import numpy as np
    env, EV = real_env(a.dataset, a.horizon, a.quota)
    print(f"真实环境：{env.n} 单元 / 每年 {env.quota} 个名额 / T={env.T}"
          f"（候选:名额 ≈ {env.n / (env.T * env.quota):.1f}:1）")
    rows = []
    drive = env
    for t in range(min(a.real_years, drive.T)):
        X, meta, cost, units = drive.pairs()
        real = [(i, m[0]) for i, m in enumerate(meta) if m[0] >= 0]
        if not real:
            if drive.step([])[2]:
                break
            continue
        real.sort(key=lambda p: -EV[p[1], min(t, EV.shape[1] - 1)])

        w_wait = copy.deepcopy(drive)
        w_wait.step([])
        q_wait = ref_rollout_real(w_wait, EV, drive.budget)

        qs = []
        for i, u in real[:a.max_cands]:
            w_act = copy.deepcopy(drive)
            w_act.step([meta[i]])
            q = float(EV[u, min(t, EV.shape[1] - 1)]) + \
                ref_rollout_real(w_act, EV, drive.budget)
            qs.append((u, q))
        u_best, q_best = max(qs, key=lambda p: p[1])
        rows.append(dict(year=t, n_cand=len(real), q_wait=q_wait,
                         q_best_act=q_best, unit_best=u_best,
                         delta_wait=q_wait - q_best))
        print(f"  t={t:>2}  Q(等)={q_wait:>8.1f}  max Q(做)={q_best:>8.1f}  "
              f"Δ等={q_wait - q_best:>+8.1f}")
        act, used = [], set()
        for i, u in real:
            if len(act) >= drive.quota:
                break
            if u in used:
                continue
            act.append(meta[i]); used.add(u)
        if drive.step(act)[2]:
            break
    D = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    D.to_csv(os.path.join(a.out, "cfcredit_real_rows.csv"), index=False,
             encoding="utf-8-sig")
    pos = int((D.delta_wait > 0).sum())
    print(f"\n真实环境 Δ等 > 0 的年份：{pos}/{len(D)}  "
          f"中位 {D.delta_wait.median():+.2f}  最大 {D.delta_wait.max():+.2f}")
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_cfcredit")
    ap.add_argument("--n-per-kind", type=int, default=6)
    ap.add_argument("--lead", type=int, default=2)
    ap.add_argument("--g-target", type=float, default=0.40)
    ap.add_argument("--value-at", default="done", choices=["init", "done"])
    ap.add_argument("--world", default="small", choices=["small", "real"],
                    help="small=三型小世界；real=717 单元真实环境（转移的关键校验）")
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--quota", type=int, default=3)
    ap.add_argument("--real-years", type=int, default=8,
                    help="真实环境里考察前多少年（每年 1+max_cands 次整程 rollout）")
    ap.add_argument("--max-cands", type=int, default=8,
                    help="每个状态最多考察多少个候选（按冻结估计取前若干），"
                         "控制反事实数量")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.world == "real":
        run_real(a)
        return

    def mk():
        w = TimingWorld(g_target=a.g_target, n_per_kind=a.n_per_kind,
                        lead=a.lead, value_at=a.value_at)
        w.reset(seed=0)
        return w

    w0 = mk()
    EV, EVf = tables(w0)
    v_orc, oplan = assign_best(EV, w0.quota)
    slots = (w0.T - w0.lead) * w0.quota
    print(f"世界：{w0.n} 地块 / {slots} 个名额 / T={w0.T} / lead={w0.lead} / "
          f"价值读{'交付年' if a.value_at == 'done' else '立项年'}")
    print(f"oracle {v_orc:.1f}")

    rows = []
    # 沿参照策略推进，在每一年做三支反事实
    drive = mk()
    for t in range(drive.T - drive.lead):
        X, meta, cost, units = drive.pairs()
        real = [(i, m[0]) for i, m in enumerate(meta) if m[0] >= 0]
        if not real:
            if drive.step([])[2]:
                break
            continue
        real.sort(key=lambda p: -EVf[p[1], t])

        # Case 1：WAIT
        w_wait = copy.deepcopy(drive)
        w_wait.step([])
        q_wait = ref_rollout(w_wait, EVf, EV)

        # Case 2/3：NOW（逐候选）
        qs = []
        for i, u in real[:a.max_cands]:
            w_act = copy.deepcopy(drive)
            w_act.step([meta[i]])
            q = float(EV[u, t]) + ref_rollout(w_act, EVf, EV)
            qs.append((u, q))
        u_best, q_best = max(qs, key=lambda p: p[1])
        u_frozen = real[0][1]                  # 参照策略自己会挑的那个
        q_frozen = dict(qs)[u_frozen]

        rows.append(dict(
            year=t, n_cand=len(real), q_wait=q_wait, q_best_act=q_best,
            unit_best=u_best, q_frozen_pick=q_frozen, unit_frozen=u_frozen,
            delta_wait=q_wait - q_best,
            delta_wait_vs_frozen=q_wait - q_frozen,
            # 参照策略挑的那个与"最优当期动作"差多少 —— Where 的可改进量
            gap_frozen_pick=q_best - q_frozen))
        print(f"  t={t:>2}  Q(等)={q_wait:>7.1f}  max Q(做)={q_best:>7.1f}"
              f"(地块{u_best})  Δ等={q_wait-q_best:>+7.1f}   "
              f"冻结近视会挑地块{u_frozen}→{q_frozen:.1f}")
        # 参照策略自己走一步，继续推进
        act = [meta[i] for i, u in real[:drive.quota] if EVf[u, t] > 0]
        if drive.step(act)[2]:
            break

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "cfcredit_rows.csv"), index=False,
             encoding="utf-8-sig")
    pos = int((D.delta_wait > 0).sum())
    print(f"\n{'='*66}\nΔ等 > 0 的年份：{pos}/{len(D)}"
          f"（占 {pos/max(len(D),1):.0%}）  "
          f"中位 Δ等 {D.delta_wait.median():+.2f}  "
          f"最大 {D.delta_wait.max():+.2f}")
    pos2 = int((D.delta_wait_vs_frozen > 0).sum())
    print(f"Δ等（对比参照策略自己会挑的那个）> 0：{pos2}/{len(D)}"
          f"  中位 {D.delta_wait_vs_frozen.median():+.2f}")
    print(f"Where 的可改进量（最优当期动作 − 冻结近视所挑）中位 "
          f"{D.gap_frozen_pick.median():+.2f}")
    json.dump(dict(vars(a), oracle=v_orc, slots=slots,
                   n_year=len(D), n_delta_pos=pos),
              open(os.path.join(a.out, "cfcredit_config.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n判读（按指南 Gate 4）：Δ等 > 0 大量存在 ⇒ 环境里确有 When 信号，"
          "模型学不会是算法问题，进入 Gate 5；若极少 ⇒ 先修环境口径，不要改 PPO。")


if __name__ == "__main__":
    main()
