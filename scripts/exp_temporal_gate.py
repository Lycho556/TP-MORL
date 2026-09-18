# -*- coding: utf-8 -*-
"""exp_temporal_gate.py —— 真实 717 单元规模上的**时序选择**闸。

## 为什么判据要从 opp_lift 换掉

上一轮打算用"PPO 的 opp_lift 是否显著高于随机"作为上服务器的最后一道闸。
这个判据不够：**它最多证明策略会挑当期机会高的单元，而这不是择时。**

反例很简单。两个单元、每年一个名额：

    A：0.4 → 0.6 → 0.9 → 0.7      （现在一般，第 3 年才到峰值）
    B：0.8 → 0.8 → 0.8 → 0.8      （一直不错）

一个完全没有时序推理的策略"每年挑当期最高的"会先做 B、再做 A，
它的 opp_at_init 已经不差，于是 `opp_lift > 随机` 轻易成立。但它从未学到
"A 现在只有 0.4，可我知道第 3 年会到 0.9，所以该把它留到后面"——
而后者才是本项目要证明的东西。

所以本脚本让四个策略同台比：

    随机        什么都不会时能得到什么（下限）
    近视贪心    **只看当期**机会挑最好的，不用任何未来信息（强基线）
    PPO         待检验
    oracle      知道全部未来、在配额下做全局最优时间分配（上限）

要证明的不是 PPO > 随机，而是

    PPO 相对**近视贪心**缩小了 oracle 缺口：

        缺口收窄率 = (V_PPO − V_近视) / (V_oracle − V_近视)

    = 0 意味着 PPO 与只看当期的贪心没有区别；> 0 才意味着它真的用上了时序信息。

## 为什么这一版把预算放松

本闸只问一件事：**在 717 单元、每年 3 个名额、有限机会窗下，RL 能不能做
时序选择。** 把预算、资金承诺、审批、老化、设施全部同时打开，任何一个负结果
都无法归因。故预算设为不咬（`--budget` 极大），其余制度机制保留——
审批风险率与有效期是本项目的核心机制，去掉它就不是这个问题了。

## 与三地块门槛世界的关系

两个世界的决策问题**不同**，不能互相替代：

    三地块门槛世界   3 个地块 / 11 个名额，名额松 -> "等" = 今年**留空**
    717 单元真实环境 717 单元 / 75 个名额，年年咬 -> "等" = 今年做**别的**，
                                                    把这个留到它的窗口

前者测的是"要不要空一年"（掩码指针的弱项），后者测的是"今年挑谁"
（掩码指针能表达的事）。上一轮在小世界得到的负结果不能外推到这里——
实测真实环境里合法动作数 ≤1 的年份占比是 0/75，小世界那个优势归一化的缺陷
在这里根本不存在。

用法：
    PYTHONPATH=src python scripts/exp_temporal_gate.py --out results_tgate \\
        --iters 150 --eps 8 --seeds 0 1 2
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from tpmorl.env import schedule as S
from tpmorl.eval import ev as EVM
from tpmorl.objectives.reward import OBJ_NAMES
from tpmorl.rl import env_gym as EG
from tpmorl.rl import scenario as SC
from tpmorl.rl import train_ppo as TP
from tpmorl.rl.env_gym import RenewalEnv


def gate_weights(objective, alpha):
    """本闸用的偏好权重。

    `floor`：**只押 Floor**，其余目标权重 0。这一档是判定择时能力的正解——
    oracle 与三个策略都按"期望折现交付建面"计价，若学习器优化的是 11 个目标的
    加权和（含成本、失效、生态等），那么它在本口径上得分低**未必**是不会择时，
    也可能是它按设计牺牲了建面去换别的目标。目标不一致时，任何负结果都不能
    归因到时序能力，所以这一档必须有。

    `mix`：沿用 weight_vector(alpha) 的多目标权重。它回答的是另一个问题
    ——"论文主组那套偏好下，策略的时序表现如何"——两档都报，但**判定只看 floor**。
    """
    if objective == "floor":
        w = np.zeros(len(OBJ_NAMES))
        w[OBJ_NAMES.index("Floor")] = 1.0
        return w
    return TP.weight_vector(alpha)


def build_env(dataset, T, alpha, seed, budget, objective="floor"):
    env = RenewalEnv(dataset, T=T, T_eval=SC.horizon_eval(),
                     weights=gate_weights(objective, alpha),
                     scale=np.ones(len(OBJ_NAMES)))
    env.reset(seed=seed)
    env.budget = float(budget)          # 预算不咬：本闸只测时序选择
    return env


def ev_tables(env, gamma):
    """真实 EV（读交付年的场取值）与**近视** EV（假设今年的条件一直不变）。

    近视 EV 是"没有前瞻的能干策略"该有的样子：它会正确地按当期条件排序，
    但不知道条件会变。两张表的差就是"未来信息值多少"。
    """
    base = np.asarray(env.farcap, float) * np.asarray(env.ncell, float)
    by = np.array([S.BUILD_YEARS_BY_CHANNEL[int(c)] for c in env.ch], int)
    tau = int(S.TAU_VALID + S.TAU_EXT)
    EV = EVM.ev_matrix(env.opp, base, env.T, env.T_eval, by, S.HAZARD, tau,
                       gamma=gamma)

    class Frozen:
        """把机会场冻结在第 t0 年：逐年查表一律返回 t0 年的取值。"""

        def __init__(self, opp, t0):
            self.opp, self.t0 = opp, t0
            self.kind = opp.kind

        def admit_prob(self, t):
            return self.opp.admit_prob(self.t0)

        def hazard_mult(self, t):
            return self.opp.hazard_mult(self.t0)

        def value_mult(self, u, t_init, t_done):
            return self.opp.value_mult(u, self.t0, self.t0)

    EVm = np.zeros_like(EV)
    for t0 in range(env.T):
        # 只需要第 t0 列，故对每个 t0 只算一列（T 列共 T 次，秒级）
        col = EVM.ev_matrix(Frozen(env.opp, t0), base, t0 + 1, env.T_eval, by,
                            S.HAZARD, tau, gamma=gamma)
        EVm[:, t0] = col[:, t0]
    return EV, EVm


def oracle_plan(EV, quota, eligible):
    """配额约束下的全局最优时间分配（二分图最大权指派，精确解）。

    每一年复制 `quota` 列；每个单元一行。全部价值非负，故指派自动填满
    min(合规单元数, 名额数) 个位置，"不做"由行多于列时的未指派自然表示。
    """
    n, T = EV.shape
    rows = np.where(eligible)[0]
    years = [y for y in range(T) for _ in range(quota)]
    M = EV[np.ix_(rows, years)] if len(years) else np.zeros((len(rows), 0))
    ri, ci = linear_sum_assignment(-M)
    plan = {}
    tot = 0.0
    for r, c in zip(ri, ci):
        if M[r, c] <= 0:
            continue
        plan[int(rows[r])] = years[c]
        tot += float(M[r, c])
    return tot, plan


def run_policy(env, EV, EVm, kind, net=None, rng=None, quota=None):
    """走一遍决策期，返回 {单元: 立项年} 与按 EV 计价的折现总价值。

    三种策略共用同一套掩码与同一套计价（EV[i, 立项年]），故价值可直接相减。
    """
    quota = int(env.quota if quota is None else quota)
    init = {}
    for t in range(env.T):
        X, meta, cost, units = env.pairs()
        avail = [(i, m[0]) for i, m in enumerate(meta) if m[0] >= 0]
        if not avail:
            env.step([])
            continue
        if kind == "random":
            pick_rows = list(rng.permutation([i for i, _ in avail]))[:quota]
        elif kind == "myopic":
            # **只看当期**：按"假设今年条件不变"的 EV 排序，不用任何未来信息
            order = sorted(avail, key=lambda p: -EVm[p[1], t])
            pick_rows, used = [], set()
            for i, u in order:
                if len(pick_rows) >= quota:
                    break
                if u in used:
                    continue
                pick_rows.append(i); used.add(u)
        elif kind == "ppo":
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(X))
            lg = logits.numpy()
            ut = torch.as_tensor(np.asarray([u for u, _ in meta], np.int64))
            ct = torch.as_tensor(np.asarray(cost, np.float64))
            picks, _, _ = TP.sample_action(logits, meta, cost, env.budget, quota,
                                           greedy=True, units_t=ut, cost_t=ct)
            pick_rows = [i for i in picks if meta[i][0] >= 0]
        else:
            raise ValueError(kind)
        acts = []
        seen = set()
        for i in pick_rows:
            u = meta[i][0]
            if u < 0 or u in seen:
                continue
            acts.append(meta[i]); seen.add(u)
            init.setdefault(int(u), t)
        env.step(acts)
    val = sum(EV[u, y] for u, y in init.items())
    return init, float(val)


def wait_when_beneficial(env, EV, init, top_k=10):
    """真实环境里的"等待"指标：单元合法、且明年更值，策略却把名额给了别人。

    与小世界的 P(STOP|WA>0) 不同——这里"等"不意味着整年空着，
    而是"今年把 A 留下来，名额给 B"，这才是 717 单元规模上的真实决策。

    只统计**当年真正在竞争名额**的前 top_k 个单元（按当期 EV 排序）：
    717 个单元里绝大多数任何年份都排不上，把它们算进来会让两个条件概率
    都趋近 1，指标失去分辨力。
    """
    n_up = n_up_wait = n_dn = n_dn_wait = 0
    for t in range(env.T - 1):
        cand = [u for u in range(env.n) if init.get(u, 10**9) >= t]
        if not cand:
            continue
        cand.sort(key=lambda u: -EV[u, t])
        for u in cand[:top_k]:
            if EV[u, t] <= 0:
                continue
            waited = init.get(u, 10**9) != t      # 今年没做它
            if EV[u, t + 1] > EV[u, t]:
                n_up += 1; n_up_wait += int(waited)
            else:
                n_dn += 1; n_dn_wait += int(waited)
    p_up = n_up_wait / n_up if n_up else np.nan
    p_dn = n_dn_wait / n_dn if n_dn else np.nan
    return p_up, p_dn, n_up, n_dn


def peak_distance(EV, init, oplan):
    """立项年与 (a) 该单元自身 EV 峰值年、(b) oracle 指派年 的距离。

    (a) 是逐单元独立最优，忽略名额竞争，是个**松**参照；
    (b) 计入配额，是真正该比的那一个。两个都报，避免只看松的那个。
    """
    d_peak, d_orc = [], []
    for u, y in init.items():
        d_peak.append(abs(y - int(np.argmax(EV[u, :]))))
        if u in oplan:
            d_orc.append(abs(y - oplan[u]))
    f = lambda a: (float(np.mean(a)), float(np.median(a)),
                   float(np.mean(np.asarray(a) <= 1))) if a else (np.nan,) * 3
    return f(d_peak), f(d_orc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--out", default="results_tgate")
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--eps", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--objective", default="floor", choices=["floor", "mix"],
                    help="学习器的目标。floor=只押 Floor，与本闸的计价口径一致"
                         "（判定用这一档）；mix=沿用多目标权重（目标与计价不一致，"
                         "其低分不能归因到时序能力）")
    ap.add_argument("--quota", type=int, default=3)
    ap.add_argument("--budget", type=float, default=1e12,
                    help="预算上限。默认极大 = 不咬：本闸只测时序选择，"
                         "把资金约束一并打开会让任何负结果无法归因")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--foresight", type=int, default=0)
    ap.add_argument("--amps", type=float, nargs=4, default=[2.4, 1.8, 0.9, 1.0],
                    metavar=("PLAN", "INFRA", "AGE", "READY"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    SC.reset()
    SC.apply(horizon=a.horizon, horizon_eval="auto", opp_shape="window",
             a_plan=a.amps[0], a_infra=a.amps[1], a_age=a.amps[2],
             a_ready=a.amps[3], foresight=a.foresight, quota=a.quota,
             budget=a.budget)

    env0 = build_env(a.dataset, a.horizon, a.alpha, 7, a.budget, a.objective)
    EV, EVm = ev_tables(env0, a.gamma)
    elig = np.asarray(env0.env.eligible, bool)
    v_orc, oplan = oracle_plan(EV, a.quota, elig)
    print(f"oracle 折现总价值 {v_orc:.4g}（指派 {len(oplan)} 个单元，"
          f"名额 {a.horizon * a.quota} 个）")

    rows, plans = [], []
    for seed in a.seeds:
        # 随机与近视：与 PPO 用同一张场、同一套掩码、同一套计价
        e = build_env(a.dataset, a.horizon, a.alpha, 7, a.budget, a.objective)
        i_rnd, v_rnd = run_policy(e, EV, EVm, "random",
                                  rng=np.random.default_rng(seed), quota=a.quota)
        e = build_env(a.dataset, a.horizon, a.alpha, 7, a.budget, a.objective)
        i_myo, v_myo = run_policy(e, EV, EVm, "myopic", quota=a.quota)

        e = build_env(a.dataset, a.horizon, a.alpha, 7, a.budget, a.objective)
        net, hist = TP.train(e, iters=a.iters, eps_per_iter=a.eps, seed=seed)
        e2 = build_env(a.dataset, a.horizon, a.alpha, 7, a.budget, a.objective)
        i_ppo, v_ppo = run_policy(e2, EV, EVm, "ppo", net=net, quota=a.quota)

        gap = ((v_ppo - v_myo) / (v_orc - v_myo)) if v_orc > v_myo else np.nan
        for tag, iv, vv in (("随机", i_rnd, v_rnd), ("近视贪心", i_myo, v_myo),
                            ("PPO", i_ppo, v_ppo)):
            plans.extend(dict(seed=seed, policy=tag, unit=u, year=y)
                         for u, y in iv.items())
            (pk_mae, pk_med, pk_hit), (or_mae, or_med, or_hit) = peak_distance(
                EV, iv, oplan)
            p_up, p_dn, n_up, n_dn = wait_when_beneficial(env0, EV, iv)
            rows.append(dict(seed=seed, policy=tag, value=vv,
                             ratio_oracle=vv / v_orc if v_orc else np.nan,
                             gap_closure=(gap if tag == "PPO" else
                                          (0.0 if tag == "近视贪心" else np.nan)),
                             n_init=len(iv),
                             init_mean=float(np.mean(list(iv.values()))) if iv else np.nan,
                             peak_mae=pk_mae, peak_med=pk_med, peak_hit1=pk_hit,
                             oracle_mae=or_mae, oracle_med=or_med, oracle_hit1=or_hit,
                             p_wait_up=p_up, p_wait_dn=p_dn,
                             wait_contrast=(p_up - p_dn), n_up=n_up, n_dn=n_dn,
                             curve_first=float(hist[0]) if tag == "PPO" else np.nan,
                             curve_last=float(hist[-1]) if tag == "PPO" else np.nan))
        print(f"seed={seed}  随机 {v_rnd:.4g}  近视 {v_myo:.4g}  PPO {v_ppo:.4g}  "
              f"oracle {v_orc:.4g}  => 缺口收窄率 {gap:+.3f}")

    # 落盘各策略的立项方案（单元 → 立项年）。下一步要做的价值分解需要它：
    #   把某策略选中的单元**换到它们的 oracle 指派年**，看价值补回多少
    #   —— 补得回来说明损失在时序，补不回来说明损失在"挑错了单元"。
    pd.DataFrame(plans).to_csv(os.path.join(a.out, "tgate_plans.csv"),
                               index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(unit=u, oracle_year=y) for u, y in oplan.items()]).to_csv(
        os.path.join(a.out, "tgate_oracle_plan.csv"), index=False,
        encoding="utf-8-sig")
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "tgate_runs.csv"), index=False, encoding="utf-8-sig")
    Sm = (D.groupby("policy").agg(
        n=("seed", "size"), 折现价值=("value", "mean"),
        相对oracle=("ratio_oracle", "mean"), 缺口收窄率=("gap_closure", "mean"),
        立项数=("n_init", "mean"), 立项重心=("init_mean", "mean"),
        距峰值年MAE=("peak_mae", "mean"), 距峰值1年内=("peak_hit1", "mean"),
        距oracle年MAE=("oracle_mae", "mean"), 距oracle1年内=("oracle_hit1", "mean"),
        P等待_明年更值=("p_wait_up", "mean"),
        P等待_明年更差=("p_wait_dn", "mean"),
        等待对比度=("wait_contrast", "mean")).reset_index())
    Sm.to_csv(os.path.join(a.out, "tgate_summary.csv"), index=False,
              encoding="utf-8-sig")
    print("\n" + Sm.round(4).to_string(index=False))
    json.dump(dict(vars(a), oracle_value=v_orc), 
              open(os.path.join(a.out, "tgate_config.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n判读：缺口收窄率 = (PPO − 近视贪心) / (oracle − 近视贪心)。"
          "= 0 说明 PPO 与只看当期的贪心没有区别；> 0 才说明它用上了时序信息。")


if __name__ == "__main__":
    main()
