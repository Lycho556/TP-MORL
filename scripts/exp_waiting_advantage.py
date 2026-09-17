# -*- coding: utf-8 -*-
"""exp_waiting_advantage.py —— P0 环境门槛：**不训练**，先证明环境里存在择时价值。

## 为什么这一步必须排在训练之前

到 v17 为止的教训是：在一个"等待并不真的更优"的环境里训练，再多的算法调参
也只是在测量噪声。门槛实验跑了十九档没有一档学会等，后来才查出原因在世界本身
（机会单调上升 + 名额宽松 → 贪心解已有穷举最优的 92–95%）。

所以在把 717 个单元 × 25 年 × 多种子送上服务器之前，先回答一个**纯数学问题**：

    在完全知道未来的条件下，这个环境里有多少 (单元, 年份) 状态满足
    "等一年再立项"比"现在立项"的折现期望价值更高？

这一步不涉及 PPO，也不涉及策略——它测的是**环境**，不是学习器。

    若 WA>0 的状态只占 0.5%，那 PPO 不学等待是**完全正常**的，
      该改的是环境；
    若 30% 的状态 WA>0 且中位数 +10%，PPO 仍然不等，
      那才轮到算法 / 观测 / 信用分配的问题。

## 一年等待优势的定义

对单元 i 与年份 t：

    WA_i(t) = [ EV_i(t+1) · γ^{pay(t+1)} − EV_i(t) · γ^{pay(t)} ] / ( EV_i(t) · γ^{pay(t)} )

其中 EV 是**期望**交付价值（含批得下来的概率与期内能否完工），pay 是收益落地
的年份（立项年 + 审批期望等待 + 建设期）。用相对量而不是绝对量，是为了让不同
规模的单元可比、也让不同幅度设定之间可比。

`--max-wait` > 1 时同时报"等 k 年"的最优 k，用于区分两种完全不同的世界：

    最优 k 恒等于 0            —— 没有择时问题（v16）
    最优 k 恒等于最大可等年数  —— "越晚越好"，退化成拖延（v17 的单调场）
    最优 k 分布在中间          —— 存在**内点**时间最优，这才是真正的择时问题

第二种情形最危险：它在"WA>0 的比例"这个指标上表现得和第三种一样好，
但它奖励的是"全部往后拖"这个退化解。故本脚本**必须**把最优等待年的分布报出来，
只报 WA>0 比例是不够的。

## 口径

* 交付概率用状态机的**基准** hazard 与有效期（`schedule.HAZARD` / `TAU_VALID`
  / `TAU_EXT`），叠加实施条件场的逐年调制——与环境真正执行的那一套同源。
* 价值侧只含基础设施通道（v18 起唯一的价值乘子）。
* 期内交付不了的立项年，EV 记 0：那不是"晚一点交付"，是没有交付。

用法：
    PYTHONPATH=src python scripts/exp_waiting_advantage.py --out results_wa
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from tpmorl.env import opportunity as OPP
from tpmorl.env import schedule as S
from tpmorl.rl import scenario as SC


def delivery_profile(hazard, tau_max, ready_mult):
    """给定逐年批准风险率与实施条件调制，返回 (获批概率, 期望获批等待年数)。

    `ready_mult` 是该单元在**立项那一年**的 hazard 调制因子。这里做了一处
    刻意的简化：审批期内逐年的调制都按立项年那一年算，而不是逐年重取。
    理由是本诊断要的是"立项时点好不好"这一个自变量，逐年重取会把"审批期间
    条件继续改善"这第二个效应混进来，两者无法分离。真实环境里是逐年重取的，
    故本式给出的是**保守**估计（条件改善时低估获批概率）。
    """
    p_alive, p_ok, ew = 1.0, 0.0, 0.0
    for k in range(int(tau_max)):
        h = min(float(hazard[min(k, len(hazard) - 1)]) * float(ready_mult), 1.0)
        p_ok += p_alive * h
        ew += p_alive * h * (k + 1)
        p_alive *= (1.0 - h)
    return float(p_ok), float(ew / p_ok) if p_ok > 0 else float("nan")


def build(dataset, T, amps, foresight, field_seed=None):
    """构造机会场（与环境同一构造路径，避免事后重建产生口径差）。"""
    U = pd.read_csv(os.path.join(dataset, "zones_v0", "candidate_units.csv"))
    SC.reset()
    SC.apply(horizon=T, horizon_eval="auto", foresight=foresight, **amps)
    T_eval = SC.horizon_eval()
    opp = OPP.OpportunityField(
        len(U), T, T_total=T_eval,
        row=U["row"].values if "row" in U else None,
        col=U["col"].values if "col" in U else None,
        seed=field_seed)
    return U, opp, T_eval


def waiting_advantage(U, opp, T, T_eval, gamma=0.95, build_years=5, max_wait=5):
    """逐单元逐年算 EV 与一年等待优势。返回长表。"""
    n = len(U)
    tau_max = int(S.TAU_VALID + S.TAU_EXT)
    farcap = U["farcap"].values if "farcap" in U else np.ones(n)
    ncell = U["n_cells"].values if "n_cells" in U else np.ones(n)
    base = np.asarray(farcap, float) * np.asarray(ncell, float)

    # EV[i, t]：第 t 年立项的期望折现交付价值
    EV = np.zeros((n, T))
    for t in range(T):
        hm = opp.hazard_mult(t)
        hm = np.full(n, 1.0) if np.ndim(hm) == 0 else np.asarray(hm, float)
        adm = opp.admit_prob(t)
        adm = np.full(n, 1.0) if np.ndim(adm) == 0 else np.asarray(adm, float)
        for i in range(n):
            p_ok, ew = delivery_profile(S.HAZARD, tau_max, hm[i])
            if not np.isfinite(ew):
                continue
            t_done = t + ew + 1.0 + build_years        # 获批 + 次年开工 + 建设
            if t_done > T_eval - 1:                    # 期内交付不了 = 没有交付
                continue
            # 准入概率乘进来：报不上去就不可能交付（v18 的规划通道）
            EV[i, t] = (adm[i] * p_ok * base[i] * opp.value_mult(i, t, int(round(t_done)))
                        * gamma ** t_done)

    rows = []
    for t in range(T - 1):
        cur = EV[:, t]
        nxt = EV[:, t + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            wa = np.where(cur > 0, nxt / cur - 1.0, np.nan)
        # 最优等待年数：在 [t, t+max_wait] 里 EV 最大的那一年减 t
        hi = min(t + max_wait, T - 1)
        seg = EV[:, t:hi + 1]
        best_k = np.argmax(seg, axis=1)
        rows.append(pd.DataFrame(dict(
            year=t, unit=np.arange(n), kind=opp.kind,
            ev_now=cur, ev_next=nxt, wa=wa, best_wait=best_k,
            gain_best=np.where(cur > 0, seg.max(axis=1) / cur - 1.0, np.nan))))
    return pd.concat(rows, ignore_index=True)


def summarize(D, tag):
    d = D[np.isfinite(D.wa)]
    if not len(d):
        return dict(arm=tag, n_states=0)
    return dict(
        arm=tag, n_states=int(len(d)),
        wa_pos_share=float((d.wa > 0).mean()),
        wa_mean=float(d.wa.mean()), wa_median=float(d.wa.median()),
        wa_p90=float(d.wa.quantile(0.90)), wa_max=float(d.wa.max()),
        best_wait_mean=float(d.best_wait.mean()),
        best_wait_0=float((d.best_wait == 0).mean()),
        best_wait_interior=float(((d.best_wait > 0) & (d.best_wait < d.best_wait.max())).mean()),
        best_wait_maxed=float((d.best_wait == d.best_wait.max()).mean()),
        gain_best_median=float(d.gain_best.median()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--out", default="results_wa")
    ap.add_argument("--horizons", type=int, nargs="+", default=[15, 25])
    ap.add_argument("--foresight", type=int, default=0)
    ap.add_argument("--max-wait", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.95)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # 幅度档：从全关（v16 口径）到主组设定，再到单通道
    ARMS = {
        "场全关（v16 口径）": dict(),
        "仅规划准入": dict(a_plan=0.8),
        "仅设施价值": dict(a_infra=0.6),
        "仅实施条件": dict(a_ready=0.4),
        "仅老化必要性": dict(a_age=0.3),
        "v18 主组": dict(a_plan=0.8, a_infra=0.6, a_age=0.3, a_ready=0.4),
        "v18 主组×2 幅度": dict(a_plan=1.6, a_infra=1.2, a_age=0.6, a_ready=0.8),
        # 有限机会窗（升→峰→落）。同幅度、只换形状，故与"v18 主组"逐项可比：
        # 差异只能归因到关窗，不能归因到幅度。
        "有限窗 主组幅度": dict(a_plan=0.8, a_infra=0.6, a_age=0.3, a_ready=0.4,
                          opp_shape="window"),
        "有限窗 ×2 幅度": dict(a_plan=1.6, a_infra=1.2, a_age=0.6, a_ready=0.8,
                          opp_shape="window"),
        "有限窗 ×3 幅度": dict(a_plan=2.4, a_infra=1.8, a_age=0.9, a_ready=1.0,
                          opp_shape="window"),
    }
    rows, longs = [], []
    for T in a.horizons:
        for tag, amps in ARMS.items():
            U, opp, T_eval = build(a.dataset, T, amps, a.foresight)
            D = waiting_advantage(U, opp, T, T_eval, gamma=a.gamma,
                                  max_wait=a.max_wait)
            s = summarize(D, tag)
            s["T"], s["T_eval"] = T, T_eval
            rows.append(s)
            D["arm"], D["T"] = tag, T
            longs.append(D)
            print(f"T={T:>2}  {tag:<16s}  WA>0 占比 {s.get('wa_pos_share', float('nan')):.3f}  "
                  f"中位 {s.get('wa_median', float('nan')):+.4f}  "
                  f"P90 {s.get('wa_p90', float('nan')):+.4f}  "
                  f"最优等待年均值 {s.get('best_wait_mean', float('nan')):.2f}  "
                  f"[立刻 {s.get('best_wait_0', float('nan')):.2f} / 内点 "
                  f"{s.get('best_wait_interior', float('nan')):.2f} / 拖满 "
                  f"{s.get('best_wait_maxed', float('nan')):.2f}]")

    Sm = pd.DataFrame(rows)
    Sm.to_csv(os.path.join(a.out, "wa_summary.csv"), index=False, encoding="utf-8-sig")
    pd.concat(longs, ignore_index=True).to_csv(
        os.path.join(a.out, "wa_states.csv.gz"), index=False, compression="gzip")
    json.dump(vars(a), open(os.path.join(a.out, "wa_config.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n判读：`WA>0 占比` 回答\"有没有择时机会\"；`最优等待年` 的三分布回答"
          "\"是不是真的择时\"——拖满占比高意味着环境奖励的是无脑延后，"
          "那是退化解，不是择时。")


if __name__ == "__main__":
    main()
