# -*- coding: utf-8 -*-
"""exp_where_when.py —— 合成实验：Where-only 对 Where+When（v19 简化框架）。

## 这个实验只证明一句话

> **当城市机会在时间上变化时，纯空间排序不足以复现最优的时空分配。**

不证明"RL 比所有方法都好"，也不再涉及"等待/不做"（前一轮已实测：在
717 单元 / 75 名额的容量区间里，"今年不做"被严格支配，Δ等 在真实环境的
前 8 年全为负）。等待、STOP、GAE/MC、初筛 K 扫描等一律降级为附录证据。

## 世界的构造：空间与时间分离

每个地块有

    空间适宜性   S_i          （与年份无关）
    时间机会     T_{i,t}      （逐年变化，不同地块峰值年不同）
    第 t 年立项的价值 = S_i · [(1−α) + α·T_{i,t}] · γ^t

关键设计（决定实验有没有判别力）：**S_i 分档，且每一档内部都同时含有
早峰 / 晚峰 / 很晚峰三种地块**。

* 档与档之间 —— 纯空间排序有真实信号（高 S 的确更值钱），所以它是一个
  有意义的基线，而不是随机；
* 同一档内部 —— S 完全并列，**只有时间能区分**，这正是 "when"。

若不这样配，而是让"高 S 的恰好都该早做"，纯空间排序会白捡一个正确的
时序，实验就测不出东西。

α 是消融实验唯一变动的量：α=0 机会在时间上完全不变，此时纯空间排序
**应当已经接近最优**（这是对照端）；α 越大，纯空间排序应当越吃亏。

## 五个方法（判据只有一张表）

    随机            —— 下界
    纯空间排序      —— 只按 S_i 排（Where only）
    时间感知贪心    —— 按 S_i·T_{i,t} 排（Where + When，但不跨年协调）
    TP-MORL         —— PPO 学跨年分配有限名额
    oracle          —— 名额约束下的精确指派（上界）

成功标准（按指南）：TP-MORL > 纯空间排序，且 TP-MORL ≈ oracle。
**不要求** TP-MORL 超过时间感知贪心。

用法：
    PYTHONPATH=src python scripts/exp_where_when.py --out results_ww \\
        --alphas 0 0.5 1.0 --seeds 0 1 2
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_timing_world import TimingWorld              # noqa: E402

from tpmorl.rl import train_ppo as TP                 # noqa: E402


def value_matrix(w):
    """V[i, t] = 第 t 年立项地块 i 的折现价值。奖励、oracle、全部基线共用它。

    直接调 w.dvalue —— 上一轮的教训是"价值读哪一年"必须只有一个定义：
    环境的 step() 与这里的计价都经由 TimingWorld.value_year()，
    并由 validate() 的机械自证保证两者数值相等。
    """
    return np.array([[w.dvalue(u, t) for t in range(w.T)] for u in range(w.n)])


def spatial_matrix(w):
    """纯空间分数 S_i：与年份无关，按行广播成同形矩阵，便于共用一个排序函数。

    注意**不带折现**：纯空间排序代表"只看这块地好不好"，它连"早做更值钱"
    都不知道。若给它折现，它就偷到了一点时间信息。
    """
    return np.repeat(np.asarray(w.base, float)[:, None], w.T, axis=1)


def oracle(V, quota):
    """名额约束下的最优时间分配：每年复制 quota 列的最大权指派，精确解。"""
    n, T = V.shape
    years = [y for y in range(T) for _ in range(quota)]
    M = V[:, years]
    ri, ci = linear_sum_assignment(-M)
    plan, tot = {}, 0.0
    for r, c in zip(ri, ci):
        if M[r, c] <= 0:
            continue
        plan[int(r)] = years[c]
        tot += float(M[r, c])
    return tot, plan


def run_rank(w, rank, V, rng=None):
    """按给定分数逐年贪心占满名额，走真实环境。返回 (实收价值, {地块: 立项年})。

    走真实 env.step 而不是纯表上计算：基线与 PPO 必须面对同一套约束
    （配额、每年至多一次、已立项不再出现），否则比较不成立。
    """
    w.reset(seed=0)
    init = {}
    for t in range(w.T):
        X, meta, cost, units = w.pairs()
        real = [(i, m[0]) for i, m in enumerate(meta) if m[0] >= 0]
        if not real:
            if w.step([])[2]:
                break
            continue
        if rng is not None:
            order = rng.permutation(len(real))
            real = [real[i] for i in order]
        else:
            real = sorted(real, key=lambda p: -rank[p[1], t])
        act, used = [], set()
        for i, u in real:
            if len(act) >= w.quota:
                break
            if u in used or rank[u, min(t, rank.shape[1] - 1)] <= 0:
                continue
            act.append(meta[i]); used.add(u); init.setdefault(u, t)
        if w.step(act)[2]:
            break
    return float(sum(V[u, t] for u, t in init.items())), init


def run_policy(w, net, V):
    """训练好的策略贪心走一遍（argmax），同样走真实 env.step。"""
    w.reset(seed=0)
    init = {}
    for t in range(w.T):
        X, meta, cost, units = w.pairs()
        real = [(i, m[0]) for i, m in enumerate(meta) if m[0] >= 0]
        if not real:
            if w.step([])[2]:
                break
            continue
        act, used = [], set()
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(X))
        order = torch.argsort(logits, descending=True).tolist()
        for i in order:
            if len(act) >= w.quota:
                break
            u = meta[i][0]
            if u < 0 or u in used:
                continue
            act.append(meta[i]); used.add(u); init.setdefault(u, t)
        if w.step(act)[2]:
            break
    return float(sum(V[u, t] for u, t in init.items())), init


def where_when_metrics(init, oplan, V):
    """把表现拆成 Where 与 When 两项，对应论文标题里的两个词。

    Where：选中的地块集合与 oracle 集合的重叠率（Jaccard 与召回）。
    When ：**在两者都选中的地块上**，立项年与 oracle 指派年的平均绝对误差。
           只在交集上算，否则"没选中"会混进时间误差里，两个维度就分不开。
    """
    a, b = set(init), set(oplan)
    inter = a & b
    yerr = [abs(init[u] - oplan[u]) for u in inter]
    return dict(
        where_recall=len(inter) / max(len(b), 1),
        where_jaccard=len(inter) / max(len(a | b), 1),
        when_mae=float(np.mean(yerr)) if yerr else np.nan,
        when_hit1=float(np.mean([e <= 1 for e in yerr])) if yerr else np.nan,
        n_init=len(init), n_overlap=len(inter))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_ww")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-per-kind", type=int, default=8, help="每个峰值型的地块数")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--quota", type=int, default=2)
    ap.add_argument("--suit", type=float, nargs="+", default=[1.0, 0.8, 0.6, 0.45],
                    help="空间适宜性档位（每档内部含全部峰值型）")
    ap.add_argument("--width", type=float, default=1.8, help="机会窗宽度（年）")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--eps", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for alpha in a.alphas:
        wkw = dict(n_per_kind=a.n_per_kind, T=a.horizon, quota=a.quota,
                   alpha=alpha, suit=a.suit, width=a.width, value_at="init")
        w0 = TimingWorld(**wkw); w0.reset(seed=0)
        diag = w0.validate()
        V, S = value_matrix(w0), spatial_matrix(w0)
        v_orc, oplan = oracle(V, a.quota)
        slots = a.horizon * a.quota
        print(f"\n===== α={alpha:g}  {w0.n} 地块 / {slots} 个名额 "
              f"（T={a.horizon}×{a.quota}）  oracle {v_orc:.1f} =====")
        print(f"      奖励=计价自证 {diag['reward_matches_value']}  "
              f"适宜性档 {diag['suit_levels']}")

        def record(tag, val, init, seed=-1):
            m = where_when_metrics(init, oplan, V)
            rows.append(dict(alpha=alpha, seed=seed, method=tag, value=val,
                             ratio=val / v_orc, oracle_value=v_orc, **m))
            print(f"  {tag:<14s} {val:8.1f}  {val/v_orc:6.3f}   "
                  f"Where 召回 {m['where_recall']:.3f}  "
                  f"When MAE {m['when_mae']:.2f}  "
                  f"When 1年内 {m['when_hit1']:.2f}")

        # 下界与两个启发式基线
        for seed in a.seeds:
            v, iv = run_rank(TimingWorld(**wkw), S, V,
                             rng=np.random.default_rng(seed))
            record("随机", v, iv, seed)
        v, iv = run_rank(TimingWorld(**wkw), S, V)
        record("纯空间排序", v, iv)
        v, iv = run_rank(TimingWorld(**wkw), V, V)
        record("时间感知贪心", v, iv)
        record("oracle", v_orc, oplan)

        # TP-MORL
        for seed in a.seeds:
            ww = TimingWorld(**wkw)
            net, hist = TP.train(ww, iters=a.iters, eps_per_iter=a.eps, seed=seed)
            v, iv = run_policy(TimingWorld(**wkw), net, V)
            record("TP-MORL", v, iv, seed)
            torch.save(dict(state=net.state_dict(), alpha=alpha, seed=seed),
                       os.path.join(a.out, f"net_a{alpha:g}_s{seed}.pt"))

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "ww_runs.csv"), index=False,
             encoding="utf-8-sig")
    Sm = (D.groupby(["alpha", "method"], sort=False)
          .agg(n=("value", "size"), 价值=("value", "mean"),
               相对oracle=("ratio", "mean"), 相对oracle_std=("ratio", "std"),
               Where召回=("where_recall", "mean"),
               When_MAE=("when_mae", "mean"),
               When_1年内=("when_hit1", "mean")).reset_index())
    Sm.to_csv(os.path.join(a.out, "ww_summary.csv"), index=False,
              encoding="utf-8-sig")
    print("\n" + "=" * 78)
    print(Sm.round(3).to_string(index=False))
    json.dump(dict(vars(a)), open(os.path.join(a.out, "ww_config.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n成功标准（按指南）：TP-MORL > 纯空间排序，且 TP-MORL ≈ oracle；"
          "α=0 时纯空间排序应当已接近最优（消融的对照端）。")


if __name__ == "__main__":
    main()
