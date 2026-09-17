# -*- coding: utf-8 -*-
"""exp_timing_gate.py —— 择时门槛实验：先证明学习器能学会"等"，再谈现实数据。

## 这个实验回答什么

v16 的结论是：策略的立项年份分布与均摊**无法区分**（主组卡方 p=0.999，
立项重心 6.98 对均摊基准 7.0，见 results_v16/quicklook.txt）。
诊断（docs/择时机制_诊断_v1.md §3）与外部建议都指向同一个原因：

    机会集是平稳的 —— 明年的候选集不优于今年，于是"等"永不严格占优，
    折现又始终把动作往前推。策略学不到择时，是因为**没有择时可学**。

那么在把四类非平稳机会场接进真实环境之前，必须先排除另一种可能：
**学习器本身（掩码指针 + PPO + 逐年奖励）是否有能力学会"等"？**

本脚本构造一个最小的、答案可以**穷举算出**的择时 MDP，并**直接复用**
`tpmorl.rl.train_ppo` 的 Pointer 网络与 train/evaluate——不另写算法，
这样通过与不通过都只能归因到环境，而不是"门槛实验用了别的算法"。

## 情景（手工设计，见建议 §9）

三个地块，年度配额 1（一年最多立一个项），资金不咬（本实验刻意不引入
预算约束，把"等"的唯一理由留给机会场）：

    B  现在就好，以后不变      -> 应当第 0 年立项
    A  现在一般，第 onset 年起变好 -> 应当**等到** onset 年
    C  一直不好                -> 低优先，且不该为它一直等

立项后经 lead 年建成，建成年结算收益，折现 γ^t_done。收益读**立项年**的
机会场取值（对应"上位规划在申报时点是否支持"这一读法）。

判据（pass/fail）：
    (1) A 的立项年 >= onset           —— 学会了"等"
    (2) B 的立项年 == 0               —— 没有因为学会等而变成一律拖延
    (3) 折现总收益 >= 穷举最优的 95%   —— 等的位置也大致对

第 (2) 条是必须的：一个"什么都往后推"的退化策略也能满足 (1)，
那不是择时，只是另一种均摊。

## lead 的两档是本实验的关键设计

    --lead 1   立项次年即建成。纯择时问题，几乎没有信用分配难度。
    --lead 7   与真实环境的最短通道同量级（审批中位 3 年 + 次年开工 +
               建设 5 年，见 schedule.py）。

两档一起跑才能把"学不会择时"分解成两个可分别处置的原因：
lead=1 不通过 => 学习器/奖励口径的问题，非平稳机会场接进去也不会有用；
lead=1 通过而 lead=7 不通过 => 是长延迟信用分配问题，环境改造之外还需要
算法侧手段（如势函数型奖励整形，policy-invariant，不改变最优策略）。

用法：
    PYTHONPATH=src python scripts/exp_timing_gate.py --out results_gate
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from tpmorl.objectives.reward import OBJ_NAMES
from tpmorl.rl.env_gym import N_FEAT, N_PAIR_FEAT, STOP
from tpmorl.rl import train_ppo

FLOOR_IDX = OBJ_NAMES.index("Floor")     # 借 Floor 这一维承载收益，保持 11 维口径


class GateEnv:
    """最小择时环境。刻意鸭子类型地实现 RenewalEnv 被 train_ppo 用到的那部分接口。

    只实现被用到的接口，不继承 RenewalEnv：后者要读 100 m 栅格数据集、建
    717×12 的成本矩阵，与本实验要隔离的东西无关。被用到的接口清单（逐个核过
    train_ppo.run_episode / train / evaluate）：
        T / T_eval / gamma / quota / budget / scale / ch / ncell
        reset(seed) / pairs() / step(actions) / pair_cost(u,tg)
        budget_hist / spent_hist / released_hist
    """

    # 三个地块的 (名称, 基准收益, 类型)。数值是情景设定，不是标定值。
    UNITS = (("B", 100.0, "flat_high"),
             ("A", 100.0, "ramp"),
             ("C", 100.0, "flat_low"))

    def __init__(self, T=10, lead=1, onset=4, gamma=0.95,
                 lo=0.6, hi=1.6, c_level=0.7, quota=1, reward_at="done",
                 ramp_years=0.0, foresight=0, shaping=False):
        """lo/hi = ramp 型地块 onset 前后的收益乘子；c_level = C 型地块的乘子。

        取值口径：**情景参数**。选 0.6→1.6 是为了让"等 onset 年"在 γ=0.95 下
        严格占优（1.6·0.95^4 = 1.30 > 0.6·0.95 = 0.57），即门槛实验必须先保证
        正确答案是"等"，否则测不出学习器有没有这个能力。
        """
        self.T = int(T)
        self.T_eval = int(T)          # 闭区间口径：本实验不启用尾部评价期
        self.gamma = float(gamma)
        self.quota = int(quota)
        self.lead = int(lead)
        self.onset = int(onset)
        self.lo, self.hi, self.c_level = float(lo), float(hi), float(c_level)
        # 奖励计入时点。done = 建成年（与真实环境同口径）；init = 立项年。
        # init 档把"延迟 lead 年"这一个因素单独去掉：若 init 档能学会等而 done 档
        # 不能，则问题是长延迟信用分配，而不是折现或探索。
        if str(reward_at) not in ("done", "init"):
            raise ValueError(f"reward_at 只能是 done/init，收到 {reward_at!r}")
        self.reward_at = str(reward_at)
        # 机会改善的**形状**。0 = 阶跃（onset 年当年跳变）；>0 = 逻辑斯蒂爬升的
        # 时间常数（年），与 opportunity.RAMP_YEARS 同一形状。
        #
        # 这一档是本实验最重要的设计变量。阶跃档下，"多等一年"在 onset 之前
        # **逐年都是净亏**（同样的收益推迟一年 = 乘 γ），只有连等到 onset 才
        # 回本；于是最优解与贪心解之间隔着一道**多步**壁垒，而 PPO 做的是局部
        # 策略改进，单步偏离一律被惩罚。爬升档下，只要局部增长率超过 1−γ，
        # 每多等一年**本身**就是净赚，改进路径单调，梯度就能爬上去。
        # 两档一起跑，才能把"学不会等"归因到壁垒形状而不是学习器。
        self.ramp_years = float(ramp_years)
        # 前瞻年数 K：观测里额外给出 **趋势** mult(t+K) − mult(t)。
        #
        # 为什么必须是趋势而不是未来水平：策略给所有地块打分用的是**同一个**函数，
        # 只看当期水平时，"现在一般但会变好的 A"与"一直不好的 C"在观测上无法区分
        # ——实测 A 的当期水平甚至高于 C，于是任何单调阈值规则都会先做 A 再做 C，
        # "留着 A、先做 C"这个最优解根本不在可表示的策略类里。给未来水平也不够，
        # 那只是把同一个贪心排序平移了 K 年；差值才是"会不会变好"的直接答案。
        # K=0 = 只看当期（信息消融档）。
        self.foresight = int(foresight)
        # 势函数型奖励整形（Ng, Harada & Russell 1999）。
        #
        # 为什么需要它：折现使"同样的收益推迟一年"一律变差，于是**局部**地看，
        # 等待几乎总是净亏——梯度把动作往前推，哪怕晚做的总价值高得多。
        # 势函数整形给每个"还没立项的地块"记一个持有价值
        #     φ_u(t) = max_{t' >= t} value(u, t') · γ^{pay(t') − t}
        # 即"这个地块留在手里、将来在它最好的年份动手，折到今天值多少"。
        # 整形项 F = γ·Φ(s') − Φ(s)。持有一个最好年份还在后面的地块时
        # γ·φ(t+1) ≈ φ(t)，等待不再被机械扣分；而在坏年份动手会立刻失掉 φ_u，
        # 亏损当年就可见，不必等 lead 年后才由回报体现。
        #
        # **它不改变最优策略**：PBRS 对任意势函数都保持最优策略集不变，这是
        # 该方法可以写进论文的前提——它不是"把答案喂给 agent"，而是把同一个
        # 最优解的梯度变得可跟随。这一点必须与"改奖励函数"区分开。
        self.shaping = bool(shaping)
        self.n = len(self.UNITS)
        self.names = [u[0] for u in self.UNITS]
        self.base = np.array([u[1] for u in self.UNITS], float)
        self.kind = [u[2] for u in self.UNITS]
        # 资金刻意不咬：预算恒大于任何单项成本，掩码里的资金项永不生效
        self.cost = np.zeros(self.n)
        self.scale = np.ones(len(OBJ_NAMES))
        self.ch = np.ones(self.n, int)
        self.ncell = np.ones(self.n, int)

    # ---- 机会场：本实验唯一的非平稳来源 ----
    def mult(self, u, t):
        """地块 u 在年份 t 的收益乘子（= 机会场取值）。"""
        k = self.kind[u]
        if k == "flat_high":
            return 1.0
        if k == "flat_low":
            return self.c_level
        if self.ramp_years <= 0:
            return self.hi if t >= self.onset else self.lo
        s = 1.0 / (1.0 + np.exp(-(float(t) - self.onset) / self.ramp_years))
        return self.lo + (self.hi - self.lo) * s

    def value(self, u, t_init):
        """立项年 t_init 下该地块的建成收益（未折现）。"""
        return float(self.base[u] * self.mult(u, t_init))

    def potential(self):
        """Φ(s) = 所有**尚未立项**且期内还来得及的地块的持有价值之和。"""
        if not self.shaping:
            return 0.0
        tot = 0.0
        for u in range(self.n):
            if u in self.init_year:
                continue
            cand = [self.value(u, t2) * self.gamma ** (self.pay_year(t2) - self.t)
                    for t2 in range(self.t, self.T - self.lead + 1)
                    if t2 + self.lead <= self.T - 1]
            if cand:
                tot += max(cand)
        return float(tot)

    def pay_year(self, t_init):
        """收益落在哪一年（折现指数），随 reward_at 口径走。"""
        return int(t_init) + (0 if self.reward_at == "init" else self.lead)

    def pair_cost(self, u, tg):
        return float(self.cost[int(u)])

    def reset(self, seed=None):
        self.t = 0
        self.done_year = {}            # 地块 -> 建成年
        self.init_year = {}            # 地块 -> 立项年
        self.budget = 1e9
        self.budget_hist, self.spent_hist, self.released_hist = [], [], []
        return self._obs()

    def _obs(self):
        return np.zeros((self.n, N_FEAT), dtype=np.float32)

    def _available(self):
        """本年可立项的地块：还没立过项，且**在期内建得成**。

        "建不成就不给立"是刻意的：本实验要测的是"在能建成的若干时点里选哪个"，
        把"立了也白立"的时点留在动作空间里，会混进另一个问题（窗外立项），
        那是真实环境里 LSR 指标在管的事。
        """
        return [u for u in range(self.n)
                if u not in self.init_year and self.t + self.lead <= self.T - 1]

    def pairs(self):
        av = self._available()
        rows = np.zeros((len(av) + 1, N_PAIR_FEAT), dtype=np.float32)
        for j, u in enumerate(av):
            rows[j, 0] = self.mult(u, self.t)                  # 当期机会场取值
            rows[j, 1] = self.base[u] / max(self.base.max(), 1e-9)
            if self.foresight:
                rows[j, 2] = self.mult(u, self.t + self.foresight) - self.mult(u, self.t)
            rows[j, 14] = self.t / self.T
            rows[j, 15] = 1.0
            rows[j, 16] = (self.T - 1 - self.lead - self.t) / self.T   # 剩余可立项年数
        rows[len(av), 14] = self.t / self.T
        rows[len(av), N_FEAT + 16] = 1.0                       # 「到此为止」标志
        meta = [(u, 0) for u in av] + [STOP]
        cost = np.concatenate([self.cost[av], [0.0]])
        units = np.concatenate([np.asarray(av, np.int64), np.array([-1], np.int64)])
        return rows, meta, cost, units

    def step(self, actions):
        actions = [(int(u), int(tg)) for u, tg in actions if int(u) >= 0]
        if len(actions) > self.quota:
            raise ValueError(f"超出年度配额 {self.quota}：本期请求 {len(actions)}")
        self.budget_hist.append(self.budget)
        self.spent_hist.append(0.0)
        self.released_hist.append(0.0)
        # Φ(s) 必须在**动作生效之前**取：写在立项循环之后会把本年被立项的地块
        # 从 Φ(s) 与 Φ(s') 里同时剔除，整形项退化成 (γ−1)Φ 这样一个与择时无关的
        # 近似常数项，看起来"开了整形"而实际上没有任何时序信号。
        # （本脚本第一版就是这么写错的，三个种子跑出与未整形档逐位相同的结果。）
        phi0 = self.potential()
        for u, _ in actions:
            if u in self.init_year:
                raise ValueError(f"地块 {u} 已立项")
            self.init_year[u] = self.t
            self.done_year[u] = self.t + self.lead
        # 收益结算：done 档在建成年、init 档在立项年（见 reward_at 的说明）
        if self.reward_at == "init":
            gain = sum(self.value(u, self.t) for u, _ in actions)
        else:
            gain = sum(self.value(u, self.init_year[u])
                       for u, dy in self.done_year.items() if dy == self.t)
        vec = np.zeros(len(OBJ_NAMES))
        vec[FLOOR_IDX] = gain
        self.t += 1
        # 整形项只进**策略学习用的标量奖励**，不进 vec：vec 是 11 维目标的
        # 落盘口径，整形是学习手段而非目标，混进去会让报出来的目标值不再是目标值。
        if self.shaping:
            gain = gain + self.gamma * self.potential() - phi0
        return self._obs(), float(gain), self.t >= self.T_eval, dict(vec=vec, raw={},
                                                                    events={})

    # ---- 穷举最优：判据 (3) 的分母 ----
    def brute_force_optimum(self):
        """穷举所有可行时间表，返回 (最优折现收益, 最优立项年字典)。

        可行 = 每个地块最多立一次、每年至多 quota 个、且立项年 + lead <= T-1。
        规模 (T+1)^n（含"不做"），T=10/n=3 时 1331 个组合，直接枚举即可，
        不需要 DP——把最优解写成闭式推导反而更容易出错。
        """
        years = list(range(self.T - self.lead)) + [None]
        best, best_plan = -np.inf, None
        for combo in itertools.product(years, repeat=self.n):
            used = [y for y in combo if y is not None]
            if len(used) != len(set(used)):        # quota=1 时同年不得两项
                continue
            if self.quota > 1:
                pass                               # quota>1 时上面的去重过严，暂不支持
            g = sum(self.value(u, y) * self.gamma ** self.pay_year(y)
                    for u, y in enumerate(combo) if y is not None)
            if g > best:
                best, best_plan = g, {self.names[u]: y for u, y in enumerate(combo)}
        return float(best), best_plan


def run_one(lead, seed, iters, T, onset, eps_per_iter, gamma=0.95, ent_c=0.01,
            lr=3e-3, epochs=4, reward_at="done", ramp_years=0.0, foresight=0,
            lo=0.6, hi=1.6, shaping=False):
    env = GateEnv(T=T, lead=lead, onset=onset, gamma=gamma, reward_at=reward_at,
                  ramp_years=ramp_years, foresight=foresight, lo=lo, hi=hi,
                  shaping=shaping)
    net, hist = train_ppo.train(env, iters=iters, eps_per_iter=eps_per_iter, seed=seed,
                                ent_c=ent_c, lr=lr, epochs=epochs)
    rec = []
    train_ppo.evaluate(env, net, n_ep=1, record=rec)
    R = pd.DataFrame(rec)
    R = R[R["unit"] >= 0] if len(R) else R
    init = {env.names[int(r.unit)]: int(r.year) for r in R.itertuples()}
    got = sum(env.value(u, init[nm]) * env.gamma ** env.pay_year(init[nm])
              for u, nm in enumerate(env.names) if nm in init)
    opt, opt_plan = env.brute_force_optimum()
    return dict(lead=lead, seed=seed, gamma=gamma, ent_c=ent_c, lr=lr,
                epochs=epochs, reward_at=reward_at, iters=iters,
                ramp_years=ramp_years, foresight=foresight, lo=lo, hi=hi,
                shaping=shaping,
                A_init=init.get("A"), B_init=init.get("B"), C_init=init.get("C"),
                ret=got, opt=opt, ratio=got / opt if opt else np.nan,
                opt_A=opt_plan["A"], opt_B=opt_plan["B"], opt_C=opt_plan["C"],
                pass_wait=(init.get("A") is not None and init["A"] >= onset),
                # 与穷举最优时点的距离容 1 年：爬升档下最优年可能晚于 onset，
                # 只判"是否等过了 onset"会给爬升档虚高的通过率。
                pass_opt=(init.get("A") is not None
                          and abs(init["A"] - opt_plan["A"]) <= 1),
                pass_now=(init.get("B") == 0),
                pass_ret=(got >= 0.95 * opt),
                curve_last=hist[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_gate")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--eps", type=int, default=4)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--onset", type=int, default=4)
    ap.add_argument("--leads", type=int, nargs="+", default=[1, 7])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--ent-c", type=float, default=0.01,
                    help="熵系数。默认 0.01 与 train_ppo 一致；调大用于检验"
                         "\"学不会等\"是否只是探索不足")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--reward-at", default="done", choices=["done", "init"],
                    help="收益计入时点。done=建成年（与真实环境同口径）；"
                         "init=立项年，用于单独去掉 lead 年延迟这一个因素")
    ap.add_argument("--ramp-years", type=float, default=0.0,
                    help="机会改善的形状：0=阶跃（onset 年跳变）；>0=逻辑斯蒂爬升的"
                         "时间常数（年）。阶跃档下\"多等一年\"在 onset 之前逐年净亏，"
                         "最优解与贪心解之间隔着多步壁垒；爬升档下只要局部增长率超过"
                         "1−γ，每多等一年本身就净赚，改进路径单调。这是本实验最重要的"
                         "设计变量")
    ap.add_argument("--foresight", type=int, default=0,
                    help="前瞻年数 K：观测里给出趋势 mult(t+K) − mult(t)。"
                         "0=只看当期（信息消融档）。对应\"法定图则与设施计划已公布、"
                         "规划师看得见未来几年的安排\"这一信息条件")
    ap.add_argument("--lo", type=float, default=0.6,
                    help="ramp 型地块爬升**前**的收益乘子，默认 0.6")
    ap.add_argument("--hi", type=float, default=1.6,
                    help="ramp 型地块爬升**后**的收益乘子，默认 1.6。lo/hi 之比决定"
                         "\"等对不等\"的净收益有多大；比值太小时择时收益会低于"
                         "PPO 的梯度噪声，学不到不是因为不可表示而是因为看不见")
    ap.add_argument("--shaping", action="store_true",
                    help="开启势函数型奖励整形（PBRS，Ng et al. 1999）。"
                         "**不改变最优策略集**，只把\"等待\"的梯度从机械扣分变成"
                         "中性。判据里的回报比仍按未整形的真实折现收益计算")
    ap.add_argument("--label", default="", help="本档的名字，写进结果表")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for lead in a.leads:
        # lead 过长时期内无可立项年，先挡住，避免跑出一张空表
        if a.T - lead <= a.onset:
            print(f"[跳过] lead={lead}：T={a.T} 下 onset={a.onset} 年立项已建不成，"
                  f"该档测不到\"等\"是否占优")
            continue
        for seed in a.seeds:
            r = run_one(lead, seed, a.iters, a.T, a.onset, a.eps,
                        gamma=a.gamma, ent_c=a.ent_c, lr=a.lr, epochs=a.epochs,
                        reward_at=a.reward_at, ramp_years=a.ramp_years,
                        foresight=a.foresight, lo=a.lo, hi=a.hi,
                        shaping=a.shaping)
            r["label"] = a.label or f"lead{lead}"
            rows.append(r)
            print(f"lead={lead} seed={seed}  A={r['A_init']} B={r['B_init']} "
                  f"C={r['C_init']}  (最优 A={r['opt_A']} B={r['opt_B']} "
                  f"C={r['opt_C']})  回报比={r['ratio']:.3f}")
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "gate_runs.csv"), index=False, encoding="utf-8-sig")

    summ = (D.groupby(["label", "lead"])
             .agg(n=("seed", "size"), 学会等=("pass_wait", "mean"),
                  命中最优年=("pass_opt", "mean"),
                  B不拖延=("pass_now", "mean"), 回报达标=("pass_ret", "mean"),
                  A立项年均值=("A_init", "mean"), A最优年=("opt_A", "mean"),
                  回报比均值=("ratio", "mean"))
             .reset_index())
    summ.to_csv(os.path.join(a.out, "gate_summary.csv"), index=False,
                encoding="utf-8-sig")
    print("\n" + summ.round(3).to_string(index=False))
    json.dump(dict(T=a.T, onset=a.onset, iters=a.iters, eps=a.eps,
                   leads=a.leads, seeds=a.seeds, gamma=a.gamma,
                   ent_c=a.ent_c, lr=a.lr, epochs=a.epochs,
                   reward_at=a.reward_at, ramp_years=a.ramp_years,
                   foresight=a.foresight, lo=a.lo, hi=a.hi, shaping=a.shaping,
                   label=a.label),
              open(os.path.join(a.out, "gate_config.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
