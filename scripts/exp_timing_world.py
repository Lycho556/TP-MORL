# -*- coding: utf-8 -*-
"""exp_timing_world.py —— 有限机会窗的择时世界：先在本地证明 RL 学得到"什么时候动手"。

## 为什么要有这个脚本（v17 门槛实验的教训）

v17 的门槛实验（scripts/exp_timing_gate.py）跑了十九档，没有一档学会推迟立项。
后来把机会摆幅从 1.6 倍加到 19 倍、把动作直接换成"给地块排一个立项年"、
把随时间递减的剩余年数特征去掉——都没用。再往下查才发现问题出在**世界本身**：

  1. 机会场是**单调上升**的，于是"越晚越好"，唯一拦住全体拖延的只有决策期长度。
     这种世界里根本没有"最佳时点"，只有"尽量靠后"。
  2. 名额宽松（3 个地块、10 个可用年份），早做几乎不牺牲什么，贪心解本来
     就能拿到穷举最优的 92–95%。于是"学没学会择时"被压缩进几个百分点，
     低于 PPO 的梯度噪声——测不出东西，不是学不会。
  3. 实测打分证实了这一点：训练后"动手"对"到此为止"的 logit 差 8 个单位，
     而三个地块之间只差 0.04。策略牢牢学会了"有得做就做"，对"做哪个、
     什么时候做"几乎没有分辨力。

所以本脚本换一个世界，而不是换一个算法：

    机会不是单调上升，而是**升 → 峰 → 落**的有限窗口。

于是"等太久"由机会场**自然**惩罚，不需要人为加 wait penalty（那种罚项一加，
审稿人立刻会问罚多少是怎么定的）；"全部往后拖"这种退化解也自动失效，
因为拖过峰值就开始亏。

## 本脚本的两个设计要点

### (1) 一年等待优势 G 是**解析可控**的自变量，不是幅度的代理量

机会曲线用高斯窗 O_i(t) = exp(−((t − p_i)/w_i)²)。在 t=0 处，
"等一年再做"相对"现在做"的折现收益比为

    1 + G = γ · O(1)/O(0) = γ · exp((2p − 1) / w²)

于是给定峰值年 p 与目标 G，窗宽 w 有闭式解

    w² = (2p − 1) / ln((1 + G)/γ)

也就是说 `--g-target 0.05` 造出来的世界，**一年等待优势精确等于 5%**。
扫 G 就是扫"等待到底值多少"，而不是扫一个说不清量纲的 amplitude。
这正是把问题从"调到多大 RL 才有效"换成"等待优势超过多少，RL 才稳定学到择时"。

### (2) 最优解用**指派问题**精确求解，不是枚举

每年至多 `quota` 个名额、每个地块至多立一次 —— 这是一个二分图最大权指派，
`scipy.optimize.linear_sum_assignment` 给的是精确最优。枚举法在地块数一多
就爆炸（本脚本前身在 6 个地块时就已经三百万种组合），而近似解不能用作判据：
判据依赖"与最优时点差几年"，最优解本身必须是精确的。

## 判据

    timing_acc   立项年与 oracle 最优年相差 ≤1 年的地块占比（**核心判据**）
    ratio        折现回报 / oracle 最优回报
    no_dump      是否没有退化成"全部往后拖"：早峰型地块的立项年不晚于其最优年 +1

三条都要报。只看 ratio 会把"随便做做也能拿九成"读成成功；只看 timing_acc
会把"什么都拖"读成学会了等。

## 世界自证：先证明这个世界里"等"确实可能最优

`validate()` 在训练之前检查两件事，不成立就直接报错退出——一个连最优解都不是
"等"的世界，测不出学习器有没有择时能力：

    * 晚峰型地块存在**内点**时间最优：折现价值序列先升后降（建议 §9 的
      "Act now < Wait 1 < Wait 2 > Wait 3"）。
    * 实测一年等待优势与 `--g-target` 吻合。

用法（本机）：
    PYTHONPATH=src python scripts/exp_timing_world.py --out results_world \\
        --g-scan 0 0.02 0.05 0.10 0.20 --seeds 0 1 2 3 4
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from tpmorl.objectives.reward import OBJ_NAMES
from tpmorl.rl.env_gym import N_FEAT, N_PAIR_FEAT, STOP
from tpmorl.rl import train_ppo

FLOOR_IDX = OBJ_NAMES.index("Floor")

#: 三型地块，峰值年按决策期的比例给出（T 变化时三型的相对时序不变）。
#:
#: 这个构成是本脚本判别力的来源，两条都是必须的：
#:   * **必须有早峰型**：否则"一直等"就是最优解，一个什么都往后拖的退化策略
#:     会冒充择时能力。
#:   * **峰值必须拉开**，且地块数要**少于**可用年份数。这样 oracle 的最优计划里
#:     会出现**空年**——某些年份什么都不做，把名额留给还没到窗口的地块。
#:     "主动留空"正是本实验要测的那个行为；若地块数把每年名额占满，
#:     最优计划被配额强制成"每年做一个"，策略无从做错，准确率会虚高到 1.00
#:     （本脚本第一版 6 地块 / 11 名额就是这个毛病，五档 G 全部满分）。
KINDS = (("E", "early", 0.05),     # 早峰：现在就是它最好的时候 -> 应当马上做
         ("L", "late", 0.45),      # 晚峰：还要等几年 -> 应当等
         ("V", "verylate", 0.78))  # 很晚才到窗口 -> 应当等很久，中间年份留空


def window_width(peak, g_target, gamma):
    """由目标一年等待优势反解高斯窗宽（见模块文档）。

    g_target <= γ−1 时无解（那意味着"等一年"连折现都补不回来，任何窗宽都
    做不出正的等待优势），此处直接报错而不是悄悄取一个近似值。
    """
    r = (1.0 + g_target) / gamma
    if r <= 1.0:
        raise ValueError(
            f"g_target={g_target:g} 在 γ={gamma:g} 下无解："
            f"需要 g_target > γ−1 = {gamma - 1:+.4f}，否则等一年必然净亏")
    denom = math.log(r)
    if 2.0 * peak - 1.0 <= 0:
        raise ValueError(f"峰值年 {peak:g} 太靠前，无法构造正的一年等待优势")
    return math.sqrt((2.0 * peak - 1.0) / denom)


class TimingWorld:
    """有限机会窗的最小择时世界。鸭子类型地实现 train_ppo 用到的那部分接口。

    只实现被用到的接口（逐个核过 run_episode / train / evaluate）：
        T / T_eval / gamma / quota / budget / scale / ch / ncell
        reset(seed) / pairs() / step(actions) / pair_cost(u, tg)
        budget_hist / spent_hist / released_hist
    """

    def __init__(self, n_per_kind=2, T=12, lead=1, gamma=0.95, quota=1,
                 g_target=0.05, channel="value", shape="window",
                 q_lo=0.30, q_hi=0.95, base=100.0, foresight=3, shaping=False,
                 wa_feat=False, action_mode="now"):
        self.T, self.T_eval = int(T), int(T)
        self.lead, self.gamma, self.quota = int(lead), float(gamma), int(quota)
        self.channel, self.shape = str(channel), str(shape)
        self.q_lo, self.q_hi = float(q_lo), float(q_hi)
        self.foresight, self.shaping = int(foresight), bool(shaping)
        # 是否把**一年等待优势**直接作为特征交给策略。
        #
        # 这一档回答最后一个分叉：策略不等，是因为**看不出来**等更好（信息问题），
        # 还是因为**即便看得出来也不会选**（动作选择问题）。把 WA 算好喂进去，
        # 等于把答案写在选项旁边——若策略仍然不等，那么信息不是瓶颈，
        # 后续力气应当全部花在动作选择/信用分配上，而不是继续加观测维度。
        self.wa_feat = bool(wa_feat)
        # 动作表示。now = 每年只能决定"这个地块现在立项 / 今年到此为止"（真实环境
        # 的动作空间）；schedule = 动作直接是"给这个地块排在第 y 年立项"。
        # 两档在**同一个有判别力的世界**上对照（oracle 要求留空 5~7 年、贪心只得
        # 最优的 0.64~0.93），才能回答：PPO 不会择时，是真的做不了时序推理，
        # 还是"今年停、明年再看"这种逐年表达太难——等待必须靠多次局部决策才能表达。
        if str(action_mode) not in ("now", "schedule"):
            raise ValueError(f"action_mode 只能是 now/schedule，收到 {action_mode!r}")
        self.action_mode = str(action_mode)
        self.g_target = float(g_target)

        self.names, self.kind, peaks = [], [], []
        for pre, k, frac in KINDS:
            for i in range(int(n_per_kind)):
                self.names.append(f"{pre}{i + 1}")
                self.kind.append(k)
                peaks.append(frac * self.T)
        self.n = len(self.names)
        self.peak = np.array(peaks, float)
        self.base = np.full(self.n, float(base))
        self.cost = np.zeros(self.n)          # 资金刻意不咬：把"等"的理由留给机会场
        self.scale = np.ones(len(OBJ_NAMES))
        self.ch = np.ones(self.n, int)
        self.ncell = np.ones(self.n, int)

        # 窗宽：晚峰型按目标 G 反解；早峰型用同一个宽度（保证两型可比，
        # 差别只在峰值年）；平坦型给一个很宽的窗，近似"什么时候做都差不多"。
        # 三型共用同一个窗宽：差别只在峰值年，故"哪一型该等多久"完全由峰值
        # 决定，不掺入宽度差异这个第二变量。
        w_late = window_width(KINDS[1][2] * self.T, self.g_target, self.gamma)
        self.width = np.full(self.n, w_late, float)

        yrs = np.arange(self.T + 1, dtype=float)
        if self.shape == "window":
            self.O = np.exp(-((yrs[None, :] - self.peak[:, None])
                              / self.width[:, None]) ** 2)
        elif self.shape == "rising":
            # 对照：单调上升（v17 的世界）。没有峰值，"越晚越好"。
            self.O = 1.0 / (1.0 + np.exp(-(yrs[None, :] - self.peak[:, None])
                                         / max(w_late, 1e-9)))
        elif self.shape == "static":
            # 负对照：完全平稳。此时最优时点只由折现决定 = 越早越好。
            self.O = np.ones((self.n, self.T + 1))
        else:
            raise ValueError(f"shape 只能是 window/rising/static，收到 {self.shape!r}")
        self.O = np.clip(self.O, 1e-6, 1.0)

    # ---- 价值与概率 ----
    def opp(self, u, t):
        return float(self.O[int(u), int(np.clip(t, 0, self.T))])

    def q(self, u, t):
        return self.q_lo + (self.q_hi - self.q_lo) * self.opp(u, t)

    def pay_year(self, t_init):
        return int(t_init) + self.lead

    def evalue(self, u, t_init):
        """第 t_init 年立项的**期望**收益（未折现）。

        value 档：收益随机会缩放。prob 档：收益固定，机会决定批得下来的概率。
        两档的期望式都用它，oracle 与判据才与实现同口径。
        """
        if t_init + self.lead > self.T - 1:
            return 0.0                       # 期内交付不了，等于没做
        if self.channel == "prob":
            return float(self.base[u] * self.q(u, t_init))
        return float(self.base[u] * self.opp(u, t_init))

    def dvalue(self, u, t_init):
        """折现到第 0 年的期望价值。oracle 与全部判据都用这一式。"""
        return self.evalue(u, t_init) * self.gamma ** self.pay_year(t_init)

    # ---- oracle：指派问题的精确最优 ----
    def oracle(self):
        """返回 (最优折现总价值, {地块名: 最优立项年})。

        每年至多 quota 个名额 → 把每一年复制 quota 列；每个地块至多一行。
        全部价值非负，故指派会自动填满 min(地块数, 名额数) 个位置，
        "不做"由行数多于列数时的未指派自然表示。
        """
        years = [y for y in range(self.T - self.lead) for _ in range(self.quota)]
        M = np.array([[self.dvalue(u, y) for y in years] for u in range(self.n)])
        ri, ci = linear_sum_assignment(-M)
        plan = {nm: None for nm in self.names}
        tot = 0.0
        for r, c in zip(ri, ci):
            if M[r, c] <= 0:
                continue
            plan[self.names[r]] = years[c]
            tot += M[r, c]
        return float(tot), plan

    # ---- 贪心参照：这个实例到底有没有判别力 ----
    def greedy(self):
        """逐年选**当期机会最高**的可立项地块，从不留空。返回 (折现总价值, 计划)。

        这是"不会择时"的策略能拿到的分数。若它已经接近 oracle，那么无论学习器
        多强，择时能力都被压缩进几个百分点、低于梯度噪声——这个实例就没有判别力，
        不该拿它下任何结论。上一版门槛实验正是栽在这里（贪心已有最优的 92–95%）。
        """
        done, plan, tot = set(), {nm: None for nm in self.names}, 0.0
        for t in range(self.T - self.lead):
            cand = [(self.opp(u, t), u) for u in range(self.n) if u not in done]
            if not cand:
                break
            _, u = max(cand)
            done.add(u)
            plan[self.names[u]] = t
            tot += self.dvalue(u, t)
        return float(tot), plan

    # ---- 世界自证 ----
    def validate(self, tol=0.02):
        """训练之前先证明这个世界里"等"确实可能最优。不成立就报错退出。"""
        out = {}
        # (a) 晚峰型必须存在**内点**时间最优：先升后降
        i = self.kind.index("late")
        v = np.array([self.dvalue(i, y) for y in range(self.T - self.lead)])
        k = int(np.argmax(v))
        if self.shape == "window":
            if not 0 < k < len(v) - 1:
                raise AssertionError(
                    f"晚峰型地块的折现最优时点在边界（第 {k} 年，共 {len(v)} 个可选年）："
                    "这个世界里最优解不是「等到某一年」而是「尽早/尽晚」，测不到择时")
            if not (v[0] < v[k] and v[-1] < v[k]):
                raise AssertionError("晚峰型的折现价值序列不是先升后降，机会窗没建立起来")
        out["late_peak_opt_year"] = k
        out["late_peak_profile"] = [round(float(x), 2) for x in v]

        # (b) 实测一年等待优势必须与 --g-target 吻合
        g = self.dvalue(i, 1) / max(self.dvalue(i, 0), 1e-12) - 1.0
        out["g_measured"] = float(g)
        if self.shape == "window" and abs(g - self.g_target) > tol:
            raise AssertionError(
                f"实测一年等待优势 {g:+.4f} 与目标 {self.g_target:+.4f} 相差超过 {tol}："
                "窗宽反解与实现不一致，扫描的横轴就不可信")

        # (c) 名额是否紧张：地块数 vs 可用名额。宽松时贪心本就接近最优，判别力低
        slots = max(self.T - self.lead, 0) * self.quota
        out["n_units"], out["slots"] = self.n, slots

        # (c) **判别力**：贪心解与最优解的差距。差距太小的实例不该用来下结论。
        opt, oplan = self.oracle()
        gre, gplan = self.greedy()
        out["oracle_value"], out["greedy_value"] = opt, gre
        out["greedy_ratio"] = float(gre / opt) if opt else float("nan")
        out["discriminative"] = bool(out["greedy_ratio"] <= 0.90)
        # (d) 最优计划是否要求**留空年**——"主动留空等窗口"正是要测的行为
        used = sorted(v for v in oplan.values() if v is not None)
        out["oracle_years"] = used
        out["oracle_idle_years"] = int(max(used) + 1 - len(used)) if used else 0
        return out

    # ---- 环境接口 ----
    def pair_cost(self, u, tg):
        return 0.0

    def reset(self, seed=None):
        self.arng = np.random.default_rng(12345 if seed is None else int(seed))
        self.t = 0
        self.init_year, self.done_year, self.approved = {}, {}, {}
        self.budget = 1e9
        self.budget_hist, self.spent_hist, self.released_hist = [], [], []
        return self._obs()

    def _obs(self):
        return np.zeros((self.n, N_FEAT), dtype=np.float32)

    def _available(self):
        return [u for u in range(self.n)
                if u not in self.init_year and self.t + self.lead <= self.T - 1]

    def pairs(self):
        av = self._available()
        if self.action_mode == "now":
            opts = [(u, self.t) for u in av]
        else:
            # 配额落在**被排定的那一年**上：某年已排满就不再出现，否则"排期"
            # 会退化成无约束的一次性分配，与逐年名额有限这件事不符
            used = {}
            for _u, _y in self.init_year.items():
                used[_y] = used.get(_y, 0) + 1
            opts = [(u, y) for u in av for y in range(self.t, self.T - self.lead)
                    if used.get(y, 0) < self.quota]
        rows = np.zeros((len(opts) + 1, N_PAIR_FEAT), dtype=np.float32)
        for j, (u, yy) in enumerate(opts):
            rows[j, 0] = self.opp(u, yy)                           # 该选项的机会
            if self.foresight:
                # 趋势：机会窗世界里这一维会**变号**——峰前为正、峰后为负，
                # 这正是"窗口正在关闭"的信号，单调场里它永远为正、没有信息量
                rows[j, 1] = self.opp(u, yy + self.foresight) - rows[j, 0]
            rows[j, 2] = self.base[u] / max(self.base.max(), 1e-9)
            rows[j, 5] = (yy - self.t) / self.T     # 排在多少年之后（now 档恒 0）
            if self.wa_feat:
                now = self.dvalue(u, self.t)
                best = max([self.dvalue(u, y)
                            for y in range(self.t, self.T - self.lead)] or [0.0])
                rows[j, 3] = (best / now - 1.0) if now > 0 else 0.0   # 最优等待增益
                nxt = self.dvalue(u, self.t + 1)
                rows[j, 4] = (nxt / now - 1.0) if now > 0 else 0.0    # 一年等待优势
            rows[j, 14] = self.t / self.T
            rows[j, 15] = 1.0
        rows[len(opts), 14] = self.t / self.T
        rows[len(opts), N_FEAT + 16] = 1.0
        meta = [(u, y) for u, y in opts] + [STOP]
        cost = np.zeros(len(opts) + 1)
        units = np.concatenate([np.asarray([u for u, _ in opts], np.int64),
                                np.array([-1], np.int64)])
        return rows, meta, cost, units

    def potential(self):
        """势函数 Φ(s)：未立项地块的持有价值之和（折到当年）。"""
        if not self.shaping:
            return 0.0
        tot = 0.0
        for u in range(self.n):
            if u in self.init_year:
                continue
            cand = [self.dvalue(u, y) / self.gamma ** self.t
                    for y in range(self.t, self.T - self.lead)]
            if cand:
                tot += max(cand)
        return float(tot)

    def step(self, actions):
        actions = [(int(u), int(tg)) for u, tg in actions if int(u) >= 0]
        if len(actions) > self.quota:
            raise ValueError(f"超出年度配额 {self.quota}：本期请求 {len(actions)}")
        phi0 = self.potential()      # 必须在动作生效前取
        self.budget_hist.append(self.budget)
        self.spent_hist.append(0.0)
        self.released_hist.append(0.0)
        for u, y in actions:
            y0 = self.t if self.action_mode == "now" else int(y)
            self.init_year[u] = y0
            self.done_year[u] = y0 + self.lead
            if self.channel == "prob":
                self.approved[u] = bool(self.arng.random() < self.q(u, y0))
        gain = 0.0
        for u, dy in self.done_year.items():
            if dy != self.t:
                continue
            y0 = self.init_year[u]
            if self.channel == "prob":
                gain += self.base[u] if self.approved.get(u, True) else 0.0
            else:
                gain += self.base[u] * self.opp(u, y0)
        vec = np.zeros(len(OBJ_NAMES))
        vec[FLOOR_IDX] = gain
        self.t += 1
        r = float(gain)
        if self.shaping:
            r += self.gamma * self.potential() - phi0
        return self._obs(), r, self.t >= self.T_eval, dict(vec=vec, raw={}, events={})


def stop_diagnostics(w, net):
    """贪心走一遍，记录策略在"该等的时候等不等"上的行为，以及 STOP 的打分位置。

    两个条件概率（建议 §9 要的那两项）：
        P(停 | 存在 WA>0 的候选)  该等的时候，它等了吗
        P(停 | 不存在)            不该等的时候，它是不是乱等
    一个健康的择时策略前者高、后者低；两者都低 = 从不等；都高 = 无脑拖延。

    同时记录每年 STOP 的 logit 与最高候选 logit 之差。这个差值回答的是
    "STOP 到底有没有被抬起来过"——只看最终立项年是看不出来的。
    """
    import torch
    w.reset(seed=0)
    n_pos = n_pos_stop = n_neg = n_neg_stop = 0
    gaps = []
    for t in range(w.T_eval):
        act = []
        if t < w.T:
            X, meta, cost, units = w.pairs()
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(X))
            lg = logits.numpy()
            i_best = int(np.argmax(lg[:-1])) if len(lg) > 1 else None
            gaps.append(float(lg[-1] - (lg[i_best] if i_best is not None else lg[-1])))
            chose_stop = bool(np.argmax(lg) == len(lg) - 1)
            if len(meta) <= 1:
                # 候选集为空（全部地块都已立项）：这一年"停止"是被迫的，不是选择。
                # 把它算进条件概率会让 P(停|不该等) 虚高到 1.00，读成"该停的时候
                # 都停了"——实际只是无事可做。
                _, _, done, _ = w.step([])
                if done:
                    break
                continue
            # 本年是否存在"等一年更好"的候选（只看 now 档的当期决策）
            has_pos = any(w.dvalue(u, t + 1) > w.dvalue(u, t) > 0
                          for u, _y in meta[:-1])
            if has_pos:
                n_pos += 1; n_pos_stop += int(chose_stop)
            else:
                n_neg += 1; n_neg_stop += int(chose_stop)
            if not chose_stop and i_best is not None:
                act = [meta[i_best]]
        _, _, done, _ = w.step(act)
        if done:
            break
    return dict(p_stop_given_wa_pos=(n_pos_stop / n_pos) if n_pos else np.nan,
                p_stop_given_wa_neg=(n_neg_stop / n_neg) if n_neg else np.nan,
                n_years_wa_pos=n_pos,
                stop_logit_gap_mean=float(np.mean(gaps)) if gaps else np.nan,
                stop_logit_gap_max=float(np.max(gaps)) if gaps else np.nan)


def run_one(world_kw, seed, iters, eps):
    stop_ctx = bool(world_kw.pop("stop_context", False))
    drop_nc = bool(world_kw.pop("drop_no_choice", False))
    advk = str(world_kw.pop("advantage", "gae"))
    w = TimingWorld(**world_kw)
    diag = w.validate()
    net, hist = train_ppo.train(w, iters=iters, eps_per_iter=eps, seed=seed,
                                stop_context=stop_ctx, drop_no_choice=drop_nc,
                                advantage=advk)
    rec = []
    train_ppo.evaluate(w, net, n_ep=1, record=rec)
    R = pd.DataFrame(rec)
    R = R[R["unit"] >= 0] if len(R) else R
    init = {w.names[int(r.unit)]: int(r.year) for r in R.itertuples()}
    opt, plan = w.oracle()
    got = sum(w.dvalue(u, init[nm]) for u, nm in enumerate(w.names) if nm in init)

    hit, late_ok, early_ok = [], [], []
    for u, nm in enumerate(w.names):
        if plan[nm] is None:
            continue
        if nm in init:
            hit.append(abs(init[nm] - plan[nm]) <= 1)
            (late_ok if w.kind[u] == "late" else early_ok).append(
                abs(init[nm] - plan[nm]) <= 1)
        else:
            hit.append(False)
            (late_ok if w.kind[u] == "late" else early_ok).append(False)
    # 退化解检查：早峰型不得被一起往后拖
    dump = [init[nm] <= plan[nm] + 1 for u, nm in enumerate(w.names)
            if w.kind[u] == "early" and nm in init and plan[nm] is not None]
    sd = stop_diagnostics(w, net)
    return dict(seed=seed, stop_context=stop_ctx, drop_no_choice=drop_nc,
                advantage=advk,
                action_mode=world_kw.get("action_mode", "now"), **sd,
                g_target=world_kw.get("g_target"), g_measured=diag["g_measured"],
                shape=world_kw.get("shape"), channel=world_kw.get("channel"),
                shaping=bool(world_kw.get("shaping")),
                wa_feat=bool(world_kw.get("wa_feat")),
                n_units=diag["n_units"], slots=diag["slots"],
                greedy_ratio=diag["greedy_ratio"],
                discriminative=diag["discriminative"],
                oracle_idle_years=diag["oracle_idle_years"],
                late_peak_opt_year=diag["late_peak_opt_year"],
                timing_acc=float(np.mean(hit)) if hit else np.nan,
                timing_acc_late=float(np.mean(late_ok)) if late_ok else np.nan,
                timing_acc_early=float(np.mean(early_ok)) if early_ok else np.nan,
                no_dump=float(np.mean(dump)) if dump else np.nan,
                ratio=got / opt if opt else np.nan,
                init_mean=float(np.mean(list(init.values()))) if init else np.nan,
                opt_mean=float(np.mean([v for v in plan.values() if v is not None])),
                curve_last=float(hist[-1]) if len(hist) else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_world")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--eps", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--quota", type=int, default=1)
    ap.add_argument("--n-per-kind", type=int, default=1,
                    help="每型几个地块（共 3×K）。**必须远少于可用年份数**，"
                         "否则最优计划被配额强制成每年一个，策略无从做错")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--channel", default="value", choices=["value", "prob"])
    ap.add_argument("--shape", default="window", choices=["window", "rising", "static"])
    ap.add_argument("--foresight", type=int, default=3)
    ap.add_argument("--shaping", action="store_true")
    ap.add_argument("--action-mode", default="now", choices=["now", "schedule"],
                    help="动作表示。now=每年决定现在做/今年到此为止（真实环境）；"
                         "schedule=直接给地块排一个立项年。两档在同一个有判别力的"
                         "世界上对照，回答「不会择时」是做不了时序推理，还是逐年"
                         "表达太难")
    ap.add_argument("--stop-context", action="store_true",
                    help="让「到此为止」的打分看到整个候选集合的池化表示，并单列"
                         "一个头。默认结构下 STOP 只是一行普通候选，它自己的特征里"
                         "没有「所有地块此刻时序状态如何」这个信息")
    ap.add_argument("--drop-no-choice", action="store_true",
                    help="把「当年只有到此为止一个合法动作」的无决策步从 PPO 缓冲区"
                         "剔除。这些步本来就不产生策略梯度，却进入优势归一化，"
                         "实测占到 47%、优势均值 −0.93，把「主动等」压在「立项」之下")
    ap.add_argument("--advantage", default="gae", choices=["gae", "mc"],
                    help="优势估计。gae=现有的 critic + GAE；mc=精确蒙特卡洛回报"
                         "（不用 critic 基线）。critic 读的是候选行均值池化，"
                         "地块被消耗后分不清「还剩三个都在窗口中段」与「只剩一个"
                         "正在峰值」——这一档用来判断偏差是否来自值函数基线")
    ap.add_argument("--wa-feat", action="store_true",
                    help="把一年等待优势与最优等待增益直接作为特征给策略。"
                         "用于分离「看不出来等更好」与「看得出来也不选」两种失败")
    ap.add_argument("--g-scan", type=float, nargs="+", default=None,
                    help="扫描一年等待优势 G（如 0.02 0.05 0.10 0.20）。"
                         "这是本脚本的主实验：横轴是**解析可控**的等待优势，"
                         "纵轴是择时准确率，回答「等待优势超过多少，RL 才稳定学到择时」")
    ap.add_argument("--g-target", type=float, default=0.05)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    base_kw = dict(n_per_kind=a.n_per_kind, T=a.T, lead=a.lead, gamma=a.gamma,
                   quota=a.quota, channel=a.channel, shape=a.shape,
                   foresight=a.foresight, shaping=a.shaping, wa_feat=a.wa_feat,
                   action_mode=a.action_mode, stop_context=a.stop_context,
                   drop_no_choice=a.drop_no_choice, advantage=a.advantage)
    gs = a.g_scan if a.g_scan else [a.g_target]
    rows = []
    for g in gs:
        kw = dict(base_kw, g_target=g)
        # 世界自证只关心环境，不关心策略结构：stop_context 是网络选项，须剔除
        wkw = {k: v for k, v in kw.items()
               if k not in ("stop_context", "drop_no_choice", "advantage")}
        try:
            TimingWorld(**wkw).validate()
        except (AssertionError, ValueError) as e:
            print(f"[跳过] G={g:+.3f}：{e}")
            continue
        _d = TimingWorld(**wkw).validate()
        if not _d["discriminative"]:
            print(f"[警告] G={g:+.3f}：贪心解已达最优的 {_d['greedy_ratio']:.3f}，"
                  "该实例判别力不足，结果只作参考")
        for seed in a.seeds:
            r = run_one(kw, seed, a.iters, a.eps)
            rows.append(r)
        d = pd.DataFrame(rows)
        d = d[d.g_target == g]
        print(f"G={g:+.3f}（实测 {d.g_measured.iloc[0]:+.4f}，晚峰型最优年 "
              f"{int(d.late_peak_opt_year.iloc[0])}）  择时准确率 "
              f"{d.timing_acc.mean():.2f}（晚峰型 {d.timing_acc_late.mean():.2f}）  "
              f"未退化 {d.no_dump.mean():.2f}  回报比 {d.ratio.mean():.3f}  "
              f"[贪心参照 {d.greedy_ratio.iloc[0]:.3f}  最优留空 "
              f"{int(d.oracle_idle_years.iloc[0])} 年]")
        print(f"        P(停|该等) {d.p_stop_given_wa_pos.mean():.2f}  "
              f"P(停|不该等) {d.p_stop_given_wa_neg.mean():.2f}  "
              f"STOP logit 相对最高候选 均值 {d.stop_logit_gap_mean.mean():+.2f} / "
              f"最高 {d.stop_logit_gap_max.mean():+.2f}")

    if not rows:
        raise SystemExit("没有任何一档通过世界自证，未进行训练。")
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "world_runs.csv"), index=False, encoding="utf-8-sig")
    S = (D.groupby("g_target")
          .agg(n=("seed", "size"), 实测G=("g_measured", "mean"),
               晚峰型最优年=("late_peak_opt_year", "mean"),
               择时准确率=("timing_acc", "mean"),
               晚峰型准确率=("timing_acc_late", "mean"),
               早峰型准确率=("timing_acc_early", "mean"),
               未退化=("no_dump", "mean"), 回报比=("ratio", "mean"),
               贪心参照=("greedy_ratio", "mean"),
               P停_该等=("p_stop_given_wa_pos", "mean"),
               P停_不该等=("p_stop_given_wa_neg", "mean"),
               STOP_logit差=("stop_logit_gap_mean", "mean"),
               最优留空年数=("oracle_idle_years", "mean"),
               立项重心=("init_mean", "mean"), 最优重心=("opt_mean", "mean"))
          .reset_index())
    S.to_csv(os.path.join(a.out, "world_summary.csv"), index=False,
             encoding="utf-8-sig")
    print("\n" + S.round(3).to_string(index=False))
    json.dump(vars(a), open(os.path.join(a.out, "world_config.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
