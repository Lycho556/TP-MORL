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


def build_env(dataset, T, alpha, seed, budget, objective="floor",
              unit_only=False, static_field=False, prescreen=0, foresight=0):
    env = RenewalEnv(dataset, T=T, T_eval=SC.horizon_eval(),
                     weights=gate_weights(objective, alpha),
                     scale=np.ones(len(OBJ_NAMES)))
    env.reset(seed=seed)
    env.budget = float(budget)          # 预算不咬：本闸只测时序选择
    if unit_only:
        # 空间闸的诊断档：**每个单元只保留一个目标功能**，动作空间由约 1985 个
        # (单元,目标) 配对降到 717 个单元。
        #
        # 为什么要有这一档：oracle 与近视贪心的计价口径是 EV(u, t)，只关心
        # "哪个单元、哪一年"；而 PPO 面对的是 (单元,目标) 配对的自回归选择，
        # 一年之内还要连选 3 次。两边其实不是同一个问题，PPO 额外承担了
        # 目标功能选择的难度。固定目标后两边口径一致，才能干净地问一句
        # "PPO 能不能在 717 个单元里做好当期选择"。
        #
        # **这是诊断档，不是论文最终模型**——最终模型必须保留目标功能选择。
        # 选哪个目标：按该单元 FAR 上限最高的那个（与 EV 的 base 定义一致），
        # 这样固定动作不会顺带改变计价口径。
        env.fix_one_target_per_unit()
    if prescreen:
        env.set_prescreen(int(prescreen))
    if static_field:
        # 把机会场冻结在第 0 年：admit/hazard/value 三个查询一律返回第 0 年取值。
        # 此时环境里**不存在任何时序优势**，oracle 与近视贪心应当几乎相等，
        # 问题退化为纯空间选择。这一档用来单独考核"挑谁"的能力。
        env.opp = _FrozenField(env.opp, 0)
    return env


class _FrozenField:
    """把机会场冻结在第 t0 年的包装器（诊断用，见 build_env 的 static_field）。"""

    def __init__(self, opp, t0=0):
        self._o, self.t0 = opp, int(t0)
        self.kind = opp.kind

    def admit_prob(self, t):
        return self._o.admit_prob(self.t0)

    def hazard_mult(self, t):
        return self._o.hazard_mult(self.t0)

    def value_mult(self, u, t_init, t_done):
        return self._o.value_mult(u, self.t0, self.t0)

    def obs_block(self, *a, **k):
        return self._o.obs_block(*a, **k)

    def __getattr__(self, k):
        return getattr(self._o, k)


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


def decompose(EV, init, quota, eligible, v_oracle):
    """反事实价值分解：把 PPO 的缺口拆成"挑错单元"与"放错年份"两份。

    做法是对**同一批被选中的单元**重解一次配额约束下的最优时间分配：

        V_实际      = Σ EV[u, 策略给的年份]
        V_单元_最优时 = restricted assignment(仅这批单元) 的最优值
        V_oracle    = 全体合规单元上的最优值

        放错年份的损失 = V_单元_最优时 − V_实际
        挑错单元的损失 = V_oracle    − V_单元_最优时

    两项按构造相加等于总缺口。**这是反事实分解，不是严格可加的因果分解**
    （选谁与何时本来是耦合的，重新计时也会改变彼此的竞争关系）；论文里应当
    称作 counterfactual decomposition，不要写成 causal decomposition。

    与"直接把年份换成 oracle 指派年"相比，这里重解一次指派是必要的：
    直接换年份会让多个单元挤到同一年、突破每年的名额，得到的是一个**不可行**
    的上界；重解指派给出的是"这批单元在名额约束下能达到的最好时序"，可行且可比。
    """
    if not init:
        return dict(v_actual=0.0, v_best_timing=0.0, loss_timing=np.nan,
                    loss_selection=np.nan, share_timing=np.nan)
    units = np.array(sorted(init))
    mask = np.zeros(EV.shape[0], bool)
    mask[units] = True
    mask &= np.asarray(eligible, bool)
    v_best, _ = oracle_plan(EV, quota, mask)
    v_act = float(sum(EV[u, y] for u, y in init.items()))
    l_t = v_best - v_act
    l_s = v_oracle - v_best
    tot = l_t + l_s
    return dict(v_actual=v_act, v_best_timing=float(v_best),
                loss_timing=float(l_t), loss_selection=float(l_s),
                share_timing=float(l_t / tot) if tot > 0 else np.nan)


def bc_actor(env_fn, rank, nf, n_ep=6, iters=800, lr=3e-3, seed=0):
    """监督初始化：用同一批候选行特征回归"这个候选此刻值多少"，只训 enc+score。

    为什么在 717 单元这一侧必须有它（小世界里反而无用）：
    上一轮实测，同样的 50 维特征、同样大小的两层网络，改用监督目标时
    秩相关 0.970、当年首选的价值比 1.00（随机 0.39）；而同一网络在 PPO 下
    训 400 迭代只到随机水平。每年 595 个候选选 3 个，一个标量回合回报要摊到
    上万次候选评分上，单个候选拿到的梯度信号极弱 —— 这正是监督能补的那一段。
    小世界只有 18~24 个候选，所以那里 PPO 自己就能学会，BC 无用；
    **不能据此否定 717 单元这一侧。**

    只训 actor（enc + score），critic 保持随机：critic 的目标依赖策略本身。
    教师 rank 由调用方给定 —— 传近视表则是"只看当期"，传真实 EV 则是
    "按已公布的规划与设施计划前瞻"，后者是一个需要声明的信息档。
    """
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for ep in range(n_ep):
        env = env_fn()
        for t in range(env.T):
            X, meta, cost, units = env.pairs()
            real = [(i, m[0]) for i, m in enumerate(meta) if m[0] >= 0]
            if not real:
                if env.step([])[2]:
                    break
                continue
            for i, u in real:
                Xs.append(X[i]); ys.append(rank[u, min(t, rank.shape[1] - 1)])
            # **必须把末行「今年到此为止」也作为监督样本，目标取 0。**
            #
            # 指南建议"BC 不学 STOP"（把时序留给 RL），在 18~24 个候选的小世界
            # 里无害；但在 717 单元这一侧实测会直接崩掉：只训真实候选行时，
            # STOP 的打分完全未经校准，初始化后它压过所有真实候选，
            # 一年只立项几个 —— seed 0 实测 0.147（比随机 0.716 还低），
            # 挑错单元损失 3926，立项数 40 而非 75。
            #
            # 取 0 是有道理的而不是调参：EV 全为正，"今年不做"的当期价值就是 0，
            # 故 BC 学到的序是"任何正价值的候选都优于停"，这恰好是配额未用满时
            # 该有的行为；**何时真的该停仍然由 RL 决定**（初始化只给出一个不
            # 病态的起点，不锁定行为）。
            Xs.append(X[-1]); ys.append(0.0)
            # 随机推进，覆盖 PPO 训练初期实际会遇到的状态分布
            pick = [meta[i] for i, _ in
                    [real[j] for j in rng.permutation(len(real))[:env.quota]]]
            if env.step(pick)[2]:
                break
    X = np.asarray(Xs, np.float32); y = np.asarray(ys, np.float32)
    ym, ysd = y.mean(), y.std() + 1e-8
    torch.manual_seed(seed)
    net = TP.Pointer()
    params = list(net.enc.parameters()) + list(net.score.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    Xt = torch.as_tensor(X); yt = torch.as_tensor((y - ym) / ysd)
    for _ in range(iters):
        idx = torch.randint(0, len(Xt), (min(2048, len(Xt)),))
        loss = ((net.score(net.enc(Xt[idx])).squeeze(-1) - yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pr = net.score(net.enc(Xt)).squeeze(-1).numpy()
    from scipy.stats import spearmanr
    rho = float(spearmanr(pr, y).statistic)
    sd = {k: v for k, v in net.state_dict().items()
          if k.startswith("enc.") or k.startswith("score.")}
    return sd, dict(bc_n=len(X), bc_rho=rho)


def eval_checkpoint(env_fn, net, EV, EVm, quota, v_orc, elig, gamma,
                    ref_states=None, ref_scores=None):
    """一个检查点只求指南要的那几个数，不再产生一堆指标。

        oracle_ratio   相对 oracle 的折现价值（EV 计价）
        where_wrong    挑错单元损失
        when_error     放错年份损失
        stop_rate      配额未用满的比例（"今年到此为止"被选中的频率）
        actual_reward  **环境自身**的折现奖励流 —— 用来确认评价指标与
                       环境奖励没有再次出现口径打架（上一轮就吃过这个亏）
        rank_corr_bc   与 BC 初始打分在同一批固定状态上的秩相关；
                       它回答"PPO 是不是很快就改掉了候选排序"
    """
    env = env_fn()
    init, n_stop_slot, n_slot = {}, 0, 0
    R = 0.0
    for t in range(env.T):
        X, meta, cost, units = env.pairs()
        avail = [(i, m[0]) for i, m in enumerate(meta) if m[0] >= 0]
        if not avail:
            _, r, done, _ = env.step([])
            R += (gamma ** t) * float(np.sum(r))
            if done:
                break
            continue
        with torch.no_grad():
            lg = net(torch.as_tensor(X))[0].numpy()
        order = np.argsort(-lg)
        act, used = [], set()
        stop_row = len(meta) - 1
        for i in order:
            if len(act) >= quota:
                break
            if i == stop_row:
                break                      # 选中「到此为止」：本年结束
            u = meta[i][0]
            if u < 0 or u in used:
                continue
            act.append(meta[i]); used.add(u); init.setdefault(u, t)
        n_slot += quota
        n_stop_slot += quota - len(act)
        _, r, done, _ = env.step(act)
        R += (gamma ** t) * float(np.sum(r))
        if done:
            break
    v = float(sum(EV[u, min(t2, EV.shape[1] - 1)] for u, t2 in init.items()))
    d = decompose(EV, init, quota, elig, v_orc)
    out = dict(oracle_ratio=v / v_orc, value=v,
               where_wrong=d["loss_selection"], when_error=d["loss_timing"],
               stop_rate=n_stop_slot / max(n_slot, 1), actual_reward=R,
               n_init=len(init))
    if ref_states is not None:
        from scipy.stats import spearmanr
        with torch.no_grad():
            cur = np.concatenate([net(torch.as_tensor(X))[0].numpy()
                                  for X in ref_states])
        out["rank_corr_bc"] = float(spearmanr(cur, ref_scores).statistic)
    return out


def ref_state_batch(env_fn, net, n_year=6):
    """固定一批状态与 BC 在其上的打分，供后续检查点比较候选排序的变化。"""
    env = env_fn()
    Xs, rng = [], np.random.default_rng(0)
    for t in range(n_year):
        X, meta, cost, units = env.pairs()
        avail = [(i, m[0]) for i, m in enumerate(meta) if m[0] >= 0]
        if not avail:
            if env.step([])[2]:
                break
            continue
        Xs.append(X)
        pick = [meta[i] for i, _ in
                [avail[j] for j in rng.permutation(len(avail))[:env.quota]]]
        if env.step(pick)[2]:
            break
    with torch.no_grad():
        sc = np.concatenate([net(torch.as_tensor(X))[0].numpy() for X in Xs])
    return Xs, sc


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
        elif kind == "forecast":
            # **时间感知贪心**（v19 表里的 Model B）：按**真实** EV 排序，
            # 即"知道交付时的条件"但仍然逐年贪心、不跨年协调。
            #
            # 它与近视贪心的差 = 前瞻信息值多少；它与 oracle 的差 =
            # 跨年协调值多少。这两个量把总缺口分成可解释的两段，
            # 正是论文表里"Temporal heuristic"那一行的意义。
            order = sorted(avail, key=lambda p: -EV[p[1], t])
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


def run_curve(a):
    """BC → PPO 训练长度曲线：一次训到最长迭代数，沿途在检查点求值。

    与"每个检查点各训一次"是同一条轨迹（同种子、同数据流），
    但只付一次训练成本 —— 400 迭代一个种子约 20 分钟，分开跑要 3 倍以上。
    """
    SC.reset()
    SC.apply(horizon=a.horizon, horizon_eval="auto", opp_shape="window",
             a_plan=a.amps[0], a_infra=a.amps[1], a_age=a.amps[2],
             a_ready=a.amps[3], foresight=a.foresight, quota=a.quota,
             budget=a.budget)
    env_fn = lambda: build_env(a.dataset, a.horizon, a.alpha, 7, a.budget,
                               a.objective, a.unit_only, a.static_field,
                               a.prescreen, a.foresight)
    e0 = env_fn()
    EV, EVm = ev_tables(e0, a.gamma)
    elig = np.asarray(e0.env.eligible, bool) if hasattr(e0, "env") \
        else np.ones(EV.shape[0], bool)
    v_orc, oplan = oracle_plan(EV, a.quota, elig)
    ckpts = sorted(set(int(x) for x in a.curve))
    rows = []
    for seed in a.seeds:
        init_sd, bcm = bc_actor(env_fn, EV if a.bc_init == "forecast" else EVm,
                                None, seed=seed)
        print(f"seed={seed} 监督初始化秩相关 {bcm['bc_rho']:.3f}")
        bc_net = TP.Pointer(); bc_net.load_state_dict(init_sd, strict=False)
        ref_X, ref_sc = ref_state_batch(env_fn, bc_net)

        def cb(it, net, seed=seed):
            if it not in ckpts:
                return
            m = eval_checkpoint(env_fn, net, EV, EVm, a.quota, v_orc, elig,
                                a.gamma, ref_X, ref_sc)
            rows.append(dict(seed=seed, iters=it, **m))
            print(f"  iter={it:>4}  相对oracle {m['oracle_ratio']:.3f}  "
                  f"挑错单元 {m['where_wrong']:>7.0f}  "
                  f"放错年份 {m['when_error']:>7.0f}  "
                  f"停用名额 {m['stop_rate']:.3f}  "
                  f"环境奖励 {m['actual_reward']:.3g}  "
                  f"与BC秩相关 {m.get('rank_corr_bc', float('nan')):.3f}")

        TP.train(env_fn(), iters=max(ckpts), eps_per_iter=a.eps, seed=seed,
                 init_actor=init_sd, callback=cb)

    D = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    D.to_csv(os.path.join(a.out, "curve_runs.csv"), index=False,
             encoding="utf-8-sig")
    Sm = (D.groupby("iters").agg(
        n=("oracle_ratio", "size"), 相对oracle=("oracle_ratio", "mean"),
        标准差=("oracle_ratio", "std"), 最低=("oracle_ratio", "min"),
        挑错单元=("where_wrong", "mean"), 放错年份=("when_error", "mean"),
        停用名额=("stop_rate", "mean"), 环境奖励=("actual_reward", "mean"),
        与BC秩相关=("rank_corr_bc", "mean")).reset_index())
    Sm.to_csv(os.path.join(a.out, "curve_summary.csv"), index=False,
              encoding="utf-8-sig")
    print("\n" + Sm.round(4).to_string(index=False))
    print("\n判读：第 0 行就是 BC 本身。若某个中间迭代数稳定 ≥ 第 0 行，"
          "则存在短期微调窗口；若一路下降，则直接以 BC 作为主模型，"
          "论文里诚实写 RL 微调未能稳定改善监督初始化。")
    return D


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
    ap.add_argument("--prescreen", type=int, default=0,
                    help="每年只把当期分数最高的 K 个候选送进动作空间（0=不筛）。"
                         "筛选分数只用当年可观测量（当期机会指数 × 基准交付量），"
                         "与近视贪心同信息——故结果只能读作「近视初筛之上的 RL」，"
                         "不能读作「RL 自己学会了空间选择」")
    ap.add_argument("--bc-init", default="off",
                    choices=["off", "myopic", "forecast"],
                    help="actor 的监督初始化。myopic=教师只用当期条件；"
                         "forecast=教师按已公布的规划与设施计划前瞻"
                         "（需在论文里声明为一个信息档）。唯一改动是初始化，"
                         "环境/奖励/动作空间/超参不变")
    ap.add_argument("--curve", type=int, nargs="+", default=None,
                    help="BC→PPO 训练长度曲线的检查点，例如 0 2 10 50 150 400。"
                         "第 0 点 = BC 本身（基准线）。一次训到最大值、沿途求值")
    ap.add_argument("--bc-only", action="store_true",
                    help="只做监督初始化、跳过 PPO。给出表里\"监督时空打分器\""
                         "那一行，用于分离 PPO 的净效应")
    ap.add_argument("--load-net", action="store_true",
                    help="若 --out 目录下已有 net_seed*.pt 则直接加载、跳过训练。"
                         "事后分析与训练无关，而一个种子 400 迭代要 25 分钟")
    ap.add_argument("--unit-only", action="store_true",
                    help="诊断档：每单元只留一个目标功能，动作空间 ~1985 → 717。"
                         "使 PPO 与 oracle/近视贪心面对同一个问题（只挑单元与年份）")
    ap.add_argument("--static-field", action="store_true",
                    help="诊断档：把机会场冻结在第 0 年，**去掉全部时序优势**。"
                         "此时 oracle 与近视贪心应当几乎相等，问题退化为纯空间选择："
                         "判据改为 (V_PPO − V_随机)/(V_近视 − V_随机)，即恢复了多少"
                         "空间选择能力")
    ap.add_argument("--quota", type=int, default=3)
    ap.add_argument("--budget", type=float, default=1e12,
                    help="预算上限。默认极大 = 不咬：本闸只测时序选择，"
                         "把资金约束一并打开会让任何负结果无法归因")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--foresight", type=int, default=0,
                    help="观测里给出 t+K 年的机会场取值（对应「法定图则与设施"
                         "计划已公布」的信息档）。小世界实测：交付年计价下这是"
                         "**载荷变量** —— 无前瞻时 0.780 是信息上界而非学习失败，"
                         "给了前瞻后 PPO 从随机初始化即达 0.994")
    ap.add_argument("--amps", type=float, nargs=4, default=[2.4, 1.8, 0.9, 1.0],
                    metavar=("PLAN", "INFRA", "AGE", "READY"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.curve:
        # 唯一重点：BC→PPO 训练长度曲线。其余分支保持原样不动。
        run_curve(a)
        return

    SC.reset()
    SC.apply(horizon=a.horizon, horizon_eval="auto", opp_shape="window",
             a_plan=a.amps[0], a_infra=a.amps[1], a_age=a.amps[2],
             a_ready=a.amps[3], foresight=a.foresight, quota=a.quota,
             budget=a.budget)

    env0 = build_env(a.dataset, a.horizon, a.alpha, 7, a.budget, a.objective,
                     a.unit_only, a.static_field, a.prescreen,
                     a.foresight)
    EV, EVm = ev_tables(env0, a.gamma)
    elig = np.asarray(env0.env.eligible, bool)
    v_orc, oplan = oracle_plan(EV, a.quota, elig)
    print(f"oracle 折现总价值 {v_orc:.4g}（指派 {len(oplan)} 个单元，"
          f"名额 {a.horizon * a.quota} 个）")

    rows, plans = [], []
    for seed in a.seeds:
        # 随机与近视：与 PPO 用同一张场、同一套掩码、同一套计价
        e = build_env(a.dataset, a.horizon, a.alpha, 7, a.budget, a.objective,
                     a.unit_only, a.static_field, a.prescreen,
                     a.foresight)
        i_rnd, v_rnd = run_policy(e, EV, EVm, "random",
                                  rng=np.random.default_rng(seed), quota=a.quota)
        e = build_env(a.dataset, a.horizon, a.alpha, 7, a.budget, a.objective,
                     a.unit_only, a.static_field, a.prescreen,
                     a.foresight)
        i_myo, v_myo = run_policy(e, EV, EVm, "myopic", quota=a.quota)
        e_fc = build_env(a.dataset, a.horizon, a.alpha, 7, a.budget, a.objective,
                        a.unit_only, a.static_field, a.prescreen, a.foresight)
        i_fc, v_fc = run_policy(e_fc, EV, EVm, "forecast", quota=a.quota)

        e = build_env(a.dataset, a.horizon, a.alpha, 7, a.budget, a.objective,
                     a.unit_only, a.static_field, a.prescreen,
                     a.foresight)
        ck = os.path.join(a.out, f"net_seed{seed}.pt")
        if a.load_net and os.path.exists(ck):
            # 直接加载已训网络，跳过训练。事后分析（分解、等待指标、逐年优势）
            # 与训练无关，而 400 迭代一个种子要 25 分钟——存盘复用是必须的。
            d = torch.load(ck, weights_only=False)
            for k in ("objective", "unit_only", "static_field"):
                if bool(d.get(k)) != bool(getattr(a, k)) and k != "objective":
                    raise SystemExit(f"存盘网络的 {k}={d.get(k)} 与本次 "
                                     f"--{k.replace('_','-')} 不一致，拒绝混用")
            net = TP.Pointer().to(TP.DEV)
            net.load_state_dict(d["state"])
            hist = [float("nan")]
            print(f"  seed={seed} 加载已训网络（{d.get('iters')} 迭代 × "
                  f"{d.get('eps')} 回合）")
        else:
            init_sd, bcm = (None, {})
            if a.bc_init != "off":
                rk = EV if a.bc_init == "forecast" else EVm
                init_sd, bcm = bc_actor(
                    lambda: build_env(a.dataset, a.horizon, a.alpha, 7, a.budget,
                                      a.objective, a.unit_only, a.static_field,
                                      a.prescreen, a.foresight),
                    rk, None, seed=seed)
                print(f"  seed={seed} 监督初始化（教师={a.bc_init}）"
                      f"样本 {bcm['bc_n']} 秩相关 {bcm['bc_rho']:.3f}")
            if a.bc_only:
                # **只做监督初始化、不经 PPO**。这是论文表里一行独立的模型
                # （"监督时空打分器"），不是诊断：它回答"把前瞻信息用监督方式
                # 学进打分器，能到哪里"。与 BC→PPO 并列才能看出 PPO 的净效应。
                if init_sd is None:
                    raise SystemExit("--bc-only 需要同时给 --bc-init")
                net = TP.Pointer().to(TP.DEV)
                net.load_state_dict(init_sd, strict=False)
                hist = [float("nan")]
            else:
                net, hist = TP.train(e, iters=a.iters, eps_per_iter=a.eps,
                                     seed=seed, init_actor=init_sd)
        # 存盘训练好的网络：后续所有事后分析（分解、等待指标、逐年优势诊断）
        # 都不必重训 —— 400 迭代一个种子要 25 分钟，重训是最贵的浪费。
        if not (a.load_net and os.path.exists(ck)):
            torch.save(dict(state=net.state_dict(), iters=a.iters, eps=a.eps,
                            seed=seed, objective=a.objective,
                            unit_only=a.unit_only, static_field=a.static_field,
                            prescreen=a.prescreen),
                       ck)
        e2 = build_env(a.dataset, a.horizon, a.alpha, 7, a.budget, a.objective,
                     a.unit_only, a.static_field, a.prescreen,
                     a.foresight)
        i_ppo, v_ppo = run_policy(e2, EV, EVm, "ppo", net=net, quota=a.quota)

        gap = ((v_ppo - v_myo) / (v_orc - v_myo)) if v_orc > v_myo else np.nan
        for tag, iv, vv in (("随机", i_rnd, v_rnd), ("近视贪心", i_myo, v_myo),
                            ("时间感知贪心", i_fc, v_fc),
                            ("PPO", i_ppo, v_ppo)):
            plans.extend(dict(seed=seed, policy=tag, unit=u, year=y)
                         for u, y in iv.items())
            dec = decompose(EV, iv, a.quota, elig, v_orc)
            (pk_mae, pk_med, pk_hit), (or_mae, or_med, or_hit) = peak_distance(
                EV, iv, oplan)
            p_up, p_dn, n_up, n_dn = wait_when_beneficial(env0, EV, iv)
            rows.append(dict(seed=seed, policy=tag, value=vv, **dec,
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
        单元最优时序价值=("v_best_timing", "mean"),
        放错年份损失=("loss_timing", "mean"),
        挑错单元损失=("loss_selection", "mean"),
        时序损失占比=("share_timing", "mean"),
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
