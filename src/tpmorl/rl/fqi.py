# -*- coding: utf-8 -*-
"""fqi.py —— Unit-level Double Fitted Q Iteration（v20）。

## 与既有代码的关系

**不修改** `env_gym.py` / `schedule.py` / `opportunity.py` / `reward.py` /
`train_ppo.py` / oracle / 评价指标。本模块只做一件事：把同一个城市更新 MDP
的 learner 从 PPO 换成 Double-FQI，因此论文里可以干净地说

> 同一个 MDP、同一个 reward、同一个机会场、同一套评价指标，仅替换 RL learner。

## 形式化（第一版，刻意最小）

动作 = **单元**（目标功能固定，故动作数 = 单元数而非 (单元,目标) 配对数）。
每年把配额看成一次**批次决策**：

    Q(s_t, u) 对所有合规 u 打分  →  取前 K 个  →  env.step(top-K)

即 π(s_t) = TopK_u Q(s_t, u)。第一版**不含 STOP**（每年必须选满配额）——
上一轮已实测：717 单元 / 75 名额的容量区间里主动等待被严格支配，
且 STOP 的尺度校准本身就是一个独立的坑。合成世界留作第二阶段再开。

### 即时回报取环境自己的价值函数

`r(s_t, u)` = 第 t 年立项单元 u 的期望折现交付价值，**直接取环境自身的
价值函数**（合成世界 `TimingWorld.dvalue`，光明区 `eval/ev.ev_matrix`）。
不另写一套 FQI 专用 reward —— v19 已经把"价值读哪一年"收进单一函数
`value_year()`，环境实付的折现奖励流与 `dvalue` 由 `validate()` 机械自证相等。
若这里另立一套，就会重新出现"环境 reward ≠ oracle 计价 ≠ RL target"。

### 有限期回溯与 Double 目标

    y_A(s,u) = r(s,u) + γ · Q_B( s', argmax_{u'∈pool'} Q_A(s', u') )
    y_B(s,u) = r(s,u) + γ · Q_A( s', argmax_{u'∈pool'} Q_B(s', u') )

`pool'` = 下一年的合规候选集合**去掉 u**（因为选了 u 就不能再选）。
两个完全独立的回归器，动作选择与动作评估分离，用于抑制 max 带来的高估。
决策期有限，故"剩余年数"进特征（见下），迭代 N ≥ T 次即把价值沿时间传播开。

**一处必须声明的近似**：数据由行为策略按批次推进，故对某个候选 u
只能拿到行为策略实际到达的下一年状态，而不是"假如只选了 u"的反事实状态。
本实现用同一条轨迹的下一年候选集合（去掉 u）近似 `s'`。
这是 batch/offline FQI 的标准做法，但它是近似而不是精确回溯，
论文方法部分应当写明。

## 数据覆盖

只用 temporal-greedy 轨迹会让"错误动作"几乎没有样本，Q 在未覆盖的动作上
不可靠。故数据集必须混合 **随机 + 近视 + 时间感知贪心** 三种行为策略。
"""
import numpy as np


# ---------------------------------------------------------------- 数据集

def collect(env_fn, reward_fn, rank_fns, n_ep_each=20, quota=None, seed=0,
            extra_feat=True):
    """按多种行为策略收集 (s_t, 候选集, 每单元即时回报, s_{t+1} 候选集) 四元组。

    `env_fn()`      → 新的环境实例（已 reset）
    `reward_fn(u,t)` → 第 t 年立项 u 的即时回报（环境自己的价值函数）
    `rank_fns`      → {名字: 打分表或 None}。None = 随机行为策略；
                      打分表形如 R[u, t]，按它逐年贪心取满配额。
    返回 list[dict]，每项是一年：
        X      (m, F)  该年合规候选的特征（已含附加的全局量）
        units  (m,)    对应单元 id
        r      (m,)    每个候选的即时回报
        nX / nunits    下一年的合规候选与其 id（终局为空数组）
        t, done
    """
    rng = np.random.default_rng(seed)
    data = []
    for name, R in rank_fns.items():
        for ep in range(n_ep_each):
            env = env_fn()
            q = int(env.quota if quota is None else quota)
            years = []
            for t in range(env.T):
                X, meta, cost, units = env.pairs()
                # 只取真实候选行：末行是 STOP（第一版不用），负 id 一律剔除。
                # 这一步同时继承了环境自己的合法动作掩码 —— 不合规、
                # 不在有效期、预算付不起的配对本来就不会出现在 pairs() 里。
                keep = [i for i, m in enumerate(meta) if m[0] >= 0]
                if not keep:
                    if env.step([])[2]:
                        break
                    continue
                us = np.array([meta[i][0] for i in keep], int)
                Xt = np.asarray(X, np.float32)[keep]
                if extra_feat:
                    Xt = _append_global(Xt, t, env, q)
                rt = np.array([reward_fn(int(u), t) for u in us], float)
                if R is None:
                    order = rng.permutation(len(keep))
                else:
                    order = np.argsort(-R[us, min(t, R.shape[1] - 1)])
                act, used = [], set()
                for j in order:
                    if len(act) >= q:
                        break
                    u = int(us[j])
                    if u in used:
                        continue
                    act.append(meta[keep[j]]); used.add(u)
                years.append(dict(X=Xt, units=us, r=rt, t=t, policy=name, ep=ep))
                if env.step(act)[2]:
                    break
            # 串起相邻两年：s' 用同一条轨迹的下一年候选集合
            for i, y in enumerate(years):
                nxt = years[i + 1] if i + 1 < len(years) else None
                y["nX"] = nxt["X"] if nxt is not None else np.zeros((0, y["X"].shape[1]), np.float32)
                y["nunits"] = nxt["units"] if nxt is not None else np.zeros(0, int)
                y["done"] = nxt is None
            data.extend(years)
    return data


def _append_global(X, t, env, quota):
    """补三个全局量：剩余年数比例、配额、预算比例。

    第 31 节要求"当前时间必须进 Q 的输入"，否则模型无法区分第 2 年与第 20 年
    的等待价值。`pairs()` 的候选特征里已有大量时序信息，但剩余年数是全局量、
    不随候选变化，故显式补上。树模型对重复/冗余特征不敏感，宁重复不遗漏。
    """
    T = float(env.T)
    rem = (T - t) / T
    bud = float(getattr(env, "budget", 0.0))
    g = np.tile(np.array([[rem, float(quota) / max(T, 1.0),
                           np.log1p(max(bud, 0.0)) / 30.0]], np.float32),
                (X.shape[0], 1))
    return np.hstack([X, g])


# ---------------------------------------------------------------- 训练

def fit_double(data, gamma, make_model, n_iter=None, verbose=True,
               split=True):
    """有限期 Double-FQI。返回 (Q_A, Q_B, 训练日志)。

    n_iter 默认取决策期长度：价值每迭代一次沿时间前传一年，故 N ≈ T 足够。

    ## split：两个估计器必须拟合**互不相交**的转移

    第一版不分数据时实测两个估计器的预测**最大绝对差恰好为 0**，两套目标
    也完全相同 —— 因为 ExtraTrees 在训练点上几乎插值，两个森林都精确复现
    目标，于是 argmax 永远一致，Double-Q 的高估修正从第 0 轮起就自我锁死
    （加上第 0 轮 `ya = yb` 还共享同一个数组）。

    按转移（而不是按行）二分：同一年的候选行必须整组落在同一侧，
    否则同年样本会跨侧泄漏，"独立估计器"就名不副实。
    每个估计器只在自己那一半上拟合，但目标对全体转移构造 ——
    这样 A 评估 B 的动作时，那个状态对 A 而言是**样本外**的，
    高估修正才真的起作用。
    """
    T_max = max(d["t"] for d in data) + 1
    n_iter = int(T_max if n_iter is None else n_iter)
    half = np.array([i % 2 == 0 for i in range(len(data))])
    Xa = np.vstack([d["X"] for d, m in zip(data, half) if m])
    Xb = np.vstack([d["X"] for d, m in zip(data, half) if not m])
    ra = np.concatenate([d["r"] for d, m in zip(data, half) if m])
    rb = np.concatenate([d["r"] for d, m in zip(data, half) if not m])
    if not split:
        Xa = Xb = np.vstack([d["X"] for d in data])
        ra = rb = np.concatenate([d["r"] for d in data])

    qa = qb = None
    log = []
    for k in range(n_iter):
        if qa is None:
            ya, yb = ra.copy(), rb.copy()   # 终局边界条件：Q = 即时回报
        else:
            ta, tb = _targets(data, gamma, qa, qb)
            if split:
                # 按转移切回两半（_targets 返回的是全体转移拼接后的向量）
                sizes = [len(d["r"]) for d in data]
                idx = np.concatenate([np.full(n, m) for n, m in
                                      zip(sizes, half)]).astype(bool)
                ya, yb = ta[idx], tb[~idx]
            else:
                ya, yb = ta, tb
        qa, qb = make_model(), make_model()
        qa.fit(Xa, ya); qb.fit(Xb, yb)
        Xall = np.vstack([d["X"] for d in data])
        pa, pb = qa.predict(Xall), qb.predict(Xall)
        log.append(dict(iter=k + 1, y_mean=float(ya.mean()),
                        y_max=float(ya.max()),
                        ab_corr=float(np.corrcoef(pa, pb)[0, 1]),
                        ab_maxdiff=float(np.abs(pa - pb).max()),
                        ab_sd=float(np.abs(pa - pb).max() / (pa.std() + 1e-12))))
        if verbose:
            print(f"    FQI 迭代 {k + 1:>2}/{n_iter}  目标均值 {ya.mean():>9.2f}  "
                  f"目标最大 {ya.max():>9.2f}  A/B 相关 {log[-1]['ab_corr']:.4f}  "
                  f"A/B 最大差 {log[-1]['ab_maxdiff']:.3g}"
                  f"（{log[-1]['ab_sd']:.2f} 个标准差）")
    return qa, qb, log


def _targets(data, gamma, qa, qb):
    """逐年构造 Double 目标。动作选择用一个估计器、动作评估用另一个。"""
    ya, yb = [], []
    for d in data:
        r = d["r"]
        if d["done"] or len(d["nunits"]) == 0:
            ya.append(r.copy()); yb.append(r.copy())
            continue
        nX, nu = d["nX"], d["nunits"]
        qan, qbn = qa.predict(nX), qb.predict(nX)
        # pool' = 下一年候选去掉 u。对每个 u 逐一剔除，向量化做法是
        # 先取全局前两名，若第一名正是 u 就退到第二名。
        oa = np.argsort(-qan)[:2]
        ob = np.argsort(-qbn)[:2]
        ta, tb = [], []
        for u in d["units"]:
            ia = oa[0] if nu[oa[0]] != u or len(oa) == 1 else oa[1]
            ib = ob[0] if nu[ob[0]] != u or len(ob) == 1 else ob[1]
            ta.append(qbn[ia])          # A 选动作、B 评估
            tb.append(qan[ib])          # B 选动作、A 评估
        ya.append(r + gamma * np.asarray(ta))
        yb.append(r + gamma * np.asarray(tb))
    return np.concatenate(ya), np.concatenate(yb)


# ---------------------------------------------------------------- 策略

def rollout(env_fn, qa, qb, reward_fn, quota=None, extra_feat=True,
            collect_q=False):
    """π(s) = TopK_u ½(Q_A + Q_B)。返回 ({单元: 立项年}, 环境折现奖励, Q 诊断)。"""
    env = env_fn()
    q = int(env.quota if quota is None else quota)
    init, R, diag = {}, 0.0, []
    for t in range(env.T):
        X, meta, cost, units = env.pairs()
        keep = [i for i, m in enumerate(meta) if m[0] >= 0]
        if not keep:
            _, rr, done, _ = env.step([])
            R += (env.gamma ** t) * float(np.sum(rr))
            if done:
                break
            continue
        us = np.array([meta[i][0] for i in keep], int)
        Xt = np.asarray(X, np.float32)[keep]
        if extra_feat:
            Xt = _append_global(Xt, t, env, q)
        qv = 0.5 * (qa.predict(Xt) + qb.predict(Xt))
        order = np.argsort(-qv)
        act, used, chosen = [], set(), []
        for j in order:
            if len(act) >= q:
                break
            u = int(us[j])
            if u in used:
                continue
            act.append(meta[keep[j]]); used.add(u); chosen.append(j)
            init.setdefault(u, t)
        if collect_q:
            m = np.ones(len(qv), bool); m[chosen] = False
            diag.append(dict(t=t, q_selected=float(qv[chosen].mean()),
                             q_not_selected=float(qv[m].mean()) if m.any() else np.nan,
                             q_gap=float(qv[chosen].mean()
                                         - (qv[m].mean() if m.any() else np.nan)),
                             n_cand=len(qv)))
        _, rr, done, _ = env.step(act)
        R += (env.gamma ** t) * float(np.sum(rr))
        if done:
            break
    return init, R, diag


def q_vs_value_corr(data, qa, qb, reward_fn):
    """Q 与"即时价值"的秩相关。

    这一项回答一个必须自己先问的问题：Q 到底有没有学到**超出当期价值**的东西。
    若秩相关 ≈ 1，则 Q 只是把即时回报复述了一遍，策略必然退化成时间感知贪心；
    明显小于 1 且成绩更好，才说明它学到了跨年协调。
    """
    from scipy.stats import spearmanr
    X = np.vstack([d["X"] for d in data])
    r = np.concatenate([d["r"] for d in data])
    qv = 0.5 * (qa.predict(X) + qb.predict(X))
    return float(spearmanr(qv, r).statistic)
