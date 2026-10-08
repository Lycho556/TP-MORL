# -*- coding: utf-8 -*-
"""baselines.py — 「何时立项」问题的**评价用基线**与上界。

本脚本回答的是：把\"提前量\"（能不能在决策期内建成）这一个因素单独拿掉或加上，
完工数与 11 维目标各差多少；以及在\"审批不确定性被消除\"的假想世界里最多能建成
多少。它**不是**训练的一部分，也**不参与目标归一化分母**。

## 与 scale.REF_MODES 的关系：刻意保持独立，不得并入
`src/tpmorl/rl/scale.py` 里的 9 个手工参考策略是目标归一化**分母**的来源。
往那个集合里加策略会改变分母，从而使此前所有标量化回报作废
（见 scale.py 模块文档第 2 条使用约定）。本文件的六个策略是**评价口径**的对照组，
因此独立实现、独立落盘，与 REF_MODES 无任何交集。

## 六个策略
    myopic       每年在候选集里按「归一化多目标即时收益 / 资金成本」降序贪心选，
                 **完全不判断能否在决策期内建成**。这是现实中最常见的做法，
                 也是本文要击败的对象。
    greedy_crh   排序键、配额、资金约束与 myopic 逐位相同，唯一差别是候选集先过
                 `slack >= 0` 的筛子（slack = 决策期剩余年数 − 期望交付年数，
                 用 schedule.expected_years_to_delivery(if_initiated_now=True) 算）。
                 与 myopic 成对出现，正是为了把\"提前量\"这一个因素单独隔离出来。
    frontload    排序键与 myopic 相同，只改**年度立项上限的时间分布**：把同一个
    evenspread   名义立项总量（QUOTA × PACE_YEARS 个）分别集中在前 PACE_YEARS 年 /
    backload     均匀摊到全部 T 年 / 集中在后 PACE_YEARS 年。三者用来说明\"节奏\"
                 本身能带来多少差异，故必须共用同一个名义总量，只有时间分布不同。
    oracle       上界。见下方\"上界口径\"。

## 上界口径（必读，论文里不得写成\"最优解\"）
`oracle` = **无审批不确定性 + 贪心**，即 hazard 全部置 1.0（立项次年必获批、
永不失效）下跑 `greedy_crh`。它是一个**松上界**：
  - 松在\"贪心\"：真正的组合最优要在 717 单元 × 15 年 × 每年至多 3 个立项的
    动作空间上做整数规划，不可解，本脚本不声称求到了它；
  - 紧的那一半只在\"审批不确定性\"这一个维度上——它给出的是\"如果审批完全可预期，
    同一条贪心规则能多建成多少\"，因此 myopic 与 oracle 的差距读作
    **审批不确定性 + 无提前量判断** 两者合计的代价，不是\"离最优有多远\"。

## 两个完工口径（必须分开报，见项目规矩第 6 条）
环境的决策期 T 与评价期 T_eval 是两个口径：T_eval > T 时尾部年份只推进状态机、
不立项也不进钱。于是有两个完工数：
    n_done_T     第 0..T-1 年内建成的单元数 —— **「期内完工」**。这才是与 slack
                 特征同口径的量：slack>=0 的定义就是\"期望在 T-1 年前交付\"。
    n_done_all   第 0..T_eval-1 年内建成的单元数 —— 管道排空后的总量。
自查三条 (a)(b)(c) 一律以 **n_done_T** 判定。若用 n_done_all，晚立项的单元在尾部
照样能建成，slack 与节奏的差异会被尾部抹平，(a)(c) 两条按构造就不该成立——
这不是策略的性质，是口径选错。

## 排序键的定义与其局限
即时收益取一张**静态**的 (单元 × 目标功能) 归一化收益表：

    benefit(u, tg) = Σ_k (w_k / d_k) · gain_k(u, tg),  k ∈ {Res, Emp, Gdp, Eco, Cpt, Floor}
    ratio(u, tg)   = benefit(u, tg) / pair_cost(u, tg)

  - `gain_k` (k ≠ Floor) 复用 `scale.gain_tables()`：那是 UUM 四行的线性增益与
    Cpt 的邻域相容度代理，**是排序用的代理量而非精确增量**（真实 Gdp/Eco 还乘
    边际效用递减 η(p)，真实 Cpt 走 500 m 池化后的二次型）。
  - `gain_Floor` = `Reward.floor_area(ch_u, ncell_u)`，只依赖通道与格数。
  - 权重 w 取 `train_ppo.weight_vector(alpha)`（默认 alpha=0.5），分母 d 取
    **默认情景**的 `load_scale`，且**所有策略共用这一套 d**（含 oracle 的
    hazard=1.0 情景）。理由：这样六个策略的贪心规则逐位相同，差异只来自筛子与
    节奏；而本脚本落盘的 11 维目标是**原始量纲**（`info["vec"] * env.scale`
    把分母乘了回去），故 d 的取值不影响任何落盘数字，只影响排序。
  - Aec / E2r / Disrupt / Expire 四个目标没有可用的静态逐对代理（前两个依赖
    全局池化场、后两个是时序量），故**不进排序键**。它们仍照常在结果表里被度量。

## 资金与配额约束
本脚本不自行检验约束，依赖环境侧既有的断言：
  - `env_gym.RenewalEnv.step` 的 `assert spent <= self.budget + 1e-6`（资金）；
  - `schedule.RenewalSchedule.step` 的 `ValueError`（超年度配额、违反立项掩码）；
  - staged 口径下另有承诺额台账一致性断言。
跑通即说明这些约束在全部 6 策略 × 5 种子 × T_eval 年内从未被违反。

用法:
    PYTHONPATH=src python scripts/baselines.py \
        --dataset data/processed/gm_dataset_v1 \
        --out data/processed/gm_dataset_v1/baselines_v15 \
        --horizon 15 --horizon-eval auto --budget 900 --carry 3
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from tpmorl.env.schedule import S1, S2, S3
from tpmorl.objectives.reward import OBJ_NAMES

# 排序键里有静态逐对代理的目标；其余四个目标见模块文档\"排序键的定义与其局限\"
RATIO_OBJS = ("Res", "Emp", "Gdp", "Eco", "Cpt", "Floor")
POLICIES = ("myopic", "greedy_crh", "frontload", "evenspread", "backload", "oracle")
BASE_SEEDS = (0, 1, 2, 3, 4)
PACE_YEARS = 5          # 节奏基线的集中年数；名义立项总量 = QUOTA × PACE_YEARS


# ------------------------------------------------------------------ 排序键
def pair_ratio(ds, alpha=0.5, scale=None):
    """静态的 (单元 × 12 目标功能) 收益/成本比矩阵。

    只依赖现状用地、通道与权重矩阵，与年份、种子、策略均无关，故整批只算一次。
    返回 (ratio, benefit, cost)，三者形状均为 (n_unit, 12)。
    """
    from tpmorl.objectives.reward import CELL_COST, FAR_CAP, CELL_AREA
    from tpmorl.objectives.run_reward_demo import load
    from tpmorl.rl.scale import gain_tables
    from tpmorl.rl.train_ppo import weight_vector

    LU0, cls, _, _, _, uid, U, _, _, CCM = load(ds)
    w = dict(zip(OBJ_NAMES, weight_vector(alpha)))
    d = dict(zip(OBJ_NAMES, np.ones(len(OBJ_NAMES)) if scale is None else scale))
    tabs = gain_tables(ds)

    ch = U["ch_code"].values.astype(int)
    n = len(ch)
    # 现状类别直方图与格数：与 env_gym.reset 里的 hist0/ncell 同口径（按 uid 数格）
    hist = np.zeros((n, 12))
    for r, u in enumerate(U["uid"].values):
        m = uid == u
        hist[r] = np.bincount(cls[m].ravel(), minlength=13)[1:13]
    ncell = hist.sum(1)

    benefit = np.zeros((n, 12))
    for k in RATIO_OBJS:
        if k == "Floor":
            far = np.array([FAR_CAP.get(int(c), FAR_CAP[5]) for c in ch])
            g = (far * ncell * CELL_AREA)[:, None] * np.ones((1, 12))
        else:
            g = tabs[k.lower()]
        benefit += (w[k] / max(d[k], 1e-9)) * g

    # 与 env_gym.reset 的 self.PC 同式：CELL_COST × 格数 + Σ CCM[现状, 目标] × 格数
    cost = CELL_COST * ncell[:, None] * np.ones((1, 12))
    for u in range(n):
        for f in range(12):
            if hist[u, f]:
                cost[u] += np.asarray(CCM, float)[f, :12] * hist[u, f]
    return benefit / np.maximum(cost, 1e-9), benefit, cost


# ------------------------------------------------------------------ 年度立项上限
def year_cap(policy, t, T, quota, pace_years=PACE_YEARS):
    """三个节奏基线的年度立项上限；其余策略恒为 quota。

    名义总量一律 `quota × pace_years`，只有时间分布不同——否则比较的就不是\"节奏\"
    而是\"总量\"。evenspread 用累计四舍五入摊，保证逐年上限之和恰等于名义总量。
    """
    if policy == "frontload":
        return quota if t < pace_years else 0
    if policy == "backload":
        return quota if t >= T - pace_years else 0
    if policy == "evenspread":
        tot = quota * pace_years
        return int(round(tot * (t + 1) / T) - round(tot * t / T))
    return quota


# ------------------------------------------------------------------ 单回合
def rollout(ds, policy, seed, T, T_eval, ratio, scale, quota,
            pace_years=PACE_YEARS):
    """跑一个策略一个种子，返回 11 维折扣回报（原始量纲）+ 计数。

    折现口径与 `train_ppo.evaluate` / `scale._rollout` 完全一致：
    `Σ_t gamma**t · info["vec"] · env.scale`，t 按真实年份走到 T_eval。
    """
    from tpmorl.rl.env_gym import RenewalEnv

    env = RenewalEnv(ds, T=T, T_eval=T_eval, scale=scale, seed=seed)
    env.reset(seed=seed)
    # oracle 走 greedy_crh 的规则，差别只在情景（hazard=1.0，由调用方 apply）
    use_slack = policy in ("greedy_crh", "oracle")

    g = np.zeros(len(OBJ_NAMES))
    n_init = n_done_T = n_done_all = n_expire = 0
    spend = 0.0
    init_by_year, done_by_year = [], []
    for t in range(env.T_eval):
        acts = []
        cap = quota if t >= env.T else min(
            quota, year_cap(policy, t, env.T, quota, pace_years))
        if t < env.T and cap > 0:
            _, meta, cost, _ = env.pairs()
            pu = np.asarray([m[0] for m in meta[:-1]], dtype=np.int64)
            pt = np.asarray([m[1] for m in meta[:-1]], dtype=np.int64)
            c = cost[:-1]                       # 末位是 STOP
            keep = np.ones(pu.size, bool)
            if use_slack:
                # 期望交付年数 → slack = 决策期剩余年数 − 期望交付年数
                ey = env.env.expected_years_to_delivery(if_initiated_now=True)
                keep = ((env.T - 1 - t) - ey[pu]) >= 0
            if keep.any():
                pu, pt, c = pu[keep], pt[keep], c[keep]
                # 并列成本/并列收益极多（见 scale._rollout 的说明），一律用
                # lexsort 加 (单元, 目标) 确定性次键，保证跨机可复现。
                idx = np.lexsort((pt, pu, -ratio[pu, pt]))
                left, used = env.budget, set()
                for i in idx:
                    u, tg = int(pu[i]), int(pt[i])
                    if u in used or c[i] > left + 1e-6:
                        continue
                    acts.append((u, tg)); used.add(u); left -= float(c[i])
                    if len(acts) >= cap:
                        break
        spend += sum(env.pair_cost(u, tg) for u, tg in acts)
        _, _, done, info = env.step(acts)
        g += (env.gamma ** t) * info["vec"] * env.scale
        ev = info["events"]
        n_init += ev["initiated"]; n_expire += ev["expired"]
        n_done_all += ev["completed"]
        if t < env.T:
            n_done_T += ev["completed"]
        init_by_year.append(int(ev["initiated"]))
        done_by_year.append(int(ev["completed"]))
        if done:
            break

    # T_eval 取够长时管道必须排空；否则晚立项被截断，两个完工口径都失真。
    # 与 train_ppo.run_episode 的同名断言同一判据。
    if env.T_eval > env.T:
        open_ = int(np.isin(env.env.sigma, (S1, S2, S3)).sum())
        assert open_ == 0, (
            f"{policy} s{seed}：评价期 T_eval={env.T_eval} 结束时仍有 {open_} 个"
            f"单元在管道中（S1/S2/S3），晚立项被截断")

    row = dict(policy=policy, seed=seed)
    row.update({k: float(g[i]) for i, k in enumerate(OBJ_NAMES)})
    row.update(n_init=int(n_init), n_done_T=int(n_done_T),
               n_done_all=int(n_done_all), n_expire=int(n_expire),
               spend=float(spend))
    # 期内交付率：每个立项里有多少在决策期内建成。绝对完工数把\"立了多少项\"与
    # \"立的项中有多少赶上\"混在一起，这一列把后者单独拿出来——slack 筛子直接作用
    # 于它，而绝对数还受\"筛子关掉了多少个决策年\"的影响。
    row["crh_T"] = (n_done_T / n_init) if n_init else float("nan")
    row["crh_all"] = (n_done_all / n_init) if n_init else float("nan")
    return row, init_by_year, done_by_year


# ------------------------------------------------------------------ 主流程
def run_all(ds, out, T, horizon_eval, budget, carry, growth, alpha,
            seeds=BASE_SEEDS, pace_years=PACE_YEARS, policies=POLICIES,
            **inst):
    from tpmorl.env import schedule as S
    from tpmorl.rl import env_gym, scenario
    from tpmorl.rl.scale import load_scale

    os.makedirs(out, exist_ok=True)

    def setup(oracle=False):
        """按情景改写模块常量。**每次换情景前必须先 reset()**，否则参数静默继承
        （scenario.reset 的文档记录过这类缺陷）。oracle 额外把 hazard 置 1.0。"""
        scenario.reset()
        kw = dict(inst)
        if oracle:
            # tau_approval=1.0 → 常数风险率 min(1/1, 1)=1.0，即立项次年必获批。
            # 这是\"审批不确定性被消除\"的实现方式，不改动 HAZARD 的出厂值本身。
            kw["tau_approval"] = 1.0
        scenario.apply(budget=budget, carry=carry, growth=growth,
                       horizon=T, horizon_eval=horizon_eval, **kw)
        return scenario.horizon_eval()

    # 分母：**默认情景**（非 oracle）的分母，全批共用。见模块文档\"排序键\"。
    T_eval = setup(oracle=False)
    scale = load_scale(ds, env_gym.BUDGET, env_gym.CARRY_CAP, env_gym.FAR_GROWTH)
    base_desc = scenario.describe()
    ratio, benefit, cost = pair_ratio(ds, alpha=alpha, scale=scale)
    print(base_desc + f"\n决策期 T={T}  评价期 T_eval={T_eval}  "
          f"配额 {S.QUOTA}  排序键 alpha={alpha:g}\n")

    rows, pace = [], []
    for p in policies:
        oracle = (p == "oracle")
        te = setup(oracle=oracle)
        q = S.QUOTA
        for s in seeds:
            r, iby, dby = rollout(ds, p, s, T, te, ratio, scale, q, pace_years)
            rows.append(r)
            for t, (a, b) in enumerate(zip(iby, dby)):
                pace.append(dict(policy=p, seed=s, year=t, initiated=a,
                                 completed=b))
        sub = pd.DataFrame([x for x in rows if x["policy"] == p])
        print(f"{p:<11} 立项 {sub.n_init.mean():5.1f}  期内完工 "
              f"{sub.n_done_T.mean():5.1f}  全期完工 {sub.n_done_all.mean():5.1f}  "
              f"失效 {sub.n_expire.mean():5.1f}  用款 {sub.spend.mean():7.0f}  "
              f"期内交付率 {sub.crh_T.mean():5.1%}")
    setup(oracle=False)     # 还原情景，避免向调用方泄漏 hazard=1.0

    R = pd.DataFrame(rows)
    p_csv = os.path.join(out, "baselines.csv")
    R.to_csv(p_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(pace).to_csv(os.path.join(out, "baselines_pace.csv"),
                              index=False, encoding="utf-8-sig")

    # 逐策略均值±标准差（5 种子）
    num = [c for c in R.columns if c not in ("policy", "seed")]
    M = R.groupby("policy", sort=False)[num].agg(["mean", "std"])
    M.to_csv(os.path.join(out, "baselines_summary.csv"), encoding="utf-8-sig")

    checks = self_checks(R)
    json.dump(dict(scenario=base_desc, T=T, T_eval=T_eval, quota=int(S.QUOTA),
                   budget=env_gym.BUDGET, carry_cap=env_gym.CARRY_CAP,
                   far_growth=env_gym.FAR_GROWTH, alpha=alpha,
                   pace_years=pace_years, seeds=list(seeds),
                   policies=list(policies), ratio_objs=list(RATIO_OBJS),
                   scale=dict(zip(OBJ_NAMES, np.asarray(scale).tolist())),
                   oracle_kind="无审批不确定性(hazard=1.0) + 贪心；松上界，非组合最优",
                   checks=checks),
              open(os.path.join(out, "baselines_config.json"), "w"),
              ensure_ascii=False, indent=1)

    print("\n自查（一律以**期内完工** n_done_T 判定）：")
    for k, v in checks.items():
        print(f"  {'通过' if v['pass'] else '不成立'}  {k}：{v['detail']}")
    print(f"\n已写出 {p_csv}")
    return R, M, checks


def self_checks(R):
    """三条必须成立的定性关系，一律以期内完工 n_done_T 的种子均值判定。"""
    m = R.groupby("policy")["n_done_T"].mean()
    out = {}

    def rec(key, ok, detail):
        # "pass" 是关键字，不能作 dict(...) 的关键字参数，故用字面量字典
        out[key] = {"pass": bool(ok), "detail": detail}

    others = m.drop(labels=["oracle"], errors="ignore")
    if len(others) and "oracle" in m.index:
        rec("(a) oracle 期内完工 >= 其余所有策略",
            float(m["oracle"]) >= float(others.max()) - 1e-9,
            f"oracle={m['oracle']:.1f}，其余最大={others.max():.1f}"
            f"（{others.idxmax()}）")
    # 两条都写成「hi >= lo」的统一形式：(c) 的 backload <= frontload 即 frontload >= backload
    for key, lo, hi in (("(b) greedy_crh 期内完工 >= myopic", "myopic", "greedy_crh"),
                        ("(c) backload 期内完工 <= frontload", "backload", "frontload")):
        if lo in m.index and hi in m.index:
            rec(key, float(m[hi]) >= float(m[lo]) - 1e-9,
                f"{hi}={m[hi]:.1f}，{lo}={m[lo]:.1f}")

    if "oracle" in m.index and "myopic" in m.index:
        o, my = float(m["oracle"]), float(m["myopic"])
        rec("myopic 相对 oracle 的期内完工缺口", True,
            (f"{o - my:.1f} 个（myopic {my:.1f} vs oracle {o:.1f}，"
             f"相当于 oracle 的 {100 * (o - my) / o:.0f}%）")
            if o else "oracle 期内完工为 0，缺口无法以比例表述")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="何时立项：评价用基线与（松）上界")
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--out", default="data/processed/gm_dataset_v1/baselines_v15")
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--carry", type=float, default=3.0)
    ap.add_argument("--growth", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="排序键用的权重档（train_ppo.weight_vector）。只影响贪心"
                         "排序，不影响落盘的原始量纲目标值")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(BASE_SEEDS))
    ap.add_argument("--pace-years", type=int, default=PACE_YEARS,
                    help="节奏基线的集中年数，名义立项总量 = 配额 × 本值")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    from tpmorl.rl import scenario
    scenario.add_args(ap)
    a = ap.parse_args()
    run_all(a.dataset, a.out, a.horizon, a.horizon_eval, a.budget, a.carry,
            a.growth, a.alpha, seeds=tuple(a.seeds), pace_years=a.pace_years,
            policies=tuple(a.policies), **scenario.from_args(a))
