"""目标归一化分母（scale）的构建与缓存。

## 为什么需要这个模块

在此之前，分母取 `reward_v0/discounted_return.csv` 三个基线策略折扣回报的
逐目标绝对值上界。那张表是在**无预算约束**情景下算的，而训练是在
`BUDGET`/`CARRY_CAP`/`FAR_GROWTH` 约束下进行的，两者的可达集差别很大：

    目标      旧分母 / 约束情景可达上界
    Eco       12.19×      ← 生态收益被定价到应有的 8%
    Emp        4.64×
    Floor      4.37×
    Cpt        3.34×
    E2r        1.86×
    Gdp        1.68×
    Aec        1.50×
    Cost       1.39×
    Expire     1.23×
    Disrupt    1.00×      ← 恰好正确
    Res        0.52×      ← 反向：被放大了

失真跨度约 24 倍，且**方向不一致**——收益项普遍被压低、`Res` 反被放大。
后果是标量化奖励系统性低估「行动」的价值：在旧分母下策略把 `Floor`
换成其他目标（35 次运行 0 次超过随机基线的 `Floor` 均值，见
`docs/reward_defect_v1.md` §5.1），而一个按成本降序买最大单元的平凡启发式
就能拿到 `Floor`=3.36e6，比训练出的最好值（2.82e6）高 19%。

## 做法

分母改为在**实际约束情景**下重跑一组手工参考策略，取逐目标折扣回报的
绝对值上界。公式与旧版相同，改变的只是参考集所处的情景。

参考策略集刻意保持多样（4 种选取规则 × 3 个种子）。这一点是必要的：
先前试过只用**单一**填充策略做包络，结果 `Emp` 在该策略下恰好很小，
除下来被放大 13 倍，α=1 的回报反而更负。取多策略上界后 `Emp` 变为
缩小 4.64 倍，病态消失。

参考策略**不含** RL 策略本身——否则分母依赖被评估对象，构成循环。

## 三条使用约定

1. 分母依赖 (budget, carry, growth)。**跨情景的标量化回报（`curves.csv`、
   学习曲线、`最终回报`）不可比。** 需要跨情景对照时，用
   `objectives.csv` / `pareto_front.csv` 里的 11 维目标值——那些是
   `evaluate()` 乘回 scale 后的**原始量纲**，与分母无关，可比。
2. 换分母会使此前所有标量化回报作废。旧结果不可与新结果并列在同一张表里。
3. 参考集是手工策略，不是可达集的真实上界；它只保证同量级。自 v4 起加入了
   按目标定向的贪心（见 `REF_MODES` 注释），每个目标至少有一个策略在正方向
   上推它，因此分母不再出现「只度量破坏幅度」的单侧失真。
"""
import json
import os

import numpy as np
import pandas as pd

# 选取规则：买最大/买最小/随机/不动 + 五个按目标定向的贪心。
#
# 为什么必须有定向贪心（docs/生态目标诊断_v4.md）：只有前四个时，参考集里
# **没有任何策略让生态变好**（Eco 折扣回报 −19.3 / 0 / −226.5 / 0），而
# `_estimate` 取 `abs().max()`，于是 Eco 的分母 226.5 恰是「随机」策略**破坏**
# 生态的幅度——分母度量的是「能破坏多少」而非「能改善多少」。补一个生态定向
# 贪心即得 Eco=+1554，为旧分母的 6.86 倍，即生态轴被压缩约 7 倍，帕累托前沿
# 的生态端整体失真。`Cpt` 同病（四策略全为负或 0）。
#
# `big` 已近似 Floor 定向、`small` 已近似 Cost 定向，故不重复设。
# 定向贪心同时是论文的基线族：它们在 α 上互有胜负（生态定向与 big 在 α≈0.6
# 交叉），RL 要跑赢的是这一族而非单一的 big。
REF_MODES = ("big", "small", "rand", "none",
             "eco", "gdp", "res", "emp", "cpt")
REF_SEEDS = (0, 1, 2, 3, 4)

# 参考集版本，进缓存键。R3 = 仅四个手工策略；R4 = 加入五个定向贪心。
REF_VER = "R4"

# 定向贪心的排序键：UUM 的行号（5 x 12 = res emp gdp eco liv）。
_UUM_ROW = dict(res=0, emp=1, gdp=2, eco=3)
_DIRECTED = tuple(_UUM_ROW) + ("cpt",)
_GAIN_CACHE = {}


def gain_tables(ds):
    """逐 (单元, 目标功能) 的目标增益表，供定向贪心排序。返回 {mode: (n_unit, 12)}。

    只依赖静态的现状用地与权重矩阵，故一次算好、跨种子与年份复用（候选配对的
    枚举 `_pu_all/_pt_all` 本身也是静态的，每年只是布尔选择）。

    - UUM 四行（res/emp/gdp/eco）：增益 = 格数 × UUM[行, 目标] − Σ_格 UUM[行, 现状]，
      即该单元全部格子改成目标功能后该项效用的总变化。
    - `cpt` 用邻域相容度：把单元掩膜膨胀一圈取**单元外**的类别直方图 nb，
      增益 = Σ_k (CM[目标, k] − 现状加权 CM[·, k]) × nb_k。这是 Cpt 的代理量
      而非精确增量（真实 Cpt 走 500 m 池化后的二次型），够用于排序。
    """
    if ds in _GAIN_CACHE:
        return _GAIN_CACHE[ds]
    from scipy import ndimage
    from tpmorl.objectives.run_reward_demo import load

    _, cls, _, _, _, uid, U, UUM, CM, _ = load(ds)
    UUM, CM = np.asarray(UUM, float), np.asarray(CM, float)
    ids = U["uid"].values
    n = len(ids)
    hist = np.zeros((n, 12))            # 现状类别直方图（格数）
    nb = np.zeros((n, 12))              # 膨胀一圈后、单元外的邻域类别直方图
    boxes = ndimage.find_objects(uid.astype(np.int32))
    for r, u in enumerate(ids):
        sl = boxes[int(u) - 1]
        if sl is None:
            continue
        pad = tuple(slice(max(s.start - 2, 0), min(s.stop + 2, d))
                    for s, d in zip(sl, uid.shape))
        m = uid[pad] == u
        c = cls[pad]
        hist[r] = np.bincount(c[m].ravel(), minlength=13)[1:13]
        ring = ndimage.binary_dilation(m, iterations=2) & ~m
        nb[r] = np.bincount(c[ring].ravel(), minlength=13)[1:13]

    ncell = hist.sum(1)
    tabs = {}
    for mode, row in _UUM_ROW.items():
        w = UUM[row]                                  # (12,)
        tabs[mode] = ncell[:, None] * w[None, :] - (hist * w[None, :]).sum(1)[:, None]
    cur_cm = (hist[:, :, None] * CM[None, :, :]).sum(1) / np.maximum(ncell, 1)[:, None]
    tabs["cpt"] = (nb[:, None, :] * (CM[None, :, :] - cur_cm[:, None, :])).sum(-1)
    _GAIN_CACHE[ds] = tabs
    return tabs


def _tag(budget, carry, growth):
    # 制度参数一并进键：窗口一改，参考策略集的可达上界随之改变，不能复用。
    # 读的是**调用时**的 schedule 模块常量，故 scenario.apply() 必须先于本函数。
    # 参考集版本号也进键：改 REF_MODES 会改变分母，而情景参数不变，若不入键则
    # 旧缓存被静默复用、新旧分母混在同一批结果里。改 REF_MODES/_estimate 时必须
    # 同步递增 REF_VER。
    from tpmorl.rl.scenario import inst_tag
    return (f"B{float(budget):g}_C{float(carry):g}_G{float(growth):.4g}"
            f"_{inst_tag()}_{REF_VER}")


def scale_path(ds, budget, carry, growth):
    return os.path.join(ds, "scale_v2", f"ref_{_tag(budget, carry, growth)}.csv")


def _rollout(ds, mode, seed):
    """单个手工策略跑一回合，返回 11 维折扣回报（原始量纲）。

    目标向量与权重无关（权重只进标量化奖励），故参考集无需按 alpha 重算。
    """
    from tpmorl.rl.env_gym import RenewalEnv, QUOTA
    from tpmorl.rl.train_ppo import weight_vector

    env = RenewalEnv(ds, weights=weight_vector(0.5), seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    g = np.zeros(len(env.scale))
    for t in range(env.T):
        acts = []
        if mode != "none":
            _, meta, cost, _ = env.pairs()
            c = cost[:-1]                       # 末位是 STOP
            if mode == "big":
                idx = np.argsort(-c)
            elif mode == "small":
                idx = np.argsort(c)
            elif mode in _DIRECTED:
                pu = np.asarray([m[0] for m in meta[:-1]], dtype=np.int64)
                pt = np.asarray([m[1] for m in meta[:-1]], dtype=np.int64)
                idx = np.argsort(-gain_tables(ds)[mode][pu, pt])
            else:
                idx = rng.permutation(len(c))
            left, used = env.budget, set()
            for i in idx:
                u, tg = meta[i][0], meta[i][1]
                if u in used or c[i] > left + 1e-6:
                    continue
                acts.append((u, tg)); used.add(u); left -= c[i]
                if len(acts) >= QUOTA:
                    break
        _, _, done, info = env.step(acts)
        g += (env.gamma ** t) * info["vec"] * env.scale
        if done:
            break
    return g


def reference_returns(ds, budget, carry, growth, modes=REF_MODES, seeds=REF_SEEDS):
    """在给定约束情景下跑参考策略集，返回 DataFrame（行=策略×种子）。"""
    from tpmorl.rl import env_gym
    from tpmorl.objectives.reward import OBJ_NAMES

    env_gym.BUDGET = float(budget)
    env_gym.CARRY_CAP = float(carry)
    env_gym.FAR_GROWTH = float(growth)

    rows, index = [], []
    for mode in modes:
        for s in seeds:
            rows.append(_rollout(ds, mode, s))
            index.append(f"{mode}_s{s}")
    return pd.DataFrame(rows, index=index, columns=list(OBJ_NAMES))


def _estimate(R):
    """由参考集导出分母：先按策略对种子取均值，再跨策略取绝对值上界。

    为什么不直接对全部 rollout 取 max：max 把**策略差异**与**种子运气**混在
    一起，且对重尾目标不收敛。实测 `Eco` 的 max 随种子数单调爆涨
    （3/5/8 种子 → 366.8 / 627.7 / 1239.8，每加种子近乎翻倍），分母取值
    全凭参考集大小，不可复现。

    先对种子取均值再跨策略取上界，量的正是「策略选择能把该目标推动多少」，
    而这恰是加权和里分母应有的含义。实测 3→8 种子的漂移收敛到 0.52–1.17 倍，
    多数目标在 0.7–1.0。
    """
    m = R.groupby([i.rsplit("_s", 1)[0] for i in R.index], sort=False).mean()
    sc = m.abs().max(axis=0).values.astype(float)
    sc[sc < 1e-9] = 1.0
    return sc


def build_scale(ds, budget, carry, growth, write=True):
    """构建分母并（可选）落盘。返回 (scale, 参考集 DataFrame)。"""
    R = reference_returns(ds, budget, carry, growth)
    sc = _estimate(R)
    if write:
        p = scale_path(ds, budget, carry, growth)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        R.to_csv(p, encoding="utf-8-sig")
        json.dump(dict(zip(R.columns, sc.tolist())),
                  open(p.replace(".csv", ".json"), "w"), ensure_ascii=False, indent=1)
    return sc, R


def load_scale(ds, budget, carry, growth, rebuild=False):
    """读缓存的分母；缺失时构建并写入。参考集随情景变化，故按情景键缓存。"""
    from tpmorl.objectives.reward import OBJ_NAMES

    p = scale_path(ds, budget, carry, growth)
    if rebuild or not os.path.exists(p):
        sc, _ = build_scale(ds, budget, carry, growth)
        return sc
    R = pd.read_csv(p, index_col=0)[list(OBJ_NAMES)]
    return _estimate(R)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="构建约束情景下的目标归一化分母")
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--carry", type=float, default=3.0)
    ap.add_argument("--growth", type=float, default=0.0)
    a = ap.parse_args()

    sc, R = build_scale(a.dataset, a.budget, a.carry, a.growth)
    old = pd.read_csv(os.path.join(a.dataset, "reward_v0", "discounted_return.csv"),
                      index_col=0)[list(R.columns)]
    so = old.abs().max(axis=0).values.astype(float); so[so == 0] = 1.0
    print(f"情景 {_tag(a.budget, a.carry, a.growth)}  参考策略 {len(R)} 次\n")
    print(pd.DataFrame({"旧分母": so, "新分母(约束情景)": sc, "旧/新": so / sc},
                       index=R.columns).round(3).to_string())
    print(f"\n已写出 {scale_path(a.dataset, a.budget, a.carry, a.growth)}")
