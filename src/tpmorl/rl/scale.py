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
3. 参考集是手工策略，不是可达集的真实上界；它只保证同量级。
"""
import json
import os

import numpy as np
import pandas as pd

REF_MODES = ("big", "small", "rand", "none")   # 选取规则：买最大/买最小/随机/不动
REF_SEEDS = (0, 1, 2)


def _tag(budget, carry, growth):
    return f"B{float(budget):g}_C{float(carry):g}_G{float(growth):.4g}"


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


def build_scale(ds, budget, carry, growth, write=True):
    """构建分母并（可选）落盘。返回 (scale, 参考集 DataFrame)。"""
    R = reference_returns(ds, budget, carry, growth)
    sc = R.abs().max(axis=0).values.astype(float)
    sc[sc < 1e-9] = 1.0
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
    sc = R.abs().max(axis=0).values.astype(float)
    sc[sc < 1e-9] = 1.0
    return sc


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
