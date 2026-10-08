# -*- coding: utf-8 -*-
"""verify_opportunity_neutral.py —— 回归校验：机会场关闭时动态与 v16 逐位等价。

v17 往环境里加了四张逐年变化的机会场（tpmorl/env/opportunity.py）。四个幅度
默认全 0，此时价值乘子恒为精确的 1.0、批准率调制整条跳过，因此**转移、事件
计数、Floor、Cost、逐年奖励都应当与 v16 逐位相同**。

这一条必须能被机械验证，而不是靠读代码相信。用法：

    # 1) 取出 v16 的源码
    git archive batch-v16-complete src | (mkdir -p /tmp/v16 && tar -x -C /tmp/v16)
    # 2) 两边各跑一次，比较摘要
    PYTHONPATH=/tmp/v16/src python scripts/verify_opportunity_neutral.py --tag v16
    PYTHONPATH=src           python scripts/verify_opportunity_neutral.py --tag v17
    # 3) 逐位比较
    python scripts/verify_opportunity_neutral.py --compare /tmp/neutral_v16.json /tmp/neutral_v17.json

动作序列由固定种子的 RNG 在**候选配对表**上选取，两个版本的配对枚举代码相同，
故动作序列相同；于是任何差异都只能来自环境动态本身。

**为什么不直接比较训练结果**：观测新增 4 维改变了网络第一层的参数量，torch 的
初始化抽样序列随之改变，训练轨迹必然不同。可逐位比较的是**环境**，不是策略；
把不可能相等的东西写进校验只会让校验被忽略。
"""
import argparse
import hashlib
import json

import numpy as np


def rollout(ds, T=15, T_eval=26, seed=7, act_seed=0):
    from tpmorl.rl.env_gym import RenewalEnv
    from tpmorl.rl.train_ppo import weight_vector
    from tpmorl.objectives.reward import OBJ_NAMES

    env = RenewalEnv(ds, T=T, T_eval=T_eval, weights=weight_vector(0.5),
                     scale=np.ones(len(OBJ_NAMES)))
    env.reset(seed=seed)
    rng = np.random.default_rng(act_seed)
    out = []
    for t in range(T_eval):
        act = []
        if t < T:
            X, meta, cost, units = env.pairs()
            left, used = env.budget, set()
            # 固定顺序 + 固定种子的置换：两个版本的 meta 同序，故动作序列相同
            for i in rng.permutation(len(meta)):
                if len(act) >= env.quota:
                    break
                u = meta[i][0]
                if u < 0 or u in used or cost[i] > left + 1e-6:
                    continue
                act.append(meta[i])
                used.add(u)
                left -= float(cost[i])
        _, r, done, info = env.step(act)
        out.append(dict(t=t, r=float(r),
                        ev={k: int(v) for k, v in info["events"].items()},
                        floor=float(info["raw"]["Floor"]),
                        cost=float(info["raw"]["Cost"]),
                        disrupt=float(info["raw"]["Disrupt"]),
                        sigma=[int(x) for x in env.env.sigma]))
        if done:
            break
    return out


def digest(rows):
    """把逐年记录压成一个十六进制摘要 + 几个可人读的合计。

    摘要用 repr 而非浮点格式化：格式化会把 1e-16 的差异磨平，而这里要的正是
    逐位相等。
    """
    h = hashlib.sha256(repr(rows).encode()).hexdigest()[:16]
    return dict(sha=h,
                n_years=len(rows),
                r_sum=sum(x["r"] for x in rows),
                floor_sum=sum(x["floor"] for x in rows),
                cost_sum=sum(x["cost"] for x in rows),
                completed=sum(x["ev"]["completed"] for x in rows),
                expired=sum(x["ev"]["expired"] for x in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--tag", default=None, help="落盘为 /tmp/neutral_{tag}.json")
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()

    if a.compare:
        A, B = (json.load(open(p)) for p in a.compare)
        same = A["digest"] == B["digest"]
        print(json.dumps(dict(A=A["digest"], B=B["digest"]), ensure_ascii=False,
                         indent=1))
        print("\n逐位一致" if same else "\n**不一致** —— 机会场关闭时改变了环境动态，须修")
        raise SystemExit(0 if same else 1)

    rows = rollout(a.dataset)
    d = digest(rows)
    print(json.dumps(d, ensure_ascii=False, indent=1))
    if a.tag:
        p = f"/tmp/neutral_{a.tag}.json"
        json.dump(dict(digest=d, rows=rows), open(p, "w"))
        print(f"\n已写 {p}")


if __name__ == "__main__":
    main()
