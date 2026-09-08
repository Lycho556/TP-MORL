"""把某个批次的学习策略回报整理成参考集格式，供分母的下一轮迭代并入。

背景见 `src/tpmorl/rl/scale.py` 的 `_estimate` 文档：9 个人工参考策略没有包住
全部目标的可达范围（v6 实测 `Aec` 越界 1.192 倍、`Eco` 1.041 倍），归一化后大于 1
使这两项在加权和里被系统性高估。修法是把学习策略当作新增"策略"并入同一个估计量。

**不**取全部运行的 max：`_estimate` 的文档已实测 max 随种子数单调爆涨、不可复现。
这里保持同一口径——输出仍是逐 (权重档, 种子) 的行，由 `_estimate` 内部按档取种子均值。

用法：
    python3 -m scripts.make_learned_scale \
        --src data/processed/gm_dataset_v1/exp_v6/base/objectives.csv \
        --dataset data/processed/gm_dataset_v1 --budget 900 --carry 3 --growth 0
注意 --budget/--carry/--growth 与制度参数须与**目标分母**的情景键一致（不是源批次的）。
"""
import argparse, os
import pandas as pd


def main(src, ds, budget, carry, growth):
    from tpmorl.objectives.reward import OBJ_NAMES
    from tpmorl.rl.scale import learned_path

    O = pd.read_csv(src)
    L = O[O.alpha >= 0]                      # α<0 是随机基线，不是学习策略
    if not len(L):
        raise SystemExit(f"{src} 里没有 alpha>=0 的行")

    # 索引用 rl_a<档>_s<种子>：`_estimate` 按 "_s" 右切分组，故每个 α 自成一个"策略"
    L = L.assign(tag=[f"rl_a{a:g}_s{int(s)}" for a, s in zip(L.alpha, L.seed)])
    out = L.set_index("tag")[list(OBJ_NAMES)]

    p = learned_path(ds, budget, carry, growth)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    out.to_csv(p, encoding="utf-8-sig")

    M = out.groupby([i.rsplit("_s", 1)[0] for i in out.index], sort=False).mean()
    print(f"源 {src}\n{len(out)} 次运行 / {len(M)} 个权重档 → {p}")
    print("\n各权重档的种子均值（原始量纲）：")
    print(M.round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源批次的 objectives.csv")
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--carry", type=float, default=3.0)
    ap.add_argument("--growth", type=float, default=0.0)
    from tpmorl.rl import scenario
    scenario.add_args(ap)
    a = ap.parse_args()
    # 情景键读调用时的模块常量，故必须先改写
    scenario.apply(budget=a.budget, carry=a.carry, growth=a.growth,
                   horizon=a.horizon, **scenario.from_args(a))
    main(a.src, a.dataset, a.budget, a.carry, a.growth)
