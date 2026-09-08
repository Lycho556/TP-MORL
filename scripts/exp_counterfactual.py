"""实验②：反事实评估。把交付建面的增量拆成「机械」与「策略」两部分。

动机（自评 v2 路线 C）：批次 v6/v7 测的是优化质量，**没有**检验论文的核心时序主张
——「等待有价值当且仅当未来机会集优于现在」。容积率增长率 `FAR_GROWTH` 正是
"未来机会集是否优于现在"的操作化：G>0 意味着晚做能拿到更高的容积率上限。

问题是"G 越大交付建面越多"这件事有两个来源，必须分开：
  * **机械部分**：环境本身变阔了。同一个策略、什么都不改，放到 G>0 的环境里
    也会拿到更多建面，因为每格允许的建面上限本身涨了。
  * **策略部分**：策略**因为**知道未来更阔而改变了择时——该等的等了。
只有策略部分才是论文主张的证据。机械部分是恒等式，任何策略都有。

本脚本算前者：把 G=0 训出的策略放到 G>0 的环境里评估（策略不变，环境变）。
    机械部分 = 评估(π_G0, 环境G) − 评估(π_G0, 环境0)
策略部分需要在 G>0 环境里**重训**一组（π_G），本脚本不做训练：
    策略部分 = 评估(π_G, 环境G) − 评估(π_G0, 环境G)
故 `--nets` 可给多个目录，脚本按目录标注 `trained_g`，两轮跑完即可相减。

前提：策略权重已落盘（`scripts/exp_opt_quality.py` 的 runs/ 子目录，v7 起才有；
v6 及更早只存了汇总诊断，故本脚本对 v6 无效）。

自查：以 trained_g=0 的策略在 eval_g=0 下评估，结果应与该批次 `objectives.csv`
逐档吻合（同为 5 回合贪心、同种子 90000+e），不吻合说明权重或情景键对不上。

用法：
    python3 scripts/exp_counterfactual.py \
        --nets data/processed/gm_dataset_v1/exp_v7/base/runs \
        --out results_v7/cf --eval-growths 0 0.05 0.10
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd

_KEY = re.compile(r"net_a([0-9.]+)_s(\d+)\.pt$")


def one_eval(job):
    """单个 (权重档, 种子, 评估用增长率) 的贪心评估。子进程执行。"""
    path, alpha, seed, trained_g, eval_g, ds, scen = job
    import torch
    torch.set_num_threads(1)
    from tpmorl.rl import train_ppo as T, scenario
    from tpmorl.rl.env_gym import RenewalEnv
    from tpmorl.rl.scale import load_scale
    import tpmorl.rl.env_gym as EG

    scenario.apply(budget=scen["budget"], carry=scen["carry"], growth=eval_g,
                   horizon=scen["horizon"], **scen["inst"])
    # 分母按**评估情景**取：evaluate 内部乘回 scale 还原原始量纲，故跨情景比较
    # 用的是原始量纲，与分母选择无关。这里取评估情景的分母只为让 env 自洽。
    sc = load_scale(ds, EG.BUDGET, EG.CARRY_CAP, eval_g)
    env = RenewalEnv(ds, T=scen["horizon"], weights=T.weight_vector(alpha), scale=sc)

    net = T.Pointer()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    rec = []
    with torch.no_grad():
        g = T.evaluate(env, net, record=rec)

    R = pd.DataFrame(rec)
    real = R[R.unit >= 0]
    n_ep = max(int(R.ep.nunique()), 1)
    # 立项年份分布：论文的时序主张要看「等不等」，故记首末立项年与年份均值
    yrs = real.year.values if len(real) else np.array([np.nan])
    return dict(alpha=alpha, seed=seed, trained_g=trained_g, eval_g=eval_g,
                n_initiated=len(real) / n_ep,
                first_year=float(np.nanmin(yrs)), last_year=float(np.nanmax(yrs)),
                mean_year=float(np.nanmean(yrs)),
                idle_years=float((R.groupby(["ep", "year"]).unit.max() < 0).sum()) / n_ep,
                **{k: float(v) for k, v in zip(T.OBJ_NAMES, g)})


def main(net_dirs, out, eval_growths, ds, budget, carry, workers, scen):
    import multiprocessing as mp
    os.makedirs(out, exist_ok=True)
    scen = dict(scen, budget=budget, carry=carry)

    jobs = []
    for nd in net_dirs:
        # 训练时的增长率从目录路径旁的 runs.json 拿不到，故由 --nets 的顺序与
        # --trained-growths 对齐；默认全部按 0 处理（v7 base 组即 G=0）。
        tg = scen["trained_growths"].get(nd, 0.0)
        for p in sorted(glob.glob(os.path.join(nd, "net_a*_s*.pt"))):
            m = _KEY.search(os.path.basename(p))
            a, s = float(m.group(1)), int(m.group(2))
            for eg in eval_growths:
                jobs.append((p, a, s, tg, eg, ds, scen))
    if not jobs:
        raise SystemExit(f"{net_dirs} 里没有 net_a*_s*.pt——v7 之前的批次没落盘权重")

    print(f"{len(jobs)} 个评估作业，并行 {workers}")
    rows = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one_eval, jobs), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}]", flush=True)

    D = pd.DataFrame(rows).sort_values(["trained_g", "eval_g", "alpha", "seed"])
    p = os.path.join(out, "counterfactual.csv")
    D.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"已写出 {p}")

    # 机械部分：同一策略跨评估情景的建面差
    M = D.groupby(["trained_g", "eval_g", "alpha"])[
        ["Floor", "n_initiated", "mean_year", "idle_years"]].mean().round(1)
    print("\n各 (训练增长率, 评估增长率, 权重档) 的种子均值：")
    print(M.to_string())
    return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", nargs="+", required=True, help="含 net_a*_s*.pt 的目录")
    ap.add_argument("--trained-growths", nargs="*", type=float, default=None,
                    help="与 --nets 一一对应的训练时增长率，默认全 0")
    ap.add_argument("--out", default="results_v7/cf")
    ap.add_argument("--eval-growths", nargs="+", type=float, default=[0.0, 0.05, 0.10])
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--carry", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=8)
    from tpmorl.rl import scenario
    scenario.add_args(ap)
    a = ap.parse_args()
    tg = a.trained_growths or [0.0] * len(a.nets)
    if len(tg) != len(a.nets):
        raise SystemExit("--trained-growths 个数须与 --nets 一致")
    scen = dict(horizon=a.horizon, inst=scenario.from_args(a),
                trained_growths=dict(zip(a.nets, tg)))
    main(a.nets, a.out, a.eval_growths, a.dataset, a.budget, a.carry, a.workers, scen)
