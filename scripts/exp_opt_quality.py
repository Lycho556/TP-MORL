"""实验①：优化质量。回答自评 v1 第四节的三条不利发现。

设计：7 权重 × 5 种子 × 300 迭代 × 每迭代 8 回合（旧：7 × 1 × 60 × 4）。
基准配置取 B=900、无容积率增长——即 `Floor` 输给随机基线的那个配置。

三条待查：
  4.1 交付建面输给随机基线 → 记录每次运行的立项数、交付格数、STOP 频率。
      若加大训练后 Floor 追上随机，是欠收敛；若立项数仍显著低于随机，
      则是负回报下的「少做更优」吸引子。
  4.2 前沿被支配 → 5 种子均值上重算支配关系。
  4.3 未收敛 → 300 迭代的曲线尾段斜率与种子间标准差。

随机基线同样跑 5 个种子，与 RL 同底可比。
"""
import argparse, json, os, time
import numpy as np
import pandas as pd

ALPHAS = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
SEEDS = [0, 1, 2, 3, 4]


def _load_scale(ds, OBJ_NAMES):
    base = pd.read_csv(os.path.join(ds, "reward_v0", "discounted_return.csv"), index_col=0)
    sc = np.array(base[list(OBJ_NAMES)].abs().max(axis=0).values, float)
    sc[sc == 0] = 1.0
    return sc


def one_run(job):
    """单个 (alpha, seed) 训练+评估。在子进程中执行，torch 压到单线程。"""
    alpha, seed, ds, iters, eps, budget, carry, growth = job
    import torch
    torch.set_num_threads(1)
    from tpmorl.rl import env_gym, train_ppo as T
    from tpmorl.rl.env_gym import RenewalEnv

    env_gym.BUDGET = float(budget)
    env_gym.CARRY_CAP = float(carry)
    env_gym.FAR_GROWTH = float(growth)
    sc = _load_scale(ds, T.OBJ_NAMES)
    env = RenewalEnv(ds, weights=T.weight_vector(alpha), scale=sc)

    t0 = time.time()
    net, hist = T.train(env, iters=iters, eps_per_iter=eps, seed=seed)
    rec = []
    g = T.evaluate(env, net, record=rec)
    wall = time.time() - t0

    R = pd.DataFrame(rec)
    real = R[R.unit >= 0]
    n_ep = max(int(R.ep.nunique()), 1)
    diag = dict(
        n_initiated=len(real) / n_ep,               # 每回合立项单元数
        cells=float(real.n_cells.sum()) / n_ep,     # 每回合交付格数
        stop_rate=float(R.groupby(["ep", "year"]).stopped.max().mean()),
        idle_years=float((R.groupby(["ep", "year"]).unit.max() < 0).sum()) / n_ep,
        spend_ratio=float(real.cost.sum()) / n_ep / (budget * env.T),
    )
    return dict(alpha=alpha, seed=seed, wall=wall,
                obj=dict(zip(T.OBJ_NAMES, [float(x) for x in g])),
                curve=[float(x) for x in hist], diag=diag)


def random_runs(ds, seeds, budget, carry, growth):
    import torch
    torch.set_num_threads(1)
    from tpmorl.rl import env_gym, train_ppo as T
    from tpmorl.rl.env_gym import RenewalEnv
    env_gym.BUDGET = float(budget)
    env_gym.CARRY_CAP = float(carry)
    env_gym.FAR_GROWTH = float(growth)
    sc = _load_scale(ds, T.OBJ_NAMES)
    env = RenewalEnv(ds, weights=T.weight_vector(0.5), scale=sc)
    out = []
    for s in seeds:
        g = T.eval_random(env, seed=s)
        out.append(dict(seed=s, obj=dict(zip(T.OBJ_NAMES, [float(x) for x in g]))))
    return out


def main(ds, out, iters, eps, budget, carry, growth, workers):
    os.makedirs(out, exist_ok=True)
    jobs = [(a, s, ds, iters, eps, budget, carry, growth) for a in ALPHAS for s in SEEDS]
    print(f"{len(jobs)} 个运行 × {iters} 迭代 × {eps} 回合  预算 {budget}  增长 {growth}"
          f"  并行 {workers}", flush=True)

    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(one_run, jobs), 1):
            res.append(r)
            print(f"[{i}/{len(jobs)}] a={r['alpha']} s={r['seed']} "
                  f"{r['wall']:.0f}s 立项={r['diag']['n_initiated']:.1f} "
                  f"Floor={r['obj']['Floor']:.3g}", flush=True)

    rnd = random_runs(ds, SEEDS, budget, carry, growth)
    with open(os.path.join(out, "runs.json"), "w") as f:
        json.dump(dict(config=dict(iters=iters, eps=eps, budget=budget, carry=carry,
                                   growth=growth, alphas=ALPHAS, seeds=SEEDS),
                       runs=res, random=rnd), f)

    rows = []
    for r in res:
        rows.append(dict(alpha=r["alpha"], seed=r["seed"], **r["obj"], **r["diag"]))
    for r in rnd:
        rows.append(dict(alpha=-1.0, seed=r["seed"], **r["obj"]))
    pd.DataFrame(rows).to_csv(os.path.join(out, "objectives.csv"), index=False)
    C = pd.DataFrame({f"a{r['alpha']}_s{r['seed']}": r["curve"] for r in res})
    C.index.name = "iter"
    C.to_csv(os.path.join(out, "curves.csv"))
    print("已写出", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--out", default="data/processed/gm_dataset_v1/exp_opt_v1")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--eps", type=int, default=8)
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--carry", type=float, default=3.0)
    ap.add_argument("--growth", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    main(a.dataset, a.out, a.iters, a.eps, a.budget, a.carry, a.growth, a.workers)
