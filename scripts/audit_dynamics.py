#!/usr/bin/env python3
"""批次动力学审计：用落盘权重复现每个 run 的目标值，判定它实际在哪套动力学下跑出。

为什么需要这个脚本
------------------
2026-09-12 定位到 `scale.reference_returns` 会向调用方泄漏 `env_gym.FAR_GROWTH`
（详见 docs/验收_v12_预算约束扫描.md 修正记录）。后果是 `runs.json` 里记录的
`growth` 是**命令行意图值**，可能与实际运行的动力学不符，且无法从落盘数据直接看出。
批次 v12 有三组因此作废。修复已落地，但**此前所有批次都需要审计**才能引用其结果。

判定方法
--------
对每个 (alpha, seed)：用 `runs/net_a*_s*.pt` 的权重，在候选动力学集合（默认
{growth=0, growth=0.1}）下各贪心评估一遍，与 `runs.json` 里的落盘目标值比对。
评估完全复刻 `train_ppo.evaluate`：5 个回合、种子 90000..90004、折现回报乘回 scale。
落盘值必然逐位落在某一个候选上（实测相对误差 < 1e-9）；落不上则报 `未匹配`，
说明还有本脚本未建模的第三个差异源，必须人工追查。

**每组必须在干净的情景状态下审计。** `scenario.apply` 对 None 参数不重置、只沿用
模块当前值，而各组的 gamma/cooldown/build_years 多为 None，故同一进程内连续审计
多组会被前一组的制度参数污染——审计工具本身绝不能重蹈被审计代码的覆辙。
本脚本用两道保险：`Pool(..., maxtasksperchild=1)` 保证每组一个全新解释器（Pool 的
worker 默认跨任务复用，**不是**每任务 fork 一次），组内每次评估前再调
`scenario.reset()` 显式恢复出厂值。初版只写了 Pool 而误以为已隔离，实测九组中
horizon20/relax2/relax5 共 105 个 run 因此判为"未匹配"。

用法
----
    python3 scripts/audit_dynamics.py --exp data/processed/gm_dataset_v1/exp_v11 \
        --out results_v11/v11_growth_audit.csv
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

CAND_GROWTHS = (0.0, 0.1)
EVAL_SEED0 = 90000          # 与 train_ppo.evaluate 一致
N_EP = 5                    # 与 train_ppo.evaluate 的 n_ep 默认值一致
RTOL = 1e-9


def _audit_one_group(args):
    """在子进程中审计单个组。返回逐 run 的判定记录列表。"""
    exp_dir, group, ds, obj_key = args
    import torch
    torch.set_num_threads(1)
    from tpmorl.rl import train_ppo as T, scenario
    from tpmorl.rl import env_gym as EG
    from tpmorl.rl.env_gym import RenewalEnv
    # 注意：归一化分母在本审计中**不影响结论**。evaluate 的 vec 是 raw/env.scale，
    # 复算时又乘回同一个 env.scale，逐位抵消。故即使原批次用的是另一套分母，
    # 复现出的目标原始量纲值仍然一致；分母只影响训练，而训练结果已固化在权重里。
    from tpmorl.rl.scale import load_fixed_scale

    gdir = os.path.join(exp_dir, group)
    with open(os.path.join(gdir, "runs.json")) as f:
        blob = json.load(f)
    cfg = blob["config"]
    inst = {k: cfg.get(k) for k in ("tau_valid", "tau_ext", "cooldown", "build_years")}
    horizon = int(cfg["horizon"])
    budget, carry = float(cfg["budget"]), float(cfg["carry"])
    obj_names = list(T.OBJ_NAMES)
    if obj_key not in obj_names:
        raise SystemExit(f"目标 {obj_key} 不在 OBJ_NAMES 中：{obj_names}")
    iobj = obj_names.index(obj_key)

    out = []
    for r in blob["runs"]:
        alpha, seed = float(r["alpha"]), int(r["seed"])
        rec = float(r["obj"][obj_key])
        wpath = os.path.join(gdir, "runs", f"net_a{alpha:g}_s{seed}.pt")
        if not os.path.exists(wpath):
            out.append(dict(组=group, alpha=alpha, seed=seed, 落盘=rec,
                            判定="缺权重文件"))
            continue
        vals = {}
        for g in CAND_GROWTHS:
            # apply 对 None 参数不重置、只沿用当前值，而各组的 gamma/cooldown/
            # build_years 多为 None（表示"用默认值"）。若不先 reset()，上一组显式
            # 设过的制度参数会静默继承进本组，审计结果即被自己污染。
            scenario.reset()
            scenario.apply(budget=budget, carry=carry, growth=g, horizon=horizon,
                           gamma=cfg.get("gamma"), **inst)
            sc = load_fixed_scale(ds, budget, carry)
            env = RenewalEnv(ds, T=horizon, weights=T.weight_vector(alpha), scale=sc)
            # apply 之后再兜一次底：load_fixed_scale 命中缓存时不碰全局，未命中时
            # （修复前）会改写。修复后两条路径都不改，这里断言该不变量成立。
            assert abs(EG.FAR_GROWTH - g) < 1e-12, (
                f"建分母改写了 FAR_GROWTH：期望 {g}，实际 {EG.FAR_GROWTH}")
            net = T.Pointer()
            net.load_state_dict(torch.load(wpath, map_location="cpu"))
            net.eval()
            ep = []
            for e in range(N_EP):
                tr = T.run_episode(env, net, greedy=True, seed=EVAL_SEED0 + e)
                v = np.array([x[iobj] * sc[iobj] for x in tr["vec"]])
                ep.append(sum((env.gamma ** t) * v[t] for t in range(len(v))))
            vals[g] = float(np.mean(ep))
        tol = max(1.0, abs(rec) * RTOL)
        hit = [g for g in CAND_GROWTHS if abs(rec - vals[g]) < tol]
        # 落盘值为 0 时两个候选可能都不为 0 或都为 0，判定不可靠，单独标出
        if abs(rec) < 1.0:
            verdict = "不可判定(落盘≈0)"
        elif len(hit) == 1:
            verdict = f"growth={hit[0]:g}"
        elif len(hit) > 1:
            verdict = "不可判定(候选重合)"
        else:
            verdict = "未匹配"
        out.append(dict(组=group, alpha=alpha, seed=seed, 落盘=rec,
                        **{f"复算_g{g:g}": vals[g] for g in CAND_GROWTHS},
                        判定=verdict))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="批次目录，如 .../exp_v11")
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--obj", default="Floor",
                    help="用于判别的目标；须对 FAR_GROWTH 敏感，默认 Floor")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    groups = sorted(d for d in os.listdir(a.exp)
                    if os.path.exists(os.path.join(a.exp, d, "runs.json")))
    if not groups:
        raise SystemExit(f"{a.exp} 下没有任何含 runs.json 的组")
    print(f"审计 {len(groups)} 组：{' '.join(groups)}", flush=True)

    jobs = [(a.exp, g, a.dataset, a.obj) for g in groups]
    rows = []
    # maxtasksperchild=1：Pool 的 worker 默认跨任务复用，一个进程会连跑多个组，
    # 模块级情景常量随之继承。这里强制每组一个全新解释器，与 _audit_one_group 里
    # 的 scenario.reset() 形成双保险。
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for i, res in enumerate(pool.imap(_audit_one_group, jobs), 1):
            rows += res
            vs = pd.Series([r["判定"] for r in res]).value_counts().to_dict()
            print(f"[{i}/{len(jobs)}] {res[0]['组']:12s} {vs}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False, encoding="utf-8-sig")

    print("\n===== 按组汇总 =====")
    piv = df.pivot_table(index="组", columns="判定", values="alpha",
                         aggfunc="count", fill_value=0)
    print(piv.to_string())
    bad = df[df.判定.isin(["未匹配", "缺权重文件", "不可判定(候选重合)"])]
    if len(bad):
        print(f"\n⚠ {len(bad)} 个 run 无法判定，须人工追查：")
        print(bad.head(20).to_string(index=False))
    print(f"\n明细已写入 {a.out}")


if __name__ == "__main__":
    sys.exit(main())
