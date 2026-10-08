# -*- coding: utf-8 -*-
"""exp_planning_shock.py —— 规划变更冲击：同一个策略，换一份规划时间表。

## 这个实验回答什么

前面所有实验回答的都是"策略能不能在**给定**的一张规划图上把时序排好"。
但规划本身是会变的——地铁原定 2028 年通车，改成 2032 年通车，是常事。
真正有用的策略应当**自己跟着改**：设施晚到四年，它周边那些项目就该往后排，
而不需要重新训练。

于是本脚本做一件很简单但判别力很强的事：

    用规划时间表 A 训练一个策略 → 不重新训练 → 拿到时间表 B 上评价
    同时把"在 B 上重新训练过"的策略作为上界参照

三种结果，三种结论：

  (1) 迁移后择时指标基本保持（接近"B 上重训"的水平）
      => 策略学到的是"看机会场决定早晚"这条**规则**，规划一变它自动响应。
         这是 adaptive planner，是本项目想要的结论。
  (2) 迁移后明显变差，但仍好于"完全不看机会场"的策略
      => 部分泛化：学到了一些规则，也记住了一些这张图的具体位置。
  (3) 迁移后退化到与"不看机会场"的策略无异
      => 策略记住的是**这一张图**，不是规则。那么前面所有"学到择时"的结论
         都要改写成"拟合了一张特定的规划安排"，这是必须自己先查出来的失败模式。

## 为什么冲击只改设施的投用年份

只动一个量，才知道差异是谁造成的。设施投用年份是四个场里**最有现实对应物**
的一个（地铁通车年份会变、会延期，且会公开），也是唯一还留在价值通道上的
一个（v18 起规划/老化/实施条件都走概率通道），因此它的变更同时冲击
"什么时候值钱"与"策略学到的空间格局"，是最严格的一档。

## 口径

* 迁移评价与重训评价**共用同一个分母**（机会场幅度不变，只改 onset），
  故两者的标量化回报可以直接相减。
* 冲击后的场用同一个 `FIELD_SEED`，只把设施投用年份整体推迟 `--shift` 年：
  换种子会同时改变三型划分与全部起始年，那测的就不是"规划变更"而是"另一个世界"。

用法（本机小规模）：
    PYTHONPATH=src python scripts/exp_planning_shock.py \\
        --out results_shock --iters 60 --eps 8 --seeds 0 1 2
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from tpmorl.env import opportunity as OPP
from tpmorl.eval.metrics import ScenarioSpec, timing_quality
from tpmorl.objectives.reward import OBJ_NAMES
from tpmorl.rl import env_gym as EG
from tpmorl.rl import scenario as SC
from tpmorl.rl import train_ppo as TP
from tpmorl.env import schedule as S


def make_env(ds, T, shift, alpha, seed):
    """造一个环境；`shift` = 设施投用年份整体推迟的年数（冲击幅度）。"""
    env = EG.RenewalEnv(ds, T=T, T_eval=SC.horizon_eval(),
                        weights=TP.weight_vector(alpha),
                        scale=EG.load_scale(ds, alpha=alpha))
    env.reset(seed=seed)
    if shift:
        # 只推迟设施：建成层与公布层同时右移，两者仍相差 INFRA_ANNOUNCE_LEAD 年。
        # 直接改场对象而不是改模块常量，是为了让同一个进程里两份场并存、可对比。
        o = env.opp
        o.infra_onset = o.infra_onset + float(shift)
        yr = o.years[None, :]
        lo, hi, rp = OPP.LEVEL_LOW, OPP.LEVEL_HIGH, OPP.RAMP_YEARS
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))
        o.infra = np.clip(lo + (hi - lo) * sig((yr - o.infra_onset[:, None]) / rp), 0, 1)
        o.infra_plan = np.clip(
            lo + (hi - lo) * sig((yr - (o.infra_onset[:, None]
                                        - OPP.INFRA_ANNOUNCE_LEAD)) / rp), 0, 1)
        env._phi = env._potential_table()      # 势函数表依赖场，必须重建
    return env


def run_eval(env, net, n_ep=3):
    rec = []
    TP.evaluate(env, net, n_ep=n_ep, record=rec)
    R = pd.DataFrame(rec)
    sp = ScenarioSpec(T=int(env.T), T_eval=int(env.T_eval),
                      tau_max=int(S.TAU_VALID + S.TAU_EXT), hazard=tuple(S.HAZARD),
                      build_years=dict(S.BUILD_YEARS_BY_CHANNEL),
                      budget=float(EG.BUDGET), quota=int(S.QUOTA))
    sp.gamma = float(EG.GAMMA)
    tq = timing_quality(R, sp, env.opp, ncell=env.ncell, farcap=env.farcap)
    yrs = R[R["unit"] >= 0]["year"].to_numpy(float)
    tq["init_year_mean"] = float(yrs.mean()) if len(yrs) else float("nan")
    tq["n_init"] = int(len(yrs) / max(n_ep, 1))
    return tq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--out", default="results_shock")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--eps", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--shift", type=float, default=4.0,
                    help="设施投用年份整体推迟几年（规划变更的幅度）")
    ap.add_argument("--a-plan", type=float, default=0.8)
    ap.add_argument("--a-infra", type=float, default=0.6)
    ap.add_argument("--a-age", type=float, default=0.3)
    ap.add_argument("--a-ready", type=float, default=0.4)
    ap.add_argument("--foresight", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    SC.reset()
    SC.apply(horizon=a.horizon, horizon_eval="auto", a_plan=a.a_plan,
             a_infra=a.a_infra, a_age=a.a_age, a_ready=a.a_ready,
             foresight=a.foresight)

    rows = []
    for seed in a.seeds:
        envA = make_env(a.dataset, a.horizon, 0.0, a.alpha, seed)
        netA, _ = TP.train(envA, iters=a.iters, eps_per_iter=a.eps, seed=seed)
        envB = make_env(a.dataset, a.horizon, a.shift, a.alpha, seed)
        netB, _ = TP.train(envB, iters=a.iters, eps_per_iter=a.eps, seed=seed)
        for tag, net, env in (("A策略@A场（原规划）", netA, envA),
                              ("A策略@B场（规划推迟，未重训）", netA, envB),
                              ("B策略@B场（在新规划上重训，上界参照）", netB, envB)):
            r = run_eval(env, net)
            r.update(seed=seed, arm=tag)
            rows.append(r)
            print(f"seed={seed} {tag}：立项重心 {r['init_year_mean']:.2f}  "
                  f"择时提升 {r['opp_lift']:+.4f}  后悔 {r['timing_regret']:.4f}")

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "shock_runs.csv"), index=False, encoding="utf-8-sig")
    S2 = (D.groupby("arm").agg(n=("seed", "size"),
                               立项重心=("init_year_mean", "mean"),
                               择时提升=("opp_lift", "mean"),
                               会变好型在爬升后=("ramp_post_onset", "mean"),
                               择时后悔=("timing_regret", "mean"))
          .reset_index())
    S2.to_csv(os.path.join(a.out, "shock_summary.csv"), index=False,
              encoding="utf-8-sig")
    print("\n" + S2.round(4).to_string(index=False))
    json.dump(vars(a), open(os.path.join(a.out, "shock_config.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n读法：若『A策略@B场』接近『B策略@B场』，说明策略学到的是看机会场决定"
          "早晚这条规则，规划一变自动响应；若退化到与『A策略@A场』的空间格局一致"
          "而指标变差，说明它记住的是那一张图。")


if __name__ == "__main__":
    main()
