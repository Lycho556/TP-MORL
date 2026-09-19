# -*- coding: utf-8 -*-
"""exp_srv_scenarios.py —— S5 未来规划情景（最多三个）。

回答的问题不是"预测得准不准"（未来没有真值），而是

> **规划条件变化时，模型给出的 Where 与 When 会怎么变？**

三个情景，不多做：

    情景 0  基线              主组机会场设定
    情景 1  基础设施提前       机会窗整体前移（onset 区间前移）
    情景 2  更新压力增强       老化与实施条件的幅度上调

**口径声明（必须随结果一起出现）**：四个机会场通道**全部是情景参数、
无实证标定**（单元表没有建成年份、历年规划定位与设施投用年份）。因此
情景结果只能写成条件句："若设施提前 X 年到位，则模型建议的立项时点前移 Y 年"，
不得写成对光明区未来的预测。

报告的量（不含 accuracy）：

    Where   各情景选中的单元集合、与基线的重叠率
    When    立项年的分布、重心、相对基线的平移
    目标     Floor 等目标的轨迹变化

用法：
    PYTHONPATH=src python scripts/exp_srv_scenarios.py \\
        --out results_srv_s5 --seeds 0 --ep-each 20
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_temporal_gate as G                                  # noqa: E402
from exp_fqi_local import make_trees                           # noqa: E402

from tpmorl.rl import fqi, scenario as SC                      # noqa: E402

MAIN = dict(a_plan=2.4, a_infra=1.8, a_age=0.9, a_ready=1.0)

# 三个情景。每一项都是**相对基线的单一方向改动**，便于归因；
# 数值是情景参数，不是标定值。
SCENARIOS = {
    "情景0 基线": dict(),
    # 机会窗整体前移：onset 区间由默认的 0.15T~0.60T 提到 0.05T~0.40T，
    # 对应"地铁、道路等设施比基线更早到位"。
    "情景1 基础设施提前": dict(onset_lo=0.05, onset_hi=0.40),
    # 老化与实施条件的幅度上调 50%：对应"更新必要性与可实施性系统性增强"。
    # 注意 a_ready 上界为 1.0（>1 会让条件差的年份批准率归零 —— 那是行政冻结
    # 而非压力增强），故它保持 1.0，只抬 a_age。
    "情景2 更新压力增强": dict(a_age=1.35),
}


def run_scenario(a, name, over, seed):
    SC.reset()
    kw = dict(MAIN); kw.update(over)
    SC.apply(horizon=a.horizon, horizon_eval="auto", opp_shape="window",
             foresight=a.foresight, quota=a.quota, budget=1e12, **kw)
    env_fn = lambda: G.build_env(a.dataset, a.horizon, 0.5, 7, 1e12, "floor",
                                 True, False, a.prescreen, a.foresight)
    e0 = env_fn()
    EV, EVm = G.ev_tables(e0, a.gamma)
    elig = np.asarray(e0.env.eligible, bool)
    v_orc, oplan = G.oracle_plan(EV, a.quota, elig)

    data = fqi.collect(env_fn, lambda u, t: float(EV[u, min(t, EV.shape[1] - 1)]),
                       {"random": None, "myopic": EVm, "temporal_greedy": EV},
                       n_ep_each=a.ep_each, seed=seed)
    qa, qb, _ = fqi.fit_double(data, a.gamma, make_trees(seed=seed),
                               n_iter=a.n_iter, verbose=False)
    init, R, _ = fqi.rollout(env_fn, qa, qb, None)
    v = float(sum(EV[u, min(t, EV.shape[1] - 1)] for u, t in init.items()))
    yrs = np.array(list(init.values()), float)
    # 注意：配额年年咬满时"立项年重心"是**构造常数**（0..T-1 各 quota 个，
    # 均值恒为 (T-1)/2），不可当作结果报告。有信息量的是下面与基线逐单元
    # 对比的 When 平移量，以及立项年的四分位分布。
    return dict(scenario=name, seed=seed, value=v, ratio=v / v_orc,
                oracle_value=v_orc, actual_reward=R, n_init=len(init),
                立项年重心=float(yrs.mean()), 立项年中位=float(np.median(yrs)),
                立项年P25=float(np.percentile(yrs, 25)),
                立项年P75=float(np.percentile(yrs, 75))), init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_srv_s5")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--ep-each", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=None)
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--quota", type=int, default=3)
    ap.add_argument("--prescreen", type=int, default=20)
    ap.add_argument("--foresight", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.95)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows, plans = [], {}
    for name, over in SCENARIOS.items():
        for sd in a.seeds:
            r, init = run_scenario(a, name, over, sd)
            rows.append(r); plans[(name, sd)] = init
            print(f"  {name} seed={sd}  相对oracle {r['ratio']:.3f}  "
                  f"立项年重心 {r['立项年重心']:.2f}  立项数 {r['n_init']}")

    # 与基线比：Where 重叠率 与 When 平移
    base_name = list(SCENARIOS)[0]
    for r in rows:
        b = plans[(base_name, r["seed"])]
        c = plans[(r["scenario"], r["seed"])]
        inter = set(b) & set(c)
        r["与基线_Where重叠"] = len(inter) / max(len(set(b) | set(c)), 1)
        r["与基线_When平移年"] = (float(np.mean([c[u] - b[u] for u in inter]))
                            if inter else np.nan)

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "s5_scenarios.csv"), index=False,
             encoding="utf-8-sig")
    pd.DataFrame([dict(scenario=k[0], seed=k[1], unit=u, year=y)
                  for k, iv in plans.items() for u, y in iv.items()]).to_csv(
        os.path.join(a.out, "s5_plans.csv"), index=False, encoding="utf-8-sig")
    S = (D.groupby("scenario", sort=False)
         .agg(n=("ratio", "size"), 相对oracle=("ratio", "mean"),
              立项年重心_构造常数=("立项年重心", "mean"),
              与基线_Where重叠=("与基线_Where重叠", "mean"),
              与基线_When平移年=("与基线_When平移年", "mean")).reset_index())
    S.to_csv(os.path.join(a.out, "s5_summary.csv"), index=False,
             encoding="utf-8-sig")
    print("\n" + S.round(3).to_string(index=False))
    json.dump({k: v for k, v in SCENARIOS.items()},
              open(os.path.join(a.out, "s5_config.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n口径声明：四个机会场通道全部是**情景参数、无实证标定**，"
          "结论只能写成条件句，不得写成对光明区未来的预测。")


if __name__ == "__main__":
    main()
