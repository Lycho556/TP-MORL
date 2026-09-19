# -*- coding: utf-8 -*-
"""exp_srv_mechanism.py —— S3 机制分解：改善到底来自哪里。

两个子实验，都**不重新训练**：

## 3A 信息阶梯（无需训练，秒级）

    空间 / 当期信息          冻结场 + 贪心
        ↓ 加入未来时间信息
    完美预报 + 贪心
        ↓ 加入跨年协调
    oracle

**两档筛选各算一份，不得混用。** 上一轮的 0.726 / 0.912 是**不设年度初筛**
时测的，而 Double-FQI 的 0.904 是**初筛 K=20** 下测的；把它们放进同一张阶梯
就是拿两套口径拼图。本脚本把两档都算出来并分开报告：

    无初筛档   随机 / 冻结场贪心 / 完美预报贪心 / oracle
    初筛档     同上四项（FQI 的数值来自 S2，同为初筛档，可并列）

FQI 只出现在初筛档那张表里。

## 3B Q 值诊断（读 S2 落盘的模型，不重训）

    corr(Q, 即时价值)        全局
    corr(Q, 即时价值)        **年内**（逐年算后平均）

本地实测：光明区全局 0.882；合成世界全局 +0.474 / −0.040 而年内 +0.969 /
+0.978。两者一起才说明问题 —— 年内秩相关接近 1 时，延续项在年内近似常数，
不可能改变 top-K 选择，策略必然与时间感知贪心重合。

用法：
    PYTHONPATH=src python scripts/exp_srv_mechanism.py \\
        --out results_srv_s3 --model-dir results_srv_s2 --seeds 0 1 2
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_temporal_gate as G                                  # noqa: E402

from tpmorl.rl import scenario as SC                           # noqa: E402

MAIN_AMPS = dict(a_plan=2.4, a_infra=1.8, a_age=0.9, a_ready=1.0)


def ladder(a, prescreen, seeds):
    """一档筛选下的信息阶梯。只有随机需要种子，其余三项是确定性的。"""
    SC.reset()
    SC.apply(horizon=a.horizon, horizon_eval="auto", opp_shape="window",
             foresight=a.foresight, quota=a.quota, budget=1e12, **MAIN_AMPS)
    env_fn = lambda: G.build_env(a.dataset, a.horizon, 0.5, 7, 1e12, "floor",
                                 True, False, prescreen, a.foresight)
    e0 = env_fn()
    EV, EVm = G.ev_tables(e0, a.gamma)
    elig = np.asarray(e0.env.eligible, bool)
    v_orc, oplan = G.oracle_plan(EV, a.quota, elig)
    rows = []
    for tag, kind, sds in (("随机", "random", seeds),
                           ("冻结场 + 贪心（仅当期信息）", "myopic", [seeds[0]]),
                           ("完美预报 + 贪心（加入未来时间信息）", "forecast",
                            [seeds[0]])):
        for sd in sds:
            e = env_fn()
            iv, vv = G.run_policy(e, EV, EVm, kind, quota=a.quota,
                                  rng=np.random.default_rng(sd))
            d = G.decompose(EV, iv, a.quota, elig, v_orc)
            rows.append(dict(prescreen=prescreen, method=tag, seed=sd,
                             value=vv, ratio=vv / v_orc,
                             where_wrong=d["loss_selection"],
                             when_error=d["loss_timing"], oracle_value=v_orc))
    rows.append(dict(prescreen=prescreen, method="oracle（加入跨年协调）",
                     seed=-1, value=v_orc, ratio=1.0, where_wrong=0.0,
                     when_error=0.0, oracle_value=v_orc))
    return rows


def q_diagnostics(model_dir):
    """3B：读 S2 落盘的 Q 模型，算全局与年内秩相关。不重训。"""
    import joblib
    from scipy.stats import spearmanr
    out = []
    for f in sorted(glob.glob(os.path.join(model_dir, "fqi_model_seed*.joblib"))):
        d = joblib.load(f)
        q = 0.5 * (d["qa"].predict(d["X"]) + d["qb"].predict(d["X"]))
        r, yr = d["r"], d["year"]
        within = [spearmanr(q[yr == t], r[yr == t]).statistic
                  for t in np.unique(yr) if (yr == t).sum() > 3]
        spread = [float(q[yr == t].max() - q[yr == t].min())
                  for t in np.unique(yr) if (yr == t).sum() > 3]
        ymean = [float(q[yr == t].mean()) for t in np.unique(yr)
                 if (yr == t).sum() > 3]
        out.append(dict(seed=int(d["seed"]), n=len(q),
                        全局秩相关=float(spearmanr(q, r).statistic),
                        年内秩相关=float(np.nanmean(within)),
                        年内极差均值=float(np.mean(spread)),
                        年份间落差=float(max(ymean) - min(ymean)) if ymean else np.nan))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_srv_s3")
    ap.add_argument("--model-dir", default="results_srv_s2",
                    help="S2 落盘 Q 模型的目录（--dump-model 产生）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--quota", type=int, default=3)
    ap.add_argument("--foresight", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--prescreens", type=int, nargs="+", default=[0, 20],
                    help="0 = 不设年度初筛；20 = 与 S2 主实验同档")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for k in a.prescreens:
        rows += ladder(a, k, a.seeds)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "s3_ladder_runs.csv"), index=False,
             encoding="utf-8-sig")
    S = (D.groupby(["prescreen", "method"], sort=False)
         .agg(n=("ratio", "size"), 相对oracle=("ratio", "mean"),
              标准差=("ratio", "std"), 挑错单元=("where_wrong", "mean"),
              放错年份=("when_error", "mean")).reset_index())
    S.to_csv(os.path.join(a.out, "s3_ladder_summary.csv"), index=False,
             encoding="utf-8-sig")
    print("=== 3A 信息阶梯（两档筛选分开报告，不得混用）===")
    print(S.round(3).to_string(index=False))
    print("\n注：Double-FQI 的数值来自 S2（初筛 K=20 档），"
          "只能与 prescreen=20 那几行并列；prescreen=0 档没有 FQI。")

    if os.path.isdir(a.model_dir):
        q = q_diagnostics(a.model_dir)
        if q:
            Q = pd.DataFrame(q)
            Q.to_csv(os.path.join(a.out, "s3_q_diagnostics.csv"), index=False,
                     encoding="utf-8-sig")
            print("\n=== 3B Q 值诊断（读 S2 模型，未重训）===")
            print(Q.round(4).to_string(index=False))
            print("\n判读：年内秩相关 ≈ 1 说明延续项在年内近似常数、不改变 top-K；"
                  "明显小于 1 才说明 Q 在年内区分了候选。")
        else:
            print(f"\n[跳过 3B] {a.model_dir} 下没有 fqi_model_seed*.joblib —— "
                  f"S2 需要带 --dump-model 跑")
    else:
        print(f"\n[跳过 3B] 找不到 {a.model_dir}")


if __name__ == "__main__":
    main()
