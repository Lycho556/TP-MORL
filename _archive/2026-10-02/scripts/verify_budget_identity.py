#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""资金记账恒等式的可复现验证。

为什么要有这个脚本
------------------
`docs/v15环境改造_实施记录.md` 与 `启动实验_v15.sh` 头注里引用过几个绝对金额，
用来说明两件事：

  (1) `upfront`（立项即全额扣款）下，预算的**实际支出**大于 Cost 目标所记的金额，
      差额恰为**未完工单元的成本**——即 Cost 目标只计完工单元，而预算按立项扣；
  (2) `staged`（按建设进度分期支付）下，Cost 总额与支付总额**逐位相等**。

那几个数字当初是在一个临时脚本里算的，没有随记录落盘种子与策略，评审时复现不出，
因而不可交叉验证。本脚本把该验证固定下来：动作策略、种子、情景参数全部写死在
代码里，任何人跑一遍都应得到与文档一致的数字。文档中的金额一律以本脚本的输出为准。

用法
----
    PYTHONPATH=src python3 scripts/verify_budget_identity.py
    PYTHONPATH=src python3 scripts/verify_budget_identity.py --json   # 机器可读

策略说明
--------
用**固定种子的随机可付性策略**，不是训练出来的策略：本脚本验的是环境的记账恒等式，
与策略好坏无关，用随机策略反而能覆盖更杂的立项组合。策略在每年按随机顺序遍历候选
对，跳过当年已选单元与买不起的，选到配额上限为止。
"""
from __future__ import annotations

import argparse
import json

import numpy as np

SEED = 7           # 动作策略与环境的随机种子（写死，供复现）
T = 15             # 决策期
BUDGET = 900.0
CARRY = 3.0
GROWTH = 0.0


def rollout(mode: str, ds: str) -> dict:
    """在指定预算模式下跑一整个回合，返回记账量。"""
    from tpmorl.rl import scenario
    from tpmorl.rl.env_gym import RenewalEnv
    from tpmorl.objectives.reward import OBJ_NAMES

    scenario.reset()
    scenario.apply(budget=BUDGET, carry=CARRY, growth=GROWTH, horizon=T,
                   horizon_eval="auto", budget_mode=mode)
    env = RenewalEnv(ds, T=T, T_eval=scenario.horizon_eval(),
                     weights=np.ones(len(OBJ_NAMES)) / len(OBJ_NAMES),
                     scale=np.ones(len(OBJ_NAMES)), seed=SEED)
    env.reset(seed=SEED)
    rng = np.random.default_rng(SEED)

    # 逐年累计**未折现**的目标向量：验记账恒等式要用原始金额，折现会破坏等式。
    vec_sum = np.zeros(len(OBJ_NAMES))
    for _ in range(T):
        X, meta, cost, units = env.pairs()
        left, used, acts = env.budget, set(), []
        for i in rng.permutation(len(meta)):
            u = meta[i][0]
            if u < 0 or u in used or cost[i] > left + 1e-6:
                continue
            acts.append(meta[i])
            used.add(u)
            left -= cost[i]
            if len(acts) >= env.quota:
                break
        _, _, _, info = env.step(acts)
        vec_sum += np.asarray(info["vec"], float) * env.scale

    # 推到评价期末，让在建单元完工、分期款付清（尾部年份不得有立项动作）
    while env.t < env.T_eval:
        _, _, _, info = env.step([])
        vec_sum += np.asarray(info["vec"], float) * env.scale

    # Cost 是负向目标（成本记为负回报），取绝对值才是金额口径。
    cost_obj = abs(float(vec_sum[OBJ_NAMES.index("Cost")]))

    # 直接从状态机算"已立项但期末未完工"的单元成本，用来核对差额的来源，
    # 而不是把差额倒过来定义成它。S4 = 已完工。
    from tpmorl.env.schedule import S4
    sched = env.env
    unfinished = 0.0
    for u, tg in env.plan.items():          # plan 是 {单元: 目标功能} 字典
        if int(tg) >= 0 and int(sched.sigma[int(u)]) != int(S4):
            unfinished += env.pair_cost(int(u), int(tg))
    committed_total = float(np.sum(env.spent_hist))   # 立项时的全额合同额之和
    disbursed = float(np.sum(env.disb_hist))          # 真正付出去的现金
    return dict(mode=mode,
                承诺总额=round(committed_total, 4),
                Cost目标=round(cost_obj, 4),
                实付现金=round(disbursed, 4),
                承诺减Cost=round(committed_total - cost_obj, 4),
                未完工单元成本=round(unfinished, 4),
                期末未付承诺=round(float(env.committed), 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ds", default="data/processed/gm_dataset_v1")
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    a = ap.parse_args()

    res = {m: rollout(m, a.ds) for m in ("upfront", "staged")}

    u, st = res["upfront"], res["staged"]
    # 恒等式 1：upfront 下现金在立项当年全额付出 → 实付现金 == 承诺总额
    ok1 = abs(u["实付现金"] - u["承诺总额"]) < 1e-3
    # 恒等式 2：upfront 下，承诺与 Cost 口径之差恰为未完工单元成本
    #（Cost 只在完工时计入，预算按立项全额扣）
    ok2 = abs(u["承诺减Cost"] - u["未完工单元成本"]) < 1e-3
    # 恒等式 3：staged 下按进度付款，现金流与成本口径一致 → 实付现金 == |Cost|，
    # 且期末无未付承诺（未完工单元的剩余义务在作废时被释放，不是被支付）
    ok3 = abs(st["实付现金"] - st["Cost目标"]) < 1e-3 and abs(st["期末未付承诺"]) < 1e-3

    if a.json:
        print(json.dumps(dict(种子=SEED, 决策期=T, 预算=BUDGET, 结转=CARRY,
                              结果=res, upfront_立项即付清=ok1,
                              upfront_差额等于未完工成本=ok2,
                              staged_实付等于Cost=ok3),
                         ensure_ascii=False, indent=2))
    else:
        print(f"种子={SEED}  决策期={T}  预算={BUDGET}  结转={CARRY}  增长={GROWTH}")
        for m, d in res.items():
            print(f"\n[{m}]")
            for k, v in d.items():
                if k != "mode":
                    print(f"    {k:<14} {v:>12.1f}")
        print(f"\n  upfront：实付现金 == 承诺总额 ? {ok1}")
        print(f"          承诺减 |Cost| = {u['承诺减Cost']:.1f} == 未完工单元成本 "
              f"{u['未完工单元成本']:.1f} ? {ok2}"
              f"（占承诺 {u['承诺减Cost'] / u['承诺总额'] * 100:.0f}%）")
        print(f"  staged ：实付现金 == |Cost| 且期末无未付承诺 ? {ok3}")

    if not (ok1 and ok2 and ok3):
        raise SystemExit("资金记账恒等式不成立——环境记账有缺陷，不得开跑批次")


if __name__ == "__main__":
    main()
