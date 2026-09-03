# -*- coding: utf-8 -*-
"""诊断：建设年限延长后规划期后半程是否还有可交付的动作。

对应 docs/访谈落实_v3.md 第 3 条。BUILD_YEARS 由 3 改为 5 之后，从立项到交付的
链条变成「立项 → 获批中位 3 年 → 建设 5 年」，而 T=15。若第 k 年之后立项的单元
在期内无法交付，`Floor` 恒为 0，后半程的奖励信号可能消失、策略退化成乱动。

这里只模拟制度状态机本身（不需要用地图与奖励），对每个立项年份估计：
  - 期内建成概率 P(S4 | 立项于第 y 年)
  - 条件平均建成年份
不依赖策略，因此结论对任何策略都成立，是该情景的**上界**性质。

用法：
  PYTHONPATH=src python3 scripts/diag_horizon.py                 # 默认 T=15
  PYTHONPATH=src python3 scripts/diag_horizon.py --horizon 20     # 路线 (a) 延长 T
  PYTHONPATH=src python3 scripts/diag_horizon.py --build-years 3  # 修改前口径
"""
import argparse
import numpy as np
from tpmorl.env import schedule as S


def trace(y0, T, tau_valid, tau_ext, build_years, cooldown, hazard, rng):
    """单个单元于第 y0 年立项，返回建成年份或 None（期内未建成）。"""
    tau_max = tau_valid + tau_ext
    t, tau = y0, 0
    while t < T:                                   # S1：等获批
        if rng.random() < hazard[min(tau, len(hazard) - 1)]:
            t += 1                                 # 获批当年 → S2
            t += 1                                 # S2→S3：批准后次年开工
            t += build_years                       # S3：施工
            return t if t <= T else None
        tau += 1
        t += 1
        if tau >= tau_max:                         # 失效 → 冷却 → 重报
            t += cooldown
            tau = 0
    return None


def main(a):
    hazard = np.asarray(S.HAZARD, float)
    tv = S.TAU_VALID if a.tau_valid is None else a.tau_valid
    te = S.TAU_EXT if a.tau_ext is None else a.tau_ext
    cd = S.COOLDOWN if a.cooldown is None else a.cooldown
    by = S.BUILD_YEARS_BY_CHANNEL[1] if a.build_years is None else a.build_years
    print(f"T={a.horizon}  有效期 {tv}+{te}  建设 {by} 年  冷却 {cd} 年  "
          f"链条下界 = 1(获批) + 1(开工) + {by}(施工) = {2 + by} 年")
    rng = np.random.default_rng(0)
    rows = []
    for y0 in range(a.horizon):
        done = [trace(y0, a.horizon, tv, te, by, cd, hazard, rng)
                for _ in range(a.reps)]
        ok = [d for d in done if d is not None]
        rows.append((y0 + 1, len(ok) / a.reps,
                     float(np.mean(ok)) if ok else float("nan")))
    print(f"\n{'立项年':>6} {'期内建成概率':>12} {'条件平均建成年':>15}")
    for y, p, m in rows:
        print(f"{y:>6} {p:>12.3f} {m:>15.2f}" if p else f"{y:>6} {p:>12.3f} {'—':>15}")
    dead = [y for y, p, _ in rows if p == 0.0]
    if dead:
        print(f"\n第 {dead[0]}–{dead[-1]} 年立项的单元期内**无法交付**"
              f"（{len(dead)}/{a.horizon} 年，占 {len(dead)/a.horizon:.0%}）"
              f"：这些年份的 Floor 恒为 0")
    low = [y for y, p, _ in rows if 0.0 < p < 0.2]
    if low:
        print(f"第 {low[0]}–{low[-1]} 年立项建成概率低于 0.2，奖励信号稀薄")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--tau-valid", type=int, default=None)
    ap.add_argument("--tau-ext", type=int, default=None)
    ap.add_argument("--cooldown", type=int, default=None)
    ap.add_argument("--build-years", type=int, default=None)
    main(ap.parse_args())
