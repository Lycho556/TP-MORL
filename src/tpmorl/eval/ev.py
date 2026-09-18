# -*- coding: utf-8 -*-
"""ev.py —— 逐条审批路径精确折现的期望交付价值 EV_i(t)。

## 为什么必须逐路径算

第一版 `exp_waiting_advantage.py` 是这么写的：

    p_ok, ew = 获批概率, 期望获批等待年数
    t_done   = t + ew + 1 + 建设年数
    EV       = p_ok · base · 价值乘子 · γ^t_done

这里有一个**数学错误**：

    E[γ^K] ≠ γ^{E[K]}

举个极端例子：一半项目第 1 年获批、一半第 5 年获批，正确的期望折现是
0.5·γ¹ + 0.5·γ⁵ = 0.5·0.950 + 0.5·0.774 = 0.862，而按期望年份折现给出的是
γ³ = 0.857。γ=0.95、tau_max=5 时两者差约 0.6%，而本项目里"等一年更好"的
一年等待优势中位数只有 ±2~5% 量级，**同一个数量级的偏差足以在临界状态改变
WA 的符号**。既然 WA>0 的状态占比是整套论证的第一块砖，这块砖不能是近似的。

## 本模块怎么算

对单元 i、立项年 t，按审批的**每一条可能路径**分别折现再求和：

    EV_i(t) = 准入概率(i,t)
              · Σ_{k=1..tau_max} P(第 k 个有效年获批 | 立项于 t)
                · base_i · 时机乘子(i, t, 交付年_k)
                · γ^{交付年_k}
    其中 交付年_k = t + k + 1 + 建设年数         （获批 → 次年开工 → 建设）
    且 交付年_k > T_eval − 1 的路径贡献 0         （期内交付不了 = 没有交付）

三处与第一版的实质差别：

1. **逐路径折现**（上面那条数学修正）。
2. **逐年重取实施条件调制**：h_k = HAZARD[k−1] · hazard_mult(i, t+k−1)，
   而不是全程按立项年那一年的调制算。第一版为了"只留立项时点一个自变量"
   做了简化，代价是与 `RenewalSchedule.step()` 的真实转移不一致；
   既然要用它当环境自证的基准，就应当与真实转移对齐。
3. **价值乘子读每条路径各自的交付年**，而不是读期望交付年。基础设施场在
   有限窗形状下会回落，读期望年会把"早交付赶上窗口、晚交付错过窗口"这个
   本该被区分的差别抹平。

另外第一版有一处更粗的问题：它按**期望**交付年判断"期内能不能交付"，
一旦超期就把整个 EV 记 0。于是临界单元的"有一半概率赶得上"被整体丢掉。
本模块按路径判断，赶得上的那部分概率照算。
"""
import numpy as np


def ev_matrix(opp, base, T, T_eval, build_years, hazard, tau_max, gamma=0.95,
              use_admit=True, use_value_mult=True):
    """返回 (n, T) 的期望折现交付价值矩阵 EV[i, t]。

    参数
    ----
    opp : OpportunityField
        机会场对象。用到 `admit_prob(t)`、`hazard_mult(t)`、`value_mult(i, t, td)`。
    base : (n,) 数组
        单元的基准交付量（本项目里是 FAR 上限 × 格数）。
    build_years : int 或 (n,) 数组
        建设年数。逐单元不同时传数组（真实环境里按通道不同）。
    hazard : 序列
        逐个有效年的条件批准率（`schedule.HAZARD`）。
    tau_max : int
        有效期总年数（`TAU_VALID + TAU_EXT`）。
    use_admit / use_value_mult : bool
        关掉可用于分离"准入"与"价值"两条通道各自的贡献。

    说明：为可读性保留逐单元循环，717 单元 × 25 年 × 5 条路径规模下耗时是秒级，
    没有必要为此把式子向量化到看不出物理含义。
    """
    n = len(base)
    base = np.asarray(base, float)
    by = (np.full(n, int(build_years)) if np.isscalar(build_years)
          else np.asarray(build_years, int))
    H = np.asarray(hazard, float)
    EV = np.zeros((n, int(T)))

    # 逐年预取场的取值，避免在内层循环里重复构造
    adm = {}
    hm = {}
    for t in range(int(T) + int(tau_max) + 1):
        a = opp.admit_prob(t) if use_admit else 1.0
        adm[t] = np.full(n, 1.0) if np.ndim(a) == 0 else np.asarray(a, float)
        m = opp.hazard_mult(t)
        hm[t] = np.full(n, 1.0) if np.ndim(m) == 0 else np.asarray(m, float)

    for t in range(int(T)):
        for i in range(n):
            p_alive = 1.0
            acc = 0.0
            for k in range(1, int(tau_max) + 1):
                # 第 k 个有效年的条件批准率，按**该年**的实施条件调制
                h = min(float(H[min(k - 1, len(H) - 1)])
                        * float(hm[min(t + k - 1, int(T) + int(tau_max))][i]), 1.0)
                p_k = p_alive * h
                p_alive *= (1.0 - h)
                if p_k <= 0:
                    continue
                td = t + k + 1 + int(by[i])       # 获批 → 次年开工 → 建设完成
                if td > int(T_eval) - 1:
                    continue                      # 这条路径期内交付不了，贡献 0
                mult = (opp.value_mult(i, t, td) if use_value_mult else 1.0)
                acc += p_k * base[i] * mult * gamma ** td
            EV[i, t] = float(adm[t][i]) * acc
    return EV


def waiting_advantage(EV, max_wait=5):
    """由 EV 矩阵给出一年等待优势与最优等待年数。

    返回 (WA, best_wait)：
        WA[i, t]        = EV[i, t+1] / EV[i, t] − 1（EV[i,t] ≤ 0 时为 nan）
        best_wait[i, t] = 在 [t, t+max_wait] 里 EV 最大的那一年减 t

    用相对量而非绝对量，是为了让不同规模的单元、不同幅度设定之间可比。
    """
    n, T = EV.shape
    WA = np.full((n, T - 1), np.nan)
    BW = np.zeros((n, T - 1), int)
    for t in range(T - 1):
        cur = EV[:, t]
        with np.errstate(divide="ignore", invalid="ignore"):
            WA[:, t] = np.where(cur > 0, EV[:, t + 1] / cur - 1.0, np.nan)
        hi = min(t + max_wait, T - 1)
        BW[:, t] = np.argmax(EV[:, t:hi + 1], axis=1)
    return WA, BW
