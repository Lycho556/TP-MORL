"""metrics.py —— 实施类指标（CRH / ACD / LSR / 失效率 / 资金闲置率）与三层评价指标表。

为什么要这一层
--------------
既有落盘只有两类数字：11 维原始量纲目标值（objectives.csv）与标量化折扣回报。
前者答"做出来的东西好不好"，后者答"在这组偏好下得分高不高"，但两者都**答不了**
"计划到底落地了没有"——一个策略可以立一堆项、一个也没建成，目标值却因为
Cost 罚项小而不难看。本模块把"落地程度"做成可复算的指标。

数据从哪来（关键约束）
----------------------
`rec_a*_s*.csv` 只记**立项事件**（ep, year, unit, channel, target, n_cells, cost,
budget_before, spent, stopped），**没有**完工年、没有终态。按本轮分工不得改动
`env_gym.py` / `train_ppo.py` 去补日志，因此完工年只能从状态机**推**出来。

这件事之所以能精确推出来，是因为立项之后的轨迹是**完全外生**的：
`env_gym.step()` 里 `advance=np.where(ma)[0]`——每年把**全部** S1 单元都推进一次，
agent 在立项之后再没有任何选择权。于是给定立项年 y、该单元建设年限 b，
以及 (hazard, tau_max)，交付时点只由一个随机量决定：获批发生在第几个有效年 k。

按 `schedule.RenewalSchedule.step()` 的真实执行顺序推导（已用逐步模拟逐位验证，
见 docs/实施类指标_v15.md 的"口径验证"一节）：

    获批年   t_a = y + 1 + k,        k = 0 .. tau_max-1
    完工年   t_c = t_a + 1 + b = y + k + b + 2
    失效年   t_x = y + tau_max      （k 全部落空）
    P(k)     = (∏_{j<k} (1-h_j)) * h_k
    P(失效)  = ∏_{j<tau_max} (1-h_j)        （默认档 = 0.294）

hazard 下标按 `schedule` 的做法钳到 `len(hazard)-1`。

因此本模块给出的 CRH / ACD / 失效率是**期望口径**：对每个真实发生的立项事件，
按上式把它展开成 tau_max+1 个带概率权重的结局，再做加权统计。这不是蒙特卡洛
近似，而是对同一状态机的解析积分——相比"只看这一次随机种子抽出来的结果"，
它的方差为零、可复算，且与 objectives.csv 里那一次实现的随机抽样是同分布的。
**代价**：它不反映该次运行实际抽到的 hazard 序列，故与 objectives.csv 中依赖
具体实现的量（如 Expire 罚项）不应期待逐位相等，只应同向。这一点必须在文档里写明。

两个时间口径（本项目踩过的坑）
------------------------------
决策期 T 与评价期 T_eval 是两个不同口径：T_eval > T 时，第 T 年起不再立项、
不再进钱，只推进状态机并结算建成年释放的目标。环境在 `t >= T_eval` 结束，
故实际执行的最后一步是 `t = T_eval - 1`，"期内完工"的判据是

    以 T 为界：     t_c <= T - 1
    以 T_eval 为界： t_c <= T_eval - 1

两者含义不同，混用会得出相反结论，所以 CRH 必须**同时**给出并分别命名
（`CRH_T` / `CRH_Teval`）。`T_eval == T` 的闭区间批次下两者必然相等。
由 T-1 <= T_eval-1，恒有 `CRH_Teval >= CRH_T`——这是实现自检的硬条件。

slack 的口径
------------
LSR（窗外立项率）复用 `schedule.expected_years_to_delivery(if_initiated_now=True)`
的**期望**定义，不是乐观定义：

    ey    = E[离开S1] + 1(开工) + b        （默认档 tau=0 时 E=3.33，b=5 → ey=9.33）
    slack = (T - 1 - y) - ey

这与 `env_gym.obs()` 里 F[:,19] 的 slack 特征逐字同式（同样用 T-1、同样用 T 而非
T_eval），即"策略当时看到的可交付性判断"。用乐观口径（假设次年必获批）会系统性
高估可交付性，正是本项目要诊断的行为，故不采用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from tpmorl.env.schedule import (
    HAZARD, TAU_VALID, TAU_EXT, BUILD_YEARS, BUILD_YEARS_BY_CHANNEL,
)

__all__ = [
    "ScenarioSpec", "approval_pmf", "expected_years_to_delivery_at_initiation",
    "expand_outcomes", "implementation_metrics", "funds_decomposition",
    "three_layer_table", "LAYER_ROWS",
]


# --------------------------------------------------------------------------
# 情景参数
# --------------------------------------------------------------------------
@dataclass
class ScenarioSpec:
    """重算指标所需的情景参数。

    只放**影响交付时点与资金口径**的量。不放权重 / alpha / 种子——那些不改变
    状态机。字段默认值取 `schedule` 的模块级标定值，与环境默认档一致。

    T           决策期（可立项、有预算到账的年数）
    T_eval      评价期；None 表示闭区间口径 T_eval = T
    tau_max     计划有效期上限（tau_valid + tau_ext）
    hazard      逐有效年条件批准率
    build_years {通道: 建设年限}，或标量（全体同值）
    budget      年度预算额度（资金闲置率的分母用）
    carry_cap   结转上限倍数（用于资金分解的截断项；None 表示不算截断）
    """
    T: int = 15
    T_eval: int | None = None
    tau_max: int = TAU_VALID + TAU_EXT
    hazard: Sequence[float] = tuple(HAZARD)
    build_years: Mapping[int, int] | int = field(
        default_factory=lambda: dict(BUILD_YEARS_BY_CHANNEL))
    budget: float = 900.0
    carry_cap: float | None = None

    def __post_init__(self):
        self.T = int(self.T)
        self.T_eval = self.T if self.T_eval is None else int(self.T_eval)
        if self.T_eval < self.T:
            raise ValueError(f"T_eval({self.T_eval}) 不得小于决策期 T({self.T})")
        self.tau_max = int(max(self.tau_max, 1))
        self.hazard = tuple(float(x) for x in self.hazard)

    def build_years_of(self, channel: int) -> int:
        """该通道的建设年限。缺项按 schedule.BUILD_YEARS 兜底，与状态机一致。"""
        if isinstance(self.build_years, Mapping):
            return int(self.build_years.get(int(channel), BUILD_YEARS))
        return int(self.build_years)

    @property
    def closed_interval(self) -> bool:
        """是否闭区间口径（T_eval == T）。此时两个 CRH 口径必然相等。"""
        return self.T_eval == self.T


# --------------------------------------------------------------------------
# 状态机的解析展开
# --------------------------------------------------------------------------
def approval_pmf(tau_max: int, hazard: Sequence[float]) -> tuple[np.ndarray, float]:
    """返回 (p_k, p_fail)：在第 k 个有效年获批的概率，与有效期内始终未获批的概率。

    hazard 下标按 schedule 的做法钳到末位，故 tau_max 可大于 len(hazard)。
    """
    h = np.asarray(hazard, float)
    m = int(max(tau_max, 1))
    hk = np.array([h[min(k, len(h) - 1)] for k in range(m)])
    surv = np.concatenate([[1.0], np.cumprod(1.0 - hk)])   # surv[k] = P(前 k 年都没批)
    return surv[:m] * hk, float(surv[m])


def expected_leave_s1(tau_max: int, hazard: Sequence[float]) -> float:
    """E[从 tau=0 起还需几年才离开 S1]（获批与失效都算离开）。

    与 `schedule._expected_leave_s1()[0]` 同式：E[v] = 1 + (1-h_v) E[v+1]，
    E[tau_max-1] = 1。默认档实算为 3.3303 年（见 expected_leave_s1(5, HAZARD)）。
    """
    h = np.asarray(hazard, float)
    m = int(max(tau_max, 1))
    E = 0.0
    for v in range(m - 1, -1, -1):
        E = 1.0 + (1.0 - float(h[min(v, len(h) - 1)])) * E
    return E


def expected_years_to_delivery_at_initiation(spec: ScenarioSpec, channel: int) -> float:
    """"本年立项"情形下的期望交付年数 ey = E[离开S1] + 1 + b。

    与 `schedule.expected_years_to_delivery(if_initiated_now=True)` 对候选单元
    给出的值同式（该方法对 mask_initiate() 命中的 S0 单元返回 _exp_leave[0]+1+b）。
    """
    return expected_leave_s1(spec.tau_max, spec.hazard) + 1.0 + spec.build_years_of(channel)


def expand_outcomes(rec: pd.DataFrame, spec: ScenarioSpec) -> pd.DataFrame:
    """把每个立项事件展开成带概率权重的结局行。

    输入 `rec` 为 rec_a*_s*.csv 的 DataFrame。**必须**含 unit / year / channel 列。
    本函数内部即做 `unit >= 0` 过滤（表里有 unit=-1 的闲置年占位行，不是立项）。

    输出每个立项事件 tau_max+1 行：tau_max 个"获批于第 k 个有效年"分支，
    加 1 个"有效期届满失效"分支，prob 之和为 1。列：

        ep, unit, channel, y0(立项年), b(建设年限), ey, slack,
        outcome ∈ {"approved", "expired"}, k, prob,
        t_approve, t_complete, t_expire, duration(= t_complete - y0)

    失效分支的 t_complete / duration 为 NaN。
    """
    need = {"unit", "year", "channel"}
    missing = need - set(rec.columns)
    if missing:
        raise ValueError(f"rec 缺列 {sorted(missing)}；需要 {sorted(need)}")
    real = rec[rec["unit"] >= 0].copy()          # 规矩 4：必须加 unit>=0 过滤
    if real.empty:
        return pd.DataFrame(columns=["ep", "unit", "channel", "y0", "b", "ey", "slack",
                                     "outcome", "k", "prob", "t_approve", "t_complete",
                                     "t_expire", "duration"])
    p_k, p_fail = approval_pmf(spec.tau_max, spec.hazard)
    E0 = expected_leave_s1(spec.tau_max, spec.hazard)

    ep = real["ep"].to_numpy() if "ep" in real.columns else np.zeros(len(real), int)
    y0 = real["year"].to_numpy()
    ch = real["channel"].to_numpy()
    b = np.array([spec.build_years_of(c) for c in ch], float)
    ey = E0 + 1.0 + b
    slack = (spec.T - 1 - y0) - ey               # 与 env_gym.obs() F[:,19] 同式

    base = dict(ep=ep, unit=real["unit"].to_numpy(), channel=ch,
                y0=y0, b=b, ey=ey, slack=slack)
    blocks = []
    for k in range(spec.tau_max):
        d = dict(base)
        d["outcome"] = "approved"
        d["k"] = k
        d["prob"] = float(p_k[k])
        d["t_approve"] = y0 + 1 + k
        d["t_complete"] = y0 + k + b + 2.0
        d["t_expire"] = np.nan
        blocks.append(pd.DataFrame(d))
    d = dict(base)
    d["outcome"] = "expired"
    d["k"] = -1
    d["prob"] = float(p_fail)
    d["t_approve"] = np.nan
    d["t_complete"] = np.nan
    d["t_expire"] = y0 + float(spec.tau_max)
    blocks.append(pd.DataFrame(d))

    out = pd.concat(blocks, ignore_index=True)
    out["duration"] = out["t_complete"] - out["y0"]
    return out.sort_values(["ep", "unit", "y0", "k"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# 加权分位数
# --------------------------------------------------------------------------
def _wq(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """加权分位数（下侧定义）：升序累计权重首次 >= q 的那个取值。

    结局是离散的（duration 只取 k+b+2 这几个整数），用插值型分位数会造出
    状态机不可能产生的年数，故取"下侧"这一保守定义。
    """
    if len(values) == 0 or weights.sum() <= 0:
        return float("nan")
    o = np.argsort(values, kind="stable")
    v, w = np.asarray(values)[o], np.asarray(weights)[o]
    c = np.cumsum(w) / w.sum()
    return float(v[int(np.searchsorted(c, q, side="left"))])


# --------------------------------------------------------------------------
# 资金口径
# --------------------------------------------------------------------------
def funds_decomposition(rec: pd.DataFrame, spec: ScenarioSpec) -> dict:
    """资金闲置率及其分解。

    累计到账（分母）取 `budget * T`，与本项目既有判据 `diag.spend_ratio` 同分母。
    口径依据（见 env_gym.step）：t=0 年初已到账一笔 BUDGET，此后每步末再到账一笔，
    但**尾部评价年不进钱**，且第 T-1 步末到账的那笔在决策期结束后已无处可花。
    故"决策期内可动用的到账总额" = 初始 1 笔 + 决策期内 T-1 笔续拨 = budget * T。

        资金闲置率 = 1 - 累计支出 / (budget * T)

    累计支出取 rec 中 `unit >= 0` 行的 cost 之和（即立项承诺全额，与 upfront
    口径下的 Cost 目标同口径）。这是**承诺**口径而非**支付**口径：staged 预算
    模式下二者的总额相等（env_gym 已实测 Cost 总额 == 支付总额），但年内分布不同。

    分解项（诊断用，非主判据。为可比，分母同取 budget*T）：
        clipped_ratio  因超结转上限 carry_cap*budget 被截断而**作废**的额度占比。
                       这部分不是"闲置"（想花也花不出去），须与余额区分。
                       carry_cap 为 None 时记 NaN。
        residual_ratio 期末仍挂在可承诺额度上的余额占比。
        released_ratio 有效期届满被撤的单元，其剩余未付承诺被**释放**回可承诺额度
                       的占比。upfront 下恒为 0（立项即全额付清，无未付承诺）；
                       staged 下非零。rec 缺 `released` 列时按 0 处理（v15 之前
                       的批次全是 upfront，按 0 即正确）。
        decomp_sum     = spend − released + clipped + residual。

    `decomp_sum` 的恒等值是 **(T+1)/T**，不是 1——这是口径而非误差，必须写明：
    资金实际到账 T+1 笔（t=0 年初 1 笔 + 决策期内每步末各 1 笔，共 T 笔），
    即名义到账 = budget*(T+1)，而第 T-1 步末到账的那笔在决策期结束后已无处可花。
    分母取 budget*T 是为了与既有判据 diag.spend_ratio 逐位对齐，代价就是四项之和
    等于 (T+1)/T。

    恒等式的来源是可用预算的逐年递推（env_gym 的 step 里那一行）：
        b_{t+1} = min(b_t − 承诺_t + 释放_t + B, carry_cap·B)
    记 clip_t 为当年被截断作废的额度，则 b_{t+1} = b_t − 承诺_t + 释放_t + B − clip_t。
    从 t=0（b_0 = B）望远镜求和到 t=T-1，名义到账共 B·(T+1)，得
        Σ承诺 − Σ释放 + Σclip + b_T == B·(T+1)
    除以 B·T 即上式。**释放项曾被漏掉**，于是 upfront 各组（释放恒 0）一直闭合、
    staged 组实测 decomp_sum=1.2095 而应为 1.0667，差额 0.1428 恰是释放额占比——
    这是闭合校验抓到的真实缺项，不是误报。
    v12 四组与 v15 十五个 upfront 组实测 decomp_sum = 1.066667 == 16/15，与此式一致。
    """
    real = rec[rec["unit"] >= 0]
    n_ep = int(rec["ep"].nunique()) if "ep" in rec.columns else 1
    n_ep = max(n_ep, 1)
    denom = float(spec.budget) * spec.T * n_ep
    spend = float(real["cost"].sum())
    out = dict(spend_ratio=spend / denom if denom > 0 else float("nan"))
    out["idle_ratio"] = 1.0 - out["spend_ratio"]

    if spec.carry_cap is None or not {"budget_before", "spent"} <= set(rec.columns):
        out["clipped_ratio"] = float("nan")
        out["residual_ratio"] = float("nan")
        return out
    # 逐 (ep, year) 取一行即可：budget_before / spent 在同年各行内重复。
    agg = dict(budget_before=("budget_before", "first"), spent=("spent", "first"))
    # released：有效期届满被撤的单元，其剩余未付承诺回到可承诺额度。upfront 下恒为
    # 0，staged 下非零。v15 之前的批次没有这一列——那些批次全是 upfront，按 0 处理
    # 即正确；若某个 staged 批次缺列，闭合校验会直接报错而不是静默算错，这是我们
    # 想要的失败方向。
    has_rel = "released" in rec.columns
    if has_rel:
        agg["released"] = ("released", "first")
    yr = rec.groupby(["ep", "year"], as_index=False).agg(**agg)
    cap = float(spec.carry_cap) * float(spec.budget)
    clipped = 0.0
    residual = 0.0
    released_sum = 0.0
    for _, sub in yr.groupby("ep"):
        sub = sub.sort_values("year")
        bb = sub["budget_before"].to_numpy(float)
        sp = sub["spent"].to_numpy(float)
        rl = (sub["released"].to_numpy(float) if has_rel
              else np.zeros(len(sub), float))
        raw = bb - sp + rl + float(spec.budget)     # 截断前的次年额度
        clipped += float(np.maximum(raw - cap, 0.0).sum())
        residual += float(min(raw[-1], cap)) if len(raw) else 0.0
        released_sum += float(rl.sum())
    out["clipped_ratio"] = clipped / denom if denom > 0 else float("nan")
    out["residual_ratio"] = residual / denom if denom > 0 else float("nan")
    out["released_ratio"] = released_sum / denom if denom > 0 else float("nan")
    # 闭合式按可用预算的逐年递推 b_{t+1} = min(b_t − 承诺 + 释放 + B, 上限)
    # 望远镜求和得：承诺 − 释放 + 作废 + 期末余额 = B·(T+1)。释放项在 upfront 下
    # 恒为 0，故 upfront 各组的 decomp_sum 与改动前逐位相同。
    out["decomp_sum"] = (out["spend_ratio"] - out["released_ratio"]
                         + out["clipped_ratio"] + out["residual_ratio"])
    # 恒等式自检：绝对金额下 支出+作废+期末余额 == budget*(T+1)*回合数。
    # 不闭合说明 rec 的 budget_before/spent 序列不完整，或环境的拨付规则已变。
    if np.isfinite(out["decomp_sum"]):
        expect = (spec.T + 1) / spec.T
        if abs(out["decomp_sum"] - expect) > 1e-6:
            raise AssertionError(
                f"资金分解不闭合：decomp_sum={out['decomp_sum']:.8f}，"
                f"应为 (T+1)/T={expect:.8f}。请查 rec 的 budget_before/spent 是否逐年齐全")
    return out


# --------------------------------------------------------------------------
# 主接口
# --------------------------------------------------------------------------
def implementation_metrics(rec: pd.DataFrame, spec: ScenarioSpec) -> dict:
    """第二层"实施层"的全部指标。

    返回扁平 dict（便于直接 pd.DataFrame 化）。所有比率都以**立项单元数**为分母，
    与"计划落地程度"的语义一致（不是以候选单元数为分母）。

    键：
      n_initiated          立项事件数（unit>=0 行数，含多回合汇总）
      n_ep                 回合数
      init_per_ep          每回合立项数
      CRH_T / CRH_Teval    期内完工率，两个时间口径（t_c <= T-1 / <= T_eval-1）
      ACD_T / ACD_Teval    平均交付时长（年），只对该口径下已完工单元计
      ACD_*_p25/p50/p75    同口径的四分位与中位数（加权、下侧定义）
      LSR                  窗外立项率：立项时 slack < 0 的占比
      slack_mean           立项时 slack 的均值（诊断）
      late_share_y0        "来不及了"的起始年号，由情景推出（默认档 = 5）
      late_share_late      自 late_share_y0 起立项占比（与窗外立项率同义的年份代理）
      late_share_later     自 late_share_y0+1 起立项占比
      expire_rate_T/_Teval 失效率：有效期届满进入 S5 且 t_x 落在该口径内的占比
      expire_rate_uncond   无时间截断的失效率，= ∏(1-h) 与立项年无关（默认 0.294）
      spend_ratio / idle_ratio / clipped_ratio / residual_ratio   资金口径
    """
    ev = expand_outcomes(rec, spec)
    real = rec[rec["unit"] >= 0]
    n = int(len(real))
    n_ep = max(int(rec["ep"].nunique()) if "ep" in rec.columns else 1, 1)
    out = dict(n_initiated=n, n_ep=n_ep, init_per_ep=n / n_ep if n_ep else float("nan"))
    if n == 0:
        # 资金分解在零立项下**完全良定义**（支出 0 → 闲置率 1.0），而且"一个都没立"
        # 恰是最有信息量的情形；早先随提前 return 一起丢掉会落盘成 NaN。只有依赖
        # 立项事件的比率类指标（CRH / ACD / LSR / 失效率）才该缺失。
        out.update(funds_decomposition(rec, spec))
        return out

    app = ev[ev["outcome"] == "approved"]
    exp_ = ev[ev["outcome"] == "expired"]

    for tag, D in (("T", spec.T - 1), ("Teval", spec.T_eval - 1)):
        done = app[app["t_complete"] <= D]
        w = done["prob"].to_numpy(float)
        out[f"CRH_{tag}"] = float(w.sum()) / n
        dur = done["duration"].to_numpy(float)
        out[f"ACD_{tag}"] = float((dur * w).sum() / w.sum()) if w.sum() > 0 else float("nan")
        out[f"ACD_{tag}_p25"] = _wq(dur, w, 0.25)
        out[f"ACD_{tag}_p50"] = _wq(dur, w, 0.50)
        out[f"ACD_{tag}_p75"] = _wq(dur, w, 0.75)
        xp = exp_[exp_["t_expire"] <= D]
        out[f"expire_rate_{tag}"] = float(xp["prob"].sum()) / n
    out["expire_rate_uncond"] = float(approval_pmf(spec.tau_max, spec.hazard)[1])

    # 每个立项事件的 slack 各分支相同，取每事件一行
    one = ev.drop_duplicates(subset=["ep", "unit", "y0"], keep="first")
    out["LSR"] = float((one["slack"] < 0).mean())
    out["slack_mean"] = float(one["slack"].mean())
    y = real["year"].to_numpy()
    # 阈值由情景推出，不写死。slack 的零点是"此时立项还赶不赶得上期内交付"，
    # 由期望交付时长决定：E[交付] = E[离开待批] + 建设年限 + 1。写死第 5 年只在
    # 默认档上碰巧等价（默认 E[交付]≈10.33 → 第 6 年起来不及），一旦扫审批时序
    # 或建设年限，"第 6 年及以后立项占比"与窗外立项率就不再同义，自查 2 会变成在
    # 比两个无关量。默认档下本式给出 y0 = 5，与旧写法逐位一致。
    # ey 已按通道逐事件算在 ev 表里，取实际立项组合的均值，口径与 slack 自洽。
    y0 = int(max(np.ceil(spec.T - 1 - float(one["ey"].mean())), 0))
    out["late_share_y0"] = y0
    out["late_share_late"] = float((y >= y0).mean())
    out["late_share_later"] = float((y >= y0 + 1).mean())

    out.update(funds_decomposition(rec, spec))
    return out


# --------------------------------------------------------------------------
# 三层评价指标表
# --------------------------------------------------------------------------
_OBJ_NAMES = ("Gdp", "Eco", "Res", "Emp", "Aec", "E2r", "Cpt",
              "Floor", "Cost", "Disrupt", "Expire")

#: 三层评价指标表的行定义。字段见 three_layer_table 的 docstring。
LAYER_ROWS: list[dict] = (
    [dict(层="一 目标层", 指标=f"目标 {k}（原始量纲）",
          定义=f"评价期内累计的 {k} 目标值，未做任何归一化",
          时间口径="T_eval（评价期，建成年结算）",
          数据来源=f"objectives.csv 列 {k}",
          可跨情景比="是",
          备注="原始量纲，不含情景相关分母，故可跨情景直接比较")
     for k in _OBJ_NAMES] +
    [
        dict(层="二 实施层", 指标="CRH_T（期内完成率，决策期口径）",
             定义="期望完工单元数 / 立项单元数，完工判据 t_c <= T-1",
             时间口径="T（决策期）",
             数据来源="rec_a*_s*.csv 列 unit(>=0 过滤)/year/channel + 情景参数 "
                      "(hazard, tau_max, build_years)，按 t_c=y0+k+b+2 解析展开",
             可跨情景比="是",
             备注="比率指标，分母是本组自己的立项数，不依赖参考策略集，故可跨情景比"),
        dict(层="二 实施层", 指标="CRH_Teval（期内完成率，评价期口径）",
             定义="期望完工单元数 / 立项单元数，完工判据 t_c <= T_eval-1",
             时间口径="T_eval（评价期）",
             数据来源="同 CRH_T，仅完工判据的时间界改为 T_eval-1",
             可跨情景比="是",
             备注="恒有 CRH_Teval >= CRH_T；闭区间批次(T_eval==T)下两者相等"),
        dict(层="二 实施层", 指标="ACD（平均交付时长）",
             定义="完工年 - 立项年的加权均值，只对该口径下已完工单元计；"
                  "并给出 p25/p50/p75",
             时间口径="随 CRH 分 T / T_eval 两版",
             数据来源="rec_a*_s*.csv 列 year/channel + 情景参数，duration=t_c-y0",
             可跨情景比="是",
             备注="分布右偏且**右尾被截断**：未在界内完工者（含失效单元）不计入，"
                  "故 ACD 系统性偏小，只能同口径横比"),
        dict(层="二 实施层", 指标="LSR（窗外立项率）",
             定义="立项时 slack<0 的单元占全部立项的比例；"
                  "slack=(T-1-y0)-ey，ey=E[离开S1]+1+b",
             时间口径="T（决策期，与 env_gym 的 slack 特征同式）",
             数据来源="rec_a*_s*.csv 列 year/channel + 情景参数；ey 复用 "
                      "schedule.expected_years_to_delivery(if_initiated_now=True) 的期望口径",
             可跨情景比="是",
             备注="期望口径而非乐观口径；乐观口径会高估可交付性"),
        dict(层="二 实施层", 指标="失效率（S5 占立项数）",
             定义="有效期届满仍未获批而进入 S5 的期望单元数 / 立项单元数",
             时间口径="分 T / T_eval 两版（t_x=y0+tau_max 须落在界内）；"
                      "另给无截断值 ∏(1-h)",
             数据来源="rec_a*_s*.csv 列 year(>=0 过滤) + 情景参数 (hazard, tau_max)",
             可跨情景比="是",
             备注="无截断值与立项年无关，只由 hazard×tau_max 决定；"
                  "带截断值才反映择时（临界前立项者来不及失效）"),
        dict(层="二 实施层", 指标="资金闲置率",
             定义="1 - 累计支出 / 累计到账；累计到账 = budget * T * 回合数",
             时间口径="T（决策期；尾部评价年不进钱）",
             数据来源="rec_a*_s*.csv 列 cost（unit>=0 过滤）求和作分子；"
                      "分母由 runs.json config.budget 与 config.horizon 给出",
             可跨情景比="是",
             备注="与既有判据 diag.spend_ratio 同分母，可逐位对照；"
                  "分解出 clipped_ratio（超结转上限被作废）与 residual_ratio（期末余额）"),
        dict(层="三 标量层", 指标="标量化折扣回报",
             定义="Σ_t γ^t · (w/Σw)·(符号化原始目标 / scale)，"
                  "scale 取参考策略集的可达上界",
             时间口径="T_eval（评价期）",
             数据来源="curves.csv（训练曲线）/ runs.json runs[*].curve；"
                      "分母来自 rl/scale.py 的 REF_MODES×REF_SEEDS 参考集",
             可跨情景比="否",
             备注="**不可跨情景比**：分母 scale 依赖情景（quota/hazard/budget_mode 等"
                  "均进 inst_tag 缓存键），换情景即换分母，数值不同量纲"),
    ]
)


def three_layer_table() -> pd.DataFrame:
    """三层评价指标表。行 = 指标，列 = 层 / 定义 / 时间口径 / 数据来源 / 可跨情景比 / 备注。

    三层的分工：
      一 目标层：11 维原始量纲目标值——"做出来的东西好不好"。可跨情景比。
      二 实施层：CRH / ACD / LSR / 失效率 / 资金闲置率——"计划落地了没有"。
                 全是以本组自身立项数或到账预算为分母的比率，不含参考策略集，
                 故可跨情景比。
      三 标量层：标量化折扣回报——"在这组偏好下得分高不高"。**不可跨情景比**。
    """
    return pd.DataFrame(LAYER_ROWS, columns=[
        "层", "指标", "定义", "时间口径", "数据来源", "可跨情景比", "备注"])
