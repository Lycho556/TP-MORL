# -*- coding: utf-8 -*-
"""opportunity.py —— 四类**逐年变化**的机会场，让"等一年"真的与"现在做"不同。

## 为什么需要这个模块

v16 的主判据成立（观测补齐后 LSR 降、CRH 升），但**次判据不成立**：主组的
立项年份分布与均摊无法区分（卡方 p=0.999，立项重心 6.98 对均摊基准 7.0，
见 results_v16/quicklook.txt）。诊断早已写明原因（docs/择时机制_诊断_v1.md §3，
env_gym 顶部"非平稳性"一节亦有同样结论）：

    候选集是平稳的。预算结转只改变"能做多少"，不改变"何时做更好"；
    成本与收益都与面积线性、无规模报酬，故攒钱无回报。于是"明年的机会集
    不优于今年"，而 γ<1 始终把动作往前推 —— **等待永不严格占优**。

策略学不到择时，不是因为学不会，而是因为**没有择时可学**。此前唯一的非平稳
通道是 `FAR_GROWTH`（容积率逐年上浮），但它是**全局同步**的：所有单元同时
变好，只改变"整体早做还是晚做"，不改变"先做哪个、后做哪个"。真实的择时问题
是后者。

本模块把非平稳性下沉到**逐单元**：

    上位规划     PlanningOpportunity(i,t)        政府什么时候开始重视这里
    基础设施     InfrastructureOpportunity(i,t)  这个地方什么时候真正成熟
    建筑老化     BuildingAge(i,t) -> Necessity    什么时候更新变得越来越必要
    实施条件     ImplementationReadiness(i,t)    什么时候真正比较容易做成

于是同一个地块在第 2 年与第 8 年**不是同一个项目**，"等"才有可能占优。

## 口径声明（必须写进论文，不得含糊）

**这四个场目前全部是情景参数，没有任何实证标定。** 单元表
`zones_v0/candidate_units.csv` 只有 uid/通道/格数/row/col/压力/距路m 七列，
既没有建成年份，也没有历年规划定位或设施投用年份。因此：

* 场的**形状**（三型分布 + 逻辑斯蒂爬升）是假设；
* 场的**幅度** `A_*` 是情景参数，必须扫；
* 一切由此得出的"等待有价值"只能写成**条件结论**：
  "若上位规划与基础设施按 X 幅度在 Y 年内改善，则最优策略呈现 Z 的时序结构"，
  不得写成"实证表明深圳光明区应当推迟某些项目"。

把它替换为实证数据所需的清单见 docs/大白话_v17实验与故事线.md 末节
（历年法定图则/更新单元计划/重点片区规划 → 逐年 Planning 图；地铁与道路开通
年份、学校医院商业中心投用年份 → Infrastructure；建筑建成年份 → BuildingAge）。

## 默认关闭，且关闭时**动态逐位等价**

四个幅度默认全 0，此时 `value_mult` 恒返回 **1.0**、`hazard_mult` 恒返回
**1.0**（都是精确的浮点 1.0，乘法不改变位模式），机会场的随机数发生器与
状态机的发生器相互独立、不消耗对方的抽样序列。故幅度全 0 时环境的转移、
事件计数、Floor/Cost 与 v16 **逐位相同**，历史结果不作废。

唯一不可逐位继承的是**网络权重**：观测新增 4 维（N_FEAT 22 -> 26）改变了
第一层参数量，torch 的初始化抽样序列随之改变。这与 v15 的 16 -> 22 是同一
类情况，处理方式相同 —— v16 及更早的权重不可载入，跨批次不可混在一张图里比。
v17 的结论一律由**批次内** 2×2（场开/关 × 观测开/关）给出。

## 与 FAR_GROWTH 的关系

两者并存且语义不同：`FAR_GROWTH` 是全局同步的容积率上浮（"整体政策红利"），
本模块是逐单元异步的机会改善（"这一块什么时候轮到它"）。两者相乘作用在
交付价值上，互不替代。默认 `FAR_GROWTH=0`，故 v17 主组的非平稳性全部来自本模块。
"""
import numpy as np

# ---------------------------------------------------------------- 幅度（情景参数）
# 四个场各自对结果的作用强度。**默认全 0 = 关闭**，此时环境与 v16 动态逐位等价。
#
# 幅度的含义：
#   A_PLAN  上位规划支持度对**立项可及性** P_init(i,t) 的调制幅度（读当年）。
#           **v18 起走"能不能报上去"这条通道，不再乘交付价值，也不并进批准
#           hazard。** 这是本版最重要的机制改动，也是与建议 §9 对应的那一格：
#           规划决定的是"这一年这个地块值不值得/能不能申报"——法定图则尚未
#           覆盖、不在更新单元计划名单里，报上去本就进不了流程；进了重点片区，
#           它才真正进入候选集。
#           实现成**逐年逐单元的随机可及性**而不是硬阈值：
#               可报 ~ Bernoulli(P_ADMIT_LO + (1−P_ADMIT_LO)·Plan(i,t))，
#           幅度由 A_PLAN 在 [1, P_ADMIT_LO] 之间插值。规划支持只是**提高**
#           它被纳入的机会，不保证纳入，也不禁止低支持度的地块被纳入——
#           与 Liang et al. (2018, IJGIS) 的规划发展区"提高发展概率而非强制
#           发展"同构，也正是建议 §12 强调的那一点。
#   A_READY 实施条件对**批准风险率** P_approve(i,t) 的调制幅度（读当年）。
#           报上去之后能不能在有效期内批下来、推得动，是另一件事，故与上一条
#           分属两条通道。
#   A_AGE   更新必要性 Need(i,t) 的幅度。**与建议的一处明示偏离**：建议把
#           Need 列为独立的第三种机制，但本环境里"必要性"没有独立的作用位——
#           它既不是准入（那是法定图则的事），也不改变客观收益（面积不会因为
#           房子旧而变多）。故这里把它并入**批准风险率**：越必要的项目在审批
#           与各方协调中越顺利。这是实现上的合并，不是建议的原意，凡引用
#           §9 的地方都按此标注，不得写成"完全照表实现"。
#   A_INFRA 周边设施成熟度对**交付价值**的加成（读**建成年**取值）。
#           **v18 起这是唯一保留的价值通道**，因为它是四者中最好辩护的一条：
#           价值在交付时点实现——地铁通了、学校医院商业到位了，同样的房子才
#           值那个钱。其余三者若也乘在 Floor 上，Floor 就不再是计容建筑面积，
#           而审稿人必然会问那个乘数是哪来的（v17 的写法确实有这个问题）。
#
# 于是四个因素按**阶段**分工，而不是一起乘价值（建议 §9 的表）：
#     上位规划   → 什么时候值得申报       概率通道
#     建筑老化   → 什么时候更新越来越必要   概率通道
#     实施条件   → 什么时候更容易推进成功   概率通道
#     基础设施   → 建成之后值不值钱        价值通道
#
# "等"因此有**两个互相独立**的理由，且都不需要改写奖励的定义：
#     等到规划与条件成熟 → 批得下来、不至于在有效期内失效（概率）
#     等到设施建成      → 交付时真正值那个钱（价值）
A_PLAN = 0.0
A_INFRA = 0.0
A_AGE = 0.0
A_READY = 0.0

# 观测开关。False 时机会场仍然作用于奖励与转移，但策略**看不到**这四维特征
# （置零而非删列，位宽与网络形状不变）。这是 v17 的第二个消融因子：
#   场开 + 观测开 = 主组
#   场开 + 观测关 = "有择时可做但看不见" —— 分离信息与机制
#   场关 + 观测开 = "看得见但没得做" —— 检验特征本身不是噪声
#   场关 + 观测关 = v16 口径
OBS_OPPORTUNITY = True

# 前瞻年数：观测里额外给出 t+FORESIGHT 年的场取值。
# 0 = 只看当期（默认）。策略仍可学"低就等、高才动"，因为当期取值本身就说明了
# 该地块此刻是否处在有利时点——不需要前瞻也能学到这条规则。
# FORESIGHT>0 对应"法定图则与设施计划已经公布、规划师看得见未来几年的安排"，
# 是一个信息量更大的档，作单独敏感性臂，不作默认。
FORESIGHT = 0

# ---------------------------------------------------------------- 场的形状（情景参数）
# 三型构成。**必须有"一直不好"这一型**：若所有地块最终都变好，"一直等"就是
# 最优策略，门槛实验会被一个退化解通过，测不出任何择时能力。
SHARE_NOW = 0.30      # 现在就好、以后不变    -> 应当早做
SHARE_RAMP = 0.40     # 现在一般、若干年后变好 -> 应当等
# 其余 1 - SHARE_NOW - SHARE_RAMP = 一直不好   -> 低优先

# 爬升起始年，按决策期 T 的**比例**给出，而不是写死年份。
# 这样 T=15 与 T=25 两档的场在"相对时序"上可比；写死年份会让 25 年档的场
# 全部堆在前三分之一，两档的差异就混进了"场被压缩"这一无关因素。
ONSET_LO_FRAC = 0.15
ONSET_HI_FRAC = 0.60
# 机会曲线的**形状**。这是 v18 最后一项、也是 P0 诊断直接要求的改动。
#   "rising" 单调爬升（v17 口径）：低 → 高 → 一直高。
#   "window" 有限机会窗：低 → 升 → **峰** → 回落。
#
# 为什么必须有"关窗"这一侧（P0 诊断的实测依据）：
# 单调场下，最优立项年只有两种可能——"立刻"或"能拖多久拖多久"。实测
# （scripts/exp_waiting_advantage.py，v18 主组 T=15）最优等待年的分布是
# 立刻 0.75 / 内点 0.12 / 拖满 0.13；把幅度加倍，WA>0 的状态占比从 0.25 升到
# 0.57，但"拖满"同时从 0.13 升到 0.31 —— 环境开始奖励**无脑延后**这个退化解。
# 也就是说单调场里"择时机会变多"与"延后被奖励"是同一件事，无法分开。
#
# 有限窗把这两件事分开：等过峰值开始亏，于是"等太久"由机会场**自然**惩罚，
# 不需要人为的 wait penalty（那种罚项一加，审稿人必问罚多少是怎么定的）。
# 窗宽取 3~7 年（WINDOW_YEARS_LO/HI），与 25 年决策期的关系是
# "全局 25 年期限 + 局部 3~7 年机会窗"，而不是靠延长期限制造择时空间。
OPP_SHAPE = "rising"
WINDOW_YEARS_LO = 3.0
WINDOW_YEARS_HI = 7.0

RAMP_YEARS = 3.0      # 逻辑斯蒂爬升的时间常数（年）。刻意不做成阶跃：
                      # 建议明确要求"不要写成 30 年以上=必须更新"这类硬阈值，
                      # 机会改善在现实中是逐渐的。

LEVEL_LOW = 0.15      # 爬升前的底水平
LEVEL_HIGH = 1.00     # 爬升后的高水平
LEVEL_NEVER = 0.10    # "一直不好"型的恒定水平

# 基础设施场的**空间相关性**：地铁开通、道路建成影响的是一片而不是一个地块，
# 故按 (row, col) 聚成若干片区，同片区共用一个投用年份。
# 聚类数是情景参数；没有 row/col 时退回逐单元独立抽样。
# **注意取整**：实现用 nb × nb 的等分位网格，nb = ceil(sqrt(k))，故实际片区数是
# nb² 而不是 k（k=6 时实际为 9）。`eff()` 报的是**实际**片区数，不是这个目标值。
INFRA_CLUSTERS = 6

# 设施的**公布**比**建成**早多少年。这一条是 Liang et al. 那套"未来规划替换"
# 机制在本模型里的落点：同一条地铁线有两个状态——
#     Infra_plan(i,t)    规划已公布、预期已形成（提前 INFRA_ANNOUNCE_LEAD 年）
#     Infra_active(i,t)  真正通车、价值可以兑现
# 观测给的是 plan 层（规划师确实看得见已公布的线路与建设计划），
# 交付价值用的是 active 层（地铁没通，房子就还没值那个钱）。
# 于是"规划先产生预期、建成以后才兑现"这句话在代码里是两条曲线而不是一句话。
# 取 4 年：轨道线路从公布建设计划到通车通常是数年量级；**情景参数**，可扫。
INFRA_ANNOUNCE_LEAD = 4.0

# 建筑老化。AGE0_* 是**基期建筑年龄**的抽样分布（无数据，情景参数）。
AGE0_MEAN = 22.0
AGE0_SD = 8.0
AGE_MID = 30.0        # 必要性 = logistic((age - AGE_MID) / AGE_SLOPE)
AGE_SLOPE = 6.0       # 平滑尺度：AGE_MID 处 0.5，±6 年约 0.27/0.73

# 实施条件场。与规划场独立抽样（现实中两者相关，但相关强度无数据，
# 故取独立并在文档中声明；需要时另设相关档）。
# 立项可及性的底水平：规划支持为 0 时仍有这么大的机会被纳入候选集。
# 不取 0：现实中没有"不在任何规划里就绝对报不上去"这回事，且取 0 会让
# A_PLAN 从"提高机会"变成"一票否决"，那正是建议 §12 反对的硬规则。
P_ADMIT_LO = 0.35

READY_BASE = 0.5      # 基期水平。取 0.5 使 A_READY 的调制在 hazard 上近似中性：
                      # hazard_mult = 1 + A_READY·(2·ready − 1)，ready=0.5 时恰为 1。

FIELD_SEED = 20260917  # 场的抽样种子。**与回合种子无关**：场代表这一片区客观的
                       # 规划与设施安排，在训练与评估的所有回合里必须是同一张图，
                       # 否则策略无法利用它，也就测不出择时能力。


def active():
    """四个幅度是否有任一非零。全 0 时环境与 v16 动态逐位等价。"""
    return bool(A_PLAN or A_INFRA or A_AGE or A_READY)


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


class OpportunityField:
    """逐单元 × 逐年的四张机会场。构造时一次算好，之后只做查表。

    形状均为 (n, T_total+1)，`T_total` 取评价期 T_eval，使尾部评价年也能取到值。
    """

    def __init__(self, n, T, T_total=None, row=None, col=None, seed=None):
        self.n = int(n)
        self.T = int(T)
        self.T_total = int(T if T_total is None else T_total)
        self.seed = FIELD_SEED if seed is None else int(seed)
        rng = np.random.default_rng(self.seed)
        H = self.T_total + 2                 # 多留一年，前瞻查表不必做边界分支
        self.years = np.arange(H, dtype=float)

        # ---- 三型划分（确定性给定 (n, seed)）----
        u = rng.random(self.n)
        self.kind = np.where(u < SHARE_NOW, 0,
                             np.where(u < SHARE_NOW + SHARE_RAMP, 1, 2))  # 0 早 1 等 2 从不

        # 逐单元机会窗宽度（仅 OPP_SHAPE="window" 时起作用）。**逐单元不同**：
        # 现实中有的地块窗口很短（一次规划调整的空档），有的较长。
        self.window = rng.uniform(WINDOW_YEARS_LO, WINDOW_YEARS_HI, self.n)

        lo_y = ONSET_LO_FRAC * self.T
        hi_y = ONSET_HI_FRAC * self.T
        # ---- 上位规划场 ----
        self.plan_onset = rng.uniform(lo_y, hi_y, self.n)
        self.plan = self._field(self.plan_onset)

        # ---- 基础设施场（按片区共享投用年份）----
        self.infra_cluster = self._clusters(row, col, rng)
        c_onset = rng.uniform(lo_y, hi_y, int(self.infra_cluster.max()) + 1)
        self.infra_onset = c_onset[self.infra_cluster]
        # 设施场不分三型：设施要么早就有、要么某年建成，没有"永远不好"的读法，
        # 故对全体单元一律用爬升曲线，差别只在投用年份的早晚。
        # 设施场：window 档下同样开窗。设施建成后周边价值确实会经历
        # "投用—成熟—相对优势被后建成的新片区稀释"这个过程，故回落有现实读法；
        # 但它比规划场更缓（窗宽 ×1.5），因为设施是存量、不像规划名额那样过期。
        if OPP_SHAPE == "window":
            ipk = self.infra_onset + self.window * 0.75
            self.infra = np.clip(LEVEL_LOW + (LEVEL_HIGH - LEVEL_LOW) * np.exp(
                -((self.years[None, :] - ipk[:, None])
                  / np.maximum(self.window[:, None] * 0.75, 1e-6)) ** 2), 0.0, 1.0)
        else:
            self.infra = np.clip(LEVEL_LOW + (LEVEL_HIGH - LEVEL_LOW)
                                 * _logistic((self.years[None, :]
                                              - self.infra_onset[:, None]) / RAMP_YEARS),
                                 0.0, 1.0)
        # 公布层：同一条曲线整体左移 INFRA_ANNOUNCE_LEAD 年。左移而不是另抽一条
        # 随机曲线，是因为"公布"与"通车"说的是同一件工程，两者必须同序——
        # 另抽一条会出现"先通车后公布"这种无意义的样本。
        if OPP_SHAPE == "window":
            ipk2 = self.infra_onset + self.window * 0.75 - INFRA_ANNOUNCE_LEAD
            self.infra_plan = np.clip(LEVEL_LOW + (LEVEL_HIGH - LEVEL_LOW) * np.exp(
                -((self.years[None, :] - ipk2[:, None])
                  / np.maximum(self.window[:, None] * 0.75, 1e-6)) ** 2), 0.0, 1.0)
        else:
            self.infra_plan = np.clip(
                LEVEL_LOW + (LEVEL_HIGH - LEVEL_LOW)
                * _logistic((self.years[None, :]
                             - (self.infra_onset[:, None] - INFRA_ANNOUNCE_LEAD))
                            / RAMP_YEARS), 0.0, 1.0)

        # ---- 建筑老化 -> 更新必要性 ----
        self.age0 = np.clip(rng.normal(AGE0_MEAN, AGE0_SD, self.n), 0.0, 80.0)
        self.age = self.age0[:, None] + self.years[None, :]
        self.necessity = _logistic((self.age - AGE_MID) / AGE_SLOPE)

        # ---- 实施条件场 ----
        self.ready_onset = rng.uniform(lo_y, hi_y, self.n)
        self.ready = np.clip(READY_BASE + (1.0 - READY_BASE)
                             * _logistic((self.years[None, :]
                                          - self.ready_onset[:, None]) / RAMP_YEARS),
                             0.0, 1.0)

    # ---- 场的构造 ----
    def _field(self, onset):
        """按三型给出 (n, H) 的场：早型恒高、等型爬升（或开窗）、从不型恒低。

        `OPP_SHAPE="window"` 时"等"型不再是爬升到高位不动，而是在 onset 之后
        升到峰值、再回落——峰值年 = onset + 窗宽/2，回落尺度 = 窗宽/2，
        故该单元的有效机会窗长度约等于 `self.window`（3~7 年）。
        """
        if OPP_SHAPE == "window":
            peak = onset + self.window / 2.0
            ramp = LEVEL_LOW + (LEVEL_HIGH - LEVEL_LOW) * np.exp(
                -((self.years[None, :] - peak[:, None])
                  / np.maximum(self.window[:, None] / 2.0, 1e-6)) ** 2)
        else:
            ramp = LEVEL_LOW + (LEVEL_HIGH - LEVEL_LOW) * _logistic(
                (self.years[None, :] - onset[:, None]) / RAMP_YEARS)
        F = np.empty_like(ramp)
        F[self.kind == 0] = LEVEL_HIGH
        F[self.kind == 1] = ramp[self.kind == 1]
        F[self.kind == 2] = LEVEL_NEVER
        return np.clip(F, 0.0, 1.0)

    def _clusters(self, row, col, rng):
        """按 (row, col) 把单元聚成 INFRA_CLUSTERS 片。无坐标时逐单元独立。

        聚类用简单的等分位网格而非 k-means：片区划分只需"空间上成块"，
        引入 sklearn 依赖与随机初始化换不来任何解释力。
        """
        k = max(int(INFRA_CLUSTERS), 1)
        if row is None or col is None or k == 1:
            return rng.integers(0, max(k, 1), self.n)
        r = np.asarray(row, float)
        c = np.asarray(col, float)
        nb = int(np.ceil(np.sqrt(k)))          # nb × nb 网格，取前 k 个非空块
        rb = np.clip(np.searchsorted(np.quantile(r, np.linspace(0, 1, nb + 1)[1:-1]), r),
                     0, nb - 1)
        cb = np.clip(np.searchsorted(np.quantile(c, np.linspace(0, 1, nb + 1)[1:-1]), c),
                     0, nb - 1)
        lab = rb * nb + cb
        _, inv = np.unique(lab, return_inverse=True)
        return inv.astype(int)

    # ---- 查表（供 env_gym / schedule 调用）----
    def _at(self, F, t):
        return F[:, int(np.clip(t, 0, F.shape[1] - 1))]

    def plan_at(self, t):
        return self._at(self.plan, t)

    def infra_at(self, t):
        """**建成**层：价值兑现用这一层。"""
        return self._at(self.infra, t)

    def infra_plan_at(self, t):
        """**公布**层：观测用这一层（规划师看得见已公布的线路与建设计划）。"""
        return self._at(self.infra_plan, t)

    def necessity_at(self, t):
        return self._at(self.necessity, t)

    def ready_at(self, t):
        return self._at(self.ready, t)

    def value_mult(self, u, t_init, t_done):
        """交付价值的时机乘子。**v18 起只含基础设施一项**，读建成年取值。

            M = 1 + A_INFRA · Infra_active(i, 建成年)

        用 `Infra_active`（真正建成通车的那一年起）而**不是** `Infra_plan`
        （规划公布年起）：规划先产生预期、建成才兑现价值。预期那一层只进观测，
        供策略判断"再等两年设施就到位了"，不进价值——否则等于认定规划一公布
        房子就值钱了。

        A_INFRA=0 时返回**精确的 1.0**。`t_init` 保留在签名里是为了兼容调用点，
        v18 起不再使用（规划已改走概率通道）。
        """
        if not A_INFRA:
            return 1.0
        i = int(u)
        j = int(np.clip(t_done, 0, self.infra.shape[1] - 1))
        return float(1.0 + A_INFRA * float(self.infra[i, j]))

    def hazard_mult(self, t):
        """逐年批准风险率的调制因子，长度 n。三个幅度全 0 时返回**精确的标量 1.0**。

            factor = (1 + A_AGE·(2·Need(i,t) − 1))
                   × (1 + A_READY·(2·Ready(i,t) − 1))

        **不含** A_PLAN：规划走的是立项可及性（能不能报上去），不是批准概率
        （报上去能不能批下来）。两者在建议 §9 里是两格，这里也是两条通道。

        每一项都在场取值 0.5 处**中性化**（取值 0.5 → 因子 1）。不做这个中性化，
        这三个参数会同时改变审批速度的**水平**与**时序**，两个效应无法分离——
        那样得到的"等待有价值"分不清是因为时机变好了，还是因为整体批准率被抬高了。

        两项**相乘**：必要性与实施条件同时到位才真正推得动，任一项拖后腿都会
        把机会拉回来。叠加上游的立项可及性，就得到"规划支持 ≠ 必须更新"——
        重点片区里的地块报上去了，若建筑还新、实施条件差，当年依然可能批不下来。

        实测力度（基准 HAZARD、有效期 3+2 年）：因子 0.44 时有效期内获批概率
        0.394，因子 1.8 时 0.918。即"等到时机成熟再报"能把项目活下来的机会
        提高一倍以上——这个量级足以让等待成为真正的决策，而不必动奖励定义。
        """
        if not (A_AGE or A_READY):
            return 1.0
        f = np.ones(self.n)
        if A_AGE:
            f *= 1.0 + A_AGE * (2.0 * self.necessity_at(t) - 1.0)
        if A_READY:
            f *= 1.0 + A_READY * (2.0 * self.ready_at(t) - 1.0)
        return np.clip(f, 0.0, None)

    def admit_prob(self, t):
        """立项可及性 P_init(i,t) ∈ [0,1]：这一年这个地块能不能报进候选集。

        A_PLAN=0 时返回**全 1**（人人可报，与 v16 逐位等价）。
        A_PLAN=1 时为 P_ADMIT_LO + (1−P_ADMIT_LO)·Plan(i,t)；
        中间幅度在"全 1"与该式之间线性插值，故 A_PLAN 的含义是
        "规划对准入的影响有多强"，0 = 完全不影响。
        """
        if not A_PLAN:
            return np.ones(self.n)
        full = P_ADMIT_LO + (1.0 - P_ADMIT_LO) * self.plan_at(t)
        return np.clip(1.0 + A_PLAN * (full - 1.0), 0.0, 1.0)

    def admissible(self, t, rng):
        """按 `admit_prob` 抽一次，返回本年可立项的布尔掩码。

        **逐年重抽**而不是一次定死：规划支持低的地块今年报不上去，明年可能
        报得上去；一次定死就变成了永久禁入，那是硬规则。
        抽样用调用方传入的发生器（环境的状态机 rng），A_PLAN=0 时整条跳过、
        **不消耗随机数**，这是与 v16 逐位等价的前提。
        """
        if not A_PLAN:
            return np.ones(self.n, dtype=bool)
        return rng.random(self.n) < self.admit_prob(t)

    def opportunity_index(self, t):
        """动态更新机会 O(i,t) ∈ [0,1]：这个地块在这一年有多大可能真正推得动。

        合成两条概率通道并归一化到 [0,1]：

            O(i,t) = admit_prob(i,t) · clip(hazard_mult(i,t) / MAX_FACTOR, 0, 1)

        即"报得上去"× "报上去能批下来"——这正是"这个地块这一年有多大可能
        真正进入更新流程"的字面含义。

        `MAX_FACTOR` 是三项各自取满时的上界（各场取 1 时的因子），故 O 的分母
        是**构造上的常数**而不是本次抽样的样本最大值——用样本最大值会让 O 的
        含义随场种子漂移，两个批次之间就不可比了。

        这是建议 §10 里"环境每年告诉策略：A 地块当前机会 0.35、明年 0.50"的那个量。
        它不是四个场的简单平均：合成是**乘性**的，一项拖后腿就会把机会拉回来，
        这与线性平均给出的排序并不相同，故它在观测里不是冗余维。
        """
        if not (A_PLAN or A_AGE or A_READY):
            return np.zeros(self.n, dtype=float)
        hi = 1.0
        for a in (A_AGE, A_READY):
            if a:
                hi *= 1.0 + a
        hz = self.hazard_mult(t)
        hz = np.full(self.n, 1.0) if np.ndim(hz) == 0 else hz
        return np.clip(self.admit_prob(t) * np.clip(hz / hi, 0.0, 1.0), 0.0, 1.0)

    def obs_block(self, t):
        """观测用的 (n, 8) 特征块。

            0..3  当期水平：规划 / 设施（**公布**层）/ 更新必要性 / 实施条件
            4..7  **趋势**：同四项在 t+FORESIGHT 与 t 之间的差值
            8     动态更新机会 O(i,t)：概率通道三项的乘性合成，归一化到 [0,1]
            9     O 的趋势（t+FORESIGHT 与 t 之差）

        第 1 维给的是设施的**公布**层而非建成层：规划师看得见已公布的线路与
        建设计划，这正是"未来规划替换"机制要交给决策者的那段信息；价值那一侧
        仍然只认建成层（见 `value_mult`）。
        第 8/9 维不是前八维的线性组合——概率通道的合成是乘性的，一项拖后腿会
        把机会拉回来，与线性平均给出的排序不同。

        ## 为什么必须有"趋势"这四维（门槛实验的结论）

        `scripts/exp_timing_gate.py` 的实测：只给当期水平时，试过的每一档都
        没有学会等（立项年恒为"按当期水平降序、逐年填满配额"）。试过四个方向：
        奖励改到立项年计（去掉延迟）、熵系数放大 5 倍与 20 倍、迭代数加到 1000、
        机会改善从阶跃改成平滑爬升。

        **四档的失败方式并不相同**，须分别记下：改奖励时点、熵 ×5、迭代 1000
        三档与基准逐位一致（回报比 0.743），γ=0.99 档略差（0.706）；熵 ×20 档
        则是**训练崩掉**而非学会别的东西——三个种子里一个整回合不立项
        （回报比 0.000），组均值 0.491。也就是说把探索强行加大并没有换来"等"，
        只换来退化。四档的共同点只有一条：没有任何一档的立项年出现推迟。

        真正的原因是**可表示性**：策略给所有地块打分用的是同一个函数，只看当期
        水平时，"现在一般但三年后变好的地块"与"一直不好的地块"在观测上无法区分
        （实测前者的当期水平甚至更高）。于是"留着前者、先做后者"这个最优解根本
        不在可表示的策略类里，与优化器好坏无关。

        给未来水平也不够——那只是把同一个贪心排序整体平移 K 年。**差值**才是
        "会不会变好"的直接答案，也才是"再等一年是否划得来"的可比量。

        这一段信息在现实中是**有出处**的：法定图则、城市更新单元计划、重点片区
        规划、地铁与道路建设计划都是提前公布的，规划师确实看得见未来几年的安排。
        FORESIGHT=0 档因此不是"更诚实的设定"，而是一个**信息消融**：它对应
        "假装看不见已公布的规划"，用来证明这段信息是主判据的载体。

        两个开关是**独立**的：幅度全 0 而观测开着，得到的是"看得见但没得做"
        这一档（用于检验这八维本身不是噪声）；要复现 v16 的观测口径，
        必须**同时**把幅度置 0 并关掉本开关。
        """
        B = np.zeros((self.n, 10), dtype=np.float32)
        if not OBS_OPPORTUNITY:
            return B
        t0 = int(t)
        B[:, 0] = self.plan_at(t0)
        B[:, 1] = self.infra_plan_at(t0)      # 公布层，不是建成层
        B[:, 2] = self.necessity_at(t0)
        B[:, 3] = self.ready_at(t0)
        B[:, 8] = self.opportunity_index(t0)
        if FORESIGHT:
            t1 = t0 + int(FORESIGHT)
            B[:, 4] = self.plan_at(t1) - B[:, 0]
            B[:, 5] = self.infra_plan_at(t1) - B[:, 1]
            B[:, 6] = self.necessity_at(t1) - B[:, 2]
            B[:, 7] = self.ready_at(t1) - B[:, 3]
            B[:, 9] = self.opportunity_index(t1) - B[:, 8]
        return B

    # ---- 自证 ----
    def eff(self):
        """生效值快照，供各 run 的 diag 落盘（记生效值而非命令行意图）。"""
        return dict(opp_active=active(),
                    a_plan_eff=float(A_PLAN), a_infra_eff=float(A_INFRA),
                    a_age_eff=float(A_AGE), a_ready_eff=float(A_READY),
                    obs_opportunity_eff=bool(OBS_OPPORTUNITY),
                    foresight_eff=int(FORESIGHT),
                    field_seed_eff=int(self.seed),
                    share_now_eff=float(SHARE_NOW), share_ramp_eff=float(SHARE_RAMP),
                    onset_lo_eff=float(ONSET_LO_FRAC * self.T),
                    onset_hi_eff=float(ONSET_HI_FRAC * self.T),
                    ramp_years_eff=float(RAMP_YEARS),
                    infra_clusters_eff=int(self.infra_cluster.max()) + 1,
                    infra_announce_lead_eff=float(INFRA_ANNOUNCE_LEAD),
                    opp_shape_eff=str(OPP_SHAPE),
                    window_years_eff=(float(self.window.mean())
                                      if OPP_SHAPE == "window" else None),
                    n_kind_now=int((self.kind == 0).sum()),
                    n_kind_ramp=int((self.kind == 1).sum()),
                    n_kind_never=int((self.kind == 2).sum()))

    def summary(self):
        """人读的一行摘要。"""
        return (f"机会场 {'开' if active() else '关'}  "
                f"幅度 P={A_PLAN:g}/I={A_INFRA:g}/B={A_AGE:g}/R={A_READY:g}  "
                f"观测{'开' if OBS_OPPORTUNITY else '关（置零）'}  前瞻 {FORESIGHT} 年  "
                f"三型 早{int((self.kind == 0).sum())}/等{int((self.kind == 1).sum())}/"
                f"从不{int((self.kind == 2).sum())}  片区 {int(self.infra_cluster.max()) + 1}  "
                f"种子 {self.seed}")


def tag():
    """机会场的短标识，进入分母缓存键。全关时返回空串，既有缓存键逐字不变。

    **必须入键**：四个场都改变可达上界（价值乘子改变 Floor 的上界，实施条件
    改变有多少立项能走到完工），漏入键会让对照组静默复用基线分母——本项目
    已经因为 FAR_GROWTH 漏入键栽过一次（见 scale.reference_returns 的说明）。

    观测开关 `OBS_OPPORTUNITY` 与前瞻 `FORESIGHT` **刻意不入键**：参考策略是
    手写规则、不读观测，两档的可达上界按构造相同，共用分母才能让标量化回报
    直接相减。这与 v15 的 OBS_LIFECYCLE 同一处理。
    """
    if not active():
        return ""
    s = "O"
    for k, v in (("P", A_PLAN), ("I", A_INFRA), ("B", A_AGE), ("R", A_READY)):
        if v:
            s += f"{k}{v:g}".replace(".", "")
    if FIELD_SEED != 20260917:
        s += f"S{FIELD_SEED}"
    # 场的形状参数也入键：改了三型构成或爬升时间常数，可达上界同样变
    if (SHARE_NOW, SHARE_RAMP) != (0.30, 0.40):
        s += f"H{SHARE_NOW:g}-{SHARE_RAMP:g}".replace(".", "")
    if OPP_SHAPE == "window":
        # 形状改变可达上界（关窗后晚立项的价值更低），**必须入键**
        s += f"W{WINDOW_YEARS_LO:g}-{WINDOW_YEARS_HI:g}".replace(".", "")
    if INFRA_ANNOUNCE_LEAD != 4.0:
        s += f"L{INFRA_ANNOUNCE_LEAD:g}".replace(".", "")
    if (ONSET_LO_FRAC, ONSET_HI_FRAC, RAMP_YEARS) != (0.15, 0.60, 3.0):
        s += f"N{ONSET_LO_FRAC:g}-{ONSET_HI_FRAC:g}-{RAMP_YEARS:g}".replace(".", "")
    return s
