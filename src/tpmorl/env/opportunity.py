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
#   A_PLAN  上位规划支持度对**交付价值**的最大加成（读立项年取值）。
#           读立项年而非建成年，对应"申报时点的规划定位决定了批得下来多少容积率
#           与多少价值捕获空间"这一读法。
#   A_INFRA 周边设施成熟度对**交付价值**的最大加成（读建成年取值）。
#           读建成年，对应"价值在交付时点实现——地铁通了这片房子才值钱"。
#   A_AGE   建筑老化带来的**更新必要性**对交付价值的最大加成（读建成年取值）。
#   A_READY 实施条件对**逐年批准风险率**的调制幅度（读当年取值），
#           这一条不走价值通道而走**概率通道**：条件好的年份更容易批得下来、
#           因而更不容易失效。它给出的是与价值无关的第二类"等的理由"。
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

# 建筑老化。AGE0_* 是**基期建筑年龄**的抽样分布（无数据，情景参数）。
AGE0_MEAN = 22.0
AGE0_SD = 8.0
AGE_MID = 30.0        # 必要性 = logistic((age - AGE_MID) / AGE_SLOPE)
AGE_SLOPE = 6.0       # 平滑尺度：AGE_MID 处 0.5，±6 年约 0.27/0.73

# 实施条件场。与规划场独立抽样（现实中两者相关，但相关强度无数据，
# 故取独立并在文档中声明；需要时另设相关档）。
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
        self.infra = np.clip(LEVEL_LOW + (LEVEL_HIGH - LEVEL_LOW)
                             * _logistic((self.years[None, :]
                                          - self.infra_onset[:, None]) / RAMP_YEARS),
                             0.0, 1.0)

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
        """按三型给出 (n, H) 的场：早型恒高、等型爬升、从不型恒低。"""
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
        return self._at(self.infra, t)

    def necessity_at(self, t):
        return self._at(self.necessity, t)

    def ready_at(self, t):
        return self._at(self.ready, t)

    def value_mult(self, u, t_init, t_done):
        """交付价值的时机乘子。幅度全 0 时**恒等于精确的 1.0**。

        M = (1 + A_PLAN·Plan(i, 立项年))
          × (1 + A_INFRA·Infra(i, 建成年))
          × (1 + A_AGE·Necessity(i, 建成年))

        三项相乘而非相加：三者是同一个项目在不同维度上的"时机好不好"，
        相加会让任一项单独变好就足以拉满，相乘要求它们**同时**到位，
        这才是"等到规划、设施、必要性都成熟"的那个读法。
        """
        if not active():
            return 1.0
        i = int(u)
        m = 1.0
        if A_PLAN:
            m *= 1.0 + A_PLAN * float(self.plan[i, int(np.clip(t_init, 0, self.plan.shape[1] - 1))])
        if A_INFRA:
            m *= 1.0 + A_INFRA * float(self.infra[i, int(np.clip(t_done, 0, self.infra.shape[1] - 1))])
        if A_AGE:
            m *= 1.0 + A_AGE * float(self.necessity[i, int(np.clip(t_done, 0, self.necessity.shape[1] - 1))])
        return float(m)

    def hazard_mult(self, t):
        """逐年批准风险率的调制因子，长度 n。A_READY=0 时**恒为精确的 1.0**。

        factor = 1 + A_READY·(2·Ready(i,t) − 1)
        Ready=0.5（基期水平）时恰为 1，故本参数在基期是中性的：它改变的是
        "条件成熟之后更容易批"，而不是整体抬高或压低批准率。若不做这个中性化，
        A_READY 会同时改变审批速度的**水平**与**时序**，两个效应无法分离。
        """
        if not A_READY:
            return 1.0
        return np.clip(1.0 + A_READY * (2.0 * self.ready_at(t) - 1.0), 0.0, None)

    def obs_block(self, t):
        """观测用的 (n, 8) 特征块。

            0..3  当期水平：规划 / 设施 / 更新必要性 / 实施条件
            4..7  **趋势**：同四项在 t+FORESIGHT 与 t 之间的差值

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
        B = np.zeros((self.n, 8), dtype=np.float32)
        if not OBS_OPPORTUNITY:
            return B
        t0 = int(t)
        B[:, 0] = self.plan_at(t0)
        B[:, 1] = self.infra_at(t0)
        B[:, 2] = self.necessity_at(t0)
        B[:, 3] = self.ready_at(t0)
        if FORESIGHT:
            t1 = t0 + int(FORESIGHT)
            B[:, 4] = self.plan_at(t1) - B[:, 0]
            B[:, 5] = self.infra_at(t1) - B[:, 1]
            B[:, 6] = self.necessity_at(t1) - B[:, 2]
            B[:, 7] = self.ready_at(t1) - B[:, 3]
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
    if (ONSET_LO_FRAC, ONSET_HI_FRAC, RAMP_YEARS) != (0.15, 0.60, 3.0):
        s += f"N{ONSET_LO_FRAC:g}-{ONSET_HI_FRAC:g}-{RAMP_YEARS:g}".replace(".", "")
    return s
