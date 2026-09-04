"""schedule.py — 城市更新的制度时序状态机与年度动作掩码。

把"计划有效期 / 法定顺位 / 年度配额"编码成 MDP 的转移与掩码，使 agent 必须回答
"这个单元现在动、还是再等"，而不是"这块地理想上该是什么功能"。

状态（每个候选更新单元一个）
    S0 未立项      -> agent 可选择立项（受年度配额约束）
    S1 计划有效期内 -> τ 年倒计时；agent 可推进（环境按 hazard 批准）或等待
    S2 单元规划已批 -> 待实施
    S3 实施中      -> 产生邻域施工干扰
    S4 已完成      -> 终止态，本回合不可再动
    S5 计划失效    -> 冷却期 N 年，期间完全不可动，之后回 S0

参数标定来源（见 docs/zoning_design.md、docs/参数依据分级表.md）
    HAZARD      109 对"计划公告→单元规划批准"配对的逐年批准风险率
                （累计获批比例 1 年 17.4%、2 年 34.9%、3 年 50.5%、4 年 64.2%、5 年 70.6%）
    TAU_VALID   3 年 —— 时滞中位数恰为 3.00 年
    TAU_EXT     2 年 —— 延期上限；τ 总上限 5 年可覆盖 70.6% 的实际案例，
                余下 29.4% 视为经"失效→重新申报"路径

                **适用期说明（必读）**：标定用的 109 对配对来自 2010–2018 年，
                当期适用深规土告〔2010〕16 号，**并无两年有效期之设**；深建规
                〔2025〕4 号要求各区对"公告未明确有效期"的存量计划补设有效期，
                正是这批旧计划原本没有有效期的直接证据。而深规划资源规〔2019〕
                4 号第三十二条（有效期 2 年 + 延期 1 年，法定上限 3 年）第一句
                即限定"**本规定施行后批准的**更新单元计划"，即 2019-03-15 之后。
                故实测中位 3 年**不是"条文与现实脱节"**，而是旧制度下本无此限，
                两者并不冲突。现行法定窗口另设情景对照（tau_valid=2, tau_ext=1）。
    QUOTA       3 个/年 —— 光明区 2011–2018 年计划公告均值 3.1、上限 6
    BUILD_YEARS 无实测数据支撑，为超参数。访谈（深大黄冠老师，2026-09-03）：
                从规划获批到建成涉及补偿、拆迁、招标、建设、验收，3 年偏短，
                拆除重建类"可能得 5 年往上"，微改造 3 年没问题，且项目间差异不小。
                据此由全局标量改为**按通道分档**，默认拆除重建类取 5 年。
                分档精度受数据限制，见 BUILD_YEARS_BY_CHANNEL 的说明。
    COOLDOWN    无实测数据支撑，**且无条文依据**——两份规范性文件全文检索
                "冷却"0 处、"重新申报"0 处，这条机制是本项目自设。访谈中老师
                明确表示"这个我确实还不太了解"，故不作为依据。必须纳入敏感性
                扫描，且 COOLDOWN=0 一档是对本项目核心主张的压力测试：
                "等待是有风险的、所以何时动才是个真决策"目前完全压在这条自设
                机制上。若 0 档下主要结论不变，说明主张另有支撑（配额竞争、
                有效期倒计时、交付折现）；若翻转，则须坦白主张依赖无依据设定。
"""
import numpy as np

S0, S1, S2, S3, S4, S5 = range(6)
SNAME = {S0: "未立项", S1: "有效期内", S2: "已批待实施", S3: "实施中", S4: "已完成", S5: "冷却期"}

# --- 实测标定 ---
HAZARD = (0.174, 0.212, 0.239, 0.277, 0.179)   # 第 1..5 个有效年的条件批准率
# 逐年由累计获批数反解：19/109, (38-19)/(109-19), (55-38)/(109-38),
#                       (70-55)/(109-55), (77-70)/(109-70)
TAU_VALID, TAU_EXT = 3, 2                      # 有效期 3 年 + 延期 2 年
QUOTA = 3                                      # 年度立项配额
# --- 无数据，超参数 ---
# 建设年限按**更新方式**分档（访谈依据见模块 docstring）。
BUILD_YEARS_BY_MODE = {"拆除重建": 5, "综合整治": 3}

# 但更新方式无法逐单元判定，只能按通道近似：
#   gm_renewal_units.csv 没有"更新方式／微改造"字段（`类型` 列只有"计划公告"
#   "规划批准"两值），且动作空间目前只有"原址转换为某功能"一种动作，没有
#   "综合整治"这一选项。故四个可更新通道**一律按拆除重建取 5 年**——
#   分档机制已就位（下面是按单元数组），但差异化尚未真正实现。
#
# 待办（两者任一到位即可逐单元取值，见 docs/访谈落实_v3.md 第 1、8 条）：
#   (a) 补齐"更新方式（拆除重建／综合整治）"字段；或
#   (b) 动作空间引入"综合整治"动作，届时按**所选动作**而非通道取值。
# 注意：不要用"目标功能 == 现状功能"来判定综合整治——原类重建仍是拆除重建，
# 与综合整治（不拆除、仅整治）是两回事，那样映射会把两者混为一谈。
BUILD_YEARS_BY_CHANNEL = {1: 5,    # P1 产业（拆除重建）
                          2: 5,    # P2 旧居住（拆除重建；综合整治档未进动作空间）
                          3: 5,    # P3 商服提升
                          5: 5}    # P5 农转用储备（实际改造方式未知，暂取 5）
BUILD_YEARS = BUILD_YEARS_BY_MODE["拆除重建"]   # 通道表缺项时的兜底
COOLDOWN = 2                                    # 无条文依据，须扫 {0,1,2,3}

ACTIONABLE_CHANNELS = (1, 2, 3, 5)             # P1 产业 / P2 旧居住 / P3 商服 / P5 农转用


class RenewalSchedule:
    """制度时序状态机。一个实例管理全部候选单元在 T 期内的状态演化。"""

    def __init__(self, channel_of_unit, quota=None, tau_max=None,
                 build_years=None, cooldown=None, hazard=None, seed=0):
        # 默认值一律在此处解析，**不写进函数签名**：签名里的默认值在函数定义时
        # 就已求值，CLI 事后改写模块常量（本项目情景参数的既有做法）不会生效，
        # 情景对照会静默失效。
        self.ch = np.asarray(channel_of_unit)
        self.n = len(self.ch)
        self.quota = QUOTA if quota is None else quota
        self.tau_max = (TAU_VALID + TAU_EXT) if tau_max is None else tau_max
        self.cooldown = COOLDOWN if cooldown is None else cooldown
        self.build_years = self._build_years_array(build_years)
        self.hazard = np.asarray(HAZARD if hazard is None else hazard, float)
        self.rng = np.random.default_rng(seed)
        self.eligible = np.isin(self.ch, ACTIONABLE_CHANNELS)   # 通道决定是否可能被更新
        self.reset()

    def _build_years_array(self, spec):
        """把 build_years 规格解析成长度 n 的按单元整数数组。

        接受三种形式：None（用模块级通道表）、标量（全体同值，兼容旧调用）、
        {通道: 年限} 字典，或已经是长度 n 的数组。
        """
        if spec is None:
            spec = BUILD_YEARS_BY_CHANNEL
        if isinstance(spec, dict):
            return np.array([int(spec.get(int(c), BUILD_YEARS)) for c in self.ch], "int16")
        arr = np.asarray(spec)
        if arr.ndim == 0:
            return np.full(self.n, int(arr), "int16")
        if len(arr) != self.n:
            raise ValueError(f"build_years 长度 {len(arr)} 与单元数 {self.n} 不符")
        return arr.astype("int16")

    def set_build_years(self, units, years):
        """按单元改写建设年限。

        为"按所选动作取值"预留：一旦动作空间引入"综合整治"，环境应在立项时
        调用本方法把该单元的年限改为 BUILD_YEARS_BY_MODE["综合整治"]。
        目前动作空间无此选项，故无调用方。
        """
        self.build_years[np.asarray(units, int)] = np.asarray(years, "int16")

    def max_clock(self):
        """各单元当前状态下 clock 的上限，供观测归一化用（逐单元）。

        S2 批准后次年开工，上限 1；S3 走到该单元的 build_years；S5 走到
        cooldown。build_years 分档后不能再用单一常数去除，否则特征会超过 1。
        """
        m = np.ones(self.n, float)
        m[self.sigma == S3] = np.maximum(self.build_years[self.sigma == S3], 1)
        m[self.sigma == S5] = max(self.cooldown, 1)
        return m

    def reset(self):
        self.t = 0
        self.sigma = np.full(self.n, S0, "int8")
        self.tau = np.zeros(self.n, "int8")        # S1 内已用年数
        self.clock = np.zeros(self.n, "int8")      # S2/S3/S5 内计时
        return self.state()

    # ---- 动作掩码：MDP 里"法规约束"的落地形式 ----
    def mask_initiate(self):
        """本期可立项的单元（S0 且通道可更新且不在冷却期）。"""
        return self.eligible & (self.sigma == S0)

    def mask_advance(self):
        """本期可推进的单元（处在计划有效期内）。"""
        return self.sigma == S1

    def quota_left(self, n_initiated_this_year=0):
        return max(0, self.quota - n_initiated_this_year)

    def state(self):
        return dict(t=self.t, sigma=self.sigma.copy(), tau=self.tau.copy(),
                    clock=self.clock.copy(), mask_initiate=self.mask_initiate(),
                    mask_advance=self.mask_advance())

    # ---- 转移 ----
    def step(self, initiate=None, advance=None):
        """initiate / advance 为单元下标数组。返回本期事件计数。"""
        initiate = np.asarray(initiate if initiate is not None else [], int)
        advance = np.asarray(advance if advance is not None else [], int)

        mi = self.mask_initiate()
        bad = initiate[~mi[initiate]] if len(initiate) else np.array([], int)
        if len(bad):
            raise ValueError(f"违反掩码：单元 {bad.tolist()} 本期不可立项")
        if len(initiate) > self.quota:
            raise ValueError(f"超出年度配额 {self.quota}：本期请求 {len(initiate)}")
        ma = self.mask_advance()
        bad = advance[~ma[advance]] if len(advance) else np.array([], int)
        if len(bad):
            raise ValueError(f"违反掩码：单元 {bad.tolist()} 不在有效期内")

        ev = dict(initiated=0, approved=0, started=0, completed=0, expired=0, released=0)

        # 冷却期 -> 未立项
        rel = (self.sigma == S5) & (self.clock >= self.cooldown)
        self.sigma[rel], self.clock[rel], ev["released"] = S0, 0, int(rel.sum())

        # 实施中 -> 已完成
        fin = (self.sigma == S3) & (self.clock >= self.build_years)
        self.sigma[fin], self.clock[fin], ev["completed"] = S4, 0, int(fin.sum())

        # 已批 -> 实施中（批准后次年开工）
        st = (self.sigma == S2) & (self.clock >= 1)
        self.sigma[st], self.clock[st], ev["started"] = S3, 0, int(st.sum())

        # 推进：环境按当年 hazard 决定是否获批
        for u in advance:
            k = min(int(self.tau[u]), len(self.hazard) - 1)
            if self.rng.random() < self.hazard[k]:
                self.sigma[u], self.clock[u] = S2, 0
                ev["approved"] += 1

        # 有效期倒计时；耗尽仍在 S1 -> 失效
        still = self.sigma == S1
        self.tau[still] += 1
        exp = still & (self.tau >= self.tau_max)
        self.sigma[exp], self.tau[exp], self.clock[exp] = S5, 0, 0
        ev["expired"] = int(exp.sum())

        # 立项
        if len(initiate):
            self.sigma[initiate], self.tau[initiate] = S1, 0
            ev["initiated"] = len(initiate)

        # 本步**新**进入冷却期的单元不参与自增：否则其 clock 在下一步已是 1，
        # 使 cooldown=0 与 =1 完全等价（生效值退化为 max(cooldown,1)），
        # 冷却期 0 档压力测试将测不到"无冷却期"。详见 docs/批次v4_查验_v5.md。
        tick = np.isin(self.sigma, (S2, S3, S5))
        tick[exp] = False
        self.clock[tick] += 1
        self.t += 1
        return self.state(), ev

    def counts(self):
        return {SNAME[s]: int((self.sigma == s).sum()) for s in range(6)}
