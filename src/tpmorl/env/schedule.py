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

参数标定来源（全部实测，见 docs/zoning_design.md）
    HAZARD      109 对"计划公告→单元规划批准"配对的逐年批准风险率
                （累计获批比例 1 年 17.4%、2 年 34.9%、3 年 50.5%、4 年 64.2%、5 年 70.6%）
    TAU_VALID   3 年 —— 时滞中位数恰为 3.00 年
    TAU_EXT     2 年 —— 延期上限；τ 总上限 5 年可覆盖 70.6% 的实际案例，
                余下 29.4% 视为经"失效→冷却→重新申报"路径
    QUOTA       3 个/年 —— 光明区 2011–2018 年计划公告均值 3.1、上限 6
    BUILD_YEARS / COOLDOWN 无实测数据支撑，为超参数，须做敏感性分析
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
BUILD_YEARS, COOLDOWN = 3, 2

ACTIONABLE_CHANNELS = (1, 2, 3, 5)             # P1 产业 / P2 旧居住 / P3 商服 / P5 农转用


class RenewalSchedule:
    """制度时序状态机。一个实例管理全部候选单元在 T 期内的状态演化。"""

    def __init__(self, channel_of_unit, quota=QUOTA, tau_max=TAU_VALID + TAU_EXT,
                 build_years=BUILD_YEARS, cooldown=COOLDOWN, hazard=HAZARD, seed=0):
        self.ch = np.asarray(channel_of_unit)
        self.n = len(self.ch)
        self.quota, self.tau_max = quota, tau_max
        self.build_years, self.cooldown = build_years, cooldown
        self.hazard = np.asarray(hazard, float)
        self.rng = np.random.default_rng(seed)
        self.eligible = np.isin(self.ch, ACTIONABLE_CHANNELS)   # 通道决定是否可能被更新
        self.reset()

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

        self.clock[np.isin(self.sigma, (S2, S3, S5))] += 1
        self.t += 1
        return self.state(), ev

    def counts(self):
        return {SNAME[s]: int((self.sigma == s).sum()) for s in range(6)}
