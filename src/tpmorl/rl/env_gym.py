"""env_gym.py — 把时序状态机 + 多目标奖励包成一个 RL 环境。

一个 env step = 一年。动作 = 当年从合规单元中选出至多 QUOTA 个立项（自回归采样）。

动作是 **(单元, 目标功能) 对**，而非只选单元。

一开始曾把目标功能交给与基线相同的 myopic 规则（通道允许集内取 Floor-Cost 最优），
但那个规则是**空洞的**：容积率上限只由通道决定，同一单元各目标的 Floor 完全相同，
于是规则退化为"取转换成本最低者"＝原类重建，转换成本恒为 0、空间目标增量恒为 0
（实测评估中 Gdp/Eco/Cost 全为 0）。因此把目标功能并入动作空间由策略决定。
若要恢复"仅时序"的作用域，需要按目标类别而非通道标定容积率上限，
而已批单元规划表中没有规划功能字段，无法标定。
"""
import os
import numpy as np, pandas as pd

from tpmorl.env.schedule import RenewalSchedule, QUOTA, S3, S4, TAU_EXT
from tpmorl.objectives.reward import Reward, FAR_CAP, OBJ_NAMES, SIGN, CELL_COST
from tpmorl.objectives.run_reward_demo import load, pick_target, ALLOWED

N_FEAT = 16
N_PAIR_FEAT = N_FEAT + 12 + 2 + 3     # +本对成本/预算, +当前可用预算/年度额度, +是否为「到此为止」
CH_ORDER = [1, 2, 3, 5]

# ---------------------------------------------------------------- 年度资金约束
# 计量口径与奖励里的 Cost 目标完全相同（CCM[from,to] × 格数），不引入新的标定量。
# 实测配对成本：P25=300 中位=390 P90=600 最大=1160；满配额 3 个约需 1050。
# BUDGET 是**情景参数**，不是标定值——必须扫。CARRY_CAP 决定能攒几年。
#
# 口径修正（v1）：只用 CCM 计资金是错的。CCM 对角线为 0，于是「原类重建」不花钱
# 却照样计入交付建面——策略可以用零成本动作填满配额，预算永远不咬。
# 现实里拆迁补偿主要与**拆除面积**成正比，改成什么用途只影响附加的改造成本。
# 故 资金成本 = CELL_COST × 格数 + Σ CCM[现状,目标] × 格数。
# CELL_COST 定义在 objectives/reward.py（顶部已导入），由 convert_cost 与本文件的
# PC 共用——v1 只把它加进了预算路径而漏掉 Cost 目标，现已统一。
BUDGET = 900.0
# 结转上限须 ≥ 最贵单元 / BUDGET，否则大单元永远不可达（实测最贵 2610，900×3=2700）。
CARRY_CAP = 3.0        # 可用预算上限 = CARRY_CAP × BUDGET，超出部分作废（防止无限攒钱）

# ---------------------------------------------------------------- 非平稳性
# 实测结论：单靠资金约束**不能**让「等待」成为最优决策。原因是机会集平稳——
# 预算结转只改变「能做多少」，不改变「何时做更好」；成本与收益都与面积线性，
# 无规模报酬，故攒钱无回报，且明年的候选集不优于今年，等待永不严格占优。
# 等待有价值当且仅当未来的机会集优于现在，即环境必须非平稳。
#
# FAR_GROWTH 就是这个非平稳通道：容积率上限逐年上浮（对应政策红利预期），
# 于是推迟立项可换取更高的交付建面。**这是未标定的情景参数**，默认 0（关闭）；
# 待取得深圳历年更新单元规划容积率上限的时间序列后按实测替换，不得当作实证结果报告。
FAR_GROWTH = 0.0
STOP = (-1, -1)        # 「今年到此为止」动作；没有它，只要付得起就必须花，等待不成为决策


class RenewalEnv:
    def __init__(self, ds, T=15, weights=None, scale=None, seed=0, gamma=0.95):
        (self.LU0, self.cls, self.road, self.water, self.inside,
         self.uid, self.U, self.UUM, self.CM, self.CCM) = load(ds)
        self.T, self.seed, self.gamma = T, seed, gamma
        self.n = len(self.U)
        self.ch = self.U["ch_code"].values
        # 单元格掩码只算一次：717 个单元各一张 171x161 布尔图，每回合重算会主导耗时
        self.cells = [self.uid == u for u in self.U["uid"].values]
        self.ncell = np.array([int(c.sum()) for c in self.cells])
        self.press = self.U["压力"].values.astype(float)
        self.press = self.press / max(self.press.max(), 1e-9)
        self.farcap = np.array([FAR_CAP.get(int(c), FAR_CAP[5]) for c in self.ch])
        self.weights = np.ones(len(OBJ_NAMES)) if weights is None else np.asarray(weights, float)
        self.scale = np.ones(len(OBJ_NAMES)) if scale is None else np.asarray(scale, float)

    def obs(self):
        F = np.zeros((self.n, N_FEAT), dtype=np.float32)
        F[:, 0] = self.press
        F[:, 1] = self.ncell / max(self.ncell.max(), 1)
        for j, c in enumerate(CH_ORDER):
            F[:, 2 + j] = (self.ch == c)
        for s in range(6):
            F[:, 6 + s] = (self.env.sigma == s)
        F[:, 12] = self.env.clock / max(TAU_EXT, 1)
        F[:, 13] = self.farcap / 10.0
        F[:, 14] = self.t / self.T
        F[:, 15] = self.mask_init.astype(np.float32)
        return F

    def pair_cost(self, u, tg):
        """立项该 (单元, 目标) 需占用的资金，与奖励里的 Cost 目标同口径。"""
        return float(self.PC[int(u), int(tg)])

    def pairs(self):
        """当年所有 (合规单元, 通道允许的目标功能) 组合，末行恒为「到此为止」。

        返回 (F, meta, cost)。cost 供策略在配额内逐次采样时做资金可行性掩码，
        因此资金约束与制度约束一样是**硬的**——不靠罚项软化。
        """
        F = self.obs()
        sel = self.mask_init[self._pu_all]
        pu, pt = self._pu_all[sel], self._pt_all[sel]
        cost = self._cost_all[sel]
        n = pu.size
        rows = np.zeros((n + 1, N_PAIR_FEAT), dtype=np.float32)
        rows[:n, :N_FEAT] = F[pu]
        rows[np.arange(n), N_FEAT + pt] = 1.0
        rows[:n, N_FEAT + 12] = cost / np.maximum(self.ncell[pu], 1) / 100.0
        rows[:n, N_FEAT + 13] = self.farcap[pu] / 10.0
        rows[:n, N_FEAT + 14] = cost / BUDGET
        rows[:n, N_FEAT + 15] = self.budget / BUDGET
        # 末行「到此为止」：把余额留到明年。特征只带时间进度与余额，成本为 0，永远可选。
        rows[n, 14] = self.t / self.T
        rows[n, N_FEAT + 15] = self.budget / BUDGET
        rows[n, N_FEAT + 16] = 1.0
        meta = list(zip(pu.tolist(), pt.tolist())) + [STOP]
        # units 与 meta 同序，供掩码做向量化的「该单元今年已选」判断
        units = np.concatenate([pu, np.array([-1], dtype=np.int64)])
        return rows, meta, np.concatenate([cost, [0.0]]), units

    def reset(self, seed=None):
        self.LU = self.LU0.copy()
        self.res_map = (self.LU * self.UUM[0]).sum(-1)
        self.R = Reward(self.UUM, self.CM, self.CCM, self.road, self.water,
                        self.inside, gamma=self.gamma)
        self.env = RenewalSchedule(self.ch, seed=self.seed if seed is None else seed)
        self.t = 0
        self.plan = {}          # 单元 -> 立项时选定的目标功能
        self.hist0 = [{k: v for k, v in
                       ((k, int((self.LU0[..., k][m] > 0).sum())) for k in range(12)) if v}
                      for m in self.cells]
        # (单元 × 目标) 成本矩阵：只依赖静态的 ncell/CCM/hist0，一次算好。
        # 原先每年为 ~1985 个配对逐个调用 pair_cost，是训练的主要开销之一。
        # 注意：因此在 env 构造之后再改 CELL_COST/CCM 不会生效。
        self.PC = np.zeros((len(self.hist0), 12), dtype=np.float64)
        for u, h in enumerate(self.hist0):
            for f, c in h.items():
                self.PC[u] += self.CCM[f, :12] * c
        self.PC += CELL_COST * np.asarray(self.ncell, dtype=np.float64)[:, None]
        # 全量 (单元, 目标) 配对枚举：只依赖静态的通道归属，一次算好。
        # 每年的候选集 = 用 mask_init 在这三个数组上做布尔选择，无需重新枚举。
        ch_all = np.asarray(self.ch, dtype=int)
        cnt_all = np.array([len(ALLOWED[int(c)]) for c in ch_all], dtype=np.int64)
        self._pu_all = np.repeat(np.arange(len(ch_all), dtype=np.int64), cnt_all)
        self._pt_all = np.concatenate(
            [np.asarray(ALLOWED[int(c)], dtype=np.int64) for c in ch_all])
        self._cost_all = self.PC[self._pu_all, self._pt_all]
        self.mask_init = self.env.mask_initiate()
        self.budget = BUDGET
        self.spent_hist, self.budget_hist = [], []
        return self.obs()

    def step(self, actions):
        """actions: [(单元, 目标功能), ...]，至多 QUOTA 个；STOP 项被忽略。"""
        actions = [(int(u), int(tg)) for u, tg in actions if int(u) >= 0]
        spent = sum(self.pair_cost(u, tg) for u, tg in actions)
        assert spent <= self.budget + 1e-6, f"超预算 {spent:.0f} > {self.budget:.0f}"
        self.budget_hist.append(self.budget); self.spent_hist.append(spent)
        self.budget = min(self.budget - spent + BUDGET, CARRY_CAP * BUDGET)
        for u, tg in actions:
            self.plan[int(u)] = int(tg)
        prev = self.env.sigma.copy()
        ma = self.env.mask_advance()
        _, ev = self.env.step(initiate=[u for u, _ in actions], advance=np.where(ma)[0])

        done_u = np.where((prev == S3) & (self.env.sigma == S4))[0]
        floor = cost = 0.0
        for u in done_u:
            m = self.cells[u]
            hist = self.hist0[u]
            tgt = self.plan.get(int(u))
            if tgt is None:      # 保险：无记录时退回 myopic 规则
                tgt = pick_target(self.R, self.ch[u], hist, self.ncell[u])
            # 交付建面按**建成年**的容积率上限计（FAR_GROWTH>0 时推迟立项可换更高上限）
            floor += self.R.floor_area(self.ch[u], self.ncell[u]) * (1.0 + FAR_GROWTH) ** self.t
            cost += sum(self.R.convert_cost(f, tgt, c) for f, c in hist.items())
            self.LU[m] = 0.0
            self.LU[..., tgt][m] = 1.0

        dis = self.R.disrupt(self.env.sigma, self.uid, self.res_map)
        rv = self.R.step_reward(self.LU, floor, cost, dis, ev["expired"])
        vec = np.array([SIGN[k] * rv[k] for k in OBJ_NAMES], float) / self.scale
        r = float((self.weights / self.weights.sum() * vec).sum())

        self.t += 1
        self.mask_init = self.env.mask_initiate()
        return self.obs(), r, self.t >= self.T, dict(vec=vec, raw=rv, events=ev)
