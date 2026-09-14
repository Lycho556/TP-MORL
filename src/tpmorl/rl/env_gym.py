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

from tpmorl.env.schedule import RenewalSchedule, QUOTA, S1, S2, S3, S4, S5
from tpmorl.objectives.reward import Reward, FAR_CAP, OBJ_NAMES, SIGN, CELL_COST
from tpmorl.objectives.run_reward_demo import load, pick_target, ALLOWED

# v15：16 -> 22。新增 16..21 = 剩余有效年 / 剩余建设年 / 期望交付年数 / slack /
# 期内可交付标志 / 在建管道占比。**改变网络输入维度**，v14 及更早的权重不可载入，
# 两批结果亦不可混在一张图里比较（见 docs/建议条目审计与改动清单_v15.md 第三节）。
N_FEAT = 22
N_PAIR_FEAT = N_FEAT + 12 + 2 + 3 + 1  # +本对成本/预算, +当前可用预算/年度额度, +是否为「到此为止」, +已承诺占用/年度额度
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

# 资金口径（v15 新增）。"upfront" = 立项当年全额支付，与 v14 及更早**逐位等价**；
# "staged" = 立项当年付前期款 STAGE_INIT，余额在该单元实施期（S3）内按年等额支付。
#
# 两种口径下「可用预算 self.budget」的语义都统一为**已拨付未承诺**（可承诺额度），
# 而非"账上现金"。这样做的关键理由：立项即锁定全额，后续年份的进度款不可能付不出，
# 因此不需要在掩码里做跨年现金流可行性检验——`cost <= budget` 这一条硬掩码即足以
# 保证不出现"开工后断供"。已承诺未支付额另记于 self.committed。
#
# Cost 目标的计入时点随口径走，二者**按构造一致**：upfront 记在完工年（旧行为），
# staged 记在实际支付年。绝不允许预算按分期走而 Cost 仍按完工年记——那正是
# docs/跨期结转_口径修正_v1.md 记录过的那类口径错位。
BUDGET_MODE = "upfront"
STAGE_INIT = 0.2       # staged 口径下立项当年支付的比例（前期/拆迁启动费）

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


GAMMA = 0.95      # 年度折现率。情景参数，由 scenario.apply() 改写。
                  # 对照：财政部《建设项目经济评价方法与参数》社会折现率 8% → 0.926


class RenewalEnv:
    def __init__(self, ds, T=15, weights=None, scale=None, seed=0, gamma=None,
                 T_eval=None):
        """T = 决策期（可立项、有预算到账的年数）；T_eval = 评价期（推进到管道排空）。

        T_eval 默认等于 T，即旧的闭区间口径——保证既有结果可复现。
        传 T_eval > T 时启用「立项与记分分离」：第 T 年起不再立项、不再进钱，
        仅推进状态机并结算建成年释放的目标，直到 t == T_eval。
        依据与推导见 docs/跨期结转_口径修正_v1.md。
        """
        (self.LU0, self.cls, self.road, self.water, self.inside,
         self.uid, self.U, self.UUM, self.CM, self.CCM) = load(ds)
        self.T, self.seed = T, seed
        self.T_eval = int(T if T_eval is None else T_eval)
        if self.T_eval < self.T:
            raise ValueError(f"T_eval({self.T_eval}) 不得小于决策期 T({self.T})")
        # gamma=None 表示取模块常量：子进程 import 后由 scenario.apply() 改写才生效
        self.gamma = GAMMA if gamma is None else float(gamma)
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
        # 「当前状态内的进度」∈[0,1]：S1 用有效期倒计时 tau/tau_max，其余状态用
        # clock/该状态上限（S3 的上限是**该单元的** build_years）。
        # 原先一律除以 TAU_EXT，既漏掉了 S1 的倒计时，也会在 build_years 分档
        # 取 5 之后给出 >1 的特征值。
        e = self.env
        prog = e.clock / e.max_clock()
        s1 = e.sigma == S1
        prog[s1] = e.tau[s1] / max(e.tau_max, 1)
        F[:, 12] = np.clip(prog, 0.0, 1.0)
        F[:, 13] = self.farcap / 10.0
        # 除以决策期 T（不是 T_eval）：该特征的语义是「决策期内的进度」，
        # 且必须与 T_eval=T 的旧口径逐值一致。尾部 t>=T 时钳到 1.0，
        # 否则会喂给策略网络训练时从未见过的 >1 取值。
        F[:, 14] = min(self.t / self.T, 1.0)
        F[:, 15] = self.mask_init.astype(np.float32)

        # ---- 生命周期剩余时间与可交付性（v15 新增，见 docs/建议条目审计与改动清单_v15.md）----
        # F[:,12] 给的是「当前状态内已用比例」，是**已用**量；策略要判断"现在立项还赶得上吗"
        # 需要的是**剩余**量，且必须是**逐单元**的——F[:,14] 的 t/T 是全局的，无法区分
        # 建设年限不同的单元。这是本次真正新增的信息。
        F[:, 16] = e.remaining_valid() / max(e.tau_max, 1)
        maxb = max(float(e.build_years.max()), 1.0)
        F[:, 17] = e.remaining_build() / (1.0 + maxb)
        # 期望交付所需年数：S0 候选单元按「本年立项」估（E[离开S1]+开工1年+建设年限），
        # 非候选的 S0/S5 为 inf（本年不可立项），归一化后钳到 1.0 表示"够不着"。
        ey = e.expected_years_to_delivery(if_initiated_now=True)
        F[:, 18] = np.clip(ey / (2.0 * self.T), 0.0, 1.0)
        # slack = 决策期还剩的年数 − 期望交付所需年数。>0 表示期内可交付。
        # 除以 T 归一并钳到 [-1,1]；inf 自然落到 -1。
        slack = (self.T - 1 - self.t) - ey
        F[:, 19] = np.clip(slack / self.T, -1.0, 1.0)
        F[:, 20] = (slack >= 0).astype(np.float32)
        # 全局状态：在建管道占比（S1/S2/S3）。这一项**随年份变化**，与配额、α 不同
        # （后两者在单次运行内恒定，逐 α 独立训练时作为特征不提供任何信息，故不加）。
        F[:, 21] = float(np.isin(e.sigma, (S1, S2, S3)).sum()) / max(self.n, 1)
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
        rows[:n, N_FEAT + 17] = self.committed / BUDGET
        # 末行「到此为止」：把余额留到明年。特征只带时间进度与余额，成本为 0，永远可选。
        # t/T 必须与 obs() 的 F[:,14] 同样钳到 1：尾部年份 t>=T 时不钳会给出 >1，
        # 与其余各行的量纲不一致（旧实现漏钳，v15 修正）。
        rows[n, 14] = min(self.t / self.T, 1.0)
        rows[n, N_FEAT + 15] = self.budget / BUDGET
        rows[n, N_FEAT + 16] = 1.0
        rows[n, N_FEAT + 17] = self.committed / BUDGET
        meta = list(zip(pu.tolist(), pt.tolist())) + [STOP]
        # units 与 meta 同序，供掩码做向量化的「该单元今年已选」判断
        units = np.concatenate([pu, np.array([-1], dtype=np.int64)])
        return rows, meta, np.concatenate([cost, [0.0]]), units

    def reset(self, seed=None):
        self.LU = self.LU0.copy()
        self.res_map = (self.LU * self.UUM[0]).sum(-1)
        # base_lu 传 LU0（基期用地），不是 self.LU：Aec 的分母必须是常数场，
        # 否则"拆居住"会机械抬高 Aec。见 Reward.spatial 的口径变更说明。
        self.R = Reward(self.UUM, self.CM, self.CCM, self.road, self.water,
                        self.inside, self.LU0, gamma=self.gamma)
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
        self.budget = BUDGET            # 已拨付未承诺（可承诺额度）
        self.committed = 0.0            # 已承诺未支付
        self._owe_year = {}             # 单元 -> 实施期内每年应付额
        self._owe_left = {}             # 单元 -> 剩余未付额
        self.spent_hist, self.budget_hist = [], []
        self.disb_hist, self.committed_hist = [], []
        return self.obs()

    def step(self, actions):
        """actions: [(单元, 目标功能), ...]，至多 QUOTA 个；STOP 项被忽略。"""
        actions = [(int(u), int(tg)) for u, tg in actions if int(u) >= 0]
        tail = self.t >= self.T          # 尾部评价年：不立项、不进钱
        if tail and actions:
            raise AssertionError(
                f"第 {self.t} 年已超出决策期 T={self.T}，不得立项（收到 {len(actions)} 个动作）")
        # spent = 本年新增**承诺**额（两种口径下都是全额，掩码据此硬约束）
        spent = sum(self.pair_cost(u, tg) for u, tg in actions)
        assert spent <= self.budget + 1e-6, f"超预算 {spent:.0f} > {self.budget:.0f}"
        self.budget_hist.append(self.budget); self.spent_hist.append(spent)
        self.committed_hist.append(self.committed)
        for u, tg in actions:
            self.plan[int(u)] = int(tg)
        prev = self.env.sigma.copy()
        ma = self.env.mask_advance()
        _, ev = self.env.step(initiate=[u for u, _ in actions], advance=np.where(ma)[0])

        # ---- 支付与承诺的分离（v15）----
        if BUDGET_MODE == "staged":
            disb = 0.0
            for u, tg in actions:                      # 立项当年的前期款
                c = self.pair_cost(u, tg)
                init = STAGE_INIT * c
                disb += init
                by = max(int(self.env.build_years[int(u)]), 1)
                self._owe_left[int(u)] = c - init
                self._owe_year[int(u)] = (c - init) / by
            # 实施中单元的当期进度款：以 env.step 之后处于 S3 为准，与 Disrupt
            # 的"S3 才产生施工干扰"同一判据，两个口径不会错位一年。
            for u in np.where(self.env.sigma == S3)[0]:
                left = self._owe_left.get(int(u), 0.0)
                if left > 1e-9:
                    pay = min(self._owe_year[int(u)], left)
                    self._owe_left[int(u)] = left - pay
                    disb += pay
            # 有效期届满被撤（S1->S5）：剩余承诺释放回可承诺额度，已付前期款沉没
            released = 0.0
            for u in np.where((prev == S1) & (self.env.sigma == S5))[0]:
                released += self._owe_left.pop(int(u), 0.0)
                self._owe_year.pop(int(u), None)
            self.committed += spent - disb - released
            assert self.committed > -1e-6, f"承诺额为负 {self.committed:.3f}"
            assert abs(self.committed - sum(self._owe_left.values())) < 1e-3, \
                "承诺额与逐单元未付额台账不一致"
        else:
            disb, released = spent, 0.0
        self.disb_hist.append(disb)
        # 年度预算只在决策期内到账；尾部余额冻结（既不进钱也无处可花）
        if not tail:
            self.budget = min(self.budget - spent + released + BUDGET,
                              CARRY_CAP * BUDGET)

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

        # Cost 的计入时点必须与资金口径一致（见模块顶部 BUDGET_MODE 说明）：
        # staged 下记当年实际支付额，upfront 下沿用完工年全额（旧行为，逐位等价）。
        # 两式的总额按构造相等——convert_cost 与 pair_cost/PC 同式。
        if BUDGET_MODE == "staged":
            cost = disb

        dis = self.R.disrupt(self.env.sigma, self.uid, self.res_map)
        rv = self.R.step_reward(self.LU, floor, cost, dis, ev["expired"])
        vec = np.array([SIGN[k] * rv[k] for k in OBJ_NAMES], float) / self.scale
        r = float((self.weights / self.weights.sum() * vec).sum())

        self.t += 1
        # 决策期结束后立项掩码强制全关：pairs() 因此只剩 STOP 行，
        # 采样器与随机基线都无需改动即可在尾部自然「无动作」。
        self.mask_init = (self.env.mask_initiate() if self.t < self.T
                          else np.zeros(self.n, dtype=bool))
        return self.obs(), r, self.t >= self.T_eval, dict(vec=vec, raw=rv, events=ev)
