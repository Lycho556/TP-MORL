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

# 刻意**不**从 schedule 按值导入 QUOTA 之类会被情景改写的常量：`from X import C`
# 在导入时取值，而情景参数改写的是 schedule 模块里的那一份，于是本模块会一直用
# 旧值。实测后果：`--quota 2` 时参考策略仍按 3 个立项，被状态机的配额断言打断；
# `--quota 6` 更坏——不报错，但放宽的配额根本用不上，静默按 3 跑。
# 会被情景改写的量一律**按模块属性或实例属性**在使用时取。
from tpmorl.env.schedule import RenewalSchedule, S1, S2, S3, S4, S5
# 按模块引用（不 from-import 幅度常量）：机会场的幅度是情景参数，
# scenario.apply() 改写的是 opportunity 模块里的那一份。
from tpmorl.env import opportunity as OPP
from tpmorl.objectives.reward import Reward, FAR_CAP, OBJ_NAMES, SIGN, CELL_COST
from tpmorl.objectives.run_reward_demo import load, pick_target, ALLOWED

# v15：16 -> 22。新增 16..21 = 剩余有效年 / 剩余建设年 / 期望交付年数 / slack /
# 期内可交付标志 / 在建管道占比。**改变网络输入维度**，v14 及更早的权重不可载入，
# 两批结果亦不可混在一张图里比较（见 docs/建议条目审计与改动清单_v15.md 第三节）。
# v18：22 -> 32。新增
#   22..25 机会场**当期水平**：上位规划 / 基础设施（**公布**层）/ 更新必要性 /
#          实施条件
#   26..29 同四项的**趋势**（t+FORESIGHT 与 t 之差）
#   30     动态更新机会 O(i,t)：概率通道三项的乘性合成，归一化到 [0,1]，
#          即"这个地块这一年有多大可能真正推得动"
#   31     O 的趋势
# 即四张逐年变化的机会场（tpmorl/env/opportunity.py）。趋势那四维不是冗余：
# 门槛实验（scripts/exp_timing_gate.py）实测只给当期水平时学习器在任何超参数下
# 都学不会等，因为"现在一般但会变好"与"一直不好"在观测上不可区分，最优解不在
# 可表示的策略类里。推导见 opportunity.obs_block 的文档。
# 同样**改变网络输入维度**，v16 及更早的权重不可载入，跨批次不可混在一张图里
# 比较；v17 的结论一律由批次内 2×2 给出。
N_FEAT = 32
N_PAIR_FEAT = N_FEAT + 12 + 2 + 3 + 1  # +本对成本/预算, +当前可用预算/年度额度, +是否为「到此为止」, +已承诺占用/年度额度
CH_ORDER = [1, 2, 3, 5]

# 生命周期特征消融开关。False 时把 16..21 这六维**置零**而不是删掉：
#   位宽、网络形状、参数量、初始化、优化器状态全部不变，两组之间只差
#   \"策略能否看到剩余时间与 slack\" 这一件事，构成单因子消融。
# 删掉会同时改变输入维度与参数量，差异就不再可归因。
# 三处刻意的设计约束：
#   1) **不入分母缓存键**（scenario.inst_tag 不含它）。参考策略是手写规则、
#      不读观测，消融组与主组的可达上界按构造完全相同；共用同一套分母，
#      两组的标量化回报才可直接相减。若误入键，消融组会另建一套数值上
#      相同但路径不同的分母，白跑一遍还会让人误以为不可比。
#   2) 仍由 scenario 登记并在 reset() 复原——它是模块级常量，不复原就会
#      发生本项目记过的那类静默继承（上一组设过、这一组没设）。
#   3) 承诺预算特征（pairs 的 N_FEAT+17）**不**随之置零：那是资金口径改动
#      带来的信息，属于另一个因子，混进来这组消融就不是单因子的了。
OBS_LIFECYCLE = True

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

# ---------------------------------------------------------------- 势函数型奖励整形
# REWARD_SHAPING = 0 关闭（默认，与 v16 逐位等价）；1 = 全量整形。
#
# 为什么需要：折现使"同样的收益推迟一年"一律变差，于是**局部**地看等待几乎总是
# 净亏，梯度把动作往前推——哪怕晚做的总价值高得多。门槛实验
# （scripts/exp_timing_gate.py）把这件事量化过：在一个最优解可穷举的三地块情景里，
# 试过的每一档（改奖励时点、熵系数、迭代数、机会改善形状、幅度、每轮回合数）都
# 没有把立项年推后；其中"每轮 32 个回合"这一档有 1/3 种子学会了等，但同时把三个
# 地块全推后（退化解），"整形"这一档则是唯一把折现回报做到穷举最优 95% 以上
# （3/3 种子）的一档。两者都只是部分有效，故本参数是**实验因子而非既定修复**。
#
# 整形项 F = γ·Φ(s') − Φ(s)，Φ(s) = Σ_{未立项且可立项的单元} φ_u(t)，
#     φ_u(t) = max_{t' ∈ [t, T-1]} 交付价值(u, 立项于 t') · γ^{(t'+L_u) − t}
# 即"这个单元留在手里、将来在它最好的年份动手，折到今天值多少"。L_u 用**期望**
# 时滞（E[离开S1] + 开工1年 + 该单元建设年限），与 slack 特征同一口径。
#
# 三条必须守住的性质：
#   1) **不改变最优策略集**。势函数型整形对任意 Φ 都保持最优策略不变
#      （Ng, Harada & Russell 1999），前提是终止态 Φ=0——这里由"t >= T 后
#      立项掩码全关、φ 的取值范围为空、Φ 自然为 0"保证，不需要额外分支。
#      这是它可以写进论文的前提：不是把答案喂给策略，而是把同一个最优解的
#      梯度变得可跟随。
#   2) **只进标量奖励，不进 vec**。vec 是 11 维目标的落盘口径，整形是学习手段
#      而非目标；混进去报出来的目标值就不再是目标值。
#   3) **不入分母缓存键**。参考策略不读整形项，可达的目标上界按构造不变，
#      故整形组与主组共用分母、标量化回报可直接相减（与 OBS_LIFECYCLE 同理）。
REWARD_SHAPING = 0.0


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

    @property
    def quota(self):
        """年度立项配额。取自**执行配额的那个状态机实例**，而不是模块常量快照，
        这样调用方的取值不可能与真正做校验的对象脱钩。"""
        return self.env.quota

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
        # 消融：整段置零（位宽不变，见模块头 OBS_LIFECYCLE 的说明）。
        # 写在计算之后而不是用分支跳过，是为了让两条路径的浮点运算次数一致，
        # 也避免将来有人往 16..21 里加东西却忘了加进消融范围。
        if not OBS_LIFECYCLE:
            F[:, 16:22] = 0.0

        # ---- 机会场（v17 新增，见 tpmorl/env/opportunity.py）----
        # 22..25 当期水平（规划/设施公布层/必要性/实施条件），26..29 趋势，
        # 30..31 动态更新机会 O 及其趋势。
        # 这八维回答的是 16..21 回答不了的另一个问题：不是"现在立项还赶得上吗"，
        # 而是"**现在**是不是这个地块的好时候、还是再等两年更好"。
        # 消融（OBS_OPPORTUNITY=False）时整段返回全零，位宽不变
        # （与 OBS_LIFECYCLE 同一约定）；FORESIGHT=0 时只有趋势四维为零。
        F[:, 22:32] = self.opp.obs_block(self.t)
        return F

    def set_prescreen(self, k=0, score=None):
        """**诊断用**：每年只把当期分数最高的 k 个候选送进动作空间。

        为什么值得试：监督探针显示，用同一批特征、同样大小的网络做**监督**回归，
        挑出的 3 个能拿到真前 3 的 100%（随机 39%）；而同一网络在 PPO 下训 400
        迭代只到随机水平。也就是说信息与容量都够，卡住的是信用分配——
        每年 595 个候选里选 3 个，一个标量回合回报要摊到上万次候选评分上，
        单个候选拿到的梯度信号极弱。先筛到几十个，直接检验这一解释。

        **口径声明（必须写进论文）**：筛选分数只用**当年可观测**的量
        （当期机会指数 × 基准交付量），与近视贪心用的是同一批信息。
        因此开了这一档之后的结果只能读作"在近视初筛之上的 RL"，
        **不能**读作"RL 自己学会了空间选择"。要证明后者，筛选必须关掉。
        """
        self.prescreen_k = int(k)
        self._prescreen_score = score
        return self.prescreen_k

    def _prescreen_rows(self, pu, pt, cost):
        """按当期分数取前 k 个候选行（k<=0 时不筛）。"""
        k = int(getattr(self, "prescreen_k", 0))
        if k <= 0 or pu.size <= k:
            return np.arange(pu.size)
        if self._prescreen_score is not None:
            sc = np.asarray(self._prescreen_score(self, pu, pt), float)
        else:
            oi = self.opp.opportunity_index(self.t)
            oi = (np.full(self.n, 1.0) if np.ndim(oi) == 0
                  else np.asarray(oi, float))
            sc = oi[pu] * self.farcap[pu] * self.ncell[pu]
        return np.argsort(-sc, kind="stable")[:k]

    def fix_one_target_per_unit(self):
        """**诊断用**：每个单元只保留一个目标功能，候选配对数降为单元数。

        为什么需要这一档：oracle 与近视贪心的计价口径是 EV(u, t)，只关心
        "哪个单元、哪一年"；而策略面对的是 (单元, 目标) 配对的自回归选择，
        一年之内还要连选 QUOTA 次。两边其实不是同一个问题——策略额外承担了
        目标功能选择的难度。固定目标后口径一致，才能干净地问
        "能不能在 717 个单元里做好当期选择"。

        **这是诊断档，论文最终模型必须保留目标功能选择。**

        保留哪一个：成本最低的那个。EV 的 base 是 FAR 上限 × 格数，与目标功能
        无关，故这个选择不改变计价口径；取最低成本是为了不顺带引入资金约束的
        干扰（本闸预算已放松，但保持中性更稳）。
        """
        keep, seen = [], set()
        for i in np.argsort(self._cost_all, kind="stable"):
            u = int(self._pu_all[i])
            if u in seen:
                continue
            seen.add(u)
            keep.append(int(i))
        keep = np.sort(np.asarray(keep, dtype=np.int64))
        self._pu_all = self._pu_all[keep]
        self._pt_all = self._pt_all[keep]
        self._cost_all = self._cost_all[keep]
        return len(keep)

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
        keep = self._prescreen_rows(pu, pt, cost)
        if keep.size != pu.size:
            pu, pt, cost = pu[keep], pt[keep], cost[keep]
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
        # 机会场：**用固定的场种子构造，不跟回合种子**。它代表这一片区客观的
        # 规划与设施安排，训练与评估的每个回合都必须是同一张图，否则策略无从利用，
        # 也就测不出择时能力。逐年取值覆盖到 T_eval，尾部评价年也能取到。
        self.opp = OPP.OpportunityField(
            self.n, self.T, T_total=self.T_eval,
            row=self.U["row"].values if "row" in self.U else None,
            col=self.U["col"].values if "col" in self.U else None)
        self.env = RenewalSchedule(self.ch, seed=self.seed if seed is None else seed,
                                   opp=self.opp)
        self.t = 0
        self.plan = {}          # 单元 -> 立项时选定的目标功能
        self.init_year = {}     # 单元 -> 立项年（机会场的价值乘子要读立项年取值）
        self.hist0 = [{k: v for k, v in
                       ((k, int((self.LU0[..., k][m] > 0).sum())) for k in range(12)) if v}
                      for m in self.cells]
        # (单元 × 目标) 成本矩阵：只依赖静态的 ncell/CCM/hist0，一次算好。
        # 原先每年为 ~1985 个配对逐个调用 pair_cost，是训练的主要开销之一。
        # 注意：因此在 env 构造之后再改 CELL_COST/CCM 不会生效。
        # 逐类拆除基数：按候选池构成现算（均值保持），并写回 Reward 实例，使
        # 「预算路径 PC」与「Cost 目标路径 convert_cost」读**同一个向量对象**。
        # flat 档该向量恒等于 CELL_COST，PC 与改动前逐位相同（数值中性）。
        from tpmorl.objectives.reward import cell_base_vector, CELL_COST_MODE
        self.cell_base = cell_base_vector(self.hist0, self.ncell, CELL_COST_MODE)
        self.cell_cost_mode = str(CELL_COST_MODE)
        self.R.cell_base = self.cell_base
        self.PC = np.zeros((len(self.hist0), 12), dtype=np.float64)
        for u, h in enumerate(self.hist0):
            for f, c in h.items():
                self.PC[u] += (self.CCM[f, :12] + self.cell_base[f]) * c
        # 断言两条路径按构造相等：漏掉一侧正是 v1 的缺陷类型，须当场响。
        _u = int(np.argmax(self.ncell))
        assert abs(self.PC[_u, 6] - sum(self.R.convert_cost(f, 6, c)
                                        for f, c in self.hist0[_u].items())) < 1e-9, \
            "PC 与 convert_cost 口径不一致：拆除基数只进了一条路径"
        # 全量 (单元, 目标) 配对枚举：只依赖静态的通道归属，一次算好。
        # 每年的候选集 = 用 mask_init 在这三个数组上做布尔选择，无需重新枚举。
        ch_all = np.asarray(self.ch, dtype=int)
        cnt_all = np.array([len(ALLOWED[int(c)]) for c in ch_all], dtype=np.int64)
        self._pu_all = np.repeat(np.arange(len(ch_all), dtype=np.int64), cnt_all)
        self._pt_all = np.concatenate(
            [np.asarray(ALLOWED[int(c)], dtype=np.int64) for c in ch_all])
        self._cost_all = self.PC[self._pu_all, self._pt_all]
        self.mask_init = self._init_mask()
        self.budget = BUDGET            # 已拨付未承诺（可承诺额度）
        self.committed = 0.0            # 已承诺未支付
        self._owe_year = {}             # 单元 -> 实施期内每年应付额
        self._owe_left = {}             # 单元 -> 剩余未付额
        self.spent_hist, self.budget_hist = [], []
        self.disb_hist, self.committed_hist = [], []
        # 逐年**释放**额：有效期届满被撤的单元，其剩余未付承诺回到可承诺额度。
        # 必须落盘：资金分解的闭合式里它是独立一项（承诺−释放+作废+期末余额
        # = B·(T+1)），upfront 下恒为 0，staged 下不记就无法闭合。
        self.released_hist = []
        self._phi = self._potential_table()
        return self.obs()

    def _init_mask(self):
        """本年可立项的单元：制度状态机的掩码 **∩** 上位规划的立项可及性。

        规划这一层是 v18 新增的通道（`opportunity.admissible`）：法定图则尚未
        覆盖、不在更新单元计划名单里的地块，这一年报上去本就进不了流程。
        它是**逐年重抽的随机准入**而非硬阈值——规划支持只提高被纳入的机会，
        不保证纳入，也不永久禁入低支持度的地块。

        抽样借用状态机的发生器 `self.env.rng`：A_PLAN=0 时 `admissible` 整条
        跳过、**不消耗随机数**，故场全关时与 v16 的随机序列逐位相同。
        """
        m = self.env.mask_initiate()
        if OPP.A_PLAN:
            m = m & self.opp.admissible(self.t, self.env.rng)
        return m

    def _potential_table(self):
        """(n, T+1) 的持有价值表 φ_u(t)；REWARD_SHAPING=0 时返回 None。

        φ_u(t) = max_{t' ∈ [t, T-1]} 单元 u 于第 t' 年立项的交付价值折到第 t 年。
        交付价值与 step() 里真正计入 Floor 的那一式**同构**（FAR 上限 × 格数 ×
        (1+FAR_GROWTH)^t' × 机会场时机乘子），并按 Floor 的权重与分母折算成
        标量奖励的量纲——否则整形项与奖励不同量纲，`REWARD_SHAPING=1` 的含义
        就无从解释。

        第 T 列恒为 0：t >= T 后立项掩码全关，取值范围为空。这正是势函数型整形
        要求的"终止态 Φ=0"，故最优策略集不变这一性质在此处成立，无需额外分支。
        """
        if not REWARD_SHAPING:
            return None
        j = OBJ_NAMES.index("Floor")
        w = self.weights / self.weights.sum()
        k = float(w[j]) * SIGN["Floor"] / float(self.scale[j])
        L = (self.env._exp_leave[0] + 1.0
             + self.env.build_years.astype(float))          # 期望时滞，逐单元
        V = np.zeros((self.n, self.T + 1))
        for u in range(self.n):
            if not self.env.eligible[u]:
                continue
            base = self.R.floor_area(self.ch[u], self.ncell[u]) * k
            for tp in range(self.T):
                td = tp + L[u]
                V[u, tp] = (base * (1.0 + FAR_GROWTH) ** tp
                            * self.opp.value_mult(u, tp, int(round(td)))
                            * self.gamma ** td)
        # 后缀最大值 → φ_u(t) = max_{t' >= t}，再把"折到第 t 年"的 γ^{-t} 补上
        P = np.zeros((self.n, self.T + 1))
        for t in range(self.T - 1, -1, -1):
            P[:, t] = np.maximum(P[:, t + 1], V[:, t])
        for t in range(self.T):
            P[:, t] /= self.gamma ** t
        return P

    def potential(self):
        """Φ(s) = 所有**尚未立项且本期仍可立项**的单元的持有价值之和。

        判据用 σ==S0 且通道可更新，而**不用** mask_initiate()：后者在决策期末
        被强制清空，会让 Φ 在 t=T-1→T 之间凭空掉一大块，制造一个与择时无关的
        虚假整形信号。σ==S0 的集合不受掩码开关影响，而 φ 的第 T 列本就是 0，
        终止态 Φ=0 依然成立。
        """
        if self._phi is None:
            return 0.0
        m = (self.env.sigma == 0) & self.env.eligible
        t = int(min(self.t, self.T))
        return float(self._phi[m, t].sum())

    def step(self, actions):
        """actions: [(单元, 目标功能), ...]，至多 QUOTA 个；STOP 项被忽略。"""
        actions = [(int(u), int(tg)) for u, tg in actions if int(u) >= 0]
        tail = self.t >= self.T          # 尾部评价年：不立项、不进钱
        if tail and actions:
            raise AssertionError(
                f"第 {self.t} 年已超出决策期 T={self.T}，不得立项（收到 {len(actions)} 个动作）")
        # spent = 本年新增**承诺**额（两种口径下都是全额，掩码据此硬约束）
        spent = sum(self.pair_cost(u, tg) for u, tg in actions)
        if spent > self.budget + 1e-6:   # 显式 raise：裸 assert 在 python -O 下整体失效
            raise AssertionError(f"超预算 {spent:.0f} > {self.budget:.0f}")
        # Φ(s) 必须在动作生效**之前**取。写在 env.step 之后会把本年立项的单元
        # 从 Φ(s) 与 Φ(s') 里同时剔除，整形项退化成一个与择时无关的近似常数，
        # 看起来"开了整形"而实际上没有任何时序信号。门槛实验的第一版就是这么
        # 写错的，三个种子跑出与未整形档逐位相同的结果。
        phi0 = self.potential()
        self.budget_hist.append(self.budget); self.spent_hist.append(spent)
        self.committed_hist.append(self.committed)
        for u, tg in actions:
            self.plan[int(u)] = int(tg)
            self.init_year[int(u)] = int(self.t)
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
            if self.committed <= -1e-6:
                raise AssertionError(f"承诺额为负 {self.committed:.3f}")
            if abs(self.committed - sum(self._owe_left.values())) >= 1e-3:
                raise AssertionError("承诺额与逐单元未付额台账不一致")

        else:
            disb, released = spent, 0.0
        self.disb_hist.append(disb); self.released_hist.append(released)
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
            # 交付建面按**建成年**的容积率上限计（FAR_GROWTH>0 时推迟立项可换更高上限），
            # 再乘机会场的时机乘子 M（v17 新增）。幅度全 0 时 M 恒为精确的 1.0，
            # 与 v16 逐位等价。
            #
            # **口径声明**：M>1 的档下，Floor 不再是纯粹的"计容建筑面积"，而是
            # "按时机加权的交付价值（以建面计量）"。这是情景构造，论文必须写明，
            # 不得把加权后的数值当作实测建面报告。
            mult = self.opp.value_mult(u, self.init_year.get(int(u), self.t), self.t)
            floor += (self.R.floor_area(self.ch[u], self.ncell[u])
                      * (1.0 + FAR_GROWTH) ** self.t * mult)
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
        # 整形项只加在**标量奖励**上，vec（11 维目标的落盘口径）保持不变。
        # REWARD_SHAPING=0 时整条跳过，与 v16 逐位等价。
        if REWARD_SHAPING:
            r += REWARD_SHAPING * (self.gamma * self.potential() - phi0)
        # 决策期结束后立项掩码强制全关：pairs() 因此只剩 STOP 行，
        # 采样器与随机基线都无需改动即可在尾部自然「无动作」。
        self.mask_init = (self._init_mask() if self.t < self.T
                          else np.zeros(self.n, dtype=bool))
        return self.obs(), r, self.t >= self.T_eval, dict(vec=vec, raw=rv, events=ev)
