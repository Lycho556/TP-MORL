"""reward.py — 多目标奖励函数：复用 pSO 的 7 个空间目标 + 4 个时序目标。

设计原则
    空间目标直接复用 pSO（objs.py）的实现与权重矩阵（UUM / CM / CCM），保证与竞品可比；
    时序目标是本方案新增的部分，它们才是"现在动还是再等"这一决策的载体。

与 pSO 的两处必要改写
    1) 池化半径随分辨率改写：pSO 在 5 m 栅格上取 r=100（即 500 m），
       本项目为 100 m 决策格网，故取 r=5 保持同一物理尺度 500 m。
    2) 目标在**逐年演化**的用地上求值并折扣累加，而非只在终局方案上求值一次。

类别编码
    UUM / CM / CCM 的 12 列对应 DN 1..12 = R1 R2 RC C CBD IH I1 I2 A E G U。
    DN 0 = 限建区、DN 15 = 区外，均不参与目标求值。
    光明区内 RC(DN3) / CBD(DN5) / IH(DN6) 三类格数为 0（实测）。
"""
import numpy as np
from scipy import ndimage as ndi

CLASSES = ("R1", "R2", "RC", "C", "CBD", "IH", "I1", "I2", "A", "E", "G", "U")
POOL_R = 5          # 5 格 x 100 m = 500 m，与 pSO 的 100 格 x 5 m 同尺度

# 通道相关的规划容积率上限（实测：269 条已批单元规划中 164 条可算容积率）
FAR_CAP = {1: 6.82,   # P1 产业：工业类中位 6.82 (n=63)
           2: 7.45,   # P2 旧居住：村类合计中位 7.45 (n=34)
           3: 8.37,   # P3 商服：商服类中位 8.37 (n=4，样本极小，须敏感性分析)
           5: 6.98}   # P5 农转用：无更新单元类比，取全市中位 6.98（超参数）
CELL_AREA = 1e4       # 100 m x 100 m = 10000 平方米

# 每格拆除补偿基数（情景参数，与 CCM 同量纲）。
# 必须计入：CCM 对角线为 0，若资金只按 CCM 计，「原类重建」就不花钱却照样
# 计入交付建面——1985 个配对中有 361 个（18.2%）是纯自转换，策略可以用它们
# 白拿 Floor。现实里拆迁补偿主要与**拆除面积**成正比，改成什么用途只影响
# 附加的改造成本。此常量在 v1 曾只加进 env_gym 的预算路径而漏掉了 Cost 目标，
# 导致预算咬住、Cost 仍报 0；现由 convert_cost 统一实现，两条路径口径一致。
CELL_COST = 50.0

# 逐类拆除基数（v16 新增，情景参数）。
#
# 为什么要加：上面那个 50 是**一刀切**——拆一格城中村（R2）与清一格农地（A）记同一个
# 价。CCM 确实按"转成什么"区分了成本（0–100，非对称），但实测候选池里拆除基数占
# 单元总成本的中位 62%，也就是说**类型差异只解释了 38%**，而占主导的那 62% 对
# "拆的是什么"完全不敏感。现实里拆迁补偿主要由被拆除物的性质决定（住宅要补偿产权人、
# 工业厂房补构筑物、农地基本只需征转），一刀切会系统性高估农转用、低估旧居住改造。
#
# 取值口径（**假设，非标定值**）：单元表 gm_renewal_units.csv 没有投资额／补偿额字段，
# 平台亦无逐类补偿标准数据集，故这里只能给**相对倍率**并声明为情景参数。倍率的排序
# 依据是补偿对象的有无与强弱：住宅（须补偿产权人）> 商业 > 公用设施 > 工业 > 农地 > 生态绿地。
CELL_COST_MULT = {0: 2.0,    # R1 一类居住
                  1: 2.4,    # R2 二类居住（城中村，补偿最重）
                  2: 1.8,    # RC 商住混合（光明区无此类）
                  3: 1.8,    # C  商业
                  4: 2.0,    # CBD 中心商务（光明区无此类）
                  5: 1.0,    # IH 工业配套（光明区无此类）
                  6: 1.0,    # I1 一类工业
                  7: 0.9,    # I2 二类工业
                  8: 0.5,    # A  农业
                  9: 0.35,   # E  生态
                  10: 0.35,  # G  绿地
                  11: 1.2}   # U  公用设施

# **均值保持**是本参数的关键设计：倍率按候选池的现状用地格数加权归一，使
# `Σ_u Σ_f base_f · c_uf == CELL_COST · Σ_u ncell_u`。这样 bytype 与 flat 两档的
# **总成本尺度完全相同**，差异是纯粹的"钱在类型间重新分配"，不掺入"整体变贵/变便宜"。
# 若不做归一，bytype 组同时改变了成本水平与成本结构，两个效应无法分离——而分母、
# 预算咬合、可达上界全都对成本水平敏感，那样的对照是不可解释的。
#
# 归一系数由数据现算（见 cell_base_vector），不硬写：候选池构成随数据集版本而变，
# 硬写会在换数据集时静默失配。gm_dataset_v1 上现算结果（加权均值锁定 50.0）：
#   A 25.17 · I1 50.34 · I2 45.31 · C 90.62 · R1 100.69 · R2 120.83
# 即拆一格城中村约等于清 4.8 格农地，这正是一刀切档看不见的差别。
CELL_COST_MODE = "flat"      # "flat" = 沿用单一 CELL_COST；"bytype" = 逐类基数


def cell_base_vector(hist0, ncell, mode=None):
    """按候选池构成算出 12 维逐类拆除基数向量（均值保持）。

    `hist0[u]` 是单元 u 的 {现状类别: 格数}，`Σ_f c_uf == ncell_u`（LU0 为独热，
    已在 env_gym 侧核过）。flat 档返回全体等于 CELL_COST 的向量，使两档共用同一
    段代码路径——**不留"另一条分支"**，避免两条路径日后各自漂移（v1 的 CELL_COST
    就是因为只进了预算路径、没进 Cost 路径而算错过）。
    """
    import numpy as np
    m = CELL_COST_MODE if mode is None else str(mode)
    v = np.full(12, float(CELL_COST))
    if m == "flat":
        return v
    if m != "bytype":
        raise ValueError(f"CELL_COST_MODE 只能是 flat/bytype，收到 {m!r}")
    cnt = np.zeros(12)
    for h in hist0:
        for f, c in h.items():
            cnt[f] += c
    tot = cnt.sum()
    if tot <= 0:
        raise ValueError("候选池为空，无法归一化逐类拆除基数")
    raw = np.array([CELL_COST_MULT[f] for f in range(12)], float)
    k = tot / float((raw * cnt).sum())        # 均值保持的归一系数
    return CELL_COST * raw * k

OBJ_SPATIAL = ("Gdp", "Eco", "Res", "Emp", "Aec", "E2r", "Cpt")
OBJ_TEMPORAL = ("Floor", "Cost", "Disrupt", "Expire")
OBJ_NAMES = OBJ_SPATIAL + OBJ_TEMPORAL
SIGN = dict(Gdp=+1, Eco=+1, Res=+1, Emp=+1, Aec=+1, E2r=-1, Cpt=+1,
            Floor=+1, Cost=-1, Disrupt=-1, Expire=-1)   # +1 越大越好

EXPIRE_PENALTY = 1.0    # 失效沉没成本，单位：一个单元的前期投入（超参数）
DISRUPT_RADIUS = 3      # 施工干扰半径，格（超参数）


def eta(p):
    """pSO 的边际效用递减函数，原样复用。"""
    return 1 - p * p / 3


def _pool(x, r=POOL_R):
    return ndi.uniform_filter(x.astype(float), size=r, mode="nearest")


class Reward:
    """在逐年演化的用地上求 11 维奖励向量。"""

    def __init__(self, UUM, CM, CCM, road, water, inside, base_lu, gamma=0.95):
        """`base_lu` 是**基期**用地 (H, W, 12)，用于 `Aec` 的分母，见 `spatial()`。

        刻意设为必填位置参数、不给默认值：给了默认值就会出现"没传就悄悄退回旧口径"
        的两套行为，而这正是本项目已经栽过两次的缺陷类型（FAR_GROWTH 泄漏、
        scenario.apply 的 None 继承）。宁可让旧调用点直接报错。
        """
        self.UUM = np.asarray(UUM, float)      # 5 x 12: res emp gdp eco liv
        # 基期居住承载的池化场，Aec 的固定分母。一次算好，不随 LU 演化。
        self.base_res_pool = _pool((np.asarray(base_lu, float)
                                    * np.asarray(UUM, float)[0]).sum(-1))
        self.CM = np.asarray(CM, float)        # 12 x 12 兼容性
        self.CCM = np.asarray(CCM, float)      # 12 x 12 非对称转换成本
        self.road = np.asarray(road, float)
        self.water = np.asarray(water, float)
        self.inside = np.asarray(inside, bool)
        self.gamma = gamma
        # 逐类拆除基数向量，由 env_gym 在建好 hist0 后写入（见 convert_cost）。
        # None = flat 口径，逐位等价于既往结果。
        self.cell_base = None

    # ---- 空间目标：复用 pSO ----
    def spatial(self, LU):
        """LU: (H, W, 12) one-hot 或概率。返回 7 个空间目标（已统一为越大越好）。

        **Aec 口径变更（2026-09-12）：分母由当期居住承载改为基期居住承载。**

        原式 `Σ livP / (resP + 1)` 中 resP 随 LU 逐年演化，于是"拆掉居住"能同时
        抬高 Aec 和压低 Res。两者在标量化奖励里都是正向目标，策略只要拆居住就能
        在 Aec 上白拿分——这不是规划含义上的改善，是定义本身的机械耦合。
        v12 七组实测组内 corr(Aec, Res) = −0.57 ~ −0.69，耦合确实显现在结果里。

        改为基期分母后，Aec 读作"按**基期**居住人口加权的生活服务可达性"：
        服务供给增加才涨，拆居住不再有任何贡献。分母是常数场，故 Aec 对 LU 单调
        非减，语义干净。

        未一并修改、留作待定的两项（改动会进一步影响可比性，需单独决策）：
        1. 分子仍为 `eco + liv`，与目标 `Eco` 重复计入生态效用一次；若要拆干净，
           分子应只取 `liv`。
        2. `+1` 是无量纲正则项，量级与 resP 的单位绑定，非居住区的权重因此偏大。

        **本改动使 Aec 与 v12 及之前所有批次不可比**，Livability 维度需重跑。
        """
        p = LU.sum((0, 1)); p = p / max(p.sum(), 1e-9)
        w = eta(p)
        res = (LU * self.UUM[0]).sum(-1)
        emp = (LU * self.UUM[1]).sum(-1)
        gdp = (LU * self.UUM[2] * w).sum(-1) * (1 - self.road)
        eco = (LU * self.UUM[3] * w).sum(-1) * (1 - self.water)
        liv = (LU * self.UUM[4] * w).sum(-1) * (1 - self.road)

        resP, empP, livP = _pool(res), _pool(emp), _pool(eco + liv)
        # Aec 的分母用**基期**居住承载，不用当期。2026-09-12 更正，理由见下方 Aec 一行。
        PM = np.stack([_pool(LU[..., k]) for k in range(LU.shape[-1])], -1)

        return dict(
            Gdp=float(gdp.sum()),
            Eco=float(eco.sum()),
            Res=float(res.sum()),
            Emp=float(emp.sum()),
            Aec=float((livP / (self.base_res_pool + 1)).sum()),
            E2r=float(np.abs(empP - resP).sum()),          # SIGN=-1
            Cpt=float((PM.reshape(-1, 12) @ self.CM * PM.reshape(-1, 12)).sum()))

    # ---- 时序目标：本方案新增 ----
    def floor_area(self, channel, n_cells):
        """计容积率建筑面积（平方米）。通道决定容积率上限。

        **计入时点是"建成"（S4），不是"获批"**（原文档串写为"获批时"，2026-09-09 更正）。
        调用方 env_gym 只对当年转入 S4 的单元累加本函数，`Cost` 同理。
        后果：立项 → hazard 获批（中位 3 年）→ 次年开工 → 建设 5 年，
        故立项年 > T-6 的单元在规划期内必然不产生 Floor，也不产生 Cost。
        """
        return FAR_CAP.get(int(channel), FAR_CAP[5]) * n_cells * CELL_AREA

    def convert_cost(self, from_idx, to_idx, n_cells):
        """资金成本 = (拆除补偿基数 + 非对称转换成本) × 格数。

        CCM 取自 pSO（如 A->E 85 而 E->A 20），对角线为 0；拆除基数保证
        「原类重建」也要花钱（见模块顶部 CELL_COST 的说明）。
        与 env_gym 的 pair_cost/PC 同式，两者按构造相等。

        拆除基数按**被拆除的现状类别** from_idx 取值：`self.cell_base` 由 env_gym 在
        建好 hist0 后写入（cell_base_vector，均值保持）。之所以放在实例上而不是读
        模块常量，是为了让本路径与 PC 路径**读同一个向量对象**——v1 的缺陷正是
        CELL_COST 只进了预算路径、Cost 目标那条路径漏掉，两条路径各算一套。
        未写入时回退到单一 CELL_COST（flat 口径），与既往结果逐位等价。
        """
        base = CELL_COST if self.cell_base is None else float(self.cell_base[from_idx])
        return (base + float(self.CCM[from_idx, to_idx])) * n_cells

    def disrupt(self, sigma_units, unit_id, res_map, s3_code=3):
        """施工干扰：S3 实施中的单元，对邻域居住承载造成的当期损失。"""
        building = np.isin(unit_id, np.where(sigma_units == s3_code)[0] + 1)
        if not building.any():
            return 0.0
        halo = ndi.binary_dilation(building, iterations=DISRUPT_RADIUS) & ~building
        return float(res_map[halo & self.inside].sum())

    def step_reward(self, LU, ev_floor, ev_cost, ev_disrupt, n_expired,
                    mode="delta"):
        """mode='delta'（默认）返回空间目标相对上一年的**增量**；'level' 返回绝对水平。

        为什么默认用增量：本区 717 个候选单元中 15 年只有个位数完成，
        绝对水平被未动的绝大多数格子淹没——实测三个差异极大的基线策略在
        7 个空间目标上的相对离散度仅 0.09%–0.26%，而时序目标为 22%–82%。
        绝对水平几乎不含策略信息，直接用作奖励会使空间目标退化为常数偏置。
        改用逐年增量后，γ<1 的折扣使"同样的改善发生得更早"获得更高回报，
        这恰好是时序问题需要的信号。
        """
        cur = self.spatial(LU)
        if mode == "level":
            r = dict(cur)
        else:
            base = getattr(self, "_prev", None)
            r = {k: (cur[k] - base[k]) if base else 0.0 for k in cur}
        self._prev = cur
        r.update(Floor=ev_floor, Cost=ev_cost, Disrupt=ev_disrupt,
                 Expire=EXPIRE_PENALTY * n_expired)
        return r

    # ---- 标量化 ----
    def scalarize(self, rvec, weights, mode="linear", ref=None):
        """MORL 标量化。linear=加权和；chebyshev=对参考点的加权 L-inf（更易触及非凸前沿）。"""
        v = np.array([SIGN[k] * rvec[k] for k in OBJ_NAMES], float)
        w = np.asarray(weights, float); w = w / w.sum()
        if mode == "linear":
            return float((w * v).sum())
        z = np.asarray(ref, float) if ref is not None else np.zeros_like(v)
        return -float(np.max(w * (z - v)))

    def discounted_return(self, traj):
        """traj: 逐年 reward dict 的列表。返回各目标的折扣累计（已统一为越大越好）。"""
        out = {k: 0.0 for k in OBJ_NAMES}
        for t, r in enumerate(traj):
            for k in OBJ_NAMES:
                out[k] += (self.gamma ** t) * SIGN[k] * r[k]
        return out
