"""uis.py — 深圳城市指数 UIS 的区级空间优化改造（UIS-D）。

这个模块做三件事，**不改动现有奖励函数**：

1. `IND_REGISTRY`  把 UIS 50 项指标的可优化性筛查结论落成可执行的数据结构。
   筛查准则与逐条判定见 `docs/UIS区级化_指标体系_v1.md` §1。
2. `DIM_OF_LEGACY` 现有 11 个优化目标 → UIS 六维的映射。
3. `uis_d()`       按六维聚合出 UIS-D 分数。

为什么现在只做聚合、不扩目标
    21 个可优化指标里有 20 个需要局里的空间数据（见 `docs/UIS区级化_数据需求清单.md`）。
    在数据到位前把它们写进奖励函数，会得到 20 个恒为 0 的目标：既跑不出结果，
    又会让 v11 的结论无法与新版对齐。正确顺序是 **先拿数据 → 再扩目标 → 再重跑**。

一条必须守住的口径
    缺数据的维度返回 `nan`，**绝不返回 0**。0 的含义是"这一维得分为零"，
    nan 的含义是"这一维还没法评"。两者在汇报里是完全不同的话。
"""
from __future__ import annotations

import numpy as np

from tpmorl.objectives.reward import OBJ_NAMES, SIGN

# ---------------------------------------------------------------- 六维定义

DIMS = ("Innovation", "Livability", "Aesthetics", "Resilience",
        "Humanism", "Intelligence")
DIM_CN = dict(Innovation="创新", Livability="宜居", Aesthetics="美丽",
              Resilience="韧性", Humanism="文明", Intelligence="智慧")

# 筛查三闸（见文档 §0）
#   G1 空间可归属 / G2 更新可响应 / G3 区级可得
GATES = ("G1", "G2", "G3")

# status 取值
#   "L0"  现有目标已覆盖（粗口径），无需新数据
#   "L1"  需要局里的基础空间数据（A/B 档）
#   "L2"  需要 UIS 官方口径（C 档）
#   "bg"  背景条件，不进目标函数
#   "con" 约束，不进目标函数但限制可行域


def _I(cn, en, dim, gates, status, obj=None, formula=None, data=None, note=None):
    return dict(cn=cn, en=en, dim=dim, gates=gates, status=status,
                obj=obj, formula=formula, data=data, note=note)


IND_REGISTRY = [
    # ---------------- 创新 INNOVATION（10 项 → 可优化 1）
    _I("高水平论文产出水平", "High-level Papers per Million Inhabitants",
       "Innovation", "--✓", "bg"),
    _I("专利产出水平", "Patents per Million Inhabitants",
       "Innovation", "--✓", "bg"),
    _I("研发投入强度", "R&D Intensity", "Innovation", "--✓", "bg"),
    _I("营商环境成熟度", "Business Environment Maturity",
       "Innovation", "--✓", "bg"),
    _I("外来人口吸引力", "Attractiveness & Inclusiveness for Migrant Population",
       "Innovation", "~~✓", "bg",
       note="经住房供给间接影响，已由宜居维承接，避免重复计分"),
    _I("初创企业孵化水平", "Startup Incubation Level",
       "Innovation", "✓✓✓", "L1", obj="Inn_Space",
       formula="Σ_更新单元 新增产业研发/孵化用房建面 / 全区产业用房总建面",
       data="C4 产业园区与孵化器边界"),
    _I("城市创新活力", "Urban Innovation Vitality",
       "Innovation", "---", "bg", note="合成指标，不独立入模"),
    _I("全员劳动生产率", "Total Labor Productivity (TFP)",
       "Innovation", "--✓", "bg", note="现有 Gdp 目标是其粗空间代理"),
    _I("技术交易活跃度", "Technology Transaction Activity",
       "Innovation", "--✓", "bg"),
    _I("跨境资本流动活跃度", "Direct Investment (FDI+OFDI)",
       "Innovation", "--✓", "bg"),

    # ---------------- 宜居 LIVABILITY（10 项 → 可优化 6）
    _I("住房保障水平", "Housing support", "Livability", "✓✓~", "L1",
       obj="Liv_Housing",
       formula="Σ 更新配建保障性住房建面 / Σ 更新新增总建面",
       data="B1 保障性住房台账"),
    _I("基础服务可达性", "Basic service access", "Livability", "✓✓~", "L1",
       obj="Liv_Basic",
       formula="15 分钟步行（路网）可达 小学+社康+菜市场 的人口 / 总人口",
       data="A1 POI, A7 路网, A8 人口"),
    _I("健康长寿水平", "Health and longevity", "Livability", "-✗✓", "bg"),
    _I("通勤便捷度", "Mobility efficiency", "Livability", "✓✓~", "L1",
       obj="Liv_Commute",
       formula="0.5·(1 - |E_pool - R_pool| 归一) + 0.5·轨道站 800 m 覆盖人口比",
       data="A7 轨道与路网, A8 人口",
       note="现有 E2r 是其前半段的雏形"),
    _I("提升服务可达性", "Advanced service access", "Livability", "✓✓~", "L1",
       obj="Liv_Adv",
       formula="30 分钟（含公交）可达 三甲+高中+区级文体 的人口 / 总人口",
       data="A1 POI, A7 路网, A8 人口"),
    _I("第三空间活力", "Third-space vitality", "Livability", "✓✓~", "L1",
       obj="Liv_Third",
       formula="500 m 池化的 咖啡/书店/健身/社区中心 POI 密度，按人口加权",
       data="A1 POI, A8 人口"),
    _I("开放空间品质", "Open space quality", "Livability", "✓✓~", "L1",
       obj="Liv_Open",
       formula="公园绿地 500 m 覆盖人口比 × 人均公园面积（对数压缩）",
       data="A2 绿地, A8 人口"),
    _I("人口结构活力", "Demographic structure", "Livability", "-✗✓", "bg"),
    _I("人才素质水平", "Human capital level", "Livability", "-✗✓", "bg"),
    _I("居民富裕程度", "Income level", "Livability", "-✗✓", "bg"),

    # ---------------- 美丽 AESTHETICS（9 项 → 可优化 5）
    _I("生态空间保护水平", "Protected Area to City Territory Ratio",
       "Aesthetics", "✓✓✓", "L0", obj="Aes_Eco",
       formula="现有 Eco 目标（生态效用 × (1-水体距离衰减)）直接对应",
       note="口径最接近的一条"),
    _I("建成区蓝绿空间比例", "Green-blue area to Built-up Area Ratio",
       "Aesthetics", "✓✓~", "L1", obj="Aes_GreenBlue",
       formula="(绿地面积 + 水域面积) / 建成区面积",
       data="A2 绿地水体"),
    _I("建成区本地生物多样性", "Native Biodiversity in Built-up Area",
       "Aesthetics", "✓~✗", "L1", obj="Aes_Connect",
       formula="绿地斑块有效网格大小 MESH = Σa_i² / A_total",
       data="A2 绿地",
       note="代理指标。汇报与论文中必须标注为连通度代理，不得写成生物多样性"),
    _I("城市低碳水平", "Urban Low-carbon Level", "Aesthetics", "✓✓~", "L1",
       obj="Aes_Carbon",
       formula="Σ (旧建筑碳强度 - 新建筑碳强度) × 更新建面",
       data="A4 建筑年代结构"),
    _I("全年空气质量", "PM2.5 Annual Mean Concentration",
       "Aesthetics", "✗-✓", "bg", note="区内空间差异小于测量误差"),
    _I("生活污水安全处理率", "The Proportion of Safely Treated Domestic Water Flows",
       "Aesthetics", "✓~~", "L1", obj="Aes_Sewage",
       formula="接入市政污水管网的更新单元建面 / 更新总建面",
       data="B2 市政管网"),
    _I("城市意向美誉度", "City Image Favorability", "Aesthetics", "✗✗✗", "bg"),
    _I("城市目的地吸引力", "Destination Appeal Intensity", "Aesthetics", "✗✗✗", "bg"),
    _I("街道空间视觉感知", "Streetscape Visual Perception",
       "Aesthetics", "✓✓~", "L1", obj="Aes_Street",
       formula="0.4·绿视率 + 0.3·界面连续度 + 0.3·天空开阔度（街景语义分割）",
       data="B3 街景影像或已算好的感知指标"),

    # ---------------- 韧性 RESILIENCE（8 项 → 可优化 6）
    _I("市政基础设施冗余度", "Redundancy of municipal infrastructure",
       "Resilience", "✓✓✗", "L1", obj="Res_Infra",
       formula="1 - 现状负荷 / 设计容量，按服务范围人口加权",
       data="B2 市政管网负荷"),
    _I("应急避难服务水平", "Emergency Shelter Service Level",
       "Resilience", "✓✓~", "L1", obj="Res_Shelter",
       formula="避难场所服务半径覆盖人口 / 总人口，容量不足按比例折减",
       data="A3 避难场所, A8 人口"),
    _I("自然岸线保有率", "Natural shoreline retention rate",
       "Resilience", "✓-✓", "bg",
       note="光明区无海岸线，本区不适用。建议区级版替换为河道生态岸线率"),
    _I("韧性城市顶层设计", "Top-level design of resilient city",
       "Resilience", "✗✗✓", "bg", note="制度性，全区一个数"),
    _I("建筑设防水平", "Building Fortification Level",
       "Resilience", "✓✓~", "L1", obj="Res_Fortify",
       formula="Σ (新建设防等级 - 原建设防等级) × 建面；等级由年代×结构查表",
       data="A4 建筑普查",
       note="拆旧建新的核心韧性收益，韧性维权重最高的一项"),
    _I("应急救援服务水平", "Emergency Response Service Level",
       "Resilience", "✓✓~", "L1", obj="Res_Rescue",
       formula="消防 5 min + 急救 10 min 路网可达覆盖人口比",
       data="A3 消防站, A7 路网, A8 人口"),
    _I("海绵城市建设水平", "The proportion of permeable urban area",
       "Resilience", "✓✓~", "L1", obj="Res_Sponge",
       formula="可渗透面积 / 建成区面积；更新单元按新建标准赋值",
       data="A6 不透水面"),
    _I("灾害风险综合治理", "Level of disaster risk management",
       "Resilience", "✓✓~", "L1", obj="Res_RiskExit",
       formula="高风险区内居住人口减量（更新把人从风险区搬出）",
       data="A5 风险区划, A8 人口"),

    # ---------------- 文明 HUMANISM（7 项 → 可优化 1 + 约束 1）
    _I("市民文明素养水平", "Civic Civility Behavior Level", "Humanism", "✗✗~", "bg"),
    _I("文化资源丰富度", "Cultural Heritage Resource Abundance",
       "Humanism", "✓✓~", "con", obj="Hum_Heritage",
       formula="约束：更新后文保单位/历史建筑/古树名木数量与范围不得减少",
       data="B4 历史文化遗存名录（空间化）",
       note="写成约束而非目标。写成目标会激励策略去『制造』文化设施"),
    _I("多元文化包容性", "Multicultural Inclusivity", "Humanism", "✗✗~", "bg"),
    _I("文化空间与设施共享", "Cultural Innovation & Production Capacity",
       "Humanism", "✓✓~", "L1", obj="Hum_Culture",
       formula="Σ 更新配建文化设施建面 / 常住人口（人均）",
       data="A1 POI 文化设施类"),
    _I("全球文化影响力", "Global Cultural Influence", "Humanism", "✗✗✓", "bg"),
    _I("公共文化参与与消费", "Cultural Participation & Consumption",
       "Humanism", "✗~~", "bg"),
    _I("市民公益参与度",
       "Participation rate in volunteering and public welfare activities",
       "Humanism", "✗✗~", "bg"),

    # ---------------- 智慧 INTELLIGENCE（6 项 → 可优化 1）
    _I("城市数据平台集成度", "Network Connectivity Level",
       "Intelligence", "✗✗✓", "bg"),
    _I("公共网络覆盖水平", "Digital Infrastructure Coverage Rate",
       "Intelligence", "✓~~", "L1", obj="Int_Network",
       formula="5G 基站 / 公共 WiFi 覆盖人口比，更新单元按配建标准赋值",
       data="C 档或运营商数据", note="弱响应"),
    _I("在线政务服务便利度", "Convenience of Online Government Service",
       "Intelligence", "✗✗✓", "bg"),
    _I("网络连接畅通度", "Digital Infrastructure Coverage Rate (throughput)",
       "Intelligence", "✗✗~", "bg", note="与公共网络覆盖水平重复"),
    _I("数字支付便捷度", "Digital Payment Convenience", "Intelligence", "✗✗~", "bg"),
    _I("数字公众参与活跃度", "Digital Civic Engagement Level",
       "Intelligence", "✗✗~", "bg"),
]

assert len(IND_REGISTRY) == 50, f"UIS 应有 50 项指标，登记表里是 {len(IND_REGISTRY)}"


# -------------------------------------------------- 现有 11 目标 → 六维映射

# 见 docs/UIS区级化_指标体系_v1.md §3。
# 权重是维度内的相对贡献（不必归一，uis_d 会归一）；`None` = 不进 UIS 六维。
DIM_OF_LEGACY = {
    "Gdp":     ("Innovation", 1.0),   # TFP 的粗空间代理
    "Emp":     ("Innovation", 0.5),   # 就业密度，同时也进宜居（职住）
    "Res":     ("Livability", 1.0),   # 住房承载，缺保障房结构
    "Aec":     ("Livability", 1.0),   # 环境舒适度，开放空间品质的粗代理
    "E2r":     ("Livability", 1.0),   # 职住偏离 → 通勤便捷度
    "Disrupt": ("Livability", 0.5),   # 施工期生活品质损失
    "Eco":     ("Aesthetics", 1.0),   # 生态空间保护水平，口径最接近
    "Cpt":     ("Aesthetics", 1.0),   # 用地兼容 → 空间共鸣
    "Floor":   (None, 0.0),           # 实施产出量，非 UIS 指标
    "Cost":    (None, 0.0),           # 投入量
    "Expire":  (None, 0.0),           # 制度类
}
assert set(DIM_OF_LEGACY) == set(OBJ_NAMES), "映射表与 OBJ_NAMES 不一致"


# ------------------------------------------------------------------ 聚合

def coverage():
    """各维的指标覆盖情况。返回 DataFrame 友好的 list[dict]。

    `n_L0` 是现有目标已覆盖的指标数——这是"今天就能评"的维度；
    `n_L1` 是拿到局里数据后能补上的；两者都为 0 的维度在 UIS-D 里只能是 nan。
    """
    rows = []
    for d in DIMS:
        g = [r for r in IND_REGISTRY if r["dim"] == d]
        n_opt = sum(r["status"] in ("L0", "L1", "L2") for r in g)
        rows.append(dict(
            dim=d, dim_cn=DIM_CN[d], n_ind=len(g),
            n_L0=sum(r["status"] == "L0" for r in g),
            n_L1=sum(r["status"] == "L1" for r in g),
            n_con=sum(r["status"] == "con" for r in g),
            n_bg=sum(r["status"] == "bg" for r in g),
            n_opt=n_opt, opt_rate=round(n_opt / len(g), 3),
            legacy_objs=",".join(k for k, (dd, w) in DIM_OF_LEGACY.items()
                                 if dd == d)))
    return rows


def uis_d(rvec, scale, weights=None):
    """把一次运行的 11 维目标向量聚合成 UIS-D 六维分数。

    参数
        rvec    dict 或长度 11 的数组，顺序同 `OBJ_NAMES`（折扣累计回报，
                已按 SIGN 统一为越大越好——`discounted_return` 的输出即是）
        scale   长度 11 的分母向量（`scale.load_fixed_scale` 的输出），用于归一化
        weights 可选，覆盖 `DIM_OF_LEGACY` 里的维度内权重

    返回
        dict(dim -> float)  六维分数，**无对应目标的维度为 nan**，外加
        `UIS_D_opt` = 有值维度的等权平均（可优化部分的总分）

    口径说明
        单维分数是维度内各目标"归一化值"的加权平均。归一化值可能 >1
        （分母取自参考策略包络，见 scale.load_fixed_scale 的文档），
        所以维度分也可能 >1；这不是 bug，含义是"超过了参考策略的上界"。
    """
    if isinstance(rvec, dict):
        v = np.array([rvec[k] for k in OBJ_NAMES], float)
    else:
        v = np.asarray(rvec, float)
    if v.shape != (len(OBJ_NAMES),):
        raise ValueError(f"rvec 应为 {len(OBJ_NAMES)} 维，收到 {v.shape}")
    s = np.abs(np.asarray(scale, float))
    s[s < 1e-12] = 1.0
    # rvec 已含 SIGN；归一化后同号即"越大越好"
    nv = v / s

    out = {}
    for d in DIMS:
        num = den = 0.0
        for i, k in enumerate(OBJ_NAMES):
            dd, w = DIM_OF_LEGACY[k]
            if dd != d:
                continue
            w = float(weights[k]) if weights and k in weights else w
            num += w * nv[i]
            den += w
        out[d] = num / den if den > 0 else float("nan")

    have = [out[d] for d in DIMS if not np.isnan(out[d])]
    out["UIS_D_opt"] = float(np.mean(have)) if have else float("nan")
    out["n_dim_scored"] = len(have)
    return out


def registry_table():
    """登记表转成可直接 to_csv 的 list[dict]（供文档与汇报用）。"""
    return [dict(序号=i + 1, 维度=DIM_CN[r["dim"]], 指标=r["cn"], 英文=r["en"],
                 三闸=r["gates"], 状态=r["status"], UIS_D目标=r["obj"] or "",
                 计算式=r["formula"] or "", 数据需求=r["data"] or "",
                 备注=r["note"] or "")
            for i, r in enumerate(IND_REGISTRY)]


if __name__ == "__main__":
    import pandas as pd
    print("== 各维覆盖 ==")
    print(pd.DataFrame(coverage()).to_string(index=False))
    print(f"\n可优化指标合计："
          f"{sum(r['status'] in ('L0','L1','L2') for r in IND_REGISTRY)}/50，"
          f"约束 {sum(r['status']=='con' for r in IND_REGISTRY)} 项")
