# -*- coding: utf-8 -*-
"""check_historical_data.py —— S4 历史回测的**数据可用性闸**。

指南要求的 S4 是：训练 1985–2009 → 留出验证 2010–2021，在真实更新记录上比较
Static / BC / Temporal Greedy / Double-FQI 的 Where 与 When。

**本仓库现有数据不支持这个设计。** 这个脚本不跑一个假的回测，而是把实际
覆盖情况算出来、写清楚缺什么才能解锁，并以退出码 3 标记为 BLOCKED
（区别于失败的 1）。

三条硬约束：

1. **时间覆盖不足。** 光明区街道—年份面板只有 2010–2018 有非零记录
   （2019 年以后全为 0），2010 年之前没有任何记录。
   "1985–2009 训练"没有数据可训。
2. **粒度对不上。** 记录是**街道 × 年**的汇总计划数，而模型的决策对象是
   717 个候选单元；`gm_renewal_units.csv` 里 52 个具名项目没有 uid，
   候选单元表也没有街道字段，两边无法连接。
3. **这批记录已被用作标定输入，再当留出集是循环论证。**
   `temporal_v0/calib.json` 写明：hazard 来自 109 对"计划公告→单元规划批准"
   配对的逐年条件批准率、tau_valid 来自配对时滞中位数、
   quota=3 来自"光明区 2011-2018 计划公告均值 3.1"。

解锁 S4 需要的最小数据（按优先级）：

    ① 更新单元计划批次名单（含单元边界或地块坐标 + 公告年份）
       —— 能把事件落到 717 个候选单元上，Where 与 When 才有逐单元真值
    ② 若只能到街道级：候选单元的街道归属字段（或街道边界图层做空间连接）
       —— 可退而做街道 × 年的分布比较，但 4 个街道 × 9 年 × 34 个事件
       的样本量只够做描述性对照，不足以支撑"精度改善"的论断
    ③ 2019 年以后的计划公告记录 —— 现有面板在 2019 起全为 0，
       无法判断是真的没有还是数据未更新
"""
import json
import os
import sys

import pandas as pd

BASE = "data/processed/gm_dataset_v1"
REQ = ["① 计划批次名单含单元边界/坐标 + 公告年份（逐单元真值）",
       "② 或：候选单元的街道归属字段（退到街道级对照）",
       "③ 2019 年以后的计划公告记录（现有面板 2019 起全为 0）"]


def main():
    out = os.environ.get("S4_OUT", "results_srv_s4")
    os.makedirs(out, exist_ok=True)
    rep = {"结论": "BLOCKED", "缺什么": REQ, "实际覆盖": {}}

    p = pd.read_csv(f"{BASE}/tables/sz_renewal_panel_street_year.csv",
                    encoding="utf-8-sig")
    gm = p[p["区位"].astype(str) == "光明区"].copy()
    gm["年"] = pd.to_numeric(gm["年"], errors="coerce")
    per_year = gm.groupby("年")["计划数"].sum()
    nz = per_year[per_year > 0]
    rep["实际覆盖"]["街道年面板"] = dict(
        年范围=[float(gm["年"].min()), float(gm["年"].max())],
        非零年范围=[float(nz.index.min()), float(nz.index.max())] if len(nz) else None,
        事件总数=float(per_year.sum()), 街道数=int(gm["街道N"].nunique()),
        逐年计划数={str(int(k)): float(v) for k, v in per_year.items()})

    u = pd.read_csv(f"{BASE}/zones_v0/candidate_units.csv", encoding="utf-8-sig")
    rep["实际覆盖"]["候选单元表"] = dict(
        行数=int(len(u)), 列=list(u.columns),
        有街道字段=bool(any("街道" in str(c) for c in u.columns)))

    g = pd.read_csv(f"{BASE}/tables/gm_renewal_units.csv", encoding="utf-8-sig")
    rep["实际覆盖"]["具名项目表"] = dict(
        行数=int(len(g)), 列=list(g.columns),
        有uid=bool(any(str(c).lower() == "uid" for c in g.columns)))

    mp = pd.read_csv(f"{BASE}/tables/sz_renewal_matched_pairs.csv",
                     encoding="utf-8-sig")
    g2 = mp[mp["区"].astype(str).str.contains("光明", na=False)]
    rep["实际覆盖"]["公告→批准配对"] = dict(光明条数=int(len(g2)),
                                   全市条数=int(len(mp)))

    cal = json.load(open(f"{BASE}/temporal_v0/calib.json", encoding="utf-8"))
    rep["循环论证风险"] = cal.get("calibration_source", {})

    with open(os.path.join(out, "S4_STATUS.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "S4_STATUS.md"), "w", encoding="utf-8") as f:
        f.write("# S4 历史回测：BLOCKED（数据不足，非代码问题）\n\n")
        f.write(f"- 街道年面板非零年份：{rep['实际覆盖']['街道年面板']['非零年范围']}"
                f"，事件总数 {rep['实际覆盖']['街道年面板']['事件总数']:.0f}"
                f"，街道 {rep['实际覆盖']['街道年面板']['街道数']} 个\n")
        f.write(f"- 候选单元表有街道字段：{rep['实际覆盖']['候选单元表']['有街道字段']}；"
                f"具名项目表有 uid：{rep['实际覆盖']['具名项目表']['有uid']}"
                f" → 两边无法连接\n")
        f.write(f"- 光明区公告→批准配对仅 {rep['实际覆盖']['公告→批准配对']['光明条数']} 条\n")
        f.write("- **这批记录已被用作标定输入**（见 calib.json）："
                "hazard / tau_valid / quota 皆由其导出，再当留出集是循环论证\n\n")
        f.write("## 解锁所需数据\n\n")
        for r in REQ:
            f.write(f"- {r}\n")
    print("S4 = BLOCKED（数据不足）：")
    print(f"  非零年份 {rep['实际覆盖']['街道年面板']['非零年范围']}，"
          f"事件 {rep['实际覆盖']['街道年面板']['事件总数']:.0f} 个，"
          f"单元表有街道字段 {rep['实际覆盖']['候选单元表']['有街道字段']}，"
          f"项目表有 uid {rep['实际覆盖']['具名项目表']['有uid']}")
    print(f"  详见 {out}/S4_STATUS.md")
    return 3


if __name__ == "__main__":
    sys.exit(main())
