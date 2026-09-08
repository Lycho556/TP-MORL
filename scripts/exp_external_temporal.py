"""实验③：用光明区真实项目做时序外部检验（粗粒度口径）。

## 为什么只能做到"粗粒度"

黄冠老师访谈第 4 条：实施年份本身不可精确预测，**不该拿"模型第几年"逐年去比真实
年份**，只应比先后次序（Spearman）或按 3／5 年分箱比。本脚本按此口径。

## 为什么做不了项目级检验（重要限定）

自评 v1 §结尾把这一项写成"14 个真实项目的时序外部检验"。实测该数字不成立，
且项目级检验在现有数据下**不可执行**，两条独立原因：

1. **没有空间键**。模型的 717 个单元由 100 m 格网聚合而成；真实项目表
   （`tables/gm_renewal_units.csv`）只有名称、街道名与面积，没有坐标或边界。
   更新单元边界在深圳开放数据平台上不存在（见数据缺口清单），街道边界数据集里
   也没有，故真实项目既落不到单元、也落不到街道。
2. **可用配对远少于 14**。光明区有 34 条计划公告（2010–2018）与 18 条规划批准，
   但两端日期都有的匹配对只有 8 条，其中 3 条经名称核对为误匹配（同一个
   「华泰小区更新单元」被分别匹配到薯田埔旧屋村、长兴科技工业园、轨道 13 号线
   车辆段三个不相关项目）。干净配对 **严格判据 4 条、容许命名变体 5 条**
   （「伶伦提可乐旧工业区」对应批准名「伶伦提可乐工业区」，差一个"旧"字）。
   本脚本重跑这道审核，两个数都报。

## 因此本脚本做的是：总量时序节奏比对

不需要空间键：把真实的"逐年计划公告数"与模型的"逐年立项数"当作两条时序剖面比。
它检验的是模型有没有再现真实的**节奏**（前重、后重、还是平），而这正与论文的
时序主张相关。三个口径：

* **Spearman 次序相关**：按年份的公告数／立项数排名比。
* **3 年分箱**与 **5 年分箱**：按访谈口径粗化后比份额。
* **前段份额**：前 1/3 规划期内完成的立项占比，真实与模型各算一个数。

**必须写进论文的限定**：真实 9 年（2010–2018）与模型 15 年规划期长度不同，
本脚本按"份额剖面"比而非绝对数，并把两者各自归一化；这个对齐是一项假设，
不是数据事实。另外真实公告数受当年政策批次节奏影响，不能全部归因于最优择时。

用法：
    python3 scripts/exp_external_temporal.py \
        --runs data/processed/gm_dataset_v1/exp_v7/base/runs --out results_v7/ext
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_KEY = re.compile(r"rec_a([0-9.]+)_s(\d+)\.csv$")
_SUF = re.compile(r"(城市更新单元规划|城市更新单元|更新单元规划|更新单元|规划)$")


def audit_pairs(tables):
    """重跑匹配对审核，返回 (光明全部配对, 判为干净的配对)。"""
    P = pd.read_csv(os.path.join(tables, "sz_renewal_matched_pairs.csv"))
    G = pd.read_csv(os.path.join(tables, "gm_renewal_units.csv"))
    gm = P[P.区.astype(str).str.contains("光明")].copy()
    appr = G[G.拆除用地面积.notna()][["名称", "拆除用地面积"]].copy()
    appr["key"] = appr.拆除用地面积.round(1)

    def core(s):
        return _SUF.sub("", str(s))

    rows = []
    for _, r in gm.iterrows():
        hit = appr[appr.key == round(r.拆除面积, 1)]
        nm = hit.名称.iloc[0] if len(hit) else ""
        c = core(r.计划名)
        # 干净判据：计划名去后缀后的主体名（≥2 字）须整段出现在批准项目名里。
        # 逐字符包含判据是错的——"区"字几乎每个名称都有。
        ok = len(c) >= 2 and c in str(nm)
        # 命名变体：两表对同一项目的写法有"旧"字等出入（实测「伶伦提可乐旧工业区」
        # 对应批准名「伶伦提可乐工业区」）。严格判据会漏掉它，故另给一个宽松判据，
        # 两个数都报，不择其一。
        cr = re.sub(r"[旧新]", "", c)
        loose = ok or (len(cr) >= 2 and cr in re.sub(r"[旧新]", "", str(nm)))
        rows.append(dict(计划名=r.计划名, 批准名=nm, 主体名=c,
                         公告年=r.公告年, 审批年=r.审批年,
                         拆除面积=r.拆除面积, 时滞年=r.时滞年,
                         干净=bool(ok), 宽松=bool(loose)))
    A = pd.DataFrame(rows)
    return A, A[A.宽松]


def real_profile(tables):
    """真实逐年计划公告数（光明区，2010–2018）。"""
    G = pd.read_csv(os.path.join(tables, "gm_renewal_units.csv"))
    P = G[G.类型 == "计划公告"].copy()
    P["年"] = pd.to_datetime(P.时间, errors="coerce").dt.year
    s = P.年.value_counts().sort_index()
    yrs = np.arange(int(s.index.min()), int(s.index.max()) + 1)
    return s.reindex(yrs, fill_value=0)


def model_profile(runs):
    """模型逐年立项数（按权重档，跨种子与回合取均值）。"""
    out = {}
    for p in sorted(glob.glob(os.path.join(runs, "rec_a*_s*.csv"))):
        a = float(_KEY.search(os.path.basename(p)).group(1))
        R = pd.read_csv(p)
        real = R[R.unit >= 0]
        n_ep = max(int(R.ep.nunique()), 1)
        c = real.groupby("year").size() / n_ep
        out.setdefault(a, []).append(c)
    prof = {}
    for a, lst in out.items():
        D = pd.concat(lst, axis=1).fillna(0.0)
        prof[a] = D.mean(axis=1)
    return prof


def _bins(share, k):
    """把份额剖面按 k 年一箱粗化（末箱可不足 k 年），再重新归一化。"""
    v = np.asarray(share, dtype=float)
    b = np.array([v[i:i + k].sum() for i in range(0, len(v), k)])
    return b / b.sum() if b.sum() else b


def compare(real, model, out):
    rs = (real / real.sum()).values
    rows = []
    for a, m in sorted(model.items()):
        ms_full = (m / m.sum()).values
        n = min(len(rs), len(ms_full))
        ms = ms_full[:n]
        ms = ms / ms.sum() if ms.sum() else ms          # 截断后重新归一化
        rho, pv = spearmanr(rs[:n], ms)
        row = dict(alpha=a, n_year_compared=n, spearman=rho, p=pv,
                   real_front_share=float(rs[:max(n // 3, 1)].sum()),
                   model_front_share=float(ms[:max(n // 3, 1)].sum()))
        for k in (3, 5):
            rb, mb = _bins(rs[:n], k), _bins(ms, k)
            row[f"L1_bin{k}"] = float(np.abs(rb - mb).sum())
            # 箱数 <3 时次序相关无定义（2 点的 Spearman 恒为 ±1 或 NaN），只报 L1
            row[f"spearman_bin{k}"] = (float(spearmanr(rb, mb)[0])
                                       if len(rb) >= 3 else np.nan)
        rows.append(row)
    D = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "external_temporal.csv")
    D.to_csv(p, index=False, encoding="utf-8-sig")
    return D, p


def main(runs, tables, out):
    A, clean = audit_pairs(tables)
    print("###### 匹配对审核（光明）")
    print(A[["计划名", "批准名", "公告年", "审批年", "干净", "宽松"]].to_string(index=False))
    print(f"\n配对 {len(A)} 条；严格判据干净 {int(A.干净.sum())} 条，"
          f"容许命名变体后 {int(A.宽松.sum())} 条。"
          f"自评 v1 写的「14 个真实项目」不成立。")

    real = real_profile(tables)
    print("\n###### 真实逐年计划公告数（光明）")
    print(real.to_string())

    model = model_profile(runs)
    if not model:
        raise SystemExit(f"{runs} 里没有 rec_a*_s*.csv——v7 之前的批次没落盘逐年记录")

    D, p = compare(real, model, out)
    print("\n###### 节奏比对（份额剖面，按访谈第 4 条只比次序与分箱）")
    print(D.round(3).to_string(index=False))
    print(f"\n已写出 {p}")
    clean.to_csv(os.path.join(out, "matched_pairs_audit.csv"),
                 index=False, encoding="utf-8-sig")
    return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="含 rec_a*_s*.csv 的目录")
    ap.add_argument("--tables", default="data/processed/gm_dataset_v1/tables")
    ap.add_argument("--out", default="results_v7/ext")
    a = ap.parse_args()
    main(a.runs, a.tables, a.out)
