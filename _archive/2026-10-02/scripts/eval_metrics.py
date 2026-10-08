"""eval_metrics.py —— 从既有批次的 rec_a*_s*.csv 重算实施类指标，并出三层评价指标表。

只**读**落盘结果，不训练、不改写任何情景全局、不碰 rl/scale.py 的参考策略集。
故对任何既有批次都可反复重跑，结果逐位可复算。

用法
----
    export PYTHONPATH=src
    python scripts/eval_metrics.py \
        --batch data/processed/gm_dataset_v1/exp_v12 \
        --groups carry1,carry3,carryInf,budget1500 \
        --out data/processed/gm_dataset_v1/eval_v15

产出（--out 目录下）
    三层评价指标表.csv        行=指标，列=层/定义/时间口径/数据来源/可跨情景比/备注
    实施类指标_逐运行.csv     每个 (组, alpha, seed) 一行
    实施类指标_分组.csv       每组一行：三层指标合表（实施层 + 目标层均值 + 标量层）
    自查结果.csv              两条硬自查的逐组实际结果

情景参数从哪来
--------------
优先取 runs.json 里 runs[*].diag 记录的**实际生效值**（t_dec_eff / t_eval_eff /
budget_eff / carry_eff），而不是 config 里的**意图值**。v12 有三组因分母构建泄漏
FAR_GROWTH，意图值与实际运行不符——教训是：凡有 *_eff 就用 *_eff。
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tpmorl.eval.metrics import (          # noqa: E402
    ScenarioSpec, implementation_metrics, three_layer_table,
)

_KEY = re.compile(r"rec_a([0-9.]+)_s(\d+)\.csv$")
#: exp_v12 中因容积率动力学污染而作废的组，默认拒绝参与统计（见 docs/验收_v12）。
#: 这是**批次内**的组名，不是全局黑名单：后续批次同名的组（v15 就有 budget600）
#: 是干净的。若按裸目录名跨批次匹配，会把干净的新组误判为污染而拒统——
#: 故只在批次目录确实是 exp_v12 时才启用这份名单。
TAINTED_V12 = ("carry6", "budget600", "loose")


def _tainted(batch_dir):
    """返回本批次需拒统的组名集合。仅 exp_v12 非空。"""
    return set(TAINTED_V12) if "v12" in os.path.basename(
        os.path.normpath(batch_dir)) else set()
_OBJ_NAMES = ("Gdp", "Eco", "Res", "Emp", "Aec", "E2r", "Cpt",
              "Floor", "Cost", "Disrupt", "Expire")


def spec_of_group(gdir: str, override: dict) -> tuple[ScenarioSpec, dict]:
    """从该组的 runs.json 解析情景参数。返回 (spec, 生效值来源记录)。"""
    with open(os.path.join(gdir, "runs.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cfg = meta.get("config", {})
    diags = [r.get("diag", {}) for r in meta.get("runs", []) if isinstance(r, dict)]
    diags = [d for d in diags if d]

    def eff(key, cfg_key, cast):
        """优先取 diag 里的实际生效值，其次 config 的意图值。"""
        vals = {d[key] for d in diags if key in d}
        if len(vals) == 1:
            return cast(vals.pop()), "diag"
        if len(vals) > 1:
            raise SystemExit(f"{gdir}: diag.{key} 组内不一致 {sorted(vals)}，拒绝统计")
        return cast(cfg[cfg_key]), "config"

    def _parse_hazard(v):
        """诊断里的 hazard_eff 是 "0.174,0.212,..." 形式的字符串（落盘时按 4 位
        有效数字格式化，对交付年数这类口径足够）。也兼容列表形式，便于以后改格式。
        """
        if isinstance(v, str):
            return tuple(float(x) for x in v.split(",") if x.strip())
        return tuple(float(x) for x in v)

    def _parse_build_years(v):
        """诊断里的 build_years_eff 是 "1:5,2:3,..." 形式的字符串（逐通道），
        也可能是标量或（经 json 往返后键为字符串的）字典。一律归一为
        {通道号:int -> 年数:int} 或标量 int。通道号必须转回 int，否则查表恒不
        命中而静默落到默认值。
        """
        if isinstance(v, dict):
            return {int(k): int(x) for k, x in v.items()}
        if isinstance(v, str):
            if ":" in v:
                out = {}
                for part in v.split(","):
                    if not part.strip():
                        continue
                    k, x = part.split(":")
                    out[int(k)] = int(x)
                return out
            return int(float(v))
        return int(v)

    def eff_opt(key, cast):
        """同 eff，但用于 list / dict 这类不可哈希的值，且允许缺失（返回 None）。

        缺失是**正常情况**：v14 及更早批次的 diag 没有这些字段，那些批次也没有
        扫过对应参数，落回 ScenarioSpec 的出厂默认值即正确。只有本批次扫了该
        参数、diag 里却读不到时才是隐患——那种情形由启动脚本的自检拦住。
        """
        seen = {}
        for d in diags:
            if key in d and d[key] is not None:
                seen[json.dumps(d[key], sort_keys=True)] = d[key]
        if len(seen) == 1:
            return cast(next(iter(seen.values()))), "diag"
        if len(seen) > 1:
            raise SystemExit(f"{gdir}: diag.{key} 组内不一致（{len(seen)} 种取值），拒绝统计")
        return None, "缺失"

    T, src_T = eff("t_dec_eff", "horizon", int)
    T_eval, src_Te = eff("t_eval_eff", "horizon", int)
    budget, src_B = eff("budget_eff", "budget", float)
    carry, src_C = eff("carry_eff", "carry", float)

    kw = {}

    # 有效期：诊断里的 tau_valid_eff/tau_ext_eff 是生效值，优先于 config 的意图值。
    tvd, src_tv = eff_opt("tau_valid_eff", int)
    ted, src_te2 = eff_opt("tau_ext_eff", int)
    if tvd is not None and ted is not None:
        kw["tau_max"] = tvd + ted
        src_tau = "diag"
    else:
        tv, te = cfg.get("tau_valid"), cfg.get("tau_ext")
        src_tau = "缺失"
        if tv is not None or te is not None:
            from tpmorl.env.schedule import TAU_VALID, TAU_EXT
            kw["tau_max"] = (int(TAU_VALID if tv is None else tv)
                             + int(TAU_EXT if te is None else te))
            src_tau = "config"

    # 逐有效年条件批准率：`--tau-approval` 会把它整条换成常数风险率。这一项不读
    # 生效值的后果是实测过的——同一批 rec 仅换 hazard，CRH_T 0.2991→0.3909、
    # 期满作废率 0.2940→0.1317，即审批时序那几组的实施类指标会全部算错。
    hz, src_hz = eff_opt("hazard_eff", _parse_hazard)
    if hz is not None:
        kw["hazard"] = hz

    # 建设年限：逐通道字典经 json 往返后键是字符串，必须转回 int，否则按通道查
    # 表恒不命中而静默落到默认值。
    byd, src_by = eff_opt("build_years_eff", _parse_build_years)
    if byd is not None:
        kw["build_years"] = byd
    else:
        by = cfg.get("build_years")
        if by is not None:
            src_by = "config"
            kw["build_years"] = ({int(k): int(v) for k, v in by.items()}
                                 if isinstance(by, dict) else int(by))
    # 年度配额：窗内立项位的上限之一。读生效值 quota_eff，与 hazard 同理——
    # 扫配额那几组若按默认 3 算，窗内占用率的分母就错了。
    qd, src_q = eff_opt("quota_eff", int)
    if qd is not None:
        kw["quota"] = qd
    kw.update(T=T, T_eval=T_eval, budget=budget,
              carry_cap=(None if not np.isfinite(carry) else carry))
    # CLI 覆盖最后生效（--T / --T-eval / --tau-max），故必须放在最后一步 update
    kw.update({k: v for k, v in override.items() if v is not None})
    spec = ScenarioSpec(**kw)
    prov = dict(T来源=src_T, T_eval来源=src_Te, budget来源=src_B, carry来源=src_C,
                tau_max来源=src_tau, hazard来源=src_hz, build_years来源=src_by,
                quota来源=src_q, quota=spec.quota,
                tau_max=spec.tau_max, T=spec.T, T_eval=spec.T_eval,
                budget=spec.budget, carry_cap=spec.carry_cap,
                hazard=",".join(f"{x:.4f}" for x in spec.hazard),
                build_years=json.dumps(spec.build_years, sort_keys=True))
    return spec, prov


def objective_means(gdir: str) -> dict:
    """目标层：11 维原始量纲目标值在 (alpha, seed) 上的均值。"""
    p = os.path.join(gdir, "objectives.csv")
    if not os.path.exists(p):
        return {}
    O = pd.read_csv(p)
    return {f"目标_{k}": float(O[k].mean()) for k in _OBJ_NAMES if k in O.columns}


def scalar_return_means(gdir: str) -> dict:
    """标量层：标量化折扣回报。取 runs.json runs[*].curve 的尾段均值。

    尾段而非末点：单点受回合随机性影响大。尾段长度取曲线后 10%（至少 1 点）。
    """
    p = os.path.join(gdir, "runs.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        meta = json.load(f)
    tails = []
    for r in meta.get("runs", []):
        c = r.get("curve") or []
        if c:
            k = max(1, len(c) // 10)
            tails.append(float(np.mean(c[-k:])))
    if not tails:
        return {}
    return dict(标量层_折扣回报尾段均值=float(np.mean(tails)),
                标量层_折扣回报尾段标准差=float(np.std(tails)))


def scalar_return_per_run(gdir: str) -> pd.DataFrame:
    """逐 (alpha, seed) 的标量化折扣回报（曲线尾段均值）。用于**组内**相关性分析。

    只在组内使用：跨情景不可比（分母依赖情景），跨 alpha 也不可比（权重不同），
    故下游必须同时按 (组, alpha) 分层后才做相关。
    """
    with open(os.path.join(gdir, "runs.json"), encoding="utf-8") as f:
        meta = json.load(f)
    rows = []
    for r in meta.get("runs", []):
        c = r.get("curve") or []
        if not c:
            continue
        k = max(1, len(c) // 10)
        rows.append(dict(alpha=float(r["alpha"]), seed=int(r["seed"]),
                         标量回报=float(np.mean(c[-k:]))))
    return pd.DataFrame(rows)


def separation_analysis(per_run: pd.DataFrame) -> pd.DataFrame:
    """"落地程度"与"得分"是否区分得开：组内、按 alpha 分层后的相关系数。

    必须**按 (组, alpha) 分层**再合并：跨情景分母不同、跨 alpha 权重不同，
    直接混算相关会把口径差异当成相关性。做法是在每个 (组, alpha) 内（n=种子数）
    算 Pearson r，再对各层取均值（Fisher z 变换后平均，回变）。

    r 接近 0 = 两类指标彼此独立，即实施层带来了标量层没有的信息（指标区分得开）；
    |r| 接近 1 = 实施层是标量层的冗余重述（指标没必要单列）。
    """
    out = []
    for m in ("CRH_T", "LSR", "expire_rate_T", "idle_ratio"):
        if "标量回报" not in per_run.columns:
            continue
        zs, ns = [], []
        for (_, _), sub in per_run.groupby(["组", "alpha"]):
            s = sub[[m, "标量回报"]].dropna()
            if len(s) < 3 or s[m].std() == 0 or s["标量回报"].std() == 0:
                continue
            r = float(np.corrcoef(s[m], s["标量回报"])[0, 1])
            zs.append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
            ns.append(len(s))
        if zs:
            # 报中位数与最小值而非均值：立项数为 0 的运行其比率指标是 0/0（NaN），
            # 会被 dropna 剔掉，使个别层的样本数少于种子数。取 int(均值) 会把
            # 4.96 截成 4，反而误导。v12 实测有 1 个这样的运行（budget1500/α=0/s2）。
            out.append(dict(实施层指标=m, 对比对象="标量层_折扣回报",
                            分层数=len(zs), 每层样本数中位数=int(np.median(ns)),
                            最小层样本数=int(min(ns)),
                            组内分层平均相关r=float(np.tanh(np.mean(zs))),
                            解读=("接近 0：实施层携带标量层没有的信息"
                                  if abs(np.tanh(np.mean(zs))) < 0.3
                                  else "相关较强：与标量层部分冗余")))
    return pd.DataFrame(out)


def run_group(gdir: str, group: str, override: dict) -> tuple[pd.DataFrame, dict, dict]:
    spec, prov = spec_of_group(gdir, override)
    files = sorted(glob.glob(os.path.join(gdir, "runs", "rec_a*_s*.csv")))
    if not files:
        raise SystemExit(f"{gdir}/runs 里没有 rec_a*_s*.csv——v7 之前的批次没落盘逐年记录")
    rows, recs = [], []
    for f in files:
        m = _KEY.search(os.path.basename(f))
        R = pd.read_csv(f)
        R.columns = [c.lstrip("\ufeff") for c in R.columns]   # utf-8-sig 的 BOM
        recs.append(R)
        rows.append(dict(组=group, alpha=float(m.group(1)), seed=int(m.group(2)),
                         **implementation_metrics(R, spec)))
    per_run = pd.DataFrame(rows).sort_values(["alpha", "seed"]).reset_index(drop=True)
    sr = scalar_return_per_run(gdir)
    if not sr.empty:
        per_run = per_run.merge(sr, on=["alpha", "seed"], how="left")

    # 分组汇总：把全部运行的立项事件**池化**后重算，而不是对逐运行指标取均值。
    # 比率指标的均值不等于池化比率（各运行立项数不同），池化才是"该组的落地程度"。
    pooled = pd.concat([r.assign(ep=r["ep"] + 1000 * i) for i, r in enumerate(recs)],
                       ignore_index=True)
    g = dict(组=group, n_run=len(files), **implementation_metrics(pooled, spec), **prov)
    g.update(objective_means(gdir))
    g.update(scalar_return_means(gdir))
    return per_run, g, prov


def main():
    ap = argparse.ArgumentParser(description="重算实施类指标并出三层评价指标表")
    ap.add_argument("--batch", required=True, help="批次目录，其下每个子目录是一组")
    ap.add_argument("--groups", default="", help="逗号分隔的组名；留空=自动取全部干净组")
    ap.add_argument("--out", required=True, help="产出目录")
    ap.add_argument("--allow-tainted", action="store_true",
                    help="允许统计已知被污染的组（默认拒绝）")
    ap.add_argument("--T", type=int, default=None, help="覆盖决策期（一般不要用）")
    ap.add_argument("--T-eval", type=int, default=None, help="覆盖评价期")
    ap.add_argument("--tau-max", type=int, default=None, help="覆盖有效期上限")
    a = ap.parse_args()

    tainted = _tainted(a.batch)
    groups = [x for x in a.groups.split(",") if x]
    if not groups:
        groups = sorted(d for d in os.listdir(a.batch)
                        if os.path.isdir(os.path.join(a.batch, d))
                        and d not in tainted)
    bad = [g for g in groups if g in tainted]
    if bad and not a.allow_tainted:
        raise SystemExit(f"拒绝统计 {os.path.basename(a.batch)} 的已知污染组 {bad}"
                         f"（容积率动力学污染）；确实要算请加 --allow-tainted")

    override = dict(T=a.T, T_eval=a.T_eval, tau_max=a.tau_max)
    os.makedirs(a.out, exist_ok=True)
    runs, grps, checks = [], [], []
    for g in groups:
        pr, gs, _ = run_group(os.path.join(a.batch, g), g, override)
        runs.append(pr)
        grps.append(gs)
        # ---- 自查 1：CRH 的两个口径必须满足 CRH_Teval >= CRH_T ----
        viol = int((pr["CRH_Teval"] < pr["CRH_T"] - 1e-12).sum())
        # ---- 自查 2：LSR 与"第 6 年及以后立项占比"应同向 ----
        sub = pr[["LSR", "late_share_late"]].dropna()
        rho = (float(np.corrcoef(sub["LSR"], sub["late_share_late"])[0, 1])
               if len(sub) > 2 and sub["LSR"].std() > 0 and sub["late_share_late"].std() > 0
               else float("nan"))
        checks.append(dict(组=g,
                           自查1_CRH单调_违例运行数=viol,
                           自查1_通过=bool(viol == 0),
                           组池化_CRH_T=gs["CRH_T"], 组池化_CRH_Teval=gs["CRH_Teval"],
                           闭区间口径=bool(gs["T"] == gs["T_eval"]),
                           自查2_LSR=gs["LSR"],
                           自查2_起始年号=gs["late_share_y0"],
                           自查2_起始年及以后立项占比=gs["late_share_late"],
                           自查2_再晚一年及以后立项占比=gs["late_share_later"],
                           自查2_逐运行相关系数=rho,
                           自查2_同向=bool(np.isnan(rho) or rho > 0)))

    per_run = pd.concat(runs, ignore_index=True)
    per_grp = pd.DataFrame(grps)
    chk = pd.DataFrame(checks)
    enc = "utf-8-sig"
    per_run.to_csv(os.path.join(a.out, "实施类指标_逐运行.csv"), index=False, encoding=enc)
    per_grp.to_csv(os.path.join(a.out, "实施类指标_分组.csv"), index=False, encoding=enc)
    chk.to_csv(os.path.join(a.out, "自查结果.csv"), index=False, encoding=enc)
    three_layer_table().to_csv(os.path.join(a.out, "三层评价指标表.csv"),
                               index=False, encoding=enc)
    sep = separation_analysis(per_run)
    if not sep.empty:
        sep.to_csv(os.path.join(a.out, "落地与得分_区分度.csv"), index=False, encoding=enc)

    cols = ["组", "n_initiated", "CRH_T", "CRH_Teval", "ACD_T", "ACD_T_p50",
            "LSR", "expire_rate_T", "expire_rate_uncond", "idle_ratio"]
    print(per_grp[cols].to_string(index=False))
    print()
    # 列名随 F-05 改名（阈值不再写死第 6 年，改由期望交付时长推出），这里同步。
    print(chk[["组", "自查1_通过", "自查2_LSR", "自查2_起始年号",
               "自查2_起始年及以后立项占比",
               "自查2_逐运行相关系数", "自查2_同向"]].to_string(index=False))
    if not sep.empty:
        print()
        print(sep.to_string(index=False))


if __name__ == "__main__":
    main()
