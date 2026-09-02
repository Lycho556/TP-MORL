"""
renewal_unit_ingest.py — 深圳城市更新单元计划公告 → 单元生命周期事件表

用途：把各区《城市更新单元计划一览表》(xlsx/csv/pdf) 与各类有效期/延期/失效公告，
归一为两张表，作为"层面1（地块状态生命周期）+ 层面2（法规时序动作掩码）"RL 环境的数据底座。

用法：
    python renewal_unit_ingest.py <输入目录> -o <输出目录>
输入目录可混放 .xlsx/.xls/.csv/.pdf；文件名建议含区名与年份批次，例如
    光明_2023_第一批_一览表.xlsx / 龙岗_2025_统一设定有效期公告.pdf
"""
from __future__ import annotations
import re, sys, json, argparse, unicodedata
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------- 模式定义

UNIT_SCHEMA = [
    "unit_id", "unit_name", "district", "subdistrict", "batch_year", "batch_no",
    "demolition_area_m2",        # 拆除范围用地面积
    "dev_land_area_m2",          # 开发建设用地面积
    "transfer_land_area_m2",     # 移交政府用地面积
    "renewal_direction",         # 更新方向（功能类型串）
    "applicant",                 # 申报主体
    "source_file",
]

EVENT_TYPES = [
    "plan_draft_public",   # 计划草案公示
    "plan_announce",       # 计划公告（入库）
    "validity_set",        # 设定计划有效期
    "validity_extend",     # 有效期顺延/延期调整
    "plan_adjust",         # 计划调整（范围/方向变更）
    "sp_draft_public",     # 单元规划草案公示
    "sp_approved",         # 单元规划批准
    "lapsed",              # 计划失效
]

EVENT_SCHEMA = [
    "unit_id", "unit_name", "district", "event_type", "event_date",
    "valid_from", "valid_to", "field_changed", "value_before", "value_after",
    "source_file",
]

# 中文列名 → 规范字段。匹配为"包含"关系，按顺序首次命中即用。
COLMAP = [
    ("unit_name",           ["更新单元名称", "单元名称", "项目名称", "更新单元"]),
    ("subdistrict",         ["街道"]),
    ("demolition_area_m2",  ["拆除范围用地面积", "拆除范围面积", "拆除用地面积", "拆除范围"]),
    ("dev_land_area_m2",    ["开发建设用地面积", "开发建设用地"]),
    ("transfer_land_area_m2",["移交政府用地面积", "移交用地面积", "移交用地"]),
    ("renewal_direction",   ["更新方向", "功能"]),
    ("applicant",           ["申报主体", "实施主体", "申报单位"]),
]

DATE_RE = re.compile(r"(20\d{2})\s*[年\-/\.]\s*(\d{1,2})\s*[月\-/\.]\s*(\d{1,2})")
RANGE_RE = re.compile(
    r"(20\d{2}\s*[年\-/\.]\s*\d{1,2}\s*[月\-/\.]\s*\d{1,2}\s*日?)\s*(?:至|到|—|-|~)\s*"
    r"(20\d{2}\s*[年\-/\.]\s*\d{1,2}\s*[月\-/\.]\s*\d{1,2}\s*日?)")
AREA_RE = re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:平方米|m2|㎡)")

EVENT_KEYWORDS = [
    ("lapsed",           ["失效"]),
    ("validity_extend",  ["顺延", "延期", "有效期调整", "计划调整"]),
    ("validity_set",     ["设定计划有效期", "设定有效期", "有效期的公告"]),
    ("sp_approved",      ["单元规划批准", "规划经批准", "批复"]),
    ("sp_draft_public",  ["单元规划草案", "规划草案公示"]),
    ("plan_draft_public",["计划（草案）", "计划草案", "公示"]),
    ("plan_announce",    ["计划的公告", "计划公告", "批准的公告"]),
]


def _norm(s) -> str:
    if s is None: return ""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(s)))


def _to_date(s: str) -> str | None:
    m = DATE_RE.search(_norm(s))
    if not m: return None
    y, mo, d = map(int, m.groups())
    return f"{y:04d}-{mo:02d}-{d:02d}"


def _to_float(s) -> float | None:
    if s is None or (isinstance(s, float) and pd.isna(s)): return None
    t = _norm(s).replace(",", "")
    m = re.search(r"\d+(?:\.\d+)?", t)
    return float(m.group()) if m else None


def parse_meta_from_name(path: Path) -> dict:
    n = _norm(path.stem)
    district = next((d for d in ["光明","龙岗","罗湖","宝安","龙华","南山","福田","盐田","坪山","大鹏"]
                     if d in n), None)
    year = (re.search(r"(20\d{2})", n) or [None])
    year = int(year.group(1)) if hasattr(year, "group") else None
    m = re.search(r"第([一二三四五六七八九十\d]+)批", n)
    return {"district": district, "batch_year": year, "batch_no": m.group(1) if m else None}


# ---------------------------------------------------------------- 表格读取

def read_tables(path: Path) -> list[pd.DataFrame]:
    """返回文件中所有候选表格。xlsx/csv 直读；pdf 用 pdfplumber 抽表。"""
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        return [df for df in pd.read_excel(path, sheet_name=None, header=None).values()]
    if suf == ".csv":
        return [pd.read_csv(path, header=None, encoding_errors="replace")]
    if suf == ".pdf":
        import pdfplumber                     # pip install pdfplumber
        out = []
        with pdfplumber.open(path) as pdf:
            for pg in pdf.pages:
                for t in pg.extract_tables() or []:
                    if len(t) > 1:
                        out.append(pd.DataFrame(t))
        return out
    return []


def locate_header(df: pd.DataFrame, max_scan: int = 8) -> int | None:
    """一览表常有多行标题，找到含'单元名称'类关键词的那一行作为表头。"""
    for i in range(min(max_scan, len(df))):
        row = " ".join(_norm(v) for v in df.iloc[i].tolist())
        if any(k in row for k in ["更新单元名称", "单元名称", "项目名称"]):
            return i
    return None


def map_columns(cols: list[str]) -> dict[int, str]:
    out = {}
    for j, c in enumerate(cols):
        cn = _norm(c)
        for field, keys in COLMAP:
            if field in out.values(): continue
            if any(k in cn for k in keys):
                out[j] = field; break
    return out


def extract_units(path: Path) -> pd.DataFrame:
    meta = parse_meta_from_name(path)
    rows = []
    for df in read_tables(path):
        h = locate_header(df)
        if h is None: continue
        cmap = map_columns(df.iloc[h].tolist())
        if "unit_name" not in cmap.values(): continue
        body = df.iloc[h + 1:]
        for _, r in body.iterrows():
            rec = {f: None for f in UNIT_SCHEMA}
            rec.update(meta); rec["source_file"] = path.name
            for j, field in cmap.items():
                v = r.iloc[j] if j < len(r) else None
                rec[field] = _to_float(v) if field.endswith("_m2") else (_norm(v) or None)
            if not rec["unit_name"] or len(rec["unit_name"]) < 3: continue
            if _norm(rec["unit_name"]) in {"合计", "小计", "总计"}: continue
            rec["unit_id"] = f"{rec['district'] or '?'}|{rec['unit_name']}"
            rows.append(rec)
    return pd.DataFrame(rows, columns=UNIT_SCHEMA)


# ---------------------------------------------------------------- 事件抽取

def pdf_text(path: Path) -> str:
    if path.suffix.lower() != ".pdf": return ""
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return "\n".join(pg.extract_text() or "" for pg in pdf.pages)


def classify_event(text: str) -> str | None:
    t = _norm(text)
    for et, keys in EVENT_KEYWORDS:
        if any(k in t for k in keys): return et
    return None


def extract_events(path: Path, units: pd.DataFrame) -> pd.DataFrame:
    """从公告正文抽取事件类型、事件日期、有效期起止；表格内若含'调整前/调整后'列则记为字段变更。"""
    meta = parse_meta_from_name(path)
    text = pdf_text(path)
    et = classify_event(path.stem + " " + text[:800])
    ev_date = _to_date(text[-1200:]) or _to_date(text[:1200])
    rng = RANGE_RE.search(_norm(text))
    vfrom, vto = (_to_date(rng.group(1)), _to_date(rng.group(2))) if rng else (None, None)

    names = [n for n in units["unit_name"].dropna().unique()] if len(units) else []
    rows = []
    hit = [n for n in names if _norm(n) and _norm(n) in _norm(text)]
    for n in (hit or names or [None]):
        rows.append({"unit_id": f"{meta['district'] or '?'}|{n}" if n else None,
                     "unit_name": n, "district": meta["district"], "event_type": et,
                     "event_date": ev_date, "valid_from": vfrom, "valid_to": vto,
                     "field_changed": None, "value_before": None, "value_after": None,
                     "source_file": path.name})

    # 调整类公告：若表格含"调整前/调整后"，逐字段记录变更
    for df in read_tables(path):
        flat = " ".join(_norm(v) for v in df.astype(str).values.ravel()[:400])
        if "调整前" in flat and "调整后" in flat:
            h = locate_header(df)
            if h is None: continue
            hdr = [_norm(v) for v in df.iloc[h].tolist()]
            for _, r in df.iloc[h + 1:].iterrows():
                cells = [_norm(v) for v in r.tolist()]
                nm = next((c for c in cells if len(c) > 4 and ("片区" in c or "工业区" in c or "旧村" in c)), None)
                if not nm: continue
                for j, hh in enumerate(hdr):
                    if "调整前" in hh and j + 1 < len(cells):
                        rows.append({"unit_id": f"{meta['district'] or '?'}|{nm}", "unit_name": nm,
                                     "district": meta["district"], "event_type": "plan_adjust",
                                     "event_date": ev_date, "valid_from": vfrom, "valid_to": vto,
                                     "field_changed": hh.replace("调整前", "").strip() or "unknown",
                                     "value_before": cells[j], "value_after": cells[j + 1],
                                     "source_file": path.name})
    return pd.DataFrame(rows, columns=EVENT_SCHEMA)


# ---------------------------------------------------------------- 完整度量化

def completeness(units: pd.DataFrame, events: pd.DataFrame) -> dict:
    rep: dict = {"n_units": int(len(units)), "n_events": int(len(events))}
    rep["field_missing_rate"] = {
        c: (round(float(units[c].isna().mean()), 4) if len(units) else None)
        for c in UNIT_SCHEMA if c not in {"unit_id", "source_file"}
    }
    rep["events_by_type"] = (events["event_type"].value_counts(dropna=False)
                             .rename(index=lambda k: str(k)).to_dict() if len(events) else {})
    rep["units_by_district_year"] = (units.groupby(["district", "batch_year"], dropna=False)
                                     .size().rename("n").reset_index()
                                     .astype({"batch_year": "object"}).to_dict("records")
                                     if len(units) else [])
    # 生命周期链完整性：入库 → 有效期 → （延期）→ 批复/失效
    if len(events):
        g = events.groupby("unit_id")["event_type"].apply(set)
        rep["lifecycle"] = {
            "has_entry":     int(sum("plan_announce" in s or "plan_draft_public" in s for s in g)),
            "has_validity":  int(sum("validity_set" in s or "validity_extend" in s for s in g)),
            "has_terminal":  int(sum(bool({"sp_approved", "lapsed"} & s) for s in g)),
            "full_chain":    int(sum(bool(({"plan_announce","plan_draft_public"} & s)
                                          and ({"validity_set","validity_extend"} & s)
                                          and ({"sp_approved","lapsed"} & s)) for s in g)),
            "n_units_with_events": int(len(g)),
        }
        # RL 可用的时序转移样本：同一单元上按日期排序的相邻事件对
        e = events.dropna(subset=["unit_id", "event_date"]).sort_values("event_date")
        rep["n_transitions"] = int(sum(max(0, n - 1) for n in e.groupby("unit_id").size()))
    # 有效期窗口可解析率（层面2动作掩码的直接依赖）
    if len(events):
        rep["validity_window_parsed_rate"] = round(
            float(events["valid_to"].notna().mean()), 4)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("indir"); ap.add_argument("-o", "--outdir", default="renewal_out")
    a = ap.parse_args(argv)
    ind, out = Path(a.indir), Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(ind.rglob("*"))
             if p.suffix.lower() in {".xlsx", ".xls", ".csv", ".pdf"}]
    if not files:
        raise SystemExit(f"no input tables found under {ind}")

    U = pd.concat([extract_units(p) for p in files], ignore_index=True)
    U = U.drop_duplicates(subset=["unit_id", "batch_year", "batch_no"])
    E = pd.concat([extract_events(p, U[U["district"] == parse_meta_from_name(p)["district"]])
                   for p in files], ignore_index=True)
    E = E.dropna(how="all", subset=["event_type", "event_date"]).drop_duplicates()

    U.to_csv(out / "units.csv", index=False, encoding="utf-8-sig")
    E.to_csv(out / "events.csv", index=False, encoding="utf-8-sig")
    rep = completeness(U, E)
    rep["files_parsed"] = [p.name for p in files]
    (out / "completeness_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return rep


if __name__ == "__main__":
    main()
