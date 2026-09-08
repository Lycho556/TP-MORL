#!/usr/bin/env python3
"""md2pdf_cjk.py — 把中文 Markdown 排成 PDF（reportlab + 系统黑体，无需 pandoc/LaTeX）。

支持：# / ## / ### 标题、段落、**粗体**、`代码`、管道表格、> 引用、
有序/无序列表、--- 分隔线。

用法: python md2pdf_cjk.py <input.md> <output.pdf>
"""
import sys, re, html
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, KeepTogether)

REG, BOLD = "Hei", "HeiB"
pdfmetrics.registerFont(TTFont(REG, "/System/Library/Fonts/STHeiti Light.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont(BOLD, "/System/Library/Fonts/STHeiti Medium.ttc", subfontIndex=0))

ACCENT = colors.HexColor("#1f3864")
GREY = colors.HexColor("#666666")
CODE = colors.HexColor("#9c2b2b")
RULE = colors.HexColor("#c8c8c8")
HDR_BG = colors.HexColor("#eaeef5")
ALT_BG = colors.HexColor("#f7f8fa")

S = {
    "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=17, leading=24, textColor=ACCENT,
                         spaceBefore=0, spaceAfter=7),
    "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=13, leading=19, textColor=ACCENT,
                         spaceBefore=14, spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName=BOLD, fontSize=11, leading=17,
                         textColor=colors.HexColor("#2f4f7f"), spaceBefore=10, spaceAfter=3),
    "p": ParagraphStyle("p", fontName=REG, fontSize=9.3, leading=15.6, spaceAfter=5.5),
    "meta": ParagraphStyle("meta", fontName=REG, fontSize=9, leading=15,
                           textColor=GREY, spaceAfter=2.5),
    "li": ParagraphStyle("li", fontName=REG, fontSize=9.3, leading=15.6,
                         leftIndent=13, spaceAfter=3.2, bulletIndent=2),
    "quote": ParagraphStyle("quote", fontName=REG, fontSize=9.3, leading=15.6,
                            leftIndent=11, rightIndent=6, textColor=colors.HexColor("#333333"),
                            spaceBefore=3, spaceAfter=6),
    "th": ParagraphStyle("th", fontName=BOLD, fontSize=8.3, leading=12.2, textColor=ACCENT),
    "td": ParagraphStyle("td", fontName=REG, fontSize=8.3, leading=12.2),
}
for _st in S.values():
    _st.wordWrap = "CJK"
_dummy = {}


CJK = r"\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef"


def join_lines(parts):
    """拼接软换行：中文与中文之间不加空格，中英之间加。"""
    out = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not out:
            out = p
            continue
        if re.search(rf"[{CJK}]$", out) or re.match(rf"^[{CJK}]", p):
            out += p
        else:
            out += " " + p
    return out


def inline(t: str) -> str:
    """Markdown 行内标记 → reportlab 内联标签。"""
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r'<font color="#9c2b2b">\1</font>', t)
    t = re.sub(r"\*\*(.+?)\*\*", rf'<font name="{BOLD}">\1</font>', t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)
    t = t.replace("—", "—").replace("→", "→")
    return t


def split_row(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def build_table(rows, avail):
    header, body = rows[0], rows[1:]
    ncol = len(header)
    # 按各列最长内容分配宽度
    weights = []
    for i in range(ncol):
        cells = [header[i]] + [r[i] if i < len(r) else "" for r in body]
        longest = max((len(re.sub(r"[*`]", "", c)) for c in cells), default=1)
        weights.append(max(3.0, min(longest, 34)) ** 0.72)
    tot = sum(weights)
    widths = [avail * w / tot for w in weights]
    data = [[Paragraph(inline(c), S["th"]) for c in header]]
    for r in body:
        r = r + [""] * (ncol - len(r))
        data.append([Paragraph(inline(c), S["td"]) for c in r[:ncol]])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HDR_BG),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
    ]
    for i in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), ALT_BG))
    t.setStyle(TableStyle(style))
    return t


def convert(md_path, pdf_path, title=None):
    lines = open(md_path, encoding="utf-8").read().split("\n")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=19 * mm, rightMargin=17 * mm,
                            topMargin=17 * mm, bottomMargin=16 * mm,
                            title=title or "", author="")
    avail = doc.width
    flow, i = [], 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if not s:
            i += 1
            continue

        if re.fullmatch(r"-{3,}|\*{3,}", s):
            flow.append(Spacer(1, 3))
            flow.append(HRFlowable(width="100%", thickness=0.5, color=RULE,
                                   spaceBefore=1, spaceAfter=7))
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            lvl = min(len(m.group(1)), 3)
            flow.append(Paragraph(inline(m.group(2)), S[f"h{lvl}"]))
            i += 1
            continue

        # 表格：当前行与下一行是分隔线
        if s.startswith("|") and i + 1 < len(lines) and re.fullmatch(
                r"\|[\s:|-]+\|?", lines[i + 1].strip()):
            rows = [split_row(s)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            flow.append(Spacer(1, 2))
            flow.append(build_table(rows, avail))
            flow.append(Spacer(1, 8))
            continue

        if s.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            txt = join_lines(buf)
            inner = Paragraph(inline(txt), S["quote"])
            t = Table([[inner]], colWidths=[avail])
            t.setStyle(TableStyle([
                ("LINEBEFORE", (0, 0), (0, -1), 2.2, ACCENT),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f6fa")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            flow.append(t)
            flow.append(Spacer(1, 6))
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m:
            flow.append(Paragraph(inline(m.group(2)), S["li"],
                                  bulletText=m.group(1) + "."))
            i += 1
            continue

        if re.match(r"^[-*+]\s+", s):
            flow.append(Paragraph(inline(re.sub(r"^[-*+]\s+", "", s)), S["li"],
                                  bulletText="•"))
            i += 1
            continue

        # 普通段落：合并到下一个空行/结构行
        buf = [s]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", "|", ">", "- ", "* ", "+ "))
                    or re.match(r"^\d+\.\s", nxt) or re.fullmatch(r"-{3,}", nxt)):
                break
            buf.append(nxt)
            i += 1
        txt = join_lines(buf)
        style = S["meta"] if re.match(r"^\*\*(汇报日期|案例区|目标期刊|用途)", txt) else S["p"]
        flow.append(Paragraph(inline(txt), style))

    def footer(canvas, docu):
        canvas.saveState()
        canvas.setFont(REG, 7.6)
        canvas.setFillColor(GREY)
        canvas.drawRightString(A4[0] - 17 * mm, 10 * mm, str(docu.page))
        canvas.restoreState()

    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    return pdf_path


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2],
            title=sys.argv[3] if len(sys.argv) > 3 else None)
    print("wrote", sys.argv[2])
