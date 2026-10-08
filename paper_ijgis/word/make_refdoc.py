import docx
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn
d = docx.Document("ref_default.docx")
def setfont(st, size=None, bold=None, italic=None):
    st.font.name = "Times New Roman"
    rpr = st.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
    if rf is None: rf = rpr.makeelement(qn("w:rFonts"), {}); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"): rf.set(qn(a), "Times New Roman")
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
    if size: st.font.size = Pt(size)
    if bold is not None: st.font.bold = bold
    if italic is not None: st.font.italic = italic
    st.font.color.rgb = RGBColor(0, 0, 0)
for name in ["Normal", "Body Text", "First Paragraph", "Compact", "Bibliography", "Block Text", "Footnote Text"]:
    try: setfont(d.styles[name], 12)
    except KeyError: pass
for name, sz in [("Title", 16), ("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 12)]:
    setfont(d.styles[name], sz, bold=True, italic=(name == "Heading 3"))
for name in ["Caption", "Image Caption", "Table Caption"]:
    try: setfont(d.styles[name], 10, italic=False)
    except KeyError: pass
for name in ["Body Text", "First Paragraph", "Bibliography"]:
    try: d.styles[name].paragraph_format.line_spacing = 1.5
    except KeyError: pass
d.save("reference_ijgis.docx")
