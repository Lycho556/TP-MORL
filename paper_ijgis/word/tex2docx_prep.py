"""Prepare IJGIS_manuscript_EN.tex for pandoc -> Word.
Resolves \\ref/\\eqref from the LaTeX .aux, numbers sections/figures/tables/equations
explicitly, swaps PDF figures for PNG, and simplifies class-specific markup."""
import re, sys
src, aux, out = sys.argv[1:4]
t = open(src, encoding="utf-8").read()
L = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", open(aux, encoding="utf-8").read()))
body = t[t.index("\\begin{document}") + len("\\begin{document}"):t.index("\\end{document}")]
body = re.sub(r"(?<!\\)%.*", "", body)                      # strip comments
# front matter
title = re.search(r"\\title\{(.*?)\}\s*\n\s*\n", body, re.S).group(1)
body = re.sub(r"\\articletype\{[^}]*\}|\\suppldatatrue|\\maketitle", "", body)
body = re.sub(r"\\title\{.*?\}\s*\n\s*\n", "", body, count=1, flags=re.S)
body = re.sub(r"\\author\{.*?\n\}\n", "", body, count=1, flags=re.S)
front = ("\\title{" + title + "}\n\n"
         "Yichen Li\\textsuperscript{a} and Bo Huang\\textsuperscript{a}\n\n"
         "\\textsuperscript{a}Department of Geography, The University of Hong Kong, Pokfulam, Hong Kong\n\n")
body = body.replace("\\begin{abstract}", "\\section*{Abstract}\n").replace("\\end{abstract}", "")
body = re.sub(r"\\begin\{keywords\}(.*?)\\end\{keywords\}",
              lambda m: "\\noindent\\textbf{Keywords:} " + " ".join(m.group(1).split()), body, flags=re.S)
body = front + body
# landscape / afterpage wrappers
body = re.sub(r"\\afterpage\{%?\s*\\begin\{landscape\}", "", body)
body = re.sub(r"\\end\{landscape\}\}", "", body)
body = body.replace("\\clearpage", "")
# references placement marker
def _tbc(b):
    out=[];i=0
    while True:
        j=b.find("\\tbc{",i)
        if j<0: out.append(b[i:]); break
        out.append(b[i:j]); k=j+5; depth=1
        while depth:
            depth += {"{":1,"}":-1}.get(b[k],0); k+=1
        out.append("\\textbf{[TO COMPLETE: " + b[j+5:k-1] + "]}"); i=k
    return "".join(out)
body = _tbc(body)
body = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}", "\n\nREFSMARKER\n\n", body)
# figures: png + numbered captions
def fig(m):
    s = m.group(0).replace(".pdf}", ".png}")
    lab = re.search(r"\\label\{([^}]+)\}", s)
    if lab:
        s = s.replace("\\caption{", "\\caption{Figure " + L[lab.group(1)] + ". ", 1)
    return s
body = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", fig, body, flags=re.S)
def tab(m):
    s = m.group(0)
    lab = re.search(r"\\label\{([^}]+)\}", s)
    if lab:
        s = s.replace("\\caption{", "\\caption{Table " + L[lab.group(1)] + ". ", 1)
    s = re.sub(r"\\small|\\setlength\{\\tabcolsep\}\{[^}]*\}", "", s)
    def spec(mm):
        sp = mm.group(1)
        sp = re.sub(r">\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", "", sp)
        sp = re.sub(r"@\{[^}]*\}", "", sp)
        n = len(re.findall(r"p\{[^}]*\}", sp)); sp2 = re.sub(r"p\{[^}]*\}", "", sp)
        n += len(re.findall(r"[lcr]", sp2))
        return "\\begin{tabular}{" + "l" * n + "}"
    return re.sub(r"\\begin\{tabular\}\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", spec, s)
body = re.sub(r"\\begin\{table\}.*?\\end\{table\}", tab, body, flags=re.S)
# equation numbers
def eqlab(m):
    k = m.group(1)
    return "\\qquad (" + L[k] + ")" if k in L else ""
body = re.sub(r"\\label\{(eq:[^}]+)\}", eqlab, body)
body = re.sub(r"\\begin\{equation\}", r"\\begin{equation*}", body)
body = re.sub(r"\\end\{equation\}", r"\\end{equation*}", body)
body = body.replace("\\begin{align}", "\\begin{align*}").replace("\\end{align}", "\\end{align*}")
# cross references
body = re.sub(r"\\eqref\{([^}]+)\}", lambda m: "(" + L.get(m.group(1), "??") + ")", body)
body = re.sub(r"\\ref\{([^}]+)\}", lambda m: L.get(m.group(1), "??"), body)
body = re.sub(r"\\label\{[^}]+\}", "", body)
# section numbering
out_lines, c, app = [], [0, 0, 0], False
for line in body.split("\n"):
    if line.strip() == "\\appendix":
        app = True; c = [0, 0, 0]; continue
    m = re.match(r"\\(section|subsection|subsubsection)(\*?)\{(.*)\}\s*$", line.strip())
    if m and not m.group(2):
        lev = ["section", "subsection", "subsubsection"].index(m.group(1))
        c[lev] += 1
        for j in range(lev + 1, 3): c[j] = 0
        first = chr(64 + c[0]) if app else str(c[0])
        num = ".".join([first] + [str(x) for x in c[1:lev + 1]])
        head = ("Appendix " + num + ". " if (app and lev == 0) else num + ". ") + m.group(3)
        line = "\\" + m.group(1) + "*{" + head + "}"
    out_lines.append(line)
body = "\n".join(out_lines)
assert "??" not in body, "unresolved reference"
open(out, "w", encoding="utf-8").write("\\documentclass{article}\n\\begin{document}\n" + body + "\n\\end{document}\n")
