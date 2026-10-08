import json, sys
d = json.load(open(sys.argv[1]))
def is_marker(b):
    return b["t"] == "Para" and any(x.get("t") == "Str" and x.get("c") == "REFSMARKER" for x in b["c"])
out = []
for b in d["blocks"]:
    if is_marker(b):
        out.append({"t": "Header", "c": [1, ["references", ["unnumbered"], []], [{"t": "Str", "c": "References"}]]})
        out.append({"t": "Div", "c": [["refs", [], []], []]})
    else:
        out.append(b)
d["blocks"] = out
d["meta"]["link-citations"] = {"t": "MetaBool", "c": False}
json.dump(d, open(sys.argv[2], "w"))
n = sum(1 for b in out if b["t"] == "Figure"); print("figures:", n, "tables:", sum(1 for b in out if b["t"] == "Table"))
