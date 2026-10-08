"""Build results_figures/table_main.csv (manuscript Table 4, Figure 6) from
part-B run files of exp_prescreen_ablation.py.

Usage: python scripts/build_table_main.py results_det/s*/partB_runs.csv
Random and supervised-scorer rows are not part of the part-B runs; their values
are carried over unchanged from the earlier runs reported in Table 4.
"""
import sys, glob
import pandas as pd

files = sorted(f for p in sys.argv[1:] for f in glob.glob(p))
D = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
D = D[D.prescreen == 20]
key = {"static_greedy": "static", "temporal_greedy": "temporal",
       "double_fqi": "value", "single_fqi": "single"}
rows = []
for m, k in key.items():
    d = D[D.method == m].drop_duplicates(subset=["seed", "method"])
    rows.append(dict(key=k, n=len(d), ratio=d.ratio.mean(),
                     sd=d.ratio.std() if len(d) > 1 else float("nan"),
                     lo=d.ratio.min(), hi=d.ratio.max(),
                     l_sel=d.where_loss.mean(), l_tim=d.when_loss.mean(),
                     oracle=d.oracle_value.iloc[0],
                     seeds=" ".join(str(int(s)) for s in sorted(d.seed))))
# carried over from earlier runs (not part of part B)
rows += [dict(key="random", n=3, ratio=0.757, sd=0.061, lo=0.716, hi=0.827,
              l_sel=458.6, l_tim=1209.9, oracle=float("nan"), seeds="earlier"),
         dict(key="supervised", n=3, ratio=0.846, sd=float("nan"), lo=0.819, hi=0.863,
              l_sel=254.0, l_tim=805.0, oracle=float("nan"), seeds="earlier")]
out = pd.DataFrame(rows)
out.to_csv("results_figures/table_main.csv", index=False)
print(out.round(4).to_string(index=False))
