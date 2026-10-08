# -*- coding: utf-8 -*-
"""p1_build_tables.py — consolidate P1 results into the manuscript tables.

Sources
  results_p1/rolling/rolling_results.csv   rolling horizon + greedy rules (local, deterministic)
  results_p1/fqi/summary_all_rows.csv      value-based FQI, supervised, random (server batch
                                           20261002_082308, 5 training seeds x 6 listing seeds)
Protocol: every ratio is averaged over listing seeds {7,0,1,2,3,4}; learned rules also over
training seeds 0-4. Field variants use listing seed 7 only. Output: results_p1/tables/*.csv
"""
import os
import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, "results_p1", "tables"); os.makedirs(OUT, exist_ok=True)
RH = pd.read_csv(os.path.join(R, "results_p1/rolling/rolling_results.csv"))
FQ = pd.read_csv(os.path.join(R, "results_p1/fqi/summary_all_rows.csv"))
LS = [7, 0, 1, 2, 3, 4]

def rh_rows(infra, conf, variant, M, seeds=LS):
    x = RH[(RH.infra_mode == infra) & (RH.conformity == conf) & (RH.variant == variant)
           & (RH.M == M) & RH.listing_seed.isin(seeds)]
    return x.assign(source="rolling")[["rule", "listing_seed", "ratio", "loss_selection", "loss_timing"]]

def fq_rows(field, M, seeds=LS, rules=("value_based", "supervised", "random")):
    x = FQ[(FQ["mode"] != "oos") & (FQ.eval_field == field) & (FQ.prescreen == M)
           & FQ.listing_seed.isin(seeds) & FQ.rule.isin(rules)]
    return x[["rule", "listing_seed", "seed", "ratio", "loss_selection", "loss_timing"]]

def check_greedy(field, infra, conf, variant, M, seeds):
    a = FQ[(FQ["mode"] != "oos") & (FQ.eval_field == field) & (FQ.prescreen == M)
           & FQ.listing_seed.isin(seeds) & FQ.rule.isin(["persistence", "trajectory"])]
    a = a.drop_duplicates(["rule", "listing_seed"]).set_index(["rule", "listing_seed"]).ratio
    b = rh_rows(infra, conf, variant, M, seeds)
    b = b[b.rule.isin(["persistence", "trajectory"])].set_index(["rule", "listing_seed"]).ratio
    j = a.to_frame("srv").join(b.to_frame("loc"), how="inner")
    assert len(j) and np.allclose(j.srv, j["loc"], atol=1e-12, rtol=0), (field, M, j)

def summarize(df):
    g = df.groupby("rule")
    s = pd.DataFrame({"ratio": g.ratio.mean(), "sd": g.ratio.std(ddof=1), "lo": g.ratio.min(),
                      "hi": g.ratio.max(), "n": g.ratio.count(),
                      "l_sel": g.loss_selection.mean(), "l_tim": g.loss_timing.mean()})
    s["timing_share"] = s.l_tim / (s.l_sel + s.l_tim)
    return s

ORDER = ["random", "persistence", "rh_current", "supervised", "rh_announced", "trajectory",
         "value_based", "rh_full"]

def table(field, infra, conf, variant, M, seeds=LS):
    check_greedy(field, infra, conf, variant, M, seeds)
    d = pd.concat([rh_rows(infra, conf, variant, M, seeds), fq_rows(field, M, seeds)])
    s = summarize(d).reindex([r for r in ORDER if r in set(d.rule)])
    s.insert(0, "M", M); s.insert(0, "config", field)
    return s

T = {}
T["main_M20"] = table("main", "cluster", True, "base", 20)
T["prescreen"] = pd.concat([table("main", "cluster", True, "base", M) for M in (10, 20, 50, 0)])
T["metro"] = pd.concat([table("metro", "metro", True, "base", M) for M in (20, 0)])
T["noconf"] = table("noconf", "cluster", False, "base", 20)
var = []
for v in ["base", "amp_low", "amp_high", "win_narrow", "win_broad", "onset_early", "onset_late"]:
    field = "main" if v == "base" else f"variant-{v}"
    var.append(table(field, "cluster", True, v, 20, seeds=[7]).assign(variant=v))
T["fields"] = pd.concat(var)
# paired margins (learned minus trajectory greedy, per listing seed and training seed)
def margins(field, M, seeds=LS):
    vb = FQ[(FQ["mode"] != "oos") & (FQ.eval_field == field) & (FQ.prescreen == M)
            & (FQ.rule == "value_based") & FQ.listing_seed.isin(seeds)]
    tg = (FQ[(FQ["mode"] != "oos") & (FQ.eval_field == field) & (FQ.prescreen == M)
             & (FQ.rule == "trajectory")].drop_duplicates("listing_seed")
          .set_index("listing_seed").ratio)
    m = vb.ratio.values - tg.loc[vb.listing_seed].values
    return dict(config=field, M=M, mean=m.mean(), sd=m.std(ddof=1), n=len(m),
                share_pos=(m > 0).mean(), lo=m.min(), hi=m.max())
T["margins"] = pd.DataFrame([margins(f, M) for f, M in
                             [("main", 10), ("main", 20), ("main", 50), ("main", 0),
                              ("metro", 20), ("metro", 0), ("noconf", 20)]])
# out of sample
O = FQ[FQ["mode"] == "oos"].copy()
O["kind"] = np.where(O.rule.isin(["value_based", "supervised"]),
                     O.rule + np.where(O.train_field == O.eval_field, "_in", "_oos"), O.rule)
T["oos"] = (O.drop_duplicates(["eval_field", "kind", "seed"])
            .pivot_table(index="eval_field", columns="kind", values="ratio", aggfunc="mean"))
for k, v in T.items():
    v.to_csv(os.path.join(OUT, f"{k}.csv"))
pd.set_option("display.width", 200)
for k, v in T.items():
    print(f"\n== {k}"); print(v.round(4).to_string())
