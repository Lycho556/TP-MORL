# -*- coding: utf-8 -*-
"""p5_external_check.py -- coarse external consistency check (review R3 2.4).

What the records allow. The 2010-2018 Guangming plan announcements
(tables/gm_renewal_units.csv, type '计划公告') carry a project name, a subdistrict
name and an area, but no coordinates, so they cannot be joined to the 717 candidate
units (see the archived exp_external_temporal.py for the audit). The street-year panel
was already used to calibrate K. Two attributes were never used in calibration and can
be compared with the model at an aggregate level:

  1. renewal type, classified from the project name by keywords
     (industrial -> P1, old village / residential -> P2, commercial -> P3);
  2. former subdistrict. Before the 2018 reorganization Guangming had two subdistricts,
     Guangming (光明) and Gongming (公明). The six current subdistricts are mapped back as
     Guangming <- {Guangming, Fenghuang}, Gongming <- {Gongming, Yutang, Matian};
     Xinhu was formed from parts of both and is reported separately.

Model side: the units each rule initiates in its first nine years (the length of the
2010-2018 window) and in the whole horizon, main field, listing draws 7,0,1,2,3,4,
compared with the composition of the candidate inventory.
This is a consistency check on where renewal is directed, not a validation of timing.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p1_common as P                                                  # noqa: E402
import p1_rolling_horizon as RH                                        # noqa: E402
import p5_spatial_ext as X                                             # noqa: E402

TAB = os.path.join(P.DATASET, "tables", "gm_renewal_units.csv")
IND = re.compile("工业|厂|工业园|工业区|产业|科技|电器|电子|印刷|陶瓷|电线|实业|物流|仓")
VIL = re.compile("村|屋|小区|居住|住宅|新城|围|宿舍")
COM = re.compile("商业|中心|大厦|办公|商务|广场|市场")
OLD = {"Guangming Sub-district": "Guangming (old)", "Fenghuang Sub-District": "Guangming (old)",
       "Gongming Sub-district": "Gongming (old)", "Yutang Sub-District": "Gongming (old)",
       "Matian Sub-district": "Gongming (old)", "Xinhu Sub-district": "Xinhu (mixed)"}
OLD_OBS = {"光明街道": "Guangming (old)", "凤凰街道": "Guangming (old)", "公明街道": "Gongming (old)",
           "玉塘街道": "Gongming (old)", "马田街道": "Gongming (old)", "新湖街道": "Xinhu (mixed)"}
CH = {1: "industrial (P1)", 2: "old residential (P2)", 3: "commercial (P3)", 5: "agricultural (P5)"}


def classify(name):
    n = str(name)
    if IND.search(n):
        return "industrial (P1)"
    if VIL.search(n):
        return "old residential (P2)"
    if COM.search(n):
        return "commercial (P3)"
    return "unclassified"


def main():
    out = os.path.join(X.OUT, "external")
    os.makedirs(out, exist_ok=True)
    g = pd.read_csv(TAB)
    ann = g[g["类型"] == "计划公告"].copy()
    ann["year"] = pd.to_datetime(ann["时间"]).dt.year
    ann = ann[(ann.year >= 2010) & (ann.year <= 2018)]
    ann["type"] = ann["名称"].map(classify)
    ann["old"] = ann["街道N"].map(OLD_OBS).fillna("unknown")
    ann[["名称", "街道N", "year", "面积ha", "type", "old"]].to_csv(os.path.join(out, "announcements_classified.csv"), index=False)

    G, env_fn, EV, EVm, elig, v_orc, oplan = P.setup(prescreen=X.M)
    e0 = env_fn(); T = EV.shape[1]
    sd = pd.read_csv(X.SUBD).set_index("uid").loc[e0.U["uid"].values.astype(int)]
    old = sd["subdistrict_en"].map(OLD).values
    ch = np.array([CH.get(int(c), "other") for c in e0.U["ch_code"].values])
    annT = RH.forecast_tables(e0, "announced", True)
    rules = {"persistence greedy": lambda mk: X.run(mk(), X.greedy_chooser(lambda t: EVm), T)[0],
             "ranking on announced plans": lambda mk: X.run(mk(), X.greedy_chooser(lambda t: annT[t]), T)[0],
             "trajectory-informed greedy": lambda mk: X.run(mk(), X.greedy_chooser(lambda t: EV), T)[0],
             "rolling horizon, full": lambda mk: X.run(mk(), X.rolling_chooser(lambda t: EV, T, elig), T)[0]}
    rows = []
    for ls in X.SEEDS:
        mk = P.env_factory(G, X.M, ls)
        for r, f in rules.items():
            init = f(mk)
            for win, cut in (("first 9 years", 9), ("whole horizon", T)):
                us = [u for u, y in init.items() if y < cut]
                rows.append(dict(rule=r, seed=ls, window=win, n=len(us),
                                 **{f"type:{k}": np.mean(ch[us] == k) for k in CH.values()},
                                 **{f"old:{k}": np.mean(old[us] == k) for k in sorted(set(OLD.values()))}))
    ref_us = list(oplan)
    for win, cut in (("first 9 years", 9), ("whole horizon", T)):
        us = [u for u, y in oplan.items() if y < cut]
        rows.append(dict(rule="reference schedule", seed=-1, window=win, n=len(us),
                         **{f"type:{k}": np.mean(ch[us] == k) for k in CH.values()},
                         **{f"old:{k}": np.mean(old[us] == k) for k in sorted(set(OLD.values()))}))
    cand = np.where(elig)[0]
    rows.append(dict(rule="candidate inventory", seed=-1, window="-", n=len(cand),
                     **{f"type:{k}": np.mean(ch[cand] == k) for k in CH.values()},
                     **{f"old:{k}": np.mean(old[cand] == k) for k in sorted(set(OLD.values()))}))
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(out, "model_composition_by_run.csv"), index=False)
    S = M.groupby(["rule", "window"]).mean(numeric_only=True).drop(columns="seed")
    S.to_csv(os.path.join(out, "model_composition_mean.csv"))
    # observed shares with exact binomial 95% intervals
    obs = []
    for col, key in (("type", ann["type"]), ("old", ann["old"])):
        for k, c in key.value_counts().items():
            ci = binomtest(int(c), len(key)).proportion_ci()
            obs.append(dict(attribute=col, category=k, count=int(c), n=len(key), share=c / len(key),
                            lo=ci.low, hi=ci.high))
    O = pd.DataFrame(obs); O.to_csv(os.path.join(out, "observed_shares.csv"), index=False)
    pd.set_option("display.width", 220)
    print(O.round(3).to_string(index=False))
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
