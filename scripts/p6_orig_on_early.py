# -*- coding: utf-8 -*-
"""p6_orig_on_early.py -- original (frozen-protocol) models evaluated on the early-infrastructure
test draws 1-10, to complete the cross table of results_p6_shift (A2 control cell)."""
import os, sys, joblib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p6_early_infra as A
import p3_explore_rulefeat as E
from tpmorl.env import opportunity as OPP
seed = int(sys.argv[1])
OPP.HISTORY = 1; OPP.INFRA_ANNOUNCE_LEAD = 4.0
qa, qb = joblib.load(os.path.join(HERE, "..", "results_p5_spatial", "models", f"s{seed}.joblib"))
OPP.INFRA_ONSET_YEARS = A.EARLY
rows = [A.evaluate(f"early{k}", *E.field_setup(k, 1), qa, qb, seed, 7) for k in range(1, 11)]
OPP.INFRA_ONSET_YEARS = None
pd.DataFrame(rows).to_csv(os.path.join(A.OUT, f"a2_orig_on_early_s{seed}.csv"), index=False); print("done", seed)
