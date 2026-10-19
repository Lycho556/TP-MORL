# -*- coding: utf-8 -*-
"""Re-train the frozen P3 policy (seed 0, as in p3_metro_test.py) and save its schedule on the main
field, listing draw 7, for Figure 5c. The ratio must equal results_p3_metro/main_s0.csv (listing 7)."""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1_common as P
from p1_fqi_job import ev_value
from exp_fqi_local import make_trees
from tpmorl.rl import fqi
import p3_explore_rulefeat as E
from p3_formal_test import FROZEN
from p3_metro_test import metro_setup
from tpmorl.env import opportunity as OPP
f = FROZEN; seed = 0
OPP.HISTORY = int(f["history"]); OPP.INFRA_ANNOUNCE_LEAD = float(f["lead"])
data = []
for k in [f["train_base"] + i for i in range(f["n_train"])]:
    C, ann, efn = E.field_setup(k, 1); EV = C["EV"]
    reward = (lambda Q: (lambda u, t: float(Q[u, min(t, Q.shape[1] - 1)])))(EV)
    data += fqi.collect(efn, reward, {"random": None, "myopic": C["EVm"], "temporal_greedy": EV},
                        n_ep_each=f["ep_each"], seed=seed * 10007 + k)
mk = make_trees(seed=seed)
qa, qb, _ = fqi.fit_double(data, P.BACKUP_GAMMA, lambda: E.ResidualModel(mk()), verbose=False, n_iter=f["n_iter"])
C, ann, efn = metro_setup(7, "cluster")
init, _, _ = fqi.rollout(efn, qa, qb, None, collect_q=True)
ratio = ev_value(C["EV"], init) / C["v_orc"]
ref = pd.read_csv(os.path.join(HERE, "..", "results_p3_metro", "main_s0.csv"))
ref = ref[(ref.rule == "residual_one_step") & (ref.listing_seed == 7)].ratio.iloc[0]
assert abs(ratio - ref) < 1e-9, (ratio, ref)
json.dump(dict(seed=seed, listing_seed=7, ratio=ratio, init={str(u): int(t) for u, t in init.items()}),
          open(os.path.join(HERE, "..", "results_p3_metro", "rl_schedule_main_ls7_s0.json"), "w"), indent=1)
print("saved", round(ratio, 4), len(init))
