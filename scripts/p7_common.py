# -*- coding: utf-8 -*-
"""p7_common.py — shared pieces for the P7 follow-up learning experiments (E2, E3).

Everything here reuses the frozen P3 method (scripts/p3_formal_test.FROZEN) unchanged:
  * state with H = 1 and the two announced-forecast columns f_now, f_best (p3_explore_rulefeat);
  * training on field draws 101-120, 2 episodes per behaviour rule, collect seed = seed*10007 + k;
  * M = 20, listing draw 7, announcement lead 4 years, BACKUP_GAMMA = 1.0 (EV is discounted to year 0).
Only validation draws 201-210 and the observed metro field are evaluated by P7 so far; test draws
1-10 and the main field are refused (VALIDATION / METRO below are the only allowed targets).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1_common as P                                                    # noqa: E402
from p1_fqi_job import row, ev_value                                      # noqa: E402
from exp_fqi_local import make_trees                                      # noqa: E402
from tpmorl.rl import fqi                                                 # noqa: E402
import p3_explore_rulefeat as E                                           # noqa: E402
from p3_formal_test import FROZEN                                         # noqa: E402

VALIDATION = list(range(201, 211))
TEST_FIELDS = set(range(1, 11))          # registered test draws: never evaluated by P7 before registration
METRO_LISTING = [7, 0, 1, 2, 3, 4]       # observed metro field, the paper's six listing draws
F_NOW_COL = E.ResidualModel.F_NOW_COL    # column of f_now in every candidate row (before 3 globals)
MODEL_DIR = os.environ.get("P7_MODEL_DIR", os.path.expanduser("~/tpmorl_p7_models"))


def configure():
    """Announcement lead and history exactly as the frozen P3 method; BACKUP_GAMMA must be 1."""
    from tpmorl.env import opportunity as OPP
    OPP.HISTORY = int(FROZEN["history"])
    OPP.INFRA_ANNOUNCE_LEAD = float(FROZEN["lead"])
    assert P.BACKUP_GAMMA == 1.0, "BACKUP_GAMMA must stay 1.0: EV is already discounted to year 0"


def guard_fields(fields):
    bad = sorted(set(fields) & TEST_FIELDS)
    assert not bad, f"test draws {bad} must not be evaluated before registration"


def collect_training(seed, n_train=None):
    """Pooled experience exactly as p3_formal_test.main (same fields, rules, episodes, seeds)."""
    n_train = FROZEN["n_train"] if n_train is None else n_train
    train = [FROZEN["train_base"] + i for i in range(n_train)]
    assert not set(train) & (TEST_FIELDS | set(VALIDATION))
    data = []
    for k in train:
        C, ann, efn = E.field_setup(k, 1)
        EV = C["EV"]
        reward = (lambda Q: (lambda u, t: float(Q[u, min(t, Q.shape[1] - 1)])))(EV)
        data += fqi.collect(efn, reward, {"random": None, "myopic": C["EVm"], "temporal_greedy": EV},
                            n_ep_each=FROZEN["ep_each"], seed=seed * 10007 + k)
    return data


def fit_p3(data, seed):
    """The paper's learning correction (P3 frozen): residual over f_now, one-step target."""
    mk = make_trees(seed=seed)
    qa, qb, _ = fqi.fit_double(data, P.BACKUP_GAMMA, lambda: E.ResidualModel(mk()),
                               verbose=False, n_iter=FROZEN["n_iter"])
    return qa, qb


def metro_setup(listing_seed):
    from p3_metro_test import metro_setup as ms
    return ms(listing_seed)


def model_path(seed):
    return os.path.join(MODEL_DIR, f"p3_s{seed}.joblib")


def self_check(new, ref, label):
    """Compare per-field ratios with an earlier run of the same frozen method."""
    d = (new - ref).dropna().abs()
    if d.empty:
        return f"self-check {label}: no overlapping fields"
    m = float(d.max())
    v = ("PASS (bit-identical)" if m < 1e-12 else
         "PASS_TOL (cross-platform ExtraTrees, <= 0.01)" if m <= 0.01 else "WARN: check code version")
    return f"self-check {label}: max |diff| = {m:.2e} over {len(d)} fields -> {v}"
