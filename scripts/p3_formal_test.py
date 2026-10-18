# -*- coding: utf-8 -*-
"""p3_formal_test.py — registered one-time test of residual value learning with a one-step target.

Frozen method (selected on validation draws 201-210, results_p3_explore; not changed here):
  * state: the paper's state with the past year's field changes (H = 1), plus two columns from
    the announced forecast F_t: f_now = F_t(u, t) and f_best = max_{y>=t} F_t(u, y);
  * value: Q(s, u) = f_now + dQ(s, u); double tree ensembles fit dQ to the one-step target
    r - f_now (fitted Q iteration with n_iter = 1), r = EV[u, t] as in the paper;
  * decision: each year initiate the K = 3 prescreened listed units with the highest Q;
  * training: pooled experience from field draws 101-120, 2 episodes per behaviour rule
    (random, persistence greedy, trajectory-informed greedy), training seeds 0-4;
  * announcement lead 4 years (main design); M = 20; listing draw 7.
Test: field draws 1-10 (primary) and the main field (secondary), each valued on its own EV table
and reference schedule. Comparators on the same draws: ranking on announced plans (primary
benchmark), persistence greedy, trajectory-informed greedy, rolling horizon with full information.

Built-in checks (written to the output, do not affect the decision):
  * rule_replay: the learned policy's candidate pipeline with the rule's choice
    (conservative_rollout with tau = inf) must reproduce ranking on announced plans exactly,
    which shows that both choose from the same candidates under the same capacity;
  * leak check: for test draw 1 and years 3, 10, 17 the decision-time inputs (observation block
    and announced forecast) are recomputed after replacing all field values that are not
    available at year t (planning, obsolescence, readiness and announced infrastructure after t;
    commissioned infrastructure after t + lead) by random numbers; they must be unchanged.

Usage (repo root):
  OMP_NUM_THREADS=1 PYTHONPATH=src TPMORL_NJOBS=1 python scripts/p3_formal_test.py --seed 0 [--baselines --leak-check]
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1_common as P                                                    # noqa: E402
from p1_fqi_job import row, peak_rss_mb, ev_value                         # noqa: E402
from p1_rolling_horizon import forecast_tables                            # noqa: E402
from exp_fqi_local import make_trees                                      # noqa: E402
from tpmorl.rl import fqi                                                 # noqa: E402
import p3_explore_rulefeat as E                                           # noqa: E402

FROZEN = dict(lead=4.0, history=1, n_train=20, train_base=101, ep_each=2, n_iter=1,
              prescreen=20, test_fields=list(range(1, 11)), main_field=True)


def leak_check(field_seed, years=(3, 10, 17), lead=4):
    C, ann, _ = E.field_setup(field_seed, 1)
    e0 = C["env_fn"]()
    rng = np.random.default_rng(12345)
    out = []
    for t in years:
        e1 = C["env_fn"](); e1.opp = copy.deepcopy(e0.opp); o = e1.opp
        for name, cut in (("plan", t + 1), ("necessity", t + 1), ("ready", t + 1),
                          ("infra_plan", t + 1), ("infra", t + lead + 1)):
            A = getattr(o, name)
            if A.ndim == 2 and cut < A.shape[1]:
                A = A.copy(); A[:, cut:] = rng.uniform(0, 1, A[:, cut:].shape); setattr(o, name, A)
        a0, a1 = e0.opp.obs_block(t), o.obs_block(t)
        F0 = forecast_tables(e0, "announced", True)[t][:, t:]
        F1 = forecast_tables(e1, "announced", True)[t][:, t:]
        out.append(dict(year=t, obs_max_abs_diff=float(np.abs(a0 - a1).max()),
                        forecast_max_abs_diff=float(np.abs(F0 - F1).max())))
    # negative control: commissioned infrastructure at t + lead IS available (announced); changing it must show
    t = years[1]; e1 = C["env_fn"](); e1.opp = copy.deepcopy(e0.opp)
    A = e1.opp.infra.copy(); A[:, t + lead] = rng.uniform(0, 1, A.shape[0]); e1.opp.infra = A
    d = np.abs(forecast_tables(e0, "announced", True)[t] - forecast_tables(e1, "announced", True)[t]).max()
    out.append(dict(year=f"control {t}", obs_max_abs_diff=np.nan, forecast_max_abs_diff=float(d)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--leak-check", action="store_true")
    ap.add_argument("--out", default="results_p3_formal")
    a = ap.parse_args()
    f = FROZEN
    from tpmorl.env import opportunity as OPP
    OPP.HISTORY = int(f["history"]); OPP.INFRA_ANNOUNCE_LEAD = float(f["lead"])
    os.makedirs(a.out, exist_ok=True)
    tag = f"s{a.seed}"
    if a.leak_check:
        lc = pd.DataFrame(leak_check(1, lead=int(f["lead"])))
        lc.to_csv(os.path.join(a.out, "leak_check.csv"), index=False)
        print(f"[{tag}] leak check:\n{lc.to_string(index=False)}", flush=True)
        main_rows = lc[~lc.year.astype(str).str.startswith("control")]
        assert (main_rows.obs_max_abs_diff == 0).all() and (main_rows.forecast_max_abs_diff == 0).all(), "future information reaches the inputs"
        assert lc[lc.year.astype(str).str.startswith("control")].forecast_max_abs_diff.iloc[0] > 0, "leak check is not sensitive"
    train = [f["train_base"] + i for i in range(f["n_train"])]
    assert not set(train) & set(f["test_fields"])
    t0 = time.time(); data = []
    for k in train:
        C, ann, efn = E.field_setup(k, 1)
        EV = C["EV"]
        reward = (lambda Q: (lambda u, t: float(Q[u, min(t, Q.shape[1] - 1)])))(EV)
        data += fqi.collect(efn, reward, {"random": None, "myopic": C["EVm"], "temporal_greedy": EV},
                            n_ep_each=f["ep_each"], seed=a.seed * 10007 + k)
    print(f"[{tag}] collected {len(data)} transitions in {time.time() - t0:.0f}s", flush=True)
    mk = make_trees(seed=a.seed)
    qa, qb, _ = fqi.fit_double(data, P.BACKUP_GAMMA, lambda: E.ResidualModel(mk()),
                               verbose=False, n_iter=f["n_iter"])
    print(f"[{tag}] fitted at {time.time() - t0:.0f}s, rss {peak_rss_mb():.0f} MB", flush=True)
    del data
    rows, path = [], os.path.join(a.out, f"test_{tag}.csv")
    fields = list(f["test_fields"]) + ([None] if f["main_field"] else [])
    for j in fields:
        name = "main" if j is None else f"field{j}"
        C, ann, efn = E.field_setup(j, 1)
        s0 = time.time()
        init, _, _ = fqi.rollout(efn, qa, qb, None, collect_q=True)
        val = ev_value(C["EV"], init)
        rows.append(row(C, "residual_one_step", init, val, name, a.seed, time.time() - s0))
        ri, _ = E.conservative_rollout(efn, qa, qb, np.inf)
        rows.append(row(C, "rule_replay", ri, ev_value(C["EV"], ri), name, a.seed, 0.0))
        msg = f"[{tag}] {name}: residual_one_step {val / C['v_orc']:.4f}"
        if a.baselines:
            rows += E.baselines(C, ann, name)
            rr = {x["rule"]: x["ratio"] for x in rows if x["eval_field"] == name}
            msg += f" rank_ann {rr['rank_announced']:.4f} traj {rr['trajectory']:.4f}"
        print(msg, flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[{tag}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
