"""Robustness check for the conformity gate (manuscript Section 5.4).

The case-study valuation EV (eval/ev.py, use_admit=True) multiplies the
admission probability a_{i,t} into the expected value, while listing is drawn
separately from the same probability.  This script (read-only, no model change)
  1. reports the distribution of a_{i,t} in the main configuration;
  2. removes the factor a_{i,t} (value conditional on admission), re-solves the
     reference schedule, and re-values the reported schedules held fixed.
Usage: python scripts/check_conformity_gate.py
"""
import json, sys
import numpy as np
sys.path[:0] = ["scripts", "src"]
import exp_temporal_gate as G
from tpmorl.rl import scenario as SC

SC.reset()
SC.apply(horizon=25, horizon_eval="auto", opp_shape="window", a_plan=2.4,
         a_infra=1.8, a_age=0.9, a_ready=1.0, foresight=0, quota=3, budget=1e12)
e = G.build_env("data/processed/gm_dataset_v1", 25, 0.5, 7, 1e12, "floor",
                True, False, 20, 0)
A = np.array([e.opp.admit_prob(t) for t in range(e.T)]).T          # n x T
elig = np.asarray(e.env.eligible, bool)
print(f"a=0: {(A == 0).mean():.3f}  a=1: {(A >= 0.999).mean():.3f}  "
      f"never admissible: {(A.max(1) == 0).sum()} units")

EV, _ = G.ev_tables(e, 0.95)
v2, _ = G.oracle_plan(EV, 3, elig)
EVc = np.where(A > 0, EV / np.where(A > 0, A, 1.0), 0.0)
v1, _ = G.oracle_plan(EVc, 3, elig)
print(f"reference value: two gates {v2:.2f}  listing only {v1:.2f}  (+{100*(v1/v2-1):.1f}%)")
S = json.load(open("results_figures/schedules.json"))
for k in ["static_greedy", "temporal_greedy"]:
    pr = [(int(u), int(t)) for u, t in S[k]["init"].items()]
    print(f"{k:16s} ratio two gates {sum(EV[u, t] for u, t in pr)/v2:.4f}  "
          f"listing only {sum(EVc[u, t] for u, t in pr)/v1:.4f}")
