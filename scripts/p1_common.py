# -*- coding: utf-8 -*-
"""p1_common.py — shared setup for the P1 experiments.

All P1 scripts build the Guangming main configuration through `setup()` so that
the scenario (horizon, amplitudes, quota, prescreen) is identical to the
published Table 3/4 runs. Three switches are added for P1:

  infra_mode  "cluster" (scenario field, default) | "metro" (observed stations)
  field_seed  None (main field) | int (alternative scenario draw, out-of-sample)
  conformity  True (default) | False (drop the a_{i,y} factor from the EV table)

`conformity=False` only changes the valuation table: listing still uses the same
admission probability. EV, EVm, the reference schedule and every rule that reads
the table are recomputed consistently.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)
DATASET = os.path.join(REPO, "data", "processed", "gm_dataset_v1")

DEFAULTS = dict(dataset=DATASET, horizon=25, quota=3, gamma=0.95, ep_each=10)


def args(**kw):
    d = dict(DEFAULTS); d.update(kw)
    return SimpleNamespace(**d)


LISTING_SEED = 7   # environment seed used by every published run (G.build_env)


def env_factory(G, prescreen, listing_seed=LISTING_SEED, a=None):
    """Environment builder for a given listing seed.

    Use this instead of env.reset(seed=...): reset() rebuilds the (unit, target)
    pair table and silently undoes fix_one_target_per_unit(), so a reset env
    offers several rows per unit and the prescreen then admits fewer than M
    distinct units. Building with the seed avoids that.
    """
    a = a or args()
    return lambda: G.build_env(a.dataset, a.horizon, 0.5, int(listing_seed), 1e12,
                               "floor", True, False, prescreen, 0)


def setup(prescreen=20, infra_mode="cluster", field_seed=None, conformity=True,
          a=None, listing_seed=LISTING_SEED, **scen):
    """Return (G, env_fn, EV, EVm, elig, v_orc, oplan).

    EV/EVm/reference do not depend on the listing seed; env_fn does."""
    import exp_temporal_gate as G
    from tpmorl.rl import scenario as SC
    from tpmorl.eval import ev as EVM
    a = a or args()
    SC.reset()
    kw = dict(horizon=a.horizon, horizon_eval="auto", opp_shape="window",
              a_plan=2.4, a_infra=1.8, a_age=0.9, a_ready=1.0,
              foresight=0, quota=a.quota, budget=1e12, infra_mode=infra_mode)
    if field_seed is not None:
        kw["field_seed"] = int(field_seed)
    kw.update(scen)
    SC.apply(**kw)
    env_fn = env_factory(G, prescreen, listing_seed, a)
    e0 = env_fn()
    EV, EVm = G.ev_tables(e0, a.gamma)
    if not conformity:
        # Value conditional on passing conformity review: divide out a_{i,y}
        # where it is positive; where a_{i,y}=0 the unit cannot be listed in
        # that year, so its value stays 0 (same as scripts/check_conformity_gate.py).
        A = np.array([e0.opp.admit_prob(t) for t in range(EV.shape[1])]).T
        safe = np.where(A > 0, A, 1.0)
        EV = np.where(A > 0, EV / safe, 0.0)
        EVm = np.where(A > 0, EVm / safe, 0.0)
    elig = np.asarray(e0.env.eligible, bool)
    v_orc, oplan = G.oracle_plan(EV, a.quota, elig)
    return G, env_fn, EV, EVm, elig, v_orc, oplan


