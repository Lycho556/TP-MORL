# -*- coding: utf-8 -*-
"""p1_rolling_horizon.py — rolling-horizon assignment baseline (P1).

At every decision year t the rule
  1. reads the currently available (listed, prescreened) units from env.pairs(),
     exactly as G.run_policy does;
  2. builds a forecast value table F_t[u, y], y = t..T-1, from the information
     available at t;
  3. solves a max-weight assignment (scipy linear_sum_assignment, as in
     G.oracle_plan) of the not-yet-initiated eligible units to the remaining
     annual slots (K per year); the year-t slots may only take units that are
     available now (a disallowed cell has weight 0, i.e. the slot stays empty);
  4. executes only the year-t assignments with env.step and advances.
The final schedule is valued on the TRUE EV table and decomposed with G.decompose.

Information versions of F_t
  current    all four fields frozen at t (admission, hazard, infrastructure);
             F_t[:, t] equals the persistence table EVm[:, t] by construction.
  announced  admission and hazard frozen at t; infrastructure at completion
             year y read as I(min(y, t + L)), L = INFRA_ANNOUNCE_LEAD = 4, i.e.
             commissioned infrastructure known up to t+L and held afterwards
             (infra only enters value_mult at completion).
  full       the true EV table.
With conformity=False the forecast tables drop the admission factor exactly as
p1_common.setup does for EV/EVm (value 0 where the frozen a_{i,t} is 0).

Comparators run in every configuration: persistence greedy (kind 'myopic') and
trajectory-informed greedy (kind 'forecast').

Usage
  python scripts/p1_rolling_horizon.py --job cluster      # one field config
  python scripts/p1_rolling_horizon.py --merge            # merge part files
Jobs: cluster, metro, noconf, amp_low, amp_high, win_narrow, win_broad,
      onset_early, onset_late   (or --job all to run everything serially)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p1_common as P                                                 # noqa: E402
from p1_common import setup, REPO                                      # noqa: E402

OUT = os.path.join(REPO, "results_p1", "rolling")
PARTS = os.path.join(OUT, "parts")
QUOTA, GAMMA, DEFAULT_LISTING_SEED = 3, 0.95, 7      # 7 = seed inside p1_common env_fn
RULES_RH = ("rh_current", "rh_announced", "rh_full")
RULES_GREEDY = (("persistence", "myopic"), ("trajectory", "forecast"))

# Temporal-field variants of scripts/exp_field_sensitivity.py, reproduced
# through setup(**overrides) (amplitudes scaled; a_ready capped at 1.0).
BASE_AMP = dict(a_plan=2.4, a_infra=1.8, a_age=0.9, a_ready=1.0)
FS_CONFIGS = {
    "base":        dict(),
    "amp_low":     dict(amp=0.5),
    "amp_high":    dict(amp=1.5),
    "win_narrow":  dict(window_years=(2.0, 4.0)),
    "win_broad":   dict(window_years=(6.0, 10.0)),
    "onset_early": dict(onset_lo=0.05, onset_hi=0.35),
    "onset_late":  dict(onset_lo=0.40, onset_hi=0.75),
}


def variant_overrides(name):
    c = dict(FS_CONFIGS[name])
    k = c.pop("amp", 1.0)
    if k != 1.0:
        c.update({n: (min(v * k, 1.0) if n == "a_ready" else v * k)
                  for n, v in BASE_AMP.items()})
    return c


# job -> (infra_mode, conformity, variant, M list, listing seeds for extra rows)
JOBS = {
    "cluster": ("cluster", True, "base", [10, 20, 50, 0], [0, 1, 2, 3, 4]),
    "metro":   ("metro", True, "base", [10, 20, 50, 0], [0, 1, 2, 3, 4]),
    "noconf":  ("cluster", False, "base", [20], [0, 1, 2, 3, 4]),
}
for _v in FS_CONFIGS:
    if _v != "base":
        JOBS[_v] = ("cluster", True, _v, [20], [])
SEED_M = (10, 20, 50, 0)  # prescreen sizes that also get listing seeds 0-4


# ---------------------------------------------------------------- forecasts
class _Info:
    """Field as seen at t0, re-indexed so that local year s = calendar year s+off."""

    def __init__(self, opp, t0, off, mode):
        self.o, self.t0, self.off, self.mode = opp, int(t0), int(off), mode
        self.kind = opp.kind
        self.lead = int(round(__import__("tpmorl.env.opportunity",
                                         fromlist=["x"]).INFRA_ANNOUNCE_LEAD))

    def admit_prob(self, s):
        return self.o.admit_prob(self.t0)

    def hazard_mult(self, s):
        return self.o.hazard_mult(self.t0)

    def value_mult(self, u, s_init, s_done):
        if self.mode == "current":
            return self.o.value_mult(u, self.t0, self.t0)
        td = int(s_done) + self.off
        return self.o.value_mult(u, self.t0, min(td, self.t0 + self.lead))


def forecast_tables(env, mode, conformity):
    """F[t] = (n, T) array, valid for columns y >= t (earlier columns zero).

    Uses tpmorl.eval.ev.ev_matrix on a time-shifted field (local year 0 = t), which
    is exact here because admission and hazard are frozen (the hazard index clip
    in ev_matrix never binds), and rescales by gamma**t.
    """
    from tpmorl.eval import ev as EVM
    from tpmorl.env import schedule as S
    base = np.asarray(env.farcap, float) * np.asarray(env.ncell, float)
    by = np.array([S.BUILD_YEARS_BY_CHANNEL[int(c)] for c in env.ch], int)
    tau = int(S.TAU_VALID + S.TAU_EXT)
    T, Te, n = env.T, env.T_eval, env.n
    out = []
    for t in range(T):
        raw = EVM.ev_matrix(_Info(env.opp, t, t, mode), base, T - t, Te - t, by,
                            S.HAZARD, tau, gamma=GAMMA, use_admit=False)
        A = np.asarray(env.opp.admit_prob(t), float)
        A = np.full(n, float(A)) if A.ndim == 0 else A
        fac = A if conformity else (A > 0).astype(float)
        F = np.zeros((n, T))
        F[:, t:] = raw * (GAMMA ** t) * fac[:, None]
        out.append(F)
    return out


# ---------------------------------------------------------------- policy
def run_rolling(env, Fget, elig, quota=QUOTA):
    """Rolling-horizon assignment. Fget(t) -> (n, T) forecast table at t."""
    init = {}
    for t in range(env.T):
        X, meta, cost, units = env.pairs()
        first_row = {}
        for i, m in enumerate(meta):
            if m[0] >= 0 and int(m[0]) not in init:
                first_row.setdefault(int(m[0]), i)
        F = Fget(t)
        rows = np.where(np.asarray(elig, bool)
                        & ~np.isin(np.arange(F.shape[0]), list(init)))[0]
        years = [y for y in range(t, env.T) for _ in range(quota)]
        W = F[np.ix_(rows, years)].copy()
        avail = np.isin(rows, list(first_row))
        W[~avail, :quota] = 0.0                     # year-t slots: available only
        acts = []
        if W.size:
            ri, ci = linear_sum_assignment(-W)
            for r, c in zip(ri, ci):
                if c < quota and W[r, c] > 0:
                    u = int(rows[r])
                    acts.append(meta[first_row[u]])
                    init[u] = t
        env.step(acts)
    return init


def one_config(job):
    infra_mode, conf, variant, Ms, seeds = JOBS[job]
    ov = variant_overrides(variant)
    os.makedirs(PARTS, exist_ok=True)
    rows, scheds = [], {}
    tabs, tab_sec = None, {}
    for M in Ms:
        t0 = time.time()
        G, env_fn, EV, EVm, elig, v_orc, _ = setup(prescreen=M, infra_mode=infra_mode,
                                                   conformity=conf, **ov)
        if tabs is None:                       # field-only: shared across M / seeds
            e0 = env_fn()
            tabs = {}
            for mode in ("current", "announced"):
                s0 = time.time()
                tabs[mode] = forecast_tables(e0, mode, conf)
                tab_sec[mode] = time.time() - s0
            # check: current-information column t equals persistence EVm[:, t]
            d = max(np.abs(tabs["current"][t][:, t] - EVm[:, t]).max()
                    for t in range(EV.shape[1]))
            assert d <= 1e-9 * max(1.0, np.abs(EVm).max()), d
            tab_sec["full"] = 0.0
            print(f"[{job}] tables {tab_sec['current']:.1f}s/{tab_sec['announced']:.1f}s"
                  f" |F_cur[t]-EVm|max={d:.1e}", flush=True)
        getters = {"rh_current": lambda t: tabs["current"][t],
                   "rh_announced": lambda t: tabs["announced"][t],
                   "rh_full": lambda t: EV}
        lseeds = [DEFAULT_LISTING_SEED] + (list(seeds) if M in SEED_M else [])
        for ls in lseeds:
            # Build with the listing seed; env.reset(seed=) would undo
            # fix_one_target_per_unit (see p1_common.env_factory).
            _mk = P.env_factory(G, M, ls)

            def fresh():
                e = _mk()
                assert len(e._pu_all) == len(elig)
                assert (np.asarray(e.env.eligible, bool) == elig).all()
                return e
            for rule in RULES_RH + tuple(r for r, _ in RULES_GREEDY):
                s0 = time.time()
                if rule in RULES_RH:
                    init = run_rolling(fresh(), getters[rule], elig)
                    val = float(sum(EV[u, y] for u, y in init.items()))
                    tb = tab_sec[rule.split("_", 1)[1]]
                else:
                    kind = dict(RULES_GREEDY)[rule]
                    init, val = G.run_policy(fresh(), EV, EVm, kind, quota=QUOTA,
                                             rng=np.random.default_rng(0))
                    tb = 0.0
                sec = time.time() - s0
                dec = G.decompose(EV, init, QUOTA, elig, v_orc)
                rows.append(dict(infra_mode=infra_mode, conformity=conf,
                                 variant=variant, M=M, rule=rule,
                                 ratio=val / v_orc, value=val, reference=v_orc,
                                 loss_selection=dec["loss_selection"],
                                 loss_timing=dec["loss_timing"], n_init=len(init),
                                 seconds=round(sec + tb, 2),
                                 seconds_policy=round(sec, 2),
                                 seconds_tables=round(tb, 2), listing_seed=ls))
                if M == 20 and ls == DEFAULT_LISTING_SEED and conf and variant == "base":
                    scheds[rule] = {int(u): int(y) for u, y in sorted(init.items())}
        r = [x for x in rows if x["M"] == M and x["listing_seed"] == DEFAULT_LISTING_SEED]
        print(f"[{job}] M={M} ref {v_orc:.1f} " +
              " ".join(f"{x['rule']}={x['ratio']:.4f}" for x in r) +
              f" ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(PARTS, f"{job}.csv"), index=False)
    if scheds:
        uid = [str(x) for x in env_fn().U["uid"].values]
        used = sorted({u for v in scheds.values() for u in v})
        with open(os.path.join(OUT, f"schedules_{infra_mode}_M20.json"), "w") as f:
            json.dump(dict(infra_mode=infra_mode, M=20, listing_seed=DEFAULT_LISTING_SEED,
                           note="unit index (env.U row order) -> initiation year",
                           uid_of_unit={str(u): uid[u] for u in used},
                           schedules={k: {str(u): y for u, y in v.items()}
                                      for k, v in scheds.items()}), f, indent=1)


def merge():
    order = list(JOBS)
    df = pd.concat([pd.read_csv(os.path.join(PARTS, f"{j}.csv")) for j in order
                    if os.path.exists(os.path.join(PARTS, f"{j}.csv"))],
                   ignore_index=True)
    cols = ["infra_mode", "conformity", "variant", "M", "rule", "ratio", "value",
            "reference", "loss_selection", "loss_timing", "n_init", "seconds",
            "seconds_policy", "seconds_tables", "listing_seed"]
    df = df[cols]
    df.to_csv(os.path.join(OUT, "rolling_results.csv"), index=False)
    print(len(df), "rows ->", os.path.join(OUT, "rolling_results.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=sorted(JOBS) + ["all"])
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.job:
        for j in (list(JOBS) if a.job == "all" else [a.job]):
            one_config(j)
    if a.merge or a.job == "all":
        merge()


if __name__ == "__main__":
    main()
