# -*- coding: utf-8 -*-
"""p3_explore_rulefeat.py — EXPLORATORY, validation fields only (201-210).

Two questions, not part of the registered tests:
  (A) Does giving the value-based policy the announced-plan forecast as input
      ("rule score as a feature") let it match or beat ranking on announced plans?
      Two extra per-unit columns are appended to every candidate row:
        f_now  = F_t(u, t)            announced forecast of initiating now
        f_best = max_{y>=t} F_t(u, y) best announced forecast over the remaining years
      Under the 4-year lead f_best == f_now (Proposition 1); under a longer lead
      f_best > f_now flags units that the announced plans say are worth waiting for.
  (B) Does a longer announcement lead (8 years, longer than the 7-year minimum
      delivery lag) give waiting a value that learning can use?
      INFRA_ANNOUNCE_LEAD changes the announced layer seen in the state and the
      announced forecast; the true EV table and the reference schedule do not change.

Training as p2_generalize.py: pooled experience from N_TRAIN field draws
(101, 102, ...), EP_EACH episodes per behaviour rule; evaluation on the
validation draws, M = 20, listing draw 7. Test draws 1-10 are not touched.

Usage (repo root):
  PYTHONPATH=src TPMORL_NJOBS=1 python scripts/p3_explore_rulefeat.py --seed 0 --lead 4 --rulefeat 1
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1_common as P                                                  # noqa: E402
from p1_fqi_job import build, greedy_rows, row, peak_rss_mb, ev_value   # noqa: E402
from p1_rolling_horizon import forecast_tables, run_rolling             # noqa: E402
from exp_fqi_local import make_trees                                    # noqa: E402
from tpmorl.rl import fqi                                               # noqa: E402


def wrap_env_fn(env_fn, ann):
    """Append f_now and f_best (announced forecast) to each candidate row of pairs()."""
    T = len(ann)
    best = [np.max(F[:, t:], axis=1) for t, F in enumerate(ann)]

    def make():
        e = env_fn()
        orig = e.pairs

        def pairs():
            X, meta, cost, units = orig()
            t = min(int(e.t), T - 1)
            u = np.asarray(units, int)
            ok = u >= 0
            c1 = np.zeros(len(u), np.float32); c2 = np.zeros(len(u), np.float32)
            c1[ok] = ann[t][u[ok], t]; c2[ok] = best[t][u[ok]]
            return np.column_stack([X, c1, c2]).astype(np.float32), meta, cost, units
        e.pairs = pairs
        return e
    return make


class ResidualModel:
    """Q(s,u) = b(s,u) + dQ(s,u): the tree ensemble fits only the residual over the
    announced-plan forecast b = F_t(u, t), which is column F_NOW_COL of each row
    (appended by wrap_env_fn, before the three global columns of fqi._append_global).
    b and the reward EV[u, t] are both expected floor area discounted to year 0."""
    F_NOW_COL = -5

    def __init__(self, inner):
        self.inner = inner

    def fit(self, X, y):
        self.inner.fit(X, y - X[:, self.F_NOW_COL])
        return self

    def predict(self, X):
        return X[:, self.F_NOW_COL] + self.inner.predict(X)


def conservative_rollout(env_fn, qa, qb, tau_rel):
    """Default to ranking on announced plans; deviate only when the learned value says so
    by more than a margin. Each year: start from the K listed candidates with the highest
    announced forecast b (the rule's choice); then, repeatedly, swap the selected unit with
    the lowest Q for the unselected unit with the highest Q while Q(in) - Q(out) > tau,
    with tau = tau_rel x mean b of the rule's choice that year (tau_rel = inf gives the rule).
    Returns ({unit: year}, number of swapped initiations)."""
    env = env_fn()
    q = int(env.quota)
    init, swaps = {}, 0
    for t in range(env.T):
        X, meta, cost, units = env.pairs()
        keep = [i for i, m in enumerate(meta) if m[0] >= 0]
        if not keep:
            if env.step([])[2]:
                break
            continue
        Xt = fqi._append_global(np.asarray(X, np.float32)[keep], t, env, q)
        b = Xt[:, ResidualModel.F_NOW_COL].astype(float)
        qv = 0.5 * (qa.predict(Xt) + qb.predict(Xt))
        k = min(q, len(keep))
        sel = list(np.argsort(-b)[:k])
        tau = tau_rel * float(b[sel].mean()) if np.isfinite(tau_rel) else np.inf
        rest = [j for j in np.argsort(-qv) if j not in sel]
        while rest:
            out = min(sel, key=lambda j: qv[j]); cand = rest[0]
            if qv[cand] - qv[out] > tau:
                sel[sel.index(out)] = cand; rest.pop(0); swaps += 1
            else:
                break
        act = [meta[keep[j]] for j in sel]
        for j in sel:
            init.setdefault(int(meta[keep[j]][0]), t)
        if env.step(act)[2]:
            break
    return init, swaps


def field_setup(field_seed, rulefeat):
    C = build("main", 20, field_seed=field_seed)
    ann = forecast_tables(C["env_fn"](), "announced", True)
    efn = wrap_env_fn(C["env_fn"], ann) if rulefeat else C["env_fn"]
    return C, ann, efn


def baselines(C, ann, name):
    rows = greedy_rows(C, name, 0, with_random=False)
    EV, elig = C["EV"], C["elig"]

    def now_only(F, t):
        G = np.zeros_like(F); G[:, t] = F[:, t]; return G
    for rule, get in (("rank_announced", lambda t: now_only(ann[t], t)),
                      ("rh_announced", lambda t: ann[t]),
                      ("rh_full", lambda t: EV)):
        t0 = time.time()
        init = run_rolling(C["env_fn"](), get, elig)
        rows.append(row(C, rule, init, ev_value(EV, init), name, -1, time.time() - t0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lead", type=float, default=4.0)
    ap.add_argument("--rulefeat", type=int, default=0)
    ap.add_argument("--history", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--train-base", type=int, default=101)
    ap.add_argument("--ep-each", type=int, default=2)
    ap.add_argument("--val-fields", type=int, nargs="+", default=list(range(201, 211)))
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--residual", type=int, default=0,
                    help="1 = learn only the residual over the announced forecast (needs --rulefeat 1)")
    ap.add_argument("--taus", type=float, nargs="*", default=[],
                    help="relative margins for the conservative rule (needs --residual 1)")
    ap.add_argument("--n-iter", type=int, default=0,
                    help="FQI iterations (0 = horizon length, as in the paper; 1 = one-step target r only)")
    ap.add_argument("--out", default="results_p3_explore")
    a = ap.parse_args()
    from tpmorl.env import opportunity as OPP
    OPP.HISTORY = int(a.history)
    OPP.INFRA_ANNOUNCE_LEAD = float(a.lead)
    assert not set(range(1, 11)) & set(a.val_fields), "test draws must not be used here"
    os.makedirs(a.out, exist_ok=True)
    assert not a.residual or a.rulefeat, "--residual needs --rulefeat 1"
    assert not a.taus or a.residual, "--taus needs --residual 1"
    tag = f"L{a.lead:g}_rf{a.rulefeat}{'_res' if a.residual else ''}{'_cons' if a.taus else ''}{f'_it{a.n_iter}' if a.n_iter else ''}_h{a.history}_s{a.seed}"
    path = os.path.join(a.out, f"explore_{tag}.csv")
    train = [a.train_base + i for i in range(a.n_train)]
    t0 = time.time()
    data = []
    for k in train:
        C, ann, efn = field_setup(k, a.rulefeat)
        EV = C["EV"]
        reward = (lambda E: (lambda u, t: float(E[u, min(t, E.shape[1] - 1)])))(EV)
        data += fqi.collect(efn, reward,
                            {"random": None, "myopic": C["EVm"], "temporal_greedy": EV},
                            n_ep_each=a.ep_each, seed=a.seed * 10007 + k)
    print(f"[{tag}] collected {len(data)} transitions in {time.time() - t0:.0f}s", flush=True)
    mk = make_trees(seed=a.seed)
    factory = (lambda: ResidualModel(mk())) if a.residual else mk
    qa, qb, _ = fqi.fit_double(data, P.BACKUP_GAMMA, factory, verbose=False,
                               n_iter=(a.n_iter or None))
    print(f"[{tag}] fitted at {time.time() - t0:.0f}s, rss {peak_rss_mb():.0f} MB", flush=True)
    del data
    rows = []
    for j in a.val_fields:
        name = f"field{j}"
        C, ann, efn = field_setup(j, a.rulefeat)
        s0 = time.time()
        init, _, _ = fqi.rollout(efn, qa, qb, None, collect_q=True)
        val = ev_value(C["EV"], init)
        rows.append(row(C, "value_based", init, val, name, a.seed, time.time() - s0,
                        lead=a.lead, rulefeat=a.rulefeat, residual=a.residual, n_iter=a.n_iter, history=a.history,
                        n_train_fields=a.n_train, ep_each=a.ep_each))
        msg = f"[{tag}] {name}: vb {val / C['v_orc']:.4f}"
        for tr in a.taus:
            s0 = time.time()
            ci, nsw = conservative_rollout(efn, qa, qb, tr)
            cv = ev_value(C["EV"], ci)
            rows.append(row(C, f"conservative_tau{tr:g}", ci, cv, name, a.seed, time.time() - s0,
                            lead=a.lead, rulefeat=a.rulefeat, residual=a.residual, n_iter=a.n_iter,
                            history=a.history, tau_rel=tr, n_swaps=nsw))
            msg += f" tau{tr:g} {cv / C['v_orc']:.4f}({nsw})"
        if a.baselines:
            for r in baselines(C, ann, name):
                r.update(lead=a.lead); rows.append(r)
            rr = {x["rule"]: x["ratio"] for x in rows if x["eval_field"] == name}
            msg += f" rank_ann {rr['rank_announced']:.4f} rh_ann {rr['rh_announced']:.4f} traj {rr['trajectory']:.4f}"
        print(msg, flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[{tag}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
