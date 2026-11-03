# -*- coding: utf-8 -*-
"""p7_safe.py — E2: deviate from ranking on announced plans only when confident (EXPLORATORY).

The paper's learning correction (P3) lost to ranking on announced plans on the observed metro field
(0.9164 vs 0.9408): the announced plans were already accurate there and the model "corrected" them
anyway. E2 adds a brake. Uncertainty is the disagreement between the five P3 models saved by
p7_rank.py (training seeds 0-4). For every candidate u in year t:
    dQ_s(u) = Q_s(u) - f_now(u),   m(u) = mean_s dQ_s(u),   sd(u) = std_s dQ_s(u)  (ddof = 1)
  E2a  safe_lam{l}:  score = f_now + m - l * sd, initiate the K highest; l in {0, 0.5, 1, 2}.
                     l = 0 is plain averaging of the five P3 models.
  E2b  safe_swap_z{z}: start from the rule's choice (K highest f_now); while the best unselected
                     candidate's lower bound f_now + m - z*sd exceeds the worst selected unit's
                     upper bound f_now + m + z*sd, swap them ("only swap, never reshuffle",
                     close to SPIBB-style safe policy improvement); z in {1, 2}.
                     z = inf reproduces the rule exactly (fairness check against rank_announced).
Selection (fixed here, before any result is seen): within each family, the variant with the highest
mean ratio over validation draws 201-210; ties within 0.002 go to the more conservative one
(larger l or z). The observed metro field was already used by p3_metro_test.py, so it is reported for
every variant and never used to choose.

Usage (repo root, after the five p7_rank.py jobs):
  OMP_NUM_THREADS=1 PYTHONPATH=src TPMORL_NJOBS=1 python scripts/p7_safe.py
"""
import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7_common as C7                                                    # noqa: E402
from p7_common import E, fqi, row, ev_value, F_NOW_COL                    # noqa: E402

LAMBDAS = [0.0, 0.5, 1.0, 2.0]
ZS = [1.0, 2.0]
TIE = 0.002


def ensemble_stats(models, Xt):
    f = Xt[:, F_NOW_COL].astype(float)
    Q = np.stack([0.5 * (qa.predict(Xt) + qb.predict(Xt)) for qa, qb in models])
    dQ = Q - f
    return f, dQ.mean(axis=0), dQ.std(axis=0, ddof=1)


def topk_unique(order, us, k):
    sel, used = [], set()
    for j in order:
        if len(sel) >= k:
            break
        if int(us[j]) in used:
            continue
        sel.append(int(j)); used.add(int(us[j]))
    return sel


def rollout_safe(env_fn, models, kind, par):
    """kind 'lam': score f + m - par*sd; kind 'swap': rule start, swap while LCB_in > UCB_out (z = par).
    Returns ({unit: year}, initiations that differ from the rule's choice)."""
    env = env_fn()
    q = int(env.quota)
    init, dev = {}, 0
    for t in range(env.T):
        X, meta, cost, units = env.pairs()
        keep = [i for i, m in enumerate(meta) if m[0] >= 0]
        if not keep:
            if env.step([])[2]:
                break
            continue
        us = np.array([meta[i][0] for i in keep], int)
        Xt = fqi._append_global(np.asarray(X, np.float32)[keep], t, env, q)
        f, m, sd = ensemble_stats(models, Xt)
        rule = topk_unique(np.argsort(-f, kind="stable"), us, q)
        if kind == "lam":
            sel = topk_unique(np.argsort(-(f + m - par * sd), kind="stable"), us, q)
        else:
            sel = list(rule)
            if np.isfinite(par):
                lcb, ucb = f + m - par * sd, f + m + par * sd
                rest = [int(j) for j in np.argsort(-lcb, kind="stable")
                        if j not in sel and int(us[j]) not in {int(us[s]) for s in sel}]
                while rest:
                    out = min(sel, key=lambda j: ucb[j]); cand = rest[0]
                    if lcb[cand] > ucb[out]:
                        sel[sel.index(out)] = cand; rest.pop(0)
                        rest = [j for j in rest if int(us[j]) != int(us[cand])]
                    else:
                        break
        dev += len(set(sel) - set(rule))
        for j in sel:
            init.setdefault(int(us[j]), t)
        if env.step([meta[keep[j]] for j in sel])[2]:
            break
    return init, dev


def variants():
    for l in LAMBDAS:
        yield f"safe_lam{l:g}", "lam", l
    for z in ZS:
        yield f"safe_swap_z{z:g}", "swap", z
    yield "safe_swap_zinf", "swap", np.inf                             # fairness: equals the rule


def choose(D):
    v = D[(D.field_set == "validation") & (D.rule != "safe_swap_zinf")]
    mean = v.groupby("rule").ratio.mean()
    out = {}
    for fam, key in (("E2a", "safe_lam"), ("E2b", "safe_swap_z")):
        s = mean[[r for r in mean.index if r.startswith(key)]]
        if s.empty:
            continue
        best = s.max()
        near = [r for r in s.index if best - s[r] <= TIE]
        out[fam] = max(near, key=lambda r: float(r[len(key):]))       # most conservative within the tie
    return mean, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--val-fields", type=int, nargs="+", default=C7.VALIDATION)
    ap.add_argument("--metro-listing", type=int, nargs="*", default=C7.METRO_LISTING)
    ap.add_argument("--rank-dir", default="results_p7_rank")
    ap.add_argument("--out", default="results_p7_safe")
    a = ap.parse_args()
    C7.configure()
    C7.guard_fields(a.val_fields)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    models = []
    for s in a.seeds:
        d = joblib.load(C7.model_path(s))
        models.append((d["qa"], d["qb"]))
    print(f"[safe] loaded {len(models)} P3 models (seeds {a.seeds}) in {time.time() - t0:.0f}s", flush=True)
    path = os.path.join(a.out, "safe.csv")
    targets = [(f"field{j}", "validation", None, j) for j in a.val_fields] + \
              [("metro", "metro", ls, None) for ls in a.metro_listing]
    rows = []
    for name, fs, ls, j in targets:
        C, ann, efn = (E.field_setup(j, 1) if fs == "validation" else C7.metro_setup(ls))
        msg = f"[safe] {name}{'' if ls is None else f' ls{ls}'}:"
        for rule, kind, par in variants():
            s0 = time.time()
            init, dev = rollout_safe(efn, models, kind, par)
            v = ev_value(C["EV"], init)
            extra = dict(field_set=fs, family=("E2a" if kind == "lam" else "E2b"), param=par, n_deviate=dev,
                         n_seeds=len(models))
            if ls is not None:
                extra["listing_seed"] = ls
            rows.append(row(C, rule, init, v, name, -2, time.time() - s0, **extra))
            msg += f" {rule} {v / C['v_orc']:.4f}({dev})"
        print(msg, flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    D = pd.DataFrame(rows)
    rk = os.path.join(a.rank_dir, "rank_s0.csv")
    if os.path.exists(rk):                                            # fairness: z = inf must equal the rule
        R = pd.read_csv(rk)
        R = R[R.rule == "rank_announced"]
        key = lambda X: X.assign(k=X.eval_field + "_" + X.get("listing_seed", pd.Series(-1, index=X.index))
                                 .fillna(-1).astype(int).astype(str)).set_index("k").ratio
        mx = float((key(D[D.rule == "safe_swap_zinf"]) - key(R)).dropna().abs().max())
        print(f"[safe] fairness: safe_swap_zinf vs rank_announced max |diff| = {mx:.2e} -> "
              f"{'PASS' if mx < 1e-12 else 'FAIL'}", flush=True)
    mean, chosen = choose(D)
    lines = ["validation mean ratio by variant (201-210):"] + [f"  {r:18s} {v:.4f}" for r, v in mean.items()]
    lines += [f"chosen on validation (tie {TIE} -> more conservative): " +
              ", ".join(f"{k} = {v}" for k, v in chosen.items()),
              "metro is reported for every variant and was not used to choose (already seen by p3_metro_test)."]
    open(os.path.join(a.out, "selection.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(f"[safe] {x}" for x in lines), flush=True)
    print(f"[safe] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
