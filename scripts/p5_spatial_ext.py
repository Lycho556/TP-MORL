# -*- coding: utf-8 -*-
"""p5_spatial_ext.py -- spatial extensions requested in the rev31 review (Phase 2).

E1  Subdistrict capacity. Units are assigned to the six Guangming subdistricts
    (OpenStreetMap admin_level 8, data/raw/subdistrict/unit_subdistrict.csv).
    Two side constraints on top of the district capacity K = 3 per year:
      'cap1'      at most one initiation per subdistrict per year;
      'balanced'  each subdistrict receives a share of the K*T slots
                  proportional to its number of candidates (largest remainder).
    Both keep the problem a linear (network-flow) assignment; it is solved
    exactly with HiGHS (scipy.optimize.milp) for the reference schedule and,
    every year, for the rolling-horizon rules. Greedy rules rank and skip
    units whose subdistrict limit is reached.

E2  Neighbour spillover (controlled). Two adjacent units (sharing a cell edge
    on the 100 m grid) that are both initiated within Delta = 2 years of each
    other earn a joint-renewal bonus beta * min(EV_i, EV_j) (values in their
    chosen years). beta in {0, 0.1, 0.25, 0.5}. The objective becomes
    quadratic in the assignment (a quadratic assignment problem). We evaluate
    the additive rules on the spillover objective, spillover-aware greedy
    rules, and a best-found schedule from local search (a heuristic
    yardstick, not a proven optimum).

E3  Choice-set-matched full-information benchmark. The reference schedule may
    pick any unit in any year. Here the assignment is re-solved with full
    information but restricted (a) to unit-years with positive admission
    probability, and (b) to the prescreened pools a rule actually faced in
    each year of its run. Listing and approval draws share one random stream
    in the simulator, so pools depend on earlier decisions; (b) is therefore
    computed on the pools of the full-information rolling horizon itself.

Usage:  python scripts/p5_spatial_ext.py [--quick]
Output: results_p5_spatial/*.csv, *.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, milp, LinearConstraint, Bounds
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import p1_common as P                                                  # noqa: E402
import p1_rolling_horizon as RH                                        # noqa: E402

REPO = P.REPO
OUT = os.path.join(REPO, "results_p5_spatial")
DATA = os.path.join(REPO, "data", "processed", "gm_dataset_v1")
SUBD = os.path.join(REPO, "data", "raw", "subdistrict", "unit_subdistrict.csv")
K, M, GAMMA = 3, 20, 0.95
SEEDS = (7, 0, 1, 2, 3, 4)
DELTA = 2
BETAS = (0.0, 0.1, 0.25, 0.5)


# ------------------------------------------------------------------ data
def load_space(env):
    uids = env.U["uid"].values.astype(int)
    sd = pd.read_csv(SUBD).set_index("uid").loc[uids]
    names = sorted(sd["subdistrict_en"].unique())
    s_of = np.array([names.index(x) for x in sd["subdistrict_en"]])
    raster = np.load(os.path.join(DATA, "zones_v0", "unit_id.npy"))
    pos = {u: i for i, u in enumerate(uids)}
    E = set()
    for a, b in ((raster[:, :-1], raster[:, 1:]), (raster[:-1, :], raster[1:, :])):
        k = (a > 0) & (b > 0) & (a != b)
        for x, y in zip(a[k], b[k]):
            i, j = pos[int(x)], pos[int(y)]
            E.add((min(i, j), max(i, j)))
    nbr = [[] for _ in uids]
    for i, j in E:
        nbr[i].append(j); nbr[j].append(i)
    xy = np.c_[sd["x"].values, sd["y"].values]
    return dict(s_of=s_of, names=names, edges=sorted(E), nbr=nbr, xy=xy)


def balanced_quota(s_of, n_slots):
    cnt = np.bincount(s_of)
    raw = n_slots * cnt / cnt.sum()
    q = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - q))[: n_slots - q.sum()]:
        q[i] += 1
    return q


# ------------------------------------------------------------------ exact solvers
def solve_assignment(W, years, s_of=None, mode=None, cap_left=None, avail0=None,
                     t0=None, k=K):
    """max sum W[u, y] z[u, y], W shape (n, len(years)); z binary.

    sum_y z <= 1 per unit; sum_u z <= k per year; side constraints by mode:
      cap1      sum_{u in s} z[u, y] <= 1 per subdistrict-year
      balanced  sum_{u in s, y} z[u, y] <= cap_left[s]
    avail0: boolean (n,) -- only these units may take the first year (t0).
    Returns {u: year}.
    """
    n, Y = W.shape
    W = W.copy()
    if avail0 is not None:
        W[~avail0, 0] = 0.0
    keep = W > 0
    ui, yi = np.nonzero(keep)
    if len(ui) == 0:
        return {}
    if mode is None:
        rows = np.unique(ui)
        cols = [c for c in range(Y) for _ in range(k)]
        Wm = W[np.ix_(rows, cols)]
        r, c = linear_sum_assignment(-Wm)
        return {int(rows[a]): int(years[cols[b]]) for a, b in zip(r, c) if Wm[a, b] > 0}
    nv = len(ui)
    c = -W[ui, yi]
    A, lb, ub = [], [], []
    A.append(sparse.csr_matrix((np.ones(nv), (ui, np.arange(nv))), shape=(n, nv))); ub.append(np.ones(n))
    A.append(sparse.csr_matrix((np.ones(nv), (yi, np.arange(nv))), shape=(Y, nv))); ub.append(np.full(Y, k))
    S = int(s_of.max()) + 1
    if mode == "cap1":
        r = s_of[ui] * Y + yi
        A.append(sparse.csr_matrix((np.ones(nv), (r, np.arange(nv))), shape=(S * Y, nv))); ub.append(np.ones(S * Y))
    elif mode == "balanced":
        A.append(sparse.csr_matrix((np.ones(nv), (s_of[ui], np.arange(nv))), shape=(S, nv))); ub.append(np.asarray(cap_left, float))
    A = sparse.vstack(A).tocsr(); ub = np.concatenate(ub)
    res = milp(c, constraints=LinearConstraint(A, -np.inf, ub), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options=dict(disp=False, time_limit=300, mip_rel_gap=0.0))
    assert res.status in (0, 1), res.message
    z = res.x > 0.5
    return {int(u): int(years[y]) for u, y, on in zip(ui, yi, z) if on}


# ------------------------------------------------------------------ rule runner
def run(env, choose, T):
    """choose(t, avail_units(list), init(dict)) -> list of units (<= K)."""
    init, pools = {}, []
    for t in range(T):
        X, meta, cost, units = env.pairs()
        first = {}
        for i, m in enumerate(meta):
            if m[0] >= 0 and int(m[0]) not in init:
                first.setdefault(int(m[0]), i)
        avail = sorted(first)
        pools.append(avail)
        pick = choose(t, avail, init) if avail else []
        acts = []
        for u in pick[:K]:
            acts.append(meta[first[u]]); init[u] = t
        env.step(acts)
    return init, pools


def greedy_chooser(table_at, s_of=None, mode=None, quota=None, bonus=None):
    """Rank available units by table_at(t)[u] (+ bonus(t, u, init)), respecting limits."""
    def choose(t, avail, init):
        F = table_at(t)
        sc = {u: F[u, t] + (bonus(t, u, init, F) if bonus else 0.0) for u in avail}
        used_s, taken = {}, []
        if mode == "balanced":
            used_tot = np.bincount([s_of[u] for u in init], minlength=len(quota))
        for u in sorted(avail, key=lambda u: -sc[u]):
            if len(taken) == K or sc[u] <= 0:
                break
            s = s_of[u] if s_of is not None else None
            if mode == "cap1" and used_s.get(s, 0) >= 1:
                continue
            if mode == "balanced" and used_tot[s] + used_s.get(s, 0) >= quota[s]:
                continue
            taken.append(u); used_s[s] = used_s.get(s, 0) + 1
            if bonus is not None and hasattr(bonus, "seen"):
                bonus.seen[u] = float(F[u, t])
        return taken
    return choose


def rolling_chooser(table_at, T, elig, s_of=None, mode=None, quota=None):
    def choose(t, avail, init):
        F = table_at(t)
        rows = np.where(elig & ~np.isin(np.arange(F.shape[0]), list(init)))[0]
        years = list(range(t, T))
        W = F[np.ix_(rows, years)]
        av = np.isin(rows, avail)
        cap_left = None
        if mode == "balanced":
            cap_left = quota - np.bincount([s_of[u] for u in init], minlength=len(quota))
        plan = solve_assignment(W, years, s_of[rows] if s_of is not None else None, mode,
                                cap_left, av, t)
        return [int(rows[r]) for r, y in plan.items() if y == t]
    return choose


def value(EV, init):
    return float(sum(EV[u, y] for u, y in init.items()))


# ------------------------------------------------------------------ E2 helpers
def spill_value(EV, init, edges, beta, delta=DELTA):
    base = value(EV, init)
    if beta == 0:
        return base, 0.0
    b = 0.0
    for i, j in edges:
        if i in init and j in init and abs(init[i] - init[j]) <= delta:
            b += beta * min(EV[i, init[i]], EV[j, init[j]])
    return base + b, b


def spill_bonus(nbr, beta, delta=DELTA):
    """Spillover term of a greedy score, computed only from the rule's own information:
    the neighbour's value is the score the rule gave it when it was initiated
    (stored in bonus.seen), never its true value."""
    seen = {}

    def bonus(t, u, init, F):
        return sum(beta * min(F[u, t], seen.get(j, 0.0)) for j in nbr[u]
                   if j in init and abs(init[j] - t) <= delta)
    bonus.seen = seen
    return bonus


def local_search(EV, start, edges, nbr, beta, elig, T, iters=60000, seed=0, delta=DELTA):
    """Best-found schedule under the spillover objective (move / swap / replace)."""
    rng = np.random.default_rng(seed)
    plan = dict(start)
    cur, _ = spill_value(EV, plan, edges, beta, delta)
    load = np.bincount(list(plan.values()), minlength=T)
    cand = np.where(elig & (EV.max(1) > 0))[0]

    def unit_val(u, y, pl):
        v = EV[u, y]
        for j in nbr[u]:
            if j in pl and abs(pl[j] - y) <= delta:
                v += beta * min(EV[u, y], EV[j, pl[j]])
        return v
    for _ in range(iters):
        r = rng.random()
        if r < 0.4:                       # move a unit to another year with a free slot
            u = rng.choice(list(plan)); y0 = plan[u]
            y = int(rng.integers(T))
            if y == y0 or load[y] >= K:
                continue
            d = -unit_val(u, y0, plan)
            del plan[u]; d += unit_val(u, y, plan)
            if d > 1e-9:
                plan[u] = y; load[y0] -= 1; load[y] += 1; cur += d
            else:
                plan[u] = y0
        elif r < 0.7:                     # swap the years of two units
            a, b = rng.choice(list(plan), 2, replace=False)
            ya, yb = plan[a], plan[b]
            if ya == yb:
                continue
            old = value(EV, {a: ya, b: yb})
            trial = dict(plan); trial[a], trial[b] = yb, ya
            nv, _ = spill_value(EV, trial, edges, beta, delta)
            if nv > cur + 1e-9:
                plan, cur = trial, nv
        else:                             # replace a selected unit by an unselected neighbour or candidate
            u = rng.choice(list(plan)); y = plan[u]
            pool = [j for j in nbr[u] if j not in plan] if rng.random() < 0.5 else []
            w = int(rng.choice(pool)) if pool else int(rng.choice(cand))
            if w in plan or EV[w, y] <= 0:
                continue
            trial = dict(plan); del trial[u]; trial[w] = y
            nv, _ = spill_value(EV, trial, edges, beta, delta)
            if nv > cur + 1e-9:
                plan, cur = trial, nv
    return plan, cur


def morans_i(vals, xy, band=1500.0, perms=999, seed=0):
    n = len(vals)
    if n < 4:
        return np.nan, np.nan
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    Wm = ((d > 0) & (d <= band)).astype(float)
    S0 = Wm.sum()
    if S0 == 0:
        return np.nan, np.nan
    z = vals - vals.mean()

    def I(zz):
        return n / S0 * (zz @ Wm @ zz) / (zz @ zz)
    obs = I(z)
    rng = np.random.default_rng(seed)
    sims = np.array([I(rng.permutation(z)) for _ in range(perms)])
    p = (1 + np.sum(np.abs(sims - sims.mean()) >= abs(obs - sims.mean()))) / (perms + 1)
    return float(obs), float(p)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only-e1", action="store_true")
    a = ap.parse_args()
    seeds = SEEDS[:1] if a.quick else SEEDS
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    G, env_fn, EV, EVm, elig, v_orc, _ = P.setup(prescreen=M)
    e0 = env_fn()
    T = EV.shape[1]
    sp = load_space(e0)
    s_of, edges, nbr, xy = sp["s_of"], sp["edges"], sp["nbr"], sp["xy"]
    quota = balanced_quota(s_of, K * T)
    ann = RH.forecast_tables(e0, "announced", True)
    cur = RH.forecast_tables(e0, "current", True)
    tab = {"persistence": lambda t: EVm, "announced": lambda t: ann[t],
           "trajectory": lambda t: EV}
    rtab = {"rh_current": lambda t: cur[t], "rh_announced": lambda t: ann[t],
            "rh_full": lambda t: EV}
    print(f"setup {time.time() - t0:.0f}s  v_orc={v_orc:.1f}  subdistricts="
          f"{dict(zip(sp['names'], np.bincount(s_of)))}  quota={quota.tolist()}  "
          f"edges={len(edges)}", flush=True)
    meta = dict(names=sp["names"], n_per_subdistrict=np.bincount(s_of).tolist(),
                balanced_quota=quota.tolist(), n_edges=len(edges), delta=DELTA,
                v_ref_unconstrained=v_orc, seeds=list(seeds))

    # ---------------- E1 ----------------
    rows = []
    refs = {None: v_orc}
    for mode in ("cap1", "balanced"):
        s1 = time.time()
        plan = solve_assignment(EV[:, :] * elig[:, None], list(range(T)), s_of, mode,
                                quota if mode == "balanced" else None)
        refs[mode] = value(EV, plan)
        print(f"[E1] reference {mode}: {refs[mode]:.1f} ({refs[mode] / v_orc:.4f} of unconstrained, "
              f"{time.time() - s1:.1f}s)", flush=True)
    for mode in (None, "cap1", "balanced"):
        for ls in seeds:
            mk = P.env_factory(G, M, ls)
            for name, tf in tab.items():
                init, _ = run(mk(), greedy_chooser(tf, s_of, mode, quota), T)
                rows.append(dict(exp="E1", mode=str(mode or "none"), seed=ls, rule=name + "_greedy",
                                 value=value(EV, init), ratio=value(EV, init) / refs[mode],
                                 ratio_unconstr=value(EV, init) / v_orc, n=len(init)))
            for name, tf in rtab.items():
                s1 = time.time()
                init, _ = run(mk(), rolling_chooser(tf, T, elig, s_of, mode, quota), T)
                rows.append(dict(exp="E1", mode=str(mode or "none"), seed=ls, rule=name,
                                 value=value(EV, init), ratio=value(EV, init) / refs[mode],
                                 ratio_unconstr=value(EV, init) / v_orc, n=len(init),
                                 sec=time.time() - s1))
            print(f"[E1] mode={mode} seed={ls} done ({time.time() - t0:.0f}s)", flush=True)
    e1 = pd.DataFrame(rows)
    e1.to_csv(os.path.join(OUT, "E1_subdistrict_capacity.csv"), index=False)
    meta["E1_reference"] = {str(k): v for k, v in refs.items()}

    if a.only_e1:
        print('E1 only', flush=True)
        return
    # ---------------- E3 ----------------
    # Choice-set-matched full-information benchmark: the full-information
    # rolling horizon run on the same listing draws and the same prescreen M as
    # the practical rules. (Pools recorded during a run shrink as units are
    # initiated, so an assignment restricted to them cannot schedule a chosen
    # unit later and is not a valid upper bound; it is not used.)
    rows = []
    for Mx in ((20,) if a.quick else (10, 20, 50, 0)):
        Gx, env_fx, EVx, EVmx, eligx, v_orcx, _ = P.setup(prescreen=Mx)
        assert abs(v_orcx - v_orc) < 1e-6
        for ls in seeds:
            mk = P.env_factory(Gx, Mx, ls)
            r = dict(exp="E3", M=Mx if Mx else "all", seed=ls, ref=v_orc)
            r["rh_full"] = value(EV, run(mk(), rolling_chooser(rtab["rh_full"], T, elig), T)[0])
            r["rh_announced"] = value(EV, run(mk(), rolling_chooser(rtab["rh_announced"], T, elig), T)[0])
            r["trajectory_greedy"] = value(EV, run(mk(), greedy_chooser(tab["trajectory"]), T)[0])
            r["persistence_greedy"] = value(EV, run(mk(), greedy_chooser(tab["persistence"]), T)[0])
            rows.append(r)
        print(f"[E3] M={Mx} done ({time.time() - t0:.0f}s)", flush=True)
    e3 = pd.DataFrame(rows)
    for c in ["rh_full", "rh_announced", "trajectory_greedy", "persistence_greedy"]:
        e3[c + "_vs_ref"] = e3[c] / e3["ref"]
        e3[c + "_vs_matched"] = e3[c] / e3["rh_full"]
    e3.to_csv(os.path.join(OUT, "E3_choice_set_benchmark.csv"), index=False)

    # ---------------- E2 ----------------
    rows, sched = [], {}
    for beta in BETAS:
        bon_true = spill_bonus(nbr, beta, EV)
        start = solve_assignment(EV * elig[:, None], list(range(T)))
        if beta > 0:
            ref_plan, v_ref = local_search(EV, start, edges, nbr, beta, elig, T,
                                           iters=20000 if a.quick else 80000)
        else:
            ref_plan, v_ref = start, value(EV, start)
        for ls in seeds:
            mk = P.env_factory(G, M, ls)
            runs = {}
            for name, tf in tab.items():
                runs[name + "_greedy"] = run(mk(), greedy_chooser(tf), T)[0]
                if beta > 0:
                    runs[name + "_greedy_aware"] = run(mk(), greedy_chooser(tf, bonus=spill_bonus(nbr, beta, EV)), T)[0]
            runs["rh_announced"] = run(mk(), rolling_chooser(rtab["rh_announced"], T, elig), T)[0]
            runs["rh_full"] = run(mk(), rolling_chooser(rtab["rh_full"], T, elig), T)[0]
            if beta > 0:
                runs["rh_full_then_ls"] = local_search(EV, runs["rh_full"], edges, nbr, beta,
                                                       elig, T, iters=5000 if a.quick else 20000,
                                                       seed=ls)[0]
            for name, init in runs.items():
                v, b = spill_value(EV, init, edges, beta)
                sel = np.array(sorted(init))
                yrs = np.array([init[u] for u in sel], float)
                mi, mp = morans_i(yrs, xy[sel]) if len(sel) else (np.nan, np.nan)
                pairs = sum(1 for i, j in edges if i in init and j in init)
                close = sum(1 for i, j in edges if i in init and j in init and abs(init[i] - init[j]) <= DELTA)
                rows.append(dict(exp="E2", beta=beta, seed=ls, rule=name, value=v, bonus=b,
                                 ratio=v / v_ref, morans_I_year=mi, morans_p=mp,
                                 adjacent_pairs=pairs, adjacent_within_delta=close, n=len(init)))
            if ls == seeds[0]:
                sched[str(beta)] = {k: {str(u): int(y) for u, y in v.items()} for k, v in runs.items()}
        sched[f"ref_{beta}"] = {str(u): int(y) for u, y in ref_plan.items()}
        vr, br = spill_value(EV, ref_plan, edges, beta)
        rows.append(dict(exp="E2", beta=beta, seed=-1, rule="best_found", value=vr, bonus=br,
                         ratio=1.0, n=len(ref_plan),
                         adjacent_within_delta=sum(1 for i, j in edges if i in ref_plan and j in ref_plan and abs(ref_plan[i] - ref_plan[j]) <= DELTA)))
        print(f"[E2] beta={beta} best-found {v_ref:.1f} (bonus {br:.1f}) ({time.time() - t0:.0f}s)", flush=True)
    e2 = pd.DataFrame(rows)
    e2.to_csv(os.path.join(OUT, "E2_neighbour_spillover.csv"), index=False)
    json.dump(sched, open(os.path.join(OUT, "E2_schedules_seed7.json"), "w"))
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
