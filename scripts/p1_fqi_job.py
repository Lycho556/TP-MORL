# -*- coding: utf-8 -*-
"""p1_fqi_job.py — one value-based (double FQI) job for the P1 server sweep.

Every configuration is built through p1_common.setup, so the scenario is the
published main configuration plus exactly one P1 switch:

  main    plain reproduction (cluster field, conformity on). Seed 0 must give
          0.9095038725980372 bit for bit (self-check; exits non-zero if not).
  noconf  conformity=False: FQI rewards, greedy tables, reference and all
          valuations use the conditional EV; listing is unchanged.
  metro   infra_mode='metro' (observed stations).
  oos     train double FQI on the main field (field_seed=None), then WITHOUT
          retraining roll the trained estimators out on setup(field_seed=k) for
          every k in --eval-field-seeds, valued on field k's own EV/reference.
          Persistence / trajectory-informed greedy and the reference are
          recomputed on each field k. --insample-too additionally trains and
          evaluates on field k (same seed) for an in-sample comparison.

Output: ONE csv <out>/<mode>_M<prescreen>_s<seed>.csv with one row per
(rule, eval_field): ratio, value, reference, loss_selection, loss_timing,
n_init, seconds (+ diagnostics). For seed 0 the value-based schedule is also
written as <out>/<mode>_M<prescreen>_s0_schedule.json.

Scenario state (field seed, infra mode, amplitudes) is module-global and is
read when an environment is built, so each setup() call must precede the
rollouts on that configuration; the oos loop is ordered accordingly.

Usage (from repo root):
  PYTHONPATH=src TPMORL_NJOBS=1 python scripts/p1_fqi_job.py --mode main --seed 0
  PYTHONPATH=src TPMORL_NJOBS=1 python scripts/p1_fqi_job.py --mode oos --seed 0 \
      --eval-field-seeds 1 2 --insample-too
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1_common as P                                                  # noqa: E402
from exp_fqi_local import make_trees                                   # noqa: E402

from tpmorl.rl import fqi                                              # noqa: E402

MAIN_SEED0 = 0.9095038725980372
# ExtraTrees 训练跨平台（本地 arm64 / 服务器 x86_64）与跨 sklearn 版本不逐位一致：
# 服务器 sklearn 1.7.2 实测 0.9091809875451883，差 3.2e-4，约为种子间标准差（v20 S2
# 服务器 5 种子 0.0097）的 3%。环境、oracle、规则基线仍由启动脚本自检逐位校验，
# 这里只确认 learner 落在同一量级、代码版本没拿错。可用环境变量 P1_REPRO_TOL 覆盖。
REPRO_TOL = float(os.environ.get("P1_REPRO_TOL", "0.01"))


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024.0   # bytes vs KiB


VARIANT = None   # set from --variant in main()


def build(mode, prescreen, field_seed=None):
    kw = dict(prescreen=prescreen)
    if mode == "variant":
        from p1_rolling_horizon import variant_overrides
        kw.update(variant_overrides(VARIANT))
    if mode == "noconf":
        kw["conformity"] = False
    elif mode == "metro":
        kw["infra_mode"] = "metro"
    if field_seed is not None:
        kw["field_seed"] = int(field_seed)
    G, env_fn, EV, EVm, elig, v_orc, oplan = P.setup(**kw)
    return dict(G=G, env_fn=env_fn, EV=EV, EVm=EVm, elig=elig, v_orc=v_orc,
                oplan=oplan)


def ev_value(EV, init):
    return float(sum(EV[u, min(t, EV.shape[1] - 1)] for u, t in init.items()))


def row(C, rule, init, value, field, seed, secs, **extra):
    dec = C["G"].decompose(C["EV"], init, P.DEFAULTS["quota"], C["elig"], C["v_orc"])
    r = dict(rule=rule, eval_field=field, seed=seed, ratio=value / C["v_orc"],
             value=value, reference=C["v_orc"],
             loss_selection=dec["loss_selection"], loss_timing=dec["loss_timing"],
             n_init=len(init), seconds=round(secs, 1))
    r.update(extra)
    return r


def train_fqi(C, seed, ep_each, gamma):
    EV = C["EV"]
    data = fqi.collect(C["env_fn"], lambda u, t: float(EV[u, min(t, EV.shape[1] - 1)]),
                       {"random": None, "myopic": C["EVm"], "temporal_greedy": EV},
                       n_ep_each=ep_each, seed=seed)
    qa, qb, _ = fqi.fit_double(data, gamma, make_trees(seed=seed), verbose=False)
    return qa, qb, data


def eval_fqi(C, qa, qb):
    init, R, diag = fqi.rollout(C["env_fn"], qa, qb, None, collect_q=True)
    return init, ev_value(C["EV"], init), diag


def train_supervised(C, seed):
    G = C["G"]
    sd, bcm = G.bc_actor(C["env_fn"], C["EV"], None, seed=seed)
    net = G.TP.Pointer(); net.load_state_dict(sd, strict=False)
    return net, bcm


def eval_supervised(C, net):
    iv, v = C["G"].run_policy(C["env_fn"](), C["EV"], C["EVm"], "ppo", net=net,
                              quota=P.DEFAULTS["quota"])
    return iv, float(v)


def greedy_rows(C, field, seed, with_random):
    out = []
    kinds = [("persistence", "myopic"), ("trajectory", "forecast")]
    if with_random:
        kinds.append(("random", "random"))
    for tag, kind in kinds:
        t0 = time.time()
        rng = np.random.default_rng(seed if kind == "random" else 0)
        iv, vv = C["G"].run_policy(C["env_fn"](), C["EV"], C["EVm"], kind,
                                   quota=P.DEFAULTS["quota"], rng=rng)
        out.append(row(C, tag, iv, float(vv), field, seed if kind == "random" else -1,
                       time.time() - t0))
    out.append(row(C, "reference", dict(C["oplan"]), C["v_orc"], field, -1, 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["main", "noconf", "metro", "oos", "variant"], required=True)
    ap.add_argument("--variant", default=None,
                    help="mode variant: amp_low|amp_high|win_narrow|win_broad|onset_early|onset_late")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--prescreen", type=int, default=20, help="M (annual prescreen)")
    ap.add_argument("--eval-field-seeds", type=int, nargs="+", default=[],
                    help="oos only: alternative field draws to evaluate on")
    ap.add_argument("--insample-too", action="store_true",
                    help="oos only: also train+evaluate on each field k")
    ap.add_argument("--out", default="results_p1/fqi")
    ap.add_argument("--with-supervised", action="store_true")
    ap.add_argument("--with-random", action="store_true")
    ap.add_argument("--listing-seeds", type=int, nargs="+", default=[],
                    help="also evaluate every rule (trained once) on these listing "
                         "seeds of the home configuration; rows carry listing_seed")
    a = ap.parse_args()
    global VARIANT
    VARIANT = a.variant
    if a.mode == "variant" and not a.variant:
        ap.error("--mode variant needs --variant")
    if a.mode == "oos" and not a.eval_field_seeds:
        ap.error("--mode oos needs --eval-field-seeds")
    os.makedirs(a.out, exist_ok=True)
    tag = (f"{a.mode}_M{a.prescreen}_s{a.seed}" if a.mode != "variant"
           else f"variant-{a.variant}_M{a.prescreen}_s{a.seed}")
    fn = os.path.join(a.out, f"{tag}.csv")
    ep_each, gamma = P.DEFAULTS["ep_each"], P.DEFAULTS["gamma"]
    T0 = time.time()
    rows = []

    def flush():
        D = pd.DataFrame(rows)
        D.insert(0, "mode", a.mode); D.insert(1, "prescreen", a.prescreen)
        D["peak_rss_mb"] = round(peak_rss_mb(), 1)
        D["job_seconds"] = round(time.time() - T0, 1)
        D.to_csv(fn, index=False)

    # ---------------- training configuration (main field for oos)
    train_mode = "main" if a.mode == "oos" else a.mode
    home = ("main" if a.mode in ("main", "oos")
            else (f"variant-{a.variant}" if a.mode == "variant" else a.mode))
    C = build(train_mode, a.prescreen)
    t0 = time.time()
    qa, qb, data = train_fqi(C, a.seed, ep_each, gamma)
    t_train = time.time() - t0
    t0 = time.time()
    init, v, diag = eval_fqi(C, qa, qb)
    rows.append(row(C, "value_based", init, v, home, a.seed,
                    t_train + time.time() - t0, train_field=home,
                    train_seconds=round(t_train, 1),
                    n_samples=sum(len(d["r"]) for d in data),
                    q_vs_value_rho=fqi.q_vs_value_corr(data, qa, qb, None)))
    print(f"[{tag}] value_based on {home}: {v / C['v_orc']!r}  ({t_train:.0f}s train)",
          flush=True)
    sched = {home: dict(ratio=v / C["v_orc"], value=v, reference=C["v_orc"],
                        init={str(u): int(t) for u, t in sorted(init.items())})}
    if a.mode == "main" and a.prescreen == 20 and a.seed == 0:
        got = v / C["v_orc"]; delta = got - MAIN_SEED0
        verdict = ("PASS" if got == MAIN_SEED0
                   else "PASS_TOL" if abs(delta) <= REPRO_TOL else "FAIL")
        rows[-1]["repro_check"] = verdict          # PASS=逐位，PASS_TOL=容差内
        rows[-1]["repro_expected"] = MAIN_SEED0
        rows[-1]["repro_delta"] = delta
        flush()
        if verdict == "FAIL":
            sys.exit(f"REPRODUCTION FAILED: got {got!r}, expected {MAIN_SEED0!r} "
                     f"(|delta|={abs(delta):.3g} > tol {REPRO_TOL})")
        print(f"[{tag}] reproduction check {verdict}: got {got!r}, "
              f"delta {delta:+.3g} (tol {REPRO_TOL})", flush=True)
    del data

    net = None
    if a.with_supervised:
        t0 = time.time()
        net, bcm = train_supervised(C, a.seed)
        iv, vs = eval_supervised(C, net)
        rows.append(row(C, "supervised", iv, vs, home, a.seed, time.time() - t0,
                        train_field=home, n_samples=bcm.get("bc_n", np.nan)))
    rows += greedy_rows(C, home, a.seed, a.with_random)
    for r_ in rows:
        r_.setdefault("listing_seed", P.LISTING_SEED)
    flush()

    # ---------------- other listing seeds of the home configuration (no retraining)
    for ls in [x for x in a.listing_seeds if x != P.LISTING_SEED]:
        Cl = dict(C); Cl["env_fn"] = P.env_factory(C["G"], a.prescreen, ls)
        n0 = len(rows)
        t0 = time.time()
        il, vl, _ = eval_fqi(Cl, qa, qb)
        rows.append(row(Cl, "value_based", il, vl, home, a.seed, time.time() - t0,
                        train_field=home))
        if net is not None:
            t0 = time.time()
            iv, vs = eval_supervised(Cl, net)
            rows.append(row(Cl, "supervised", iv, vs, home, a.seed, time.time() - t0,
                            train_field=home))
        rows += greedy_rows(Cl, home, a.seed, a.with_random)
        for r_ in rows[n0:]:
            r_["listing_seed"] = ls
        print(f"[{tag}] listing seed {ls}: value_based {vl / C['v_orc']:.4f}", flush=True)
        flush()

    # ---------------- out-of-sample fields
    if a.mode == "oos":
        for k in a.eval_field_seeds:
            fk = f"field{k}"
            Ck = build("main", a.prescreen, field_seed=k)   # sets global field seed k
            assert not np.allclose(Ck["EV"], C["EV"]), f"field {k} EV equals main EV"
            t0 = time.time()
            ik, vk, _ = eval_fqi(Ck, qa, qb)
            rows.append(row(Ck, "value_based", ik, vk, fk, a.seed, time.time() - t0,
                            train_field=home))
            sched[f"{fk}_oos"] = dict(ratio=vk / Ck["v_orc"], value=vk,
                                      reference=Ck["v_orc"],
                                      init={str(u): int(t) for u, t in sorted(ik.items())})
            msg = f"[{tag}] {fk} oos {vk / Ck['v_orc']:.4f}"
            if net is not None:
                t0 = time.time()
                iv, vs = eval_supervised(Ck, net)
                rows.append(row(Ck, "supervised", iv, vs, fk, a.seed, time.time() - t0,
                                train_field=home))
            rows += greedy_rows(Ck, fk, a.seed, a.with_random)
            for r_ in rows:
                r_.setdefault("listing_seed", P.LISTING_SEED)
            if a.insample_too:
                t0 = time.time()
                qa_k, qb_k, data_k = train_fqi(Ck, a.seed, ep_each, gamma)
                tt = time.time() - t0
                ik2, vk2, _ = eval_fqi(Ck, qa_k, qb_k)
                rows.append(row(Ck, "value_based", ik2, vk2, fk, a.seed,
                                time.time() - t0, train_field=fk,
                                train_seconds=round(tt, 1),
                                n_samples=sum(len(d["r"]) for d in data_k),
                                q_vs_value_rho=fqi.q_vs_value_corr(data_k, qa_k, qb_k,
                                                                   None)))
                sched[f"{fk}_insample"] = dict(
                    ratio=vk2 / Ck["v_orc"], value=vk2, reference=Ck["v_orc"],
                    init={str(u): int(t) for u, t in sorted(ik2.items())})
                msg += f"  in-sample {vk2 / Ck['v_orc']:.4f}"
                del data_k, qa_k, qb_k
            print(msg, flush=True)
            flush()

    for r_ in rows:
        r_.setdefault("listing_seed", P.LISTING_SEED)
    flush()
    if a.seed == 0:
        json.dump(dict(mode=a.mode, prescreen=a.prescreen, seed=a.seed,
                       schedules=sched),
                  open(os.path.join(a.out, f"{tag}_schedule.json"), "w"), indent=1)
    print(f"[{tag}] done {time.time() - T0:.0f}s  peak RSS {peak_rss_mb():.0f} MB  -> {fn}",
          flush=True)


if __name__ == "__main__":
    main()
