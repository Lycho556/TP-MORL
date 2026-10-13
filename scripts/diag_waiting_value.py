"""Diagnostic only: how much timing value is left for a rule that sees the announced
information, if it also knew (a) the true P/B/C trajectories or (b) the true infrastructure.
Uses the project's own forecast machinery; no training, no change to repo code."""
import os, sys, time, json
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) + "/"
sys.path.insert(0, R + "scripts"); sys.path.insert(0, R + "src")
import p1_common as P
from p1_fqi_job import build
from p1_rolling_horizon import run_rolling, GAMMA
from tpmorl.eval import ev as EVM
from tpmorl.env import schedule as S
from tpmorl.env import opportunity as OPPM
LEAD = int(round(OPPM.INFRA_ANNOUNCE_LEAD))

class Info:
    def __init__(self, opp, t0, pbc_true, infra):
        self.o, self.t0, self.off, self.kind = opp, int(t0), int(t0), opp.kind
        self.pbc_true, self.infra = pbc_true, infra          # infra: 'frozen' | 'announced' | 'true'
    def admit_prob(self, s):
        return self.o.admit_prob(s + self.off if self.pbc_true else self.t0)
    def hazard_mult(self, s):
        return self.o.hazard_mult(s + self.off if self.pbc_true else self.t0)
    def value_mult(self, u, s_init, s_done):
        td = int(s_done) + self.off
        if self.infra == "frozen":   return self.o.value_mult(u, self.t0, self.t0)
        if self.infra == "announced": return self.o.value_mult(u, self.t0, min(td, self.t0 + LEAD))
        return self.o.value_mult(u, self.t0, td)

def tables(env, pbc_true, infra):
    base = np.asarray(env.farcap, float) * np.asarray(env.ncell, float)
    by = np.array([S.BUILD_YEARS_BY_CHANNEL[int(c)] for c in env.ch], int)
    tau = int(S.TAU_VALID + S.TAU_EXT); T, Te, n = env.T, env.T_eval, env.n
    out = []
    for t in range(T):
        raw = EVM.ev_matrix(Info(env.opp, t, pbc_true, infra), base, T - t, Te - t, by,
                            S.HAZARD, tau, gamma=GAMMA, use_admit=False)
        if pbc_true:
            fac = np.stack([np.broadcast_to(np.asarray(env.opp.admit_prob(y), float), (n,)) for y in range(t, T)], 1)
        else:
            A = np.broadcast_to(np.asarray(env.opp.admit_prob(t), float), (n,))
            fac = np.repeat(A[:, None], T - t, 1)
        F = np.zeros((n, T)); F[:, t:] = raw * (GAMMA ** t) * fac
        out.append(F)
    return out

def now_only(F, t):
    G = np.zeros_like(F); G[:, t] = F[:, t]; return G

VARS = [("announced (rule in paper)", False, "announced"),
        ("+ true P/B/C, announced infra", True, "announced"),
        ("frozen P/B/C, + true infra", False, "true"),
        ("true P/B/C and true infra", True, "true")]
fields = [int(x) for x in sys.argv[1:]] or list(range(1, 11))
rows = []
for j in fields:
    t0 = time.time()
    C = build("main", 20, field_seed=j); EV, elig, v = C["EV"], C["elig"], C["v_orc"]
    e0 = C["env_fn"]()
    for name, pbc, inf in VARS:
        Ft = tables(e0, pbc, inf)
        for mode in ("rank", "rolling"):
            get = (lambda t, Ft=Ft: now_only(Ft[t], t)) if mode == "rank" else (lambda t, Ft=Ft: Ft[t])
            init = run_rolling(C["env_fn"](), get, elig)
            rows.append(dict(field=j, info=name, use=mode, ratio=sum(EV[u, y] for u, y in init.items()) / v))
    # true table directly (should equal trajectory-informed greedy / rh_full)
    for mode in ("rank", "rolling"):
        get = (lambda t: now_only(EV, t)) if mode == "rank" else (lambda t: EV)
        init = run_rolling(C["env_fn"](), get, elig)
        rows.append(dict(field=j, info="true EV table (check)", use=mode, ratio=sum(EV[u, y] for u, y in init.items()) / v))
    print(f"field{j} done in {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(R + "results_p2_diag/waiting_value.csv", index=False)
