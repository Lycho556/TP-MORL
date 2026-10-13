import sys, time, numpy as np, pandas as pd
src = open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'diag_waiting_value.py')).read().split("VARS = [")[0]
exec(src)
rows=[]
for j in range(1,11):
    C = build("main", 20, field_seed=j); EV, elig, v = C["EV"], C["elig"], C["v_orc"]; e0 = C["env_fn"]()
    for L in (4, 8, 12):
        globals()['LEAD'] = L
        for pbc in (False, True):
            Ft = tables(e0, pbc, "announced")
            for mode in ("rank","rolling"):
                get = (lambda t, Ft=Ft: now_only(Ft[t], t)) if mode=="rank" else (lambda t, Ft=Ft: Ft[t])
                init = run_rolling(C["env_fn"](), get, elig)
                rows.append(dict(field=j, lead=L, pbc_true=pbc, use=mode, ratio=sum(EV[u,y] for u,y in init.items())/v))
    print(j, flush=True)
    pd.DataFrame(rows).to_csv(R + 'results_p2_diag/announce_lead.csv', index=False)
