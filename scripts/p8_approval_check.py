# -*- coding: utf-8 -*-
"""p8_approval_check.py -- approval timing against the record, pre-registered in
results_p8_approval/PREREGISTRATION.md (with the amendment). Window-adjusted maximum
likelihood of the announcement-to-approval lag (bins ceil(lag) = 1..12) for pairs observed only
if approved in 2018-2022; 5-fold cross-validated log score against the paper's hazards; bootstrap
intervals. Output: results_p8_approval/{summary.json, bins.csv, cv_pairs.csv}."""
import json, os
import numpy as np, pandas as pd
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..")
OUT = os.path.join(R, "results_p8_approval"); J = 12; YEARS = range(2018, 2023)
PAPER_H = np.array([0.174, 0.212, 0.239, 0.277, 0.179])
d = pd.read_csv(os.path.join(R, "data/processed/gm_dataset_v1/tables/sz_renewal_matched_pairs.csv")).dropna(subset=["时滞年"]).reset_index(drop=True)
a = d["公告年"].values; Y = d["审批年"].values.astype(int)
jb = lambda lag: np.clip(np.maximum(np.ceil(lag), 1), 1, J).astype(int)
jobs = jb(Y - a); feas = np.array([[jb(y - ai) for y in YEARS] for ai in a])

def fit(idx):
    def nll(th):
        f = np.exp(th - th.max()); f /= f.sum()
        return -np.sum(np.log(f[jobs[idx] - 1]) - np.log(f[feas[idx] - 1].sum(1)))
    th = minimize(nll, np.zeros(J), method="L-BFGS-B").x
    f = np.exp(th - th.max()); return f / f.sum()

def hazards(f):
    h, S = [], 1.0
    for j in range(5): h.append(f[j] / S); S -= f[j]
    return np.array(h)

def paper_f():
    f, S = [], 1.0
    for h in PAPER_H: f.append(S * h); S -= S * h
    return np.array(f + [S / (J - 5)] * (J - 5))

def lifetable_f(idx):
    """The paper's life table (hazards from observed lags, no window adjustment), refitted on idx."""
    j = jobs[idx]; h, risk = [], len(j)
    for b in range(1, 6): k = (j == b).sum(); h.append(k / risk); risk -= k
    f, S = [], 1.0
    for x in h: f.append(S * x); S -= S * x
    return np.array(f + [S / (J - 5)] * (J - 5))

def logscore(f, idx):
    f = np.maximum(f, 1e-6); f = f / f.sum()
    return np.log(f[jobs[idx] - 1]) - np.log(f[feas[idx] - 1].sum(1))

def main():
    os.makedirs(OUT, exist_ok=True)
    allidx = np.arange(len(d)); f = fit(allidx); cum = np.cumsum(f)
    rng = np.random.default_rng(0); B = []
    for _ in range(1000):
        fb = fit(rng.integers(0, len(d), len(d))); cb = np.cumsum(fb)
        B.append((cb[4], int(np.argmax(cb >= 0.5)) + 1, *hazards(fb)))
    B = np.array(B)
    # 5-fold CV stratified by announcement year
    ay = np.floor(a).astype(int); folds = np.empty(len(d), int)
    for yr in np.unique(ay):
        ix = np.where(ay == yr)[0]; rng2 = np.random.default_rng(0 + yr); ix = rng2.permutation(ix)
        folds[ix] = np.arange(len(ix)) % 5
    s_adj = np.empty(len(d)); s_lt = np.empty(len(d)); s_pap = logscore(paper_f(), allidx)
    for k in range(5):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        s_adj[te] = logscore(fit(tr), te); s_lt[te] = logscore(lifetable_f(tr), te)
    d2 = s_adj - s_lt; se2 = d2.std(ddof=1) / np.sqrt(len(d2))
    diff = s_adj - s_pap; se = diff.std(ddof=1) / np.sqrt(len(diff))
    pd.DataFrame(dict(announce=a, approve=Y, bin=jobs, fold=folds, score_adjusted=s_adj, score_paper=s_pap, score_lifetable_refit=s_lt)).to_csv(os.path.join(OUT, "cv_pairs.csv"), index=False)
    pd.DataFrame(dict(bin=np.arange(1, J + 1), f_adjusted=f, f_paper=paper_f(), observed=np.bincount(jobs, minlength=J + 1)[1:])).to_csv(os.path.join(OUT, "bins.csv"), index=False)
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    S = dict(n=len(d), share5_adjusted=float(cum[4]), share5_ci=q(B[:, 0]), share5_paper=float(paper_f()[:5].sum()),
             median_bin_adjusted=int(np.argmax(cum >= 0.5)) + 1, median_bin_ci=q(B[:, 1]),
             hazards_adjusted=hazards(f).round(3).tolist(), hazards_ci=[q(B[:, 2 + j]) for j in range(5)], hazards_paper=PAPER_H.tolist(),
             cv_logscore_adjusted=float(s_adj.mean()), cv_logscore_paper=float(s_pap.mean()),
             cv_diff=float(diff.mean()), cv_diff_ci=[float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)],
             naive_share5=float((jobs <= 5).mean()), naive_median_lag=float(np.median(Y - a)))
    S["posthoc_cv_logscore_lifetable_refit"] = float(s_lt.mean())
    S["posthoc_cv_diff_adjusted_minus_lifetable_refit"] = float(d2.mean())
    S["posthoc_cv_diff_ci"] = [float(d2.mean() - 1.96 * se2), float(d2.mean() + 1.96 * se2)]
    S["posthoc_note"] = "not pre-registered: the paper life table refitted on the training folds, for a like-for-like comparison"
    S["verdict_check1"] = "supported" if abs(S["share5_adjusted"] - 0.706) <= 0.10 and abs(S["median_bin_adjusted"] - 3) <= 1 else "sensitive"
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(S, indent=1))

if __name__ == "__main__":
    main()
