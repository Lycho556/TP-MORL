# results_p3_explore — exploratory runs on the validation draws (not a registered test)

All runs: validation field draws 201–210 only (test draws 1–10 not used), listing draw 7, M = 20,
training on 20 field draws (101–120, 2 episodes per behaviour rule), past-year field changes in the
state (H = 1). Script: `scripts/p3_explore_rulefeat.py`; summaries: `scripts/p3_explore_summary.py`.

Variants (ratio to the reference schedule, mean over draws; ranking on announced plans = 0.8715,
trajectory-informed greedy = 0.9012):

| Variant | Seeds | Mean | Minus announced (95% t-interval) | Draws above |
|---|---|---|---|---|
| original FQI (lead 4) | 2 | 0.857 | −0.014 (−0.030 to +0.002) | 4 |
| FQI + announced forecast as input | 2 | 0.860 | −0.011 (−0.027 to +0.004) | 3 |
| residual FQI (Q = announced forecast + learned residual) | 5 | 0.866 | −0.006 (−0.022 to +0.011) | 5 |
| residual FQI + conservative deviation, tau = 0.15 | 5 | 0.870 | −0.001 (−0.017 to +0.015) | 7 |
| residual, one-step target (n_iter = 1) | 5 | 0.899 | +0.027 (+0.012 to +0.042) | 10 |
| residual, one-step target + conservative, tau = 0.05 | 5 | 0.900 | +0.028 (+0.014 to +0.042) | 10 |

Lead-8 diagnostic (announcement lead 8 years): ranking 0.920, re-planning 0.932, FQI 0.874,
FQI + forecast input 0.881.

About eight variants were compared on these ten draws, so the numbers carry selection bias.
Any claim must come from a separately registered, one-time test on draws not used here.

Files: `explore_*.csv` (one row per rule and draw), `explore_summary.csv`, `conservative_summary.csv`,
`residual_steps_summary.csv`, `logs/`.
