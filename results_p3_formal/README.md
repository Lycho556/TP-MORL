# results_p3_formal — registered one-time test (protocol: docs/P3/registration_p3_formal.md)

Code: commit 17a4a2b (`scripts/p3_formal_test.py`, registered before the run), analysis
`scripts/p3_formal_summary.py`. Run locally on macOS arm64, Python 3.11.16, NumPy 2.4.6,
scikit-learn 1.9.1, single-threaded jobs, five seeds in parallel, about 6 minutes per seed.

## Primary result (test field draws 1–10, listing draw 7, M = 20, seeds 0–4)
| Rule | Ratio to reference |
|---|---|
| Persistence greedy | 0.768 |
| Ranking on announced plans (benchmark) | 0.871 |
| Residual value learning, one-step target | 0.896 |
| Trajectory-informed greedy | 0.900 |
| Rolling horizon, full information | 0.963 |

Policy minus ranking on announced plans: +0.025, 95% t-interval +0.017 to +0.034
(t = 6.56, p = 1.0e-4; Wilcoxon p = 0.002); above the benchmark on 10 of 10 draws and 50 of 50
draw–seed pairs; seed means 0.895–0.897. Registered wording: improves on ranking on announced plans.
Main field (secondary): 0.896 against 0.865.

## Checks
- rule_replay reproduces ranking on announced plans on every draw and seed (max deviation 1e-16):
  same candidates, capacity, listing draw and pipeline. Every rule initiates 75 units.
- Ranking on announced plans equals the rolling horizon with announced information (Proposition 1).
- Leakage check (`leak_check.csv`): decision-time inputs unchanged when unavailable future field
  values are randomized (years 3, 10, 17); negative control changes the forecast (88.9).

Files: `test_s{0-4}.csv`, `formal_summary.csv`, `formal_by_field.csv`, `leak_check.csv`.
