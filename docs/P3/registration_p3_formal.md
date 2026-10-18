# Registration: one-time test of residual value learning with a one-step target (P3)

Registered before the run. The method was selected on validation draws 201–210
(`results_p3_explore/`, about eight variants compared there); nothing is changed after this commit.

## Frozen method
- State: the paper's state with the past year's field changes (H = 1), plus two columns from the
  announced forecast F_t (4-year lead): f_now = F_t(u, t), f_best = max_{y≥t} F_t(u, y).
- Value: Q(s, u) = f_now + ΔQ(s, u). Double extremely randomized trees (200 trees, as in the paper)
  fit ΔQ to the one-step target r − f_now (fitted Q iteration with n_iter = 1); r = EV[u, t].
- Decision: each year initiate the K = 3 prescreened listed units with the highest Q. No threshold.
- Training: field draws 101–120, 2 episodes per behaviour rule (random, persistence greedy,
  trajectory-informed greedy); training seeds 0–4.
- Case setting as in the paper: announcement lead 4 years, M = 20, listing draw 7.

## Test
- Primary: field draws 1–10, five training seeds. Secondary: the main field.
- Benchmark: ranking on announced plans on the same draws. Also reported: persistence greedy,
  trajectory-informed greedy, rolling horizon with full information.

## Analysis (fixed in advance)
- Per draw, the policy's mean over the five seeds minus ranking on announced plans; mean
  difference with its 95% t-interval over the ten draws; one-sample t-test and Wilcoxon signed-rank
  test; number of draws (of 10) and draw–seed pairs (of 50) above the benchmark; range of seed means.
- Wording: "improves on ranking on announced plans" only if the 95% interval lies above zero;
  otherwise "does not beat" (interval includes zero) or "falls below" (interval below zero).

## Fairness and leakage checks (reported, do not change the result)
- Same candidates: the policy and every rule choose year-t initiations from the same prescreened
  listed units, with the same capacity, listing draw and pipeline. `rule_replay` (the policy's
  candidate pipeline with the rule's choice) must reproduce ranking on announced plans exactly.
- The reference schedule (no prescreen, no listing draws) is only the common denominator of all ratios;
  differences between rules are differences in value divided by the same constant.
- Leakage: for test draw 1 and years 3, 10, 17, the decision-time inputs are recomputed after replacing
  field values unavailable at year t by random numbers; they must be unchanged (negative control:
  changing commissioned infrastructure at t + 4, which is announced, must change the forecast).

## Commands (repo root)
    OMP_NUM_THREADS=1 PYTHONPATH=src TPMORL_NJOBS=1 python scripts/p3_formal_test.py --seed 0 --baselines --leak-check
    OMP_NUM_THREADS=1 PYTHONPATH=src TPMORL_NJOBS=1 python scripts/p3_formal_test.py --seed {1,2,3,4}
Output: `results_p3_formal/test_s{seed}.csv`, `results_p3_formal/leak_check.csv`.
