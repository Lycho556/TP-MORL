# results_p3_shift and results_p3_metro — robustness of the P3 method

Method as registered in docs/P3/registration_p3_formal.md (residual value learning, one-step target).
- results_p3_metro/: policy trained on the standard scenario fields (announced layer leads by exactly
  4 years), evaluated on the main field and the observed metro field, listing draws 7, 0, 1, 2, 3, 4.
- results_p3_shift/: same, but the training scenarios draw a cluster-specific announcement lead
  uniformly in [3, 7] years (opportunity.ANNOUNCE_SHIFT_RANGE; range set in advance). This variant
  was run after the metro result of the standard training had been seen.

| Evaluation | Ranking on announced plans | Trajectory-informed | RL, standard training | RL, lead 3-7 in training |
|---|---|---|---|---|
| Validation 201-210 | 0.872 | 0.901 | 0.899 | 0.899 (+0.027, 10/10) |
| Test 1-10 | 0.871 | 0.900 | 0.896 (+0.025, 10/10) | 0.899 (+0.028, 10/10) |
| Main field, 6 draws | 0.861 | 0.901 | 0.899 (+0.039, 30/30 runs) | 0.901 (+0.040, 30/30) |
| Metro field, 6 draws | 0.941 | 0.949 | 0.916 (-0.024, 0/30) | 0.929 (-0.012, 4/30) |
