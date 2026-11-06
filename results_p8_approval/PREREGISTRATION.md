# P8-A: approval timing against the record (written before running, 2026-11-06)

Data. 109 matched pairs of renewal-plan announcement and unit-plan approval in Shenzhen
(tables/sz_renewal_matched_pairs.csv; one pair without an announcement date excluded).
All approvals fall in 2018-2022 and all announcements in 2010-2018, so the sample is selected
on the approval year: a pair announced in year a is observed only if its lag lies in
[2018 - a, 2022 - a]. The calibration in Appendix A.1 (hazard 0.174, 0.212, 0.239, 0.277,
0.179; 70.6% within five years; median lag 3.00 years) is a life table of the observed lags,
lag bin j = ceil(lag), that does not adjust for this window.

Check 1 (window-adjusted estimate). Discrete lag bins j = 1..12. Maximum likelihood of the
bin probabilities f_j under the sampling window: each pair contributes
f_{j_i} / sum_{Y=2018..2022} f_{ceil(Y - a_i)}. Report f, the implied hazards for bins 1-5,
the share within five years (of approvals within 12 years) and the median bin.
  * Calibration supported: share within five years within 0.10 of 0.706 AND median bin
    within one year of 3.
  * Calibration sensitive: otherwise. Then the main comparison (rules only, main field, six
    listing draws, M = 20) is re-run with the window-adjusted hazards for bins 1-5 as a
    sensitivity, and both are reported.

Check 2 (held-out cohorts). Fit on pairs announced 2010-2016 (n = 65); predict, for each pair
announced 2017-2018 (n = 44), its approval year among 2018-2022 given its announcement date.
Compare the mean held-out log score of (i) the paper's hazards (lag bins 1-5 from the
hazards; the remaining 0.294 spread evenly over bins 6-12) and (ii) the window-adjusted
estimate fitted on the training cohorts; also report observed versus predicted counts by
approval year. Reported whatever the outcome; no threshold.

Both checks concern the approval process only; they are not a validation of initiation timing.

## Amendment (2026-11-06, before either check was run)
Check 2 as written cannot be run: pairs announced 2010-2016 can only show lags above about
one year (2018 - a > 1), so bin 1 is not identified from the training cohorts and the
window-adjusted model cannot predict the held-out cohorts, whose lags are mostly in bin 1-2.
Replacement: 5-fold cross-validation over all 109 pairs (folds stratified by announcement
year, seed 0). In each fold both models are evaluated on the held-out pairs by the log
probability of the observed approval year given the announcement date and the window; the
paper's hazards are fixed (not refitted), the window-adjusted estimate is refitted on the
training folds (bins unidentified in a training fold get a floor of 1e-6 before renormalizing).
Reported: mean held-out log score per pair for both, and its paired difference with a 95%
interval over pairs. No threshold.
