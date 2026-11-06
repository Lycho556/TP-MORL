# P8-C: unit delineation across field draws (written before running, 2026-11-06)

Question. Supplementary Table S19 reports, for one field draw (the main field), that with
18-cell units the value of coordination (0.068) exceeds the value of information (0.041),
the reverse of the 12-cell inventory. Is this reversal a property of the coarser delineation
or of that one field draw?

Design. Targets 8, 12 and 18 cells per unit (inventories as in p5_unit_delineation.py).
Field draws: the main field and scenario draws 1-10 (11 fields). For each field, listing
draws 7, 0, 1, 2, 3, 4, M = 20. Rule-based rules only (persistence greedy, ranking on
announced plans, its rolling horizon, trajectory-informed greedy, full-information rolling
horizon); no learning, so draws 1-10 are not used to select or tune any method.
Per field: info = trajectory - persistence, coord = rh_full - trajectory, averaged over the
six listing draws.

Criteria for the 18-cell inventory (fixed now):
  * Robust reversal: coord > info in at least 9 of 11 fields AND the 95% t-interval of the
    mean (coord - info) over the 11 fields lies above 0.
  * Not robust: coord > info in at most 5 of 11 fields, OR the interval contains 0.
  * Anything else: partly robust (report the count and interval).
The same quantities are reported for 8 and 12 cells. Proposition 1 (|rh_announced -
announced|) is reported for every run.

Consequence for the text (fixed now): robust -> keep the sentence and state "in 11 field
draws"; not robust or partly robust -> state that with coarser units the balance between
information and coordination depends on the field draw, with the count and interval.
