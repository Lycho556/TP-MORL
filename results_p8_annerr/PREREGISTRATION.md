# P8-E: ranking on announced plans when announcements are wrong (written before running, 2026-11-06)

Question. In the model the announced forecast reads the commissioned infrastructure exactly four
years ahead, so announcements are never revised; Section 6.5 notes that real plans are. How much
of the value of information does ranking on announced plans keep when the announced timetable
is wrong?

Design. Rule-based rules only, M = 20. Fields: main field and scenario draws 1-10; listing draws
7, 0, 1, 2, 3, 4. For each infrastructure cluster c an error d_c (integer years) is drawn: the
realized programme runs d_c years later than announced (d_c < 0: earlier). In year t the planner's
announced forecast reads, for units whose cluster has not started commissioning by t, the
infrastructure level of year t + 4 + d_c instead of t + 4 (it believes the announced timetable);
units whose commissioning has started are read correctly. Value is always computed on the true
fields. Conditions: d_c uniform on {-e..e} (symmetric) and on {0..e} (delays only), e = 1, 2, 3;
five error draws per condition and field. e = 0 must reproduce the published ranking result.
Measure per field: share recovered = (ranking - persistence) / (trajectory-informed - persistence),
averaged over listing draws and error draws.

Criterion (fixed now): the two-thirds statement is called robust to announcement error if the mean
share over the eleven fields stays >= 0.50 under symmetric errors of +/-2 years; otherwise the text
states that the share depends on the accuracy of announcements, with the values. All conditions
are reported.
