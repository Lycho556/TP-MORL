# Pre-registration: why does the learned value correction fall below ranking on the metro field?
Written before running A1/A2 (2026-11-02). Nothing below is changed after results are seen.

Claim under test (manuscript Section 5.3): the shortfall on the observed metro field
(learned 0.916 vs ranking on announced plans 0.941, difference -0.024) is a distribution shift:
in the training scenarios infrastructure keeps being commissioned after the four-year
announcement window, whereas all 28 metro stations open in model years 1.6-7.5, so the
correction adds value the metro field does not contain.

A1 (diagnostic, saved models s0-s4 of results_p5_spatial/models, no retraining).
For every prescreened candidate in every year of the policy's own rollout, record
predicted correction dQ = Q - F_now and true residual R = EV[u,t] - F_now.
Prediction if the claim holds: on the metro field mean dQ over the policy's selections is
clearly positive while mean R is near zero or smaller (over-correction dQ - R > 0), and the
over-correction is larger on the metro field than on the ten test draws.

A2 (single-factor retraining). Identical frozen protocol (draws 101-120, two episodes per
behaviour rule, one-step target, lead 4, M = 20, seeds 0-4) except that infrastructure
commissioning onsets are mapped to model years [-1, 3] (peaks about years 1-8, as the metro
openings), all other fields unchanged.
Primary metric: metro field, mean over 5 seeds x 6 listing draws of
(learned - ranking on announced plans).
  * Supported:            difference >= -0.012 (at least half of the -0.024 gap closed)
  * Strongly supported:   difference >= 0
  * Not supported:        difference < -0.012
Secondary (reported whatever the outcome): the same difference on the ten original test
draws (expected to fall, since the retrained correction no longer matches them) and on ten
early-infrastructure test draws (field draws 1-10 with the same onset mapping).

## Outcome (appended after running; criteria above unchanged)
A1 (original models, 5 seeds): correlation of predicted correction with true residual
0.98 (test draws), 0.98 (main field), 0.19 (metro). Units swapped in by the correction:
predicted +28.7 / true +28.5 (test), +31.4 / -0.4 (metro). Over-correction on metro is
concentrated in model years 0-8 (station openings) and vanishes after year 9.
A2 (early-infrastructure training, 5 seeds x 6 listing draws): metro learned - ranking
= -0.0053 (95% t-interval over listing draws -0.0087 to -0.0019; 0 of 30 runs positive),
against -0.024 with the original training: 78% of the gap closed.
Verdict under the pre-registered criteria: SUPPORTED (>= -0.012), not strongly supported (< 0).
Secondary: original test draws +0.0068 (was +0.025); early test draws +0.0024.
