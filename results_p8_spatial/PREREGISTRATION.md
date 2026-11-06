# P8-B: spatial hold-out of the learned value correction (written before running, 2026-11-06)

Question (review R3 3.3). The learned value correction was trained on all 717 units and tested on
unseen field draws of the same units. Does the correction transfer to units it was not trained on?

Folds (fixed now, by the two subdistricts before the 2018 reorganization):
  fold G = Guangming, Fenghuang, Xinhu subdistricts (370 units)
  fold M = Gongming, Yutang, Matian subdistricts (347 units)
(Xinhu was formed from parts of both former subdistricts; it is assigned to G to balance the folds.)

Training. Frozen protocol of scripts/p5_learned_spatial.py (field draws 101-120, two episodes per
behaviour rule, one-step target, lead 4, M = 20, training seeds 0-4), with the same collected
experience; for each fold the training rows are restricted to candidates whose unit lies OUTSIDE
the fold. Two models per seed: one that has never seen fold G units, one that has never seen fold M.

Cross-fitted policy. Each year the K = 3 prescreened listed units with the highest Q are initiated,
exactly as in fqi.rollout, except that a unit in fold G is scored by the model trained without
fold G and a unit in fold M by the model trained without fold M. Every correction in the schedule
is therefore an out-of-region prediction. Sanity check: with the full model in both slots the
rollout must reproduce the published learned ratio.

Evaluation. Ten test field draws 1-10 (listing draw 7) and the observed metro field (listing draws
7, 0, 1, 2, 3, 4), each against ranking on announced plans on the same draw.
Primary metric: test draws, mean over draws of (cross-fitted - ranking on announced plans), each
draw averaged over the five seeds; 95% t-interval over the ten draws. Reference: the full model,
+0.025 (95% t-interval 0.017 to 0.034).
  * Transfer supported:          mean >= +0.0125 (half the full-model gain) AND interval above 0
  * Transfer strongly supported: mean >= +0.020 AND interval above 0
  * Transfer not supported:      interval contains 0, or mean <= 0
  * Otherwise: partly supported.
Secondary (reported whatever the outcome): metro field difference (full model -0.024); runs above
the rule; correlation of the predicted correction with the true residual (EV[u,t] - F_t(u,t)) over
the prescreened candidates of the cross-fitted rollout, by fold, against the full model's 0.98 on
the test draws (Table S26).
