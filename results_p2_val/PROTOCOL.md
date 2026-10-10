# Validation pilot (written before any run)

Purpose: check whether more training experience lets the value-based policy with
H = 1 history beat ranking on announced plans on scenario draws it has not seen.
No code changes; only existing CLI flags of scripts/p2_generalize.py.

Validation fields: 201-210 (disjoint from training 101-200 and test 1-10).
Test fields 1-10 are NOT used in this pilot.

Arms (all --history 1, M = 20, listing draw 7):
  A  --n-train 20  --ep-each 2   (current setting)          seeds 0, 1
  B  --n-train 50  --ep-each 2                              seeds 0, 1
  C  --n-train 20  --ep-each 5                              seeds 0, 1
  D  --n-train 100 --ep-each 2                              seed 0
Baselines (persistence, announced, trajectory, rh_full) written by arm A seed 0.

Selection rule: the arm with the highest mean ratio over fields 201-210
(averaged over its seeds). Proceed to the formal test (fields 1-10, seeds 0-4,
run once, reported as is) only if that arm's mean is at least the announced
mean on 201-210 and at least 0.005 above arm A.
