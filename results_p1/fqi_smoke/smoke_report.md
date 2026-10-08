# p1_fqi_job.py — local smoke test (seed 0, M = 20)

Machine: local macOS, 12 cores, 18 GiB. Every job ran single-threaded
(`TPMORL_NJOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`).
The four jobs ran concurrently, one process each. Peak RSS is `ru_maxrss` of the job process.
All jobs used `--with-supervised --with-random`.

## Reproduction check
- main, seed 0: value-based ratio = 0.9095038725980372, which matches the expected value **bit for bit (PASS)**.
  The schedule (unit -> start year) is identical to `results_figures/schedules.json["value_based"]`.
  The oos job retrains the same main-field model, and it reproduces the same number.
- Persistence 0.7745224826431258 and trajectory-informed 0.8935554552319573 match table_main.csv. The reference is 6876.246054424286.
- Reference values for metro (4941.85) and noconf (7054.04) match the verified numbers. So do their greedy ratios (0.9016 / 0.9498 and 0.7762 / 0.9059).

## Timings and memory
| job | wall (s) | FQI train (s) | peak RSS (MB) |
|---|---|---|---|
| main   | 326 | 309 | 639 |
| noconf | 326 | 310 | 636 |
| metro  | 328 | 313 | 706 |
| oos (fields 1, 2, --insample-too) | 882 | 309 + 276 + 269 | 955 |

Costs per component: data collection plus double FQI fit takes about 310 s. The supervised scorer takes about 6 s, a greedy or random rollout about 0.3 s, and an out-of-sample FQI rollout about 0.5 s per field.
An oos job therefore costs about 310 s, plus about 2 s per evaluation field, plus about 275 s per field when `--insample-too` is set.

## Smoke-test ratios (seed 0; single seed, provisional)
| rule | main | noconf | metro | field1 | field2 |
|---|---|---|---|---|---|
| value-based (double FQI) | 0.9095 | 0.8911 | 0.9501 | 0.8529 (oos) / 0.9200 (in-sample) | 0.8460 (oos) / 0.8981 (in-sample) |
| supervised scorer | 0.8634 | 0.8505 | 0.9520 | 0.8959 (oos) | 0.8212 (oos) |
| trajectory-informed | 0.8936 | 0.9059 | 0.9498 | 0.9164 | 0.9000 |
| persistence | 0.7745 | 0.7762 | 0.9016 | 0.7523 | 0.7544 |
| random (rng seed 0) | 0.7165 | 0.7243 | 0.7891 | 0.7337 | 0.7222 |
References: main 6876.25, noconf 7054.04, metro 4941.85, field1 6833.01, field2 6916.57.

## Definitions in the csv
- One row per (rule, eval_field). `train_field` gives the field on which a learned rule was trained.
  For oos rows on field k, `train_field=main` means the estimators were trained on the main field and not retrained.
- In oos mode the supervised scorer is likewise trained on main and rolled out on field k without retraining.
  Greedy rules and the reference are recomputed on each field from that field's own EV.
- `random` uses `np.random.default_rng(seed)`. Persistence and trajectory are deterministic, so their rows carry seed = -1.
- `peak_rss_mb` and `job_seconds` are job-level values, repeated on every row.
- Scenario state is module-global. In oos mode, `setup(field_seed=k)` is therefore called before every rollout on field k.
  The script also asserts that field k's EV differs from main.

## Command lines (repo root)
```
export PYTHONPATH=src TPMORL_NJOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/p1_fqi_job.py --mode main   --seed S --prescreen 20 --out results_p1/fqi --with-supervised --with-random
python scripts/p1_fqi_job.py --mode noconf --seed S --prescreen 20 --out results_p1/fqi --with-supervised --with-random
python scripts/p1_fqi_job.py --mode metro  --seed S --prescreen 20 --out results_p1/fqi --with-supervised --with-random
python scripts/p1_fqi_job.py --mode oos    --seed S --prescreen 20 --eval-field-seeds 1 2 3 4 5 --out results_p1/fqi --with-supervised --with-random [--insample-too]
```
Output: `results_p1/fqi/<mode>_M<M>_s<S>.csv`. When S = 0 the job also writes `<mode>_M<M>_s0_schedule.json`.
If the main job with seed 0 and M = 20 fails the reproduction check, it exits non-zero and records `repro_check=FAIL` in the csv.
