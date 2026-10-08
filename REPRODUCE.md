# Reproducing the manuscript results

*Spatio-temporal scheduling of urban renewal under planning and implementation constraints* (IJGIS submission).

All commands are run from the repository root. Scripts put `scripts/` on the import path themselves;
set `PYTHONPATH=src` so that the `tpmorl` package is found.

```bash
conda env create -f environment.yml && conda activate tpmorl
export PYTHONPATH=src TPMORL_NJOBS=1 OMP_NUM_THREADS=1   # single-threaded, deterministic
```

## Data

`data/processed/gm_dataset_v1/` contains everything the experiments read: the 100 m decision grid
(EPSG:2383), the 717 candidate renewal units and their institutional channels, the calibrated
institutional tables, and the metro station table (`metro/`). Raw land-use rasters are only needed to
rebuild this dataset (`bash scripts/fetch_raw.sh`, then `python -m tpmorl.data.build_zones`).

## Seeds and configuration

- Main configuration: horizon T = 25, annual capacity K = 3, prescreen M = 20, discount 0.95,
  scenario field seed 20260917.
- Listing draws: 7, 0, 1, 2, 3, 4 (six draws). Learned rules: training seeds 0-4.
- Transfer tests: training fields 101-120, unseen test fields 1-10, listing draw 7.

## Manuscript item -> script -> output

| Manuscript item | Command | Output |
|---|---|---|
| Table 4, Figure 5; Tables D3, D4, D5 (greedy and rolling-horizon rules) | `python scripts/p1_rolling_horizon.py --job cluster` (also `--job metro`, `--job noconf`, and one job per field variant), then `python scripts/p1_rolling_horizon.py --merge` | `results_p1/rolling/` |
| Table 4; Tables D3, D4, D5 (value-based policy, supervised scorer, random) | `python scripts/p1_fqi_job.py --mode main --seed S --prescreen M --with-supervised --with-random --listing-seeds 7 0 1 2 3 4` for S = 0-4 and M = 20, 10, 50, 0; modes `noconf`, `metro`, `variant --variant V` analogously. The full job list is in `启动实验_P1服务器.sh` | `results_p1/fqi/` |
| Consolidated tables | `python scripts/p1_build_tables.py` | `results_p1/tables/` (`main_M20.csv`, `prescreen.csv`, `noconf.csv`, `metro.csv`, `fields.csv`, `margins.csv`) |
| Announced-information greedy equals the announced rolling horizon (Section 3.1) | `python scripts/p2_announced_greedy.py` | `results_p2/announced_greedy.csv` |
| Spatial autocorrelation of timing errors (Section 5.1) | `python scripts/p2_timing_space.py` | `results_p2/timing_space_summary.csv` |
| Table D6, Figure 6b (transfer to unseen fields) | `python scripts/p2_generalize.py --seed S --with-single --baselines` for S = 0-4 | `results_p2/gen_s{S}.csv` |
| Table D7, Figure 6c (past observations in the state) | `python scripts/p2_generalize.py --seed S --history H` for S = 0-4, H = 1, 2, 3, 5, then `python scripts/p2_history_summary.py` | `results_p2/gen_s{S}_h{H}.csv`, `results_p2/history_sensitivity.csv` |
| Table C1 (single vs double estimator) | `python scripts/exp_prescreen_ablation.py --part B --seeds S --out results_det/sS` for S = 0-4, then `python scripts/build_table_main.py results_det/s*/partB_runs.csv` | `results_det/`, `results_figures/table_main.csv` |
| Figure D1, Table D1 (controlled simulation) | `python scripts/exp_fqi_local.py --stage L2 --seeds 0 1 2` | `results_fqi/fqi_synth_summary.csv` |
| Table D2 (candidate-to-capacity ratio) | `python scripts/exp_rho_sweep.py --mode inflate` and `python scripts/exp_rho_sweep.py --mode tighten --out results_rho_tighten` | `results_rho/`, `results_rho_tighten/` |
| Conformity-gate check (Section 2.3) | `python scripts/check_conformity_gate.py` | printed summary |
| Every number in the manuscript text and tables | `python scripts/audit_manuscript_numbers.py` | `84 checks, 0 not found` |

## Figures

Each figure is drawn from the committed result files by one script in `paper_ijgis/figsrc/`, written
to `paper_ijgis/figures/`:

| Figure | Script | Main inputs |
|---|---|---|
| 1 | `fig1_framework.py` | none (schematic) |
| 2 | `fig2_temporal_opportunity.py` | `results_figures/gm_field_curves.json`, `synth_opp_curves.json`, `fig2_example_units.json` |
| 3 | `fig4_study_area.py` | `data/processed/gm_dataset_v1/`, `results_figures/*.geojson` |
| 4 | `fig5_schedules.py` | `results_p1/rolling/schedules_cluster_M20.json`, `rolling_results.csv` |
| 5 | `fig5_ladder.py` | `results_p1/tables/main_M20.csv`, `results_p1/rolling/rolling_results.csv` |
| 6 | `fig6_learning.py` | `results_p1/fqi/`, `results_p1/rolling/`, `results_p2/gen_s*.csv` |
| A1 | `figA1_construction.py` | none (schematic) |
| D1 | `fig3_controlled_experiment.py` | `results_fqi/fqi_synth_summary.csv` |

Script file names follow an earlier figure numbering. Three small inputs were extracted from the
simulator by one-off code and are provided as data rather than regenerated:
`results_figures/fig2_example_units.json` (the two example units of Figure 2),
`results_figures/gm_field_curves.json` and `synth_opp_curves.json` (field curves of Figure 2), and
`results_p2/fig5_zoom_window.json` (per-unit shares in the Figure 4c window, used by the audit).

## Runtimes

See Table B1. Training the value-based policy on one field took 545-687 s per seed on the server and
1427-1475 s on twenty pooled fields locally; the greedy and rolling-horizon rules run in under a second
per schedule.
