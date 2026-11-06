# Reproducing the manuscript results

Capacity-constrained scheduling of urban renewal initiations in Guangming District, Shenzhen.

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
- Transfer tests: training fields 101-120, validation fields 201-210 (method selection only),
  unseen test fields 1-10 (used once, after selection), listing draw 7.
- Announcement lead 4 years; construction duration b = 5 years.

## Manuscript item -> script -> output

Table and figure numbers refer to the submitted manuscript (Tables 1-5, Figures 1-7, A1) and its
supplementary material (Tables S1-S26, Figure S1). `python scripts/audit_manuscript_numbers.py`
recomputes the numbers in the text and tables from the result files below and reports
`126 checks, 0 not found`.

### Main text

| Item | Command | Output |
|---|---|---|
| Table 3; Figure 5a (rule-based rules) | `python scripts/p1_rolling_horizon.py --job cluster` (also `--job metro`, `--job noconf`, and one job per field variant), then `python scripts/p1_rolling_horizon.py --merge` and `python scripts/p1_build_tables.py` | `results_p1/rolling/`, `results_p1/tables/main_M20.csv` |
| Table 3, Table 4 (learned value correction, main and metro fields) | `python scripts/p3_metro_test.py --seed S --field main` and `--field metro` (add `--baselines` for S = 0) for S = 0-4 | `results_p3_metro/` |
| Table 4, Figure 7 (pre-specified test, ten unseen field draws) | `python scripts/p3_formal_test.py --seed S` for S = 0-4 (`--baselines` once, `--leak-check` once), then `python scripts/p3_formal_summary.py`; registration in `docs/P3/registration_p3_formal.md` | `results_p3_formal/` |
| Table 5 (spatial extensions) | `python scripts/p5_spatial_ext.py`; learned correction under the same constraints `python scripts/p5_learned_spatial.py --seed S` for S = 0-4 | `results_p5_spatial/` |
| Figure 6 (timing corridors, observed metro field) | `python scripts/p3_timing_space_metro.py` | `results_p2/timing_stake_units.csv`, `timing_stake_by_distance.csv`, `start_now_by_distance.csv` |
| Proposition 1 (announced greedy equals the announced rolling horizon) | `python scripts/p2_announced_greedy.py`, `python scripts/p2_equivalence_check.py` | `results_p2/announced_greedy.csv`, `equivalence_check.csv` |
| Conformity gate (Section 2) | `python scripts/check_conformity_gate.py` | printed summary |

### Supplementary material

| Item | Command | Output |
|---|---|---|
| S1 (prescreens, no gate), S2 (observed metro field), S3 (field variants) | P1 jobs as for Table 3; learned rows of S2 from `p3_metro_test.py` | `results_p1/tables/prescreen.csv`, `noconf.csv`, `metro.csv`, `fields.csv`; `results_p3_metro/` |
| S4 (transfer of the earlier multi-step policy) | `python scripts/p2_generalize.py --seed S --with-single --baselines` for S = 0-4 | `results_p2/gen_s{S}.csv` |
| S5 (past observations in the state) | `python scripts/p2_generalize.py --seed S --history H` for S = 0-4, H = 1, 2, 3, 5, then `python scripts/p2_history_summary.py` | `results_p2/history_sensitivity.csv`; 100-draw run in `results_p2_formal_v2/` |
| S6 (information ablation, announcement lead) | `python scripts/diag_announce_lead.py`, `python scripts/diag_waiting_value.py` | `results_p2_diag/` |
| S7 (selection on validation draws 201-210) | `python scripts/p3_explore_rulefeat.py --seed S ...` (variants as listed in the table), then `python scripts/p3_explore_summary.py`; mixed-lead retraining `python scripts/p3_shift_test.py --seed S` | `results_p3_explore/`, `results_p3_shift/` |
| S8 (capacity and discount), S9 (metro premium and catchment), S10 (Moran's I) | `python scripts/p4_robustness.py --part quota`, `--part gamma`, `--part metro`, `--part spatial` | `results_p4/` |
| S11-S13 (subdistrict limits, spillovers, choice-set benchmark) | `python scripts/p5_spatial_ext.py` | `results_p5_spatial/E1_*.csv`, `E2_*`, `E3_choice_set_benchmark.csv` |
| S14-S15 (spatial extensions across fields) | `python scripts/p5_spatial_fields.py --field main`, `--field metro`, `--field test` | `results_p5_spatial/fields/` |
| Section S8 text | `python scripts/p5_make_supp.py` | `paper_ijgis/supp_S8.tex` |
| S16 (announced plans 2010-2018) | `python scripts/p5_external_check.py` | `results_p5_spatial/external/` |
| S17 (without the agricultural-conversion channel) | `python scripts/p5_no_p5.py` | `results_p5_spatial/external/no_P5_rules.csv` |
| S18 (approval timing against the record; pre-registered in `results_p8_approval/PREREGISTRATION.md`) | `python scripts/p8_approval_check.py` | `results_p8_approval/` |
| S21 (unit delineation across eleven fields; pre-registered in `results_p8_delineation/PREREGISTRATION.md`) | `python scripts/p8_delineation_fields.py --target T` for T = 8, 12, 18, then `python scripts/p8_delineation_summary.py` | `results_p8_delineation/` |
| S19 (construction duration b = 3, 5, 7) | `python scripts/p5_build_years.py` | `results_p5_spatial/build_years.csv` |
| S20 (unit delineation, 8 / 12 / 18 cells) | `python scripts/p5_unit_delineation.py` (rebuilds the units in `data/processed/gm_dataset_v1_units{8,18}/`, not committed) | `results_p5_spatial/unit_delineation.csv` |
| S22 (runtimes) | timings of the commands above | - |
| S24, S25, Figure S1 (controlled simulation) | `python scripts/exp_fqi_local.py --stage L2 --seeds 0 1 2` | `results_fqi/` |
| S26 (distribution-shift tests; pre-registered in `results_p6_shift/PREREGISTRATION.md`) | A1: `python scripts/p6_shift_diag.py --seed S` (models saved by `p5_learned_spatial.py`); A2: `python scripts/p6_early_infra.py --seed S`, then `python scripts/p6_shift_diag.py --seed S --models results_p6_shift/models_early --tag a1early` and `python scripts/p6_orig_on_early.py S`, for S = 0-4; summaries `python scripts/p6_summary.py` | `results_p6_shift/` |

Fitted models (`results_p5_spatial/models/`, `results_p6_shift/models_early/`, about 3 GB each)
are not committed; the scripts above recreate them.

## Figures

Each figure is drawn from the committed result files by one script in `paper_ijgis/figsrc/`, written
to `paper_ijgis/figures/`:

| Figure | Script | Main inputs |
|---|---|---|
| 1 | `fig1_framework.py` | none (schematic) |
| 2 | `fig2_temporal_opportunity.py` | `results_figures/gm_field_curves.json`, `synth_opp_curves.json`, `fig2_example_units.json` |
| 3 | `fig4_study_area.py` | `data/processed/gm_dataset_v1/`, `results_figures/*.geojson` |
| 4 | `fig5_schedules.py` | `results_p1/rolling/schedules_cluster_M20.json`, `rolling_results.csv`, `results_p3_metro/rl_schedule_main_ls7_s0.json` (from `p3_save_schedule.py`) |
| 5 | `fig5_ladder.py` | `results_p1/tables/main_M20.csv`, `results_p1/rolling/rolling_results.csv`, `results_figures/` |
| 6 | `fig7_timing_corridors.py` | `results_p2/timing_stake_units.csv` |
| 7 | `fig6_learning.py` | `results_p3_formal/test_s*.csv` |
| A1 | `figA1_construction.py` | none (schematic) |
| S1 | `fig3_controlled_experiment.py` | `results_fqi/fqi_synth_summary.csv` |

Script file names follow an earlier figure numbering. Three small inputs were extracted from the
simulator by one-off code and are provided as data rather than regenerated:
`results_figures/fig2_example_units.json` (the two example units of Figure 2),
`results_figures/gm_field_curves.json` and `synth_opp_curves.json` (field curves of Figure 2), and
`results_p2/fig5_zoom_window.json` (per-unit shares in the Figure 4c window, used by the audit).

## Runtimes

See Supplementary Table S22. The rule-based rules run in under a second per schedule; training the
learned value correction takes several minutes to about half an hour per seed, single-threaded.
