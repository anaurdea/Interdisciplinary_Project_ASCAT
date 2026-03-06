# ASCAT Interpolation

This repository contains an exploration for
`ASCAT_SSM_EASE2_25.nc`.

The scripts focus on:
- Dataset overview and quick sanity checks.
- Missing-data pattern analysis.
- Accessing time series, time slices, and data volumes.
- Geospatial visualization with EOMaps.
- Baseline training
-DenseNet Model training
- Standalone DenseNet map export and aggregated before/after overviews

## 1) Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Run the exploration scripts

From the project root:

```powershell
python scripts/00_overview.py
python scripts/01_missing_patterns.py
python scripts/02_access_patterns.py --lat 50 --lon 8 --date 2019-07-15 --start-date 2019-07-15 --end-date 2019-07-16 --lat-min 49 --lat-max 51 --lon-min 7 --lon-max 9
python scripts/03_eomaps_timeslice.py --date 2019-07-15 --sensor-index 2
```

Generated outputs are written to `outputs/figures` and `outputs/reports`.

## 3) Script overview

- `scripts/00_overview.py`
  - Prints dimensions, coordinates, primary variable, sensor labels, and a sample missing-value fraction.

- `scripts/01_missing_patterns.py`
  - Computes missingness by sensor and by time.
  - Saves a line plot and a sensor map for missingness.
  - Uses the first 4096 timestamps by default for quicker EDA. Use `--max-time-steps 0` for full-length analysis.
  - Auto-zooms the missing-by-time y-axis for sparse datasets. Use `--full-y-range` to force `[0, 1]`.

- `scripts/02_access_patterns.py`
  - Demonstrates three access patterns:
    - Time series at one grid point.
    - Time slice (map) on one date.
    - Data volume (subset by time and region) and summary stats.
  - Uses METOP-A + METOP-B mean as input and METOP-C as target.
  - Decimates long time series to 300 points by default for speed (`--timeseries-max-points`).
  - Downsamples map plots by stride 4 by default for faster rendering (`--map-step`).

- `scripts/03_eomaps_timeslice.py`
  - Visualizes one time slice with EOMaps for geospatial context.

- `scripts/10_build_training_index.py`
  - Builds a valid sample index (`[time, y, x]`) for supervised learning pairs.
  - Applies time filtering, spatial filtering, stride/downsampling, and finite-value checks.
  - Supports METOP-A/B inputs with METOP-C target.

- `scripts/11_dataloader_demo.py`
  - Creates a PyTorch `Dataset`/`DataLoader` from the index.
  - Performs feature engineering, optional normalization fitting/loading, and optional ancillary feature loading.

- `scripts/31_export_densenet_maps.py`
  - Loads a trained DenseNet checkpoint and exports maps for multiple dates without retraining.
  - Supports regular matplotlib outputs and optional EOMaps outputs.

- `scripts/32_aggregate_gapfill_overview.py`
  - Aggregates many timestamps into global before/after overview maps.
  - Reports coverage fractions and reconstructed-only fractions.

## 4) Notes on memory and chunking

The dataset is large. All scripts open it lazily with xarray + dask and conservative chunks.
Tune chunk sizes via CLI arguments if needed.

Example:

```powershell
python scripts/00_overview.py --time-chunk 128 --lat-chunk 120 --lon-chunk 120
```

## 5) Data preparation and PyTorch DataLoader

### Step A: Build a valid training index

This precomputes where both inputs and target are finite and stores positions as
`outputs/reports/train_index.npy`.

```powershell
python scripts/10_build_training_index.py --input-channels 0 1 --target-channel 2 --start-date 2019-04-01 --end-date 2019-12-31 --lat-min 40 --lat-max 60 --lon-min -10 --lon-max 20 --time-stride 3 --spatial-step 2 --time-batch-size 48
```

### Step B: Create and inspect a DataLoader

```powershell
python scripts/11_dataloader_demo.py --index outputs/reports/train_index.npy --input-channels 0 1 --target-channel 2 --batch-size 64 --max-batches 2 --include-input-std --include-input-range
```

### Step C (optional): Fit normalization stats from sampled training points

```powershell
python scripts/11_dataloader_demo.py --index outputs/reports/train_index.npy --fit-stats-samples 20000 --save-stats-json outputs/reports/normalization_stats.json
```

Then reuse the saved stats:

```powershell
python scripts/11_dataloader_demo.py --index outputs/reports/train_index.npy --stats-json outputs/reports/normalization_stats.json
```

Important: normalization stats must be fitted with the same feature flags used later (for example, `--include-input-std` and `--include-input-range`).



## 6) Baseline benchmarks (interpolation + PyTorch)

Run reproducible baselines with train/validation/test split, cross-validation, and held-out KPIs:

```powershell
python scripts/20_baseline_benchmarks.py --dataset Dataset/ASCAT_SSM_EASE2_25.nc --index outputs/reports/train_index.npy --seed 42 --split-mode time --train-ratio 0.7 --val-ratio 0.15 --cv-folds 3 --include-input-std --include-input-range --batch-size 64 --epochs 100 --patience 15
```

This reports:
- Interpolation baselines on held-out test: temporal linear interpolation and spatial IDW.
- Optional kriging baseline with `--run-kriging` (requires `pykrige` installed).
- PyTorch supervised baseline (`torch_mlp`) with early stopping and learning-rate scheduling.
- Cross-validation summaries (RMSE/MAE mean and std) on development folds.

Outputs:
- `outputs/reports/20_baseline_results.json`

## 7) DenseNet training and evaluation

Train a DenseNet gap-filling model with reproducible seed, train/val/test split, and cross-validation:

```powershell
python scripts/30_train_densenet_gapfill.py --dataset Dataset/ASCAT_SSM_EASE2_25.nc --index outputs/reports/train_index.npy --baseline-json outputs/reports/20_baseline_results.json --seed 42 --split-mode time --train-ratio 0.7 --val-ratio 0.15 --cv-folds 3 --epochs 80 --cv-epochs 40 --patience 12 --batch-size 64 --map-date 2019-07-15 --output-json outputs/reports/30_densenet_results.json --save-model outputs/models/30_densenet_model.pt
```

This produces:
- DenseNet held-out RMSE/MAE and CV summary in `outputs/reports/30_densenet_results.json`
- Trained checkpoint in `outputs/models/30_densenet_model.pt`
- Baseline-vs-DenseNet metric plot in `outputs/figures/30_baseline_vs_densenet_metrics.png`
- Gap-filling map visualization in `outputs/figures/30_densenet_gapfilled_map.png`

## 8) Standalone map export from a trained DenseNet

Creates more map dates for the report without rerunning training.

```powershell
python scripts/31_export_densenet_maps.py --dataset Dataset/ASCAT_SSM_EASE2_25.nc --checkpoint outputs/models/30_densenet_model.pt --index outputs/reports/train_index_full.npy --step-days 30 --max-dates 12 --save-eomaps --output-json outputs/reports/31_densenet_map_exports.json
```

outputs:
- Multi-date map figures in `outputs/figures/`
- Optional EOMaps figures when `--save-eomaps` is enabled
- Export summary JSON in `outputs/reports/31_densenet_map_exports.json`

## 9) Aggregated before/after overview maps

Summarizes reconstruction behavior across many timestamps.

Day-based selection example:

```powershell
python scripts/32_aggregate_gapfill_overview.py --dataset Dataset/ASCAT_SSM_EASE2_25.nc --checkpoint outputs/models/30_densenet_model.pt --index outputs/reports/train_index_full.npy --step-days 30 --max-dates 24 --fig-path outputs/figures/32_before_after_overview.png --summary-json outputs/reports/32_before_after_overview.json
```

Improved timestamp-based selection example (recommended for broader spatial support):

```powershell
python scripts/32_aggregate_gapfill_overview.py --dataset Dataset/ASCAT_SSM_EASE2_25.nc --checkpoint outputs/models/30_densenet_model.pt --index outputs/reports/train_index_full.npy --selection-mode timestamp --step-timestamps 1 --max-timestamps 0 --save-eomaps --eomaps-prefix 32_aggregate_improved --fig-path outputs/figures/32_before_after_overview_all_dates_improved.png --summary-json outputs/reports/32_before_after_overview_all_dates_improved.json --summary-npz outputs/reports/32_before_after_overview_all_dates_improved.npz
```

outputs:
- Aggregated overview figure in `outputs/figures/`
- Summary statistics in `outputs/reports/*.json`
- Optional aggregated arrays in `outputs/reports/*.npz`

