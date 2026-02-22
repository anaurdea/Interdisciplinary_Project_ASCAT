from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_train_module(script_path: Path):
    spec = importlib.util.spec_from_file_location('densenet_train_module', script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load module from: {script_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build aggregated before/after global overview maps from DenseNet predictions.'
    )
    parser.add_argument('--dataset', default='Dataset/ASCAT_SSM_EASE2_25.nc')
    parser.add_argument('--checkpoint', required=True, help='Path to DenseNet checkpoint (.pt).')
    parser.add_argument(
        '--index',
        default=None,
        help='Optional index (.npy) used to select available timestamps. If omitted, full timeline is used.',
    )
    parser.add_argument(
        '--selection-mode',
        choices=['day', 'timestamp'],
        default='day',
        help='Select one timestamp per day (day) or aggregate directly over timestamps (timestamp).',
    )
    parser.add_argument('--dates', nargs='*', default=None, help='Optional explicit dates (YYYY-MM-DD).')
    parser.add_argument('--step-days', type=int, default=30, help='Auto-select one date every N days.')
    parser.add_argument('--max-dates', type=int, default=24, help='Max number of auto-selected dates. Use 0 for all.')
    parser.add_argument(
        '--step-timestamps',
        type=int,
        default=1,
        help='When --selection-mode timestamp is used, keep every Nth timestamp.',
    )
    parser.add_argument(
        '--max-timestamps',
        type=int,
        default=0,
        help='When --selection-mode timestamp is used, optional cap on selected timestamps (0 = no cap).',
    )
    parser.add_argument('--start-date', default=None)
    parser.add_argument('--end-date', default=None)
    parser.add_argument('--fig-path', default='outputs/figures/32_before_after_overview.png')
    parser.add_argument('--summary-json', default='outputs/reports/32_before_after_overview.json')
    parser.add_argument('--summary-npz', default='outputs/reports/32_before_after_overview_arrays.npz')
    parser.add_argument('--save-eomaps', action='store_true', help='Also export aggregated EOMaps layers.')
    parser.add_argument('--eomaps-dir', default='outputs/figures/eomaps_aggregate')
    parser.add_argument('--eomaps-prefix', default='32_aggregate_improved')
    parser.add_argument('--time-chunk', type=int, default=256)
    parser.add_argument('--lat-chunk', type=int, default=200)
    parser.add_argument('--lon-chunk', type=int, default=200)
    parser.add_argument('--map-row-chunk', type=int, default=24)
    parser.add_argument('--map-batch-size', type=int, default=2048)
    return parser.parse_args()


def _filter_days(days: np.ndarray, start_date: str | None, end_date: str | None) -> np.ndarray:
    out = days
    if start_date:
        out = out[out >= np.datetime64(start_date, 'D')]
    if end_date:
        out = out[out <= np.datetime64(end_date, 'D')]
    return out


def _sample_days(days: np.ndarray, step_days: int, max_dates: int) -> list[str]:
    if days.size == 0:
        return []
    if max_dates == 0:
        return [str(d.astype('datetime64[D]')) for d in days[:: max(1, step_days)]]

    selected: list[np.datetime64] = [days[0]]
    for day in days[1:]:
        if (day - selected[-1]) >= np.timedelta64(max(1, step_days), 'D'):
            selected.append(day)
        if len(selected) >= max_dates:
            break
    return [str(d.astype('datetime64[D]')) for d in selected]


def _select_dates(
    cube_time_values: np.ndarray,
    index_path: str | None,
    selection_mode: str,
    explicit_dates: list[str] | None,
    step_days: int,
    max_dates: int,
    step_timestamps: int,
    max_timestamps: int,
    start_date: str | None,
    end_date: str | None,
) -> list[str]:
    if explicit_dates:
        return explicit_dates

    if index_path:
        index_arr = np.load(index_path, mmap_mode='r')
        if index_arr.ndim != 2 or index_arr.shape[1] != 3:
            raise ValueError('Index file must have shape [N,3].')
        time_indices = np.unique(index_arr[:, 0].astype(np.int64))
        timestamps = cube_time_values[time_indices]
    else:
        timestamps = np.asarray(cube_time_values)

    if selection_mode == 'timestamp':
        ts = np.unique(np.asarray(timestamps))
        if start_date:
            ts = ts[ts >= np.datetime64(start_date)]
        if end_date:
            ts = ts[ts <= np.datetime64(end_date)]
        if ts.size == 0:
            return []
        ts = ts[:: max(1, step_timestamps)]
        if max_timestamps > 0:
            ts = ts[:max_timestamps]
        return [str(np.datetime_as_string(t, unit='s')) for t in ts]

    days = np.unique(np.asarray(timestamps, dtype='datetime64[D]'))
    days = _filter_days(days, start_date=start_date, end_date=end_date)
    return _sample_days(days, step_days=step_days, max_dates=max_dates)


def nanmean_from_sum_count(sum_arr: np.ndarray, count_arr: np.ndarray) -> np.ndarray:
    out = np.full(sum_arr.shape, np.nan, dtype=np.float32)
    valid = count_arr > 0
    out[valid] = (sum_arr[valid] / count_arr[valid]).astype(np.float32)
    return out


def robust_limits(arrays: list[np.ndarray], low_q: float = 0.02, high_q: float = 0.98) -> tuple[float, float]:
    finite_vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size > 0])
    if finite_vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.quantile(finite_vals, low_q))
    vmax = float(np.quantile(finite_vals, high_q))
    if vmax <= vmin:
        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def try_save_eomaps_figure(map_obj, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for saver in (
        lambda: map_obj.savefig(out_path, dpi=180),
        lambda: map_obj.f.savefig(out_path, dpi=180),
    ):
        try:
            saver()
            return True
        except Exception:
            continue
    return False


def set_eomaps_title(map_obj, title: str) -> None:
    for setter in (
        lambda: map_obj.ax.set_title(title),
        lambda: map_obj.f.suptitle(title),
    ):
        try:
            setter()
            return
        except Exception:
            continue


def export_eomaps_layer(
    data: np.ndarray,
    lon_2d: np.ndarray | None,
    lat_2d: np.ndarray | None,
    *,
    title: str,
    label: str,
    out_path: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> bool:
    if lon_2d is None or lat_2d is None:
        return False
    try:
        from eomaps import Maps
    except ImportError:
        return False

    m = Maps(crs=4326)
    m.set_data(
        data,
        x=lon_2d,
        y=lat_2d,
        crs=4326,
    )
    plot_kwargs = {'cmap': cmap}
    if vmin is not None and vmax is not None and np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        plot_kwargs['vmin'] = float(vmin)
        plot_kwargs['vmax'] = float(vmax)
    m.plot_map(**plot_kwargs)
    m.add_feature.preset.coastline()
    m.add_colorbar(label=label)
    set_eomaps_title(m, title)
    return try_save_eomaps_figure(m, out_path)


def plot_panel(
    ax,
    data: np.ndarray,
    title: str,
    lon_2d: np.ndarray | None,
    lat_2d: np.ndarray | None,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    masked = np.ma.masked_invalid(data)
    use_geo = (
        lon_2d is not None
        and lat_2d is not None
        and lon_2d.shape == data.shape
        and lat_2d.shape == data.shape
    )
    if use_geo:
        im = ax.pcolormesh(lon_2d, lat_2d, masked, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    else:
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    train_script = root / 'scripts' / '30_train_densenet_gapfill.py'
    train_module = load_train_module(train_script)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})
    model_args = SimpleNamespace(
        input_channels=checkpoint.get('input_channels', config.get('input_channels', [0, 1])),
        growth_rate=int(config.get('growth_rate', 16)),
        block_layers=tuple(config.get('block_layers', [4, 4])),
        init_features=int(config.get('init_features', 32)),
        dropout=float(config.get('dropout', 0.1)),
        head_hidden=int(config.get('head_hidden', 64)),
    )
    model = train_module.build_model(model_args)
    model.load_state_dict(checkpoint['model_state_dict'])

    input_channels = [int(v) for v in checkpoint.get('input_channels', config.get('input_channels', [0, 1]))]
    target_channel = int(checkpoint.get('target_channel', config.get('target_channel', 2)))
    patch_size = int(checkpoint.get('patch_size', config.get('patch_size', 9)))
    target_mean = float(checkpoint['target_mean'])
    target_std = float(checkpoint['target_std'])

    cube_args = SimpleNamespace(
        dataset=args.dataset,
        variable=config.get('variable', None),
        time_chunk=args.time_chunk,
        lat_chunk=args.lat_chunk,
        lon_chunk=args.lon_chunk,
    )
    cube = train_module.open_cube_context(cube_args)
    dates = _select_dates(
        cube_time_values=cube.time_values,
        index_path=args.index,
        selection_mode=args.selection_mode,
        explicit_dates=args.dates,
        step_days=args.step_days,
        max_dates=args.max_dates,
        step_timestamps=args.step_timestamps,
        max_timestamps=args.max_timestamps,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if not dates:
        raise ValueError('No dates selected for aggregation. Check filters or provide --dates.')

    obs_sum = None
    obs_count = None
    gap_sum = None
    gap_count = None
    pred_only_count = None
    lon_2d = None
    lat_2d = None
    selected_times: list[str] = []

    for i, date in enumerate(dates, start=1):
        target_map, _, _, gap_filled, lon_out, lat_out, selected_time = train_module.predict_map(
            model,
            cube,
            input_channels=input_channels,
            target_channel=target_channel,
            patch_size=patch_size,
            target_mean=target_mean,
            target_std=target_std,
            map_date=date,
            row_chunk=args.map_row_chunk,
            infer_batch_size=args.map_batch_size,
        )
        selected_times.append(selected_time)
        if obs_sum is None:
            shape = target_map.shape
            obs_sum = np.zeros(shape, dtype=np.float64)
            obs_count = np.zeros(shape, dtype=np.int32)
            gap_sum = np.zeros(shape, dtype=np.float64)
            gap_count = np.zeros(shape, dtype=np.int32)
            pred_only_count = np.zeros(shape, dtype=np.int32)
            lon_2d = lon_out
            lat_2d = lat_out

        obs_valid = np.isfinite(target_map)
        gap_valid = np.isfinite(gap_filled)
        pred_only = (~obs_valid) & gap_valid

        obs_sum[obs_valid] += target_map[obs_valid]
        obs_count[obs_valid] += 1
        gap_sum[gap_valid] += gap_filled[gap_valid]
        gap_count[gap_valid] += 1
        pred_only_count[pred_only] += 1

        print(f'Aggregated {i}/{len(dates)}: requested={date} selected={selected_time}')

    assert obs_sum is not None
    assert obs_count is not None
    assert gap_sum is not None
    assert gap_count is not None
    assert pred_only_count is not None

    n_dates = len(dates)
    observed_mean = nanmean_from_sum_count(obs_sum, obs_count)
    gapfilled_mean = nanmean_from_sum_count(gap_sum, gap_count)
    difference = np.where(
        np.isfinite(observed_mean) & np.isfinite(gapfilled_mean),
        gapfilled_mean - observed_mean,
        np.nan,
    ).astype(np.float32)
    observed_coverage = (obs_count / float(n_dates)).astype(np.float32)
    gapfilled_coverage = (gap_count / float(n_dates)).astype(np.float32)
    reconstructed_fraction = (pred_only_count / float(n_dates)).astype(np.float32)

    mean_vmin, mean_vmax = robust_limits([observed_mean, gapfilled_mean])
    diff_abs = np.abs(difference[np.isfinite(difference)])
    diff_lim = float(np.quantile(diff_abs, 0.98)) if diff_abs.size > 0 else 1.0
    if diff_lim <= 0:
        diff_lim = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    cmap_main = plt.get_cmap('viridis').copy()
    cmap_main.set_bad(color='#d9d9d9')
    cmap_diff = plt.get_cmap('RdBu_r').copy()
    cmap_diff.set_bad(color='#d9d9d9')
    cmap_cov = plt.get_cmap('magma').copy()
    cmap_cov.set_bad(color='#d9d9d9')

    plot_panel(
        axes[0, 0],
        observed_mean,
        'Observed mean (Metop-C)',
        lon_2d,
        lat_2d,
        cmap=cmap_main,
        vmin=mean_vmin,
        vmax=mean_vmax,
    )
    plot_panel(
        axes[0, 1],
        gapfilled_mean,
        'Gap-filled mean (DenseNet)',
        lon_2d,
        lat_2d,
        cmap=cmap_main,
        vmin=mean_vmin,
        vmax=mean_vmax,
    )
    plot_panel(
        axes[0, 2],
        difference,
        'Difference (after - before)',
        lon_2d,
        lat_2d,
        cmap=cmap_diff,
        vmin=-diff_lim,
        vmax=diff_lim,
    )
    plot_panel(
        axes[1, 0],
        observed_coverage,
        'Observed coverage fraction',
        lon_2d,
        lat_2d,
        cmap=cmap_cov,
        vmin=0.0,
        vmax=1.0,
    )
    plot_panel(
        axes[1, 1],
        gapfilled_coverage,
        'Gap-filled coverage fraction',
        lon_2d,
        lat_2d,
        cmap=cmap_cov,
        vmin=0.0,
        vmax=1.0,
    )
    plot_panel(
        axes[1, 2],
        reconstructed_fraction,
        'Reconstructed-only fraction',
        lon_2d,
        lat_2d,
        cmap=cmap_cov,
        vmin=0.0,
        vmax=1.0,
    )
    fig.suptitle(
        f'DenseNet before/after aggregated overview | n_dates={n_dates}',
        fontsize=14,
    )

    fig_path = Path(args.fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=170)
    plt.close(fig)

    npz_path = Path(args.summary_npz)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        observed_mean=observed_mean,
        gapfilled_mean=gapfilled_mean,
        difference=difference,
        observed_coverage=observed_coverage,
        gapfilled_coverage=gapfilled_coverage,
        reconstructed_fraction=reconstructed_fraction,
        lon_2d=lon_2d if lon_2d is not None else np.array([], dtype=np.float32),
        lat_2d=lat_2d if lat_2d is not None else np.array([], dtype=np.float32),
        selected_times=np.asarray(selected_times, dtype='datetime64[s]'),
    )

    summary = {
        'dataset': args.dataset,
        'checkpoint': args.checkpoint,
        'selection_mode': args.selection_mode,
        'n_dates': n_dates,
        'selected_dates_requested': dates,
        'selected_timestamps_actual': selected_times,
        'artifacts': {
            'overview_figure': str(fig_path),
            'overview_arrays_npz': str(npz_path),
        },
        'stats': {
            'observed_coverage_mean': float(np.nanmean(observed_coverage)),
            'gapfilled_coverage_mean': float(np.nanmean(gapfilled_coverage)),
            'reconstructed_fraction_mean': float(np.nanmean(reconstructed_fraction)),
            'mean_difference_global': float(np.nanmean(difference)),
        },
    }
    if args.save_eomaps:
        eomaps_dir = Path(args.eomaps_dir)
        eomaps_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.eomaps_prefix
        if not prefix.endswith('_improved'):
            prefix = f'{prefix}_improved'
        eomaps_files: dict[str, str | None] = {
            'observed_mean': None,
            'gapfilled_mean': None,
            'difference': None,
            'observed_coverage': None,
            'gapfilled_coverage': None,
            'reconstructed_fraction': None,
        }
        export_plan = [
            ('observed_mean', observed_mean, 'Observed mean (Metop-C)', 'SSM', 'viridis', mean_vmin, mean_vmax),
            ('gapfilled_mean', gapfilled_mean, 'Gap-filled mean (DenseNet)', 'SSM', 'viridis', mean_vmin, mean_vmax),
            ('difference', difference, 'Difference (after - before)', 'Delta SSM', 'RdBu_r', -diff_lim, diff_lim),
            ('observed_coverage', observed_coverage, 'Observed coverage fraction', 'Coverage', 'magma', 0.0, 1.0),
            ('gapfilled_coverage', gapfilled_coverage, 'Gap-filled coverage fraction', 'Coverage', 'magma', 0.0, 1.0),
            ('reconstructed_fraction', reconstructed_fraction, 'Reconstructed-only fraction', 'Fraction', 'magma', 0.0, 1.0),
        ]
        for key, arr, title, label, cmap, vmin, vmax in export_plan:
            out_png = eomaps_dir / f'{prefix}_{key}.png'
            ok = export_eomaps_layer(
                data=arr,
                lon_2d=lon_2d,
                lat_2d=lat_2d,
                title=title,
                label=label,
                out_path=out_png,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if ok:
                eomaps_files[key] = str(out_png)
        summary['artifacts']['eomaps_layers'] = eomaps_files

    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    cube.ds.close()
    print(f'Saved overview figure: {fig_path}')
    print(f'Saved overview arrays: {npz_path}')
    print(f'Saved summary JSON: {summary_json}')


if __name__ == '__main__':
    main()
