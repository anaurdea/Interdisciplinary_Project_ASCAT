from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from ascat_utils import (
    detect_sensor_dim,
    detect_spatial_dims,
    ensure_dir,
    infer_primary_variable,
    map_plot_axes,
    open_dataset,
    sensor_labels,
    with_time_coordinate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Estimate how much of the Earth is observed in ASCAT data.'
    )
    parser.add_argument('--dataset', default='Dataset/ASCAT_SSM_EASE2_25.nc')
    parser.add_argument('--variable', default=None)
    parser.add_argument('--target-channel', type=int, default=2)
    parser.add_argument('--start-date', default=None, help='Optional start date (YYYY-MM-DD).')
    parser.add_argument('--end-date', default=None, help='Optional end date (YYYY-MM-DD).')
    parser.add_argument(
        '--max-time-steps',
        type=int,
        default=4096,
        help='Limit to first N selected timestamps for faster exploratory runs. Use 0 for full period.',
    )
    parser.add_argument('--time-chunk', type=int, default=256)
    parser.add_argument('--lat-chunk', type=int, default=200)
    parser.add_argument('--lon-chunk', type=int, default=200)
    parser.add_argument('--fig-dir', default='outputs/figures')
    parser.add_argument('--report-dir', default='outputs/reports')
    return parser.parse_args()


def subset_time(
    da,
    time_dim: str,
    *,
    start_date: str | None,
    end_date: str | None,
    max_time_steps: int,
):
    tvals = np.asarray(da[time_dim].values)
    mask = np.ones(tvals.shape[0], dtype=bool)
    if start_date:
        mask &= tvals >= np.datetime64(start_date)
    if end_date:
        mask &= tvals <= np.datetime64(end_date)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError('No timestamps matched the requested time filter.')
    if max_time_steps > 0:
        idx = idx[:max_time_steps]
    return da.isel({time_dim: idx}), idx


def plot_observed_fraction_by_time(observed_by_time, out_path: Path, title: str) -> None:
    time_dim = observed_by_time.dims[-1] if observed_by_time.ndim > 1 else observed_by_time.dims[0]
    x = observed_by_time[time_dim].values
    y = observed_by_time.values

    fig, ax = plt.subplots(figsize=(12, 4))
    if observed_by_time.ndim == 2:
        sensor_dim = [d for d in observed_by_time.dims if d != time_dim][0]
        for i in range(observed_by_time.sizes[sensor_dim]):
            ax.plot(x, observed_by_time.isel({sensor_dim: i}).values, linewidth=0.9, alpha=0.8)
    else:
        ax.plot(x, y, linewidth=1.2)

    if np.issubdtype(np.asarray(x).dtype, np.datetime64):
        locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    ax.set_title(title)
    ax.set_ylabel('Observed fraction')
    ax.set_xlabel('Time')
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_map(data, out_path: Path, title: str, *, cmap: str, vmin: float, vmax: float, cbar_label: str) -> None:
    x_axis, y_axis = map_plot_axes(data)
    kwargs = {'cmap': cmap, 'vmin': vmin, 'vmax': vmax, 'cbar_kwargs': {'label': cbar_label}}
    if x_axis and y_axis:
        kwargs['x'] = x_axis
        kwargs['y'] = y_axis

    fig, ax = plt.subplots(figsize=(10, 5))
    data.astype('float32').plot(ax=ax, **kwargs)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_map_by_sensor(data, labels: list[str], out_path: Path, title: str) -> None:
    sensor_dim = detect_sensor_dim(data)
    if sensor_dim is None:
        return
    x_axis, y_axis = map_plot_axes(data)
    kwargs = {'cmap': 'magma', 'vmin': 0.0, 'vmax': 1.0}
    if x_axis and y_axis:
        kwargs.update({'x': x_axis, 'y': y_axis})

    n = data.sizes[sensor_dim]
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), constrained_layout=True, squeeze=False)
    mappable = None
    for i in range(n):
        ax = axes[0, i]
        panel = data.isel({sensor_dim: i})
        mappable = panel.astype('float32').plot(ax=ax, add_colorbar=False, **kwargs)
        label = labels[i] if i < len(labels) else f'sensor {i}'
        ax.set_title(label)
    if mappable is not None:
        fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02, label='Observed fraction')
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def finite_stats(arr: np.ndarray) -> dict[str, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {'mean': float('nan'), 'p10': float('nan'), 'p50': float('nan'), 'p90': float('nan')}
    return {
        'mean': float(np.mean(finite)),
        'p10': float(np.quantile(finite, 0.10)),
        'p50': float(np.quantile(finite, 0.50)),
        'p90': float(np.quantile(finite, 0.90)),
    }


def main() -> None:
    args = parse_args()
    fig_dir = ensure_dir(args.fig_dir)
    report_dir = ensure_dir(args.report_dir)

    ds = open_dataset(
        args.dataset,
        time_chunk=args.time_chunk,
        lat_chunk=args.lat_chunk,
        lon_chunk=args.lon_chunk,
    )
    var_name = infer_primary_variable(ds, args.variable)
    da = ds[var_name]
    da, time_dim, _ = with_time_coordinate(da, ds)
    if time_dim is None:
        raise ValueError('Could not detect a time-like dimension.')
    y_dim, x_dim = detect_spatial_dims(da)
    if y_dim is None or x_dim is None:
        raise ValueError('Could not detect spatial dimensions.')

    da, selected_idx = subset_time(
        da,
        time_dim,
        start_date=args.start_date,
        end_date=args.end_date,
        max_time_steps=args.max_time_steps,
    )
    sensor_dim = detect_sensor_dim(da)
    labels = sensor_labels(ds)

    if sensor_dim is not None:
        if args.target_channel < 0 or args.target_channel >= da.sizes[sensor_dim]:
            raise ValueError(
                f'--target-channel {args.target_channel} is out of range [0, {da.sizes[sensor_dim]-1}].'
            )
        target_finite = (~da.isel({sensor_dim: args.target_channel}).isnull())
        combined_finite = (~da.isnull()).any(dim=sensor_dim)
        per_sensor_finite = (~da.isnull())
    else:
        target_finite = (~da.isnull())
        combined_finite = target_finite
        per_sensor_finite = None

    reduce_spatial = [y_dim, x_dim]
    target_by_time = target_finite.mean(dim=reduce_spatial).compute()
    target_freq_map = target_finite.mean(dim=time_dim).compute()
    target_ever_map = target_finite.any(dim=time_dim).astype('float32').compute()

    combined_by_time = combined_finite.mean(dim=reduce_spatial).compute()
    combined_freq_map = combined_finite.mean(dim=time_dim).compute()
    combined_ever_map = combined_finite.any(dim=time_dim).astype('float32').compute()

    plot_observed_fraction_by_time(
        target_by_time,
        fig_dir / '04_observed_fraction_by_time_target.png',
        title='Observed fraction over time (target channel)',
    )
    plot_observed_fraction_by_time(
        combined_by_time,
        fig_dir / '04_observed_fraction_by_time_combined.png',
        title='Observed fraction over time (combined sensors)',
    )

    plot_map(
        target_freq_map,
        fig_dir / '04_observation_frequency_map_target.png',
        title='Observation frequency map (target channel)',
        cmap='magma',
        vmin=0.0,
        vmax=1.0,
        cbar_label='Fraction of timestamps observed',
    )
    plot_map(
        combined_freq_map,
        fig_dir / '04_observation_frequency_map_combined.png',
        title='Observation frequency map (combined sensors)',
        cmap='magma',
        vmin=0.0,
        vmax=1.0,
        cbar_label='Fraction of timestamps observed',
    )
    plot_map(
        combined_ever_map,
        fig_dir / '04_observed_ever_map_combined.png',
        title='Ever-observed map over selected period (combined sensors)',
        cmap='Greens',
        vmin=0.0,
        vmax=1.0,
        cbar_label='1 = observed at least once',
    )

    summary: dict[str, object] = {
        'dataset': str(args.dataset),
        'variable': var_name,
        'target_channel': int(args.target_channel),
        'n_selected_timestamps': int(len(selected_idx)),
        'selected_start': str(np.asarray(da[time_dim].values).min()),
        'selected_end': str(np.asarray(da[time_dim].values).max()),
        'grid_shape': {y_dim: int(da.sizes[y_dim]), x_dim: int(da.sizes[x_dim])},
        'target': {
            'ever_observed_grid_fraction': float(target_ever_map.mean().compute().item()),
            'observed_fraction_by_time_stats': finite_stats(np.asarray(target_by_time.values, dtype=np.float32)),
        },
        'combined_sensors': {
            'ever_observed_grid_fraction': float(combined_ever_map.mean().compute().item()),
            'observed_fraction_by_time_stats': finite_stats(np.asarray(combined_by_time.values, dtype=np.float32)),
        },
        'figures': {
            'target_by_time': str(fig_dir / '04_observed_fraction_by_time_target.png'),
            'combined_by_time': str(fig_dir / '04_observed_fraction_by_time_combined.png'),
            'target_frequency_map': str(fig_dir / '04_observation_frequency_map_target.png'),
            'combined_frequency_map': str(fig_dir / '04_observation_frequency_map_combined.png'),
            'combined_ever_map': str(fig_dir / '04_observed_ever_map_combined.png'),
        },
    }

    if per_sensor_finite is not None and sensor_dim is not None:
        sensor_by_time = per_sensor_finite.mean(dim=reduce_spatial).compute()
        sensor_ever = per_sensor_finite.any(dim=time_dim).astype('float32').compute()
        plot_observed_fraction_by_time(
            sensor_by_time,
            fig_dir / '04_observed_fraction_by_time_by_sensor.png',
            title='Observed fraction over time (by sensor)',
        )
        plot_map_by_sensor(
            per_sensor_finite.mean(dim=time_dim).compute(),
            labels,
            fig_dir / '04_observation_frequency_map_by_sensor.png',
            title='Observation frequency map by sensor',
        )
        plot_map_by_sensor(
            sensor_ever,
            labels,
            fig_dir / '04_observed_ever_map_by_sensor.png',
            title='Ever-observed map by sensor',
        )

        per_sensor_summary = []
        for i in range(sensor_ever.sizes[sensor_dim]):
            label = labels[i] if i < len(labels) else str(i)
            per_sensor_summary.append(
                {
                    'sensor_index': int(i),
                    'sensor_label': label,
                    'ever_observed_grid_fraction': float(sensor_ever.isel({sensor_dim: i}).mean().compute().item()),
                    'observed_fraction_by_time_stats': finite_stats(
                        np.asarray(sensor_by_time.isel({sensor_dim: i}).values, dtype=np.float32)
                    ),
                }
            )
        summary['per_sensor'] = per_sensor_summary
        summary['figures']['by_sensor_time'] = str(fig_dir / '04_observed_fraction_by_time_by_sensor.png')
        summary['figures']['by_sensor_frequency_map'] = str(fig_dir / '04_observation_frequency_map_by_sensor.png')
        summary['figures']['by_sensor_ever_map'] = str(fig_dir / '04_observed_ever_map_by_sensor.png')

    report_json = report_dir / '04_observation_coverage_summary.json'
    report_txt = report_dir / '04_observation_coverage_summary.txt'
    report_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines = [
        'Observation coverage summary',
        '',
        f'Dataset: {summary["dataset"]}',
        f'Variable: {summary["variable"]}',
        f'Selected timestamps: {summary["n_selected_timestamps"]}',
        f'Time range: {summary["selected_start"]} to {summary["selected_end"]}',
        '',
        f'Target ever-observed grid fraction: {summary["target"]["ever_observed_grid_fraction"]:.6f}',
        f'Combined-sensors ever-observed grid fraction: {summary["combined_sensors"]["ever_observed_grid_fraction"]:.6f}',
        '',
        'Main figures:',
        f'  - {summary["figures"]["target_by_time"]}',
        f'  - {summary["figures"]["combined_by_time"]}',
        f'  - {summary["figures"]["target_frequency_map"]}',
        f'  - {summary["figures"]["combined_frequency_map"]}',
        f'  - {summary["figures"]["combined_ever_map"]}',
        '',
        f'JSON summary: {report_json}',
    ]
    report_txt.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))

    ds.close()


if __name__ == '__main__':
    main()
