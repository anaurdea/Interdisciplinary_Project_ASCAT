from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    parser = argparse.ArgumentParser(description='Analyze missing-data patterns for ASCAT SSM.')
    parser.add_argument('--dataset', default='Dataset/ASCAT_SSM_EASE2_25.nc', help='Path to NetCDF dataset')
    parser.add_argument('--variable', default=None, help='Optional data variable name')
    parser.add_argument('--time-chunk', type=int, default=256)
    parser.add_argument('--lat-chunk', type=int, default=200)
    parser.add_argument('--lon-chunk', type=int, default=200)
    parser.add_argument(
        '--max-time-steps',
        type=int,
        default=4096,
        help='Limit analysis to first N time steps for faster exploratory runs. Use 0 for full series.',
    )
    parser.add_argument(
        '--full-y-range',
        action='store_true',
        help='Use fixed y-axis range [0, 1] instead of automatic zoom.',
    )
    parser.add_argument('--fig-dir', default='outputs/figures')
    parser.add_argument('--report-dir', default='outputs/reports')
    return parser.parse_args()


def plot_missing_by_time(missing_by_time, out_path: Path, full_y_range: bool = False) -> None:
    time_dim = missing_by_time.dims[0]
    x_values = missing_by_time[time_dim].values
    y_values = missing_by_time.values

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_values, y_values, linewidth=1.2)
    ax.set_title('Missing fraction over time')
    ax.set_ylabel('Missing fraction')
    ax.set_xlabel('Time')

    if np.issubdtype(np.asarray(x_values).dtype, np.datetime64):
        locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    if full_y_range:
        ax.set_ylim(0.0, 1.0)
    else:
        finite = np.asarray(y_values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size > 0:
            y_min = float(finite.min())
            y_max = float(finite.max())
            spread = y_max - y_min
            pad = max(0.002, spread * 0.15)
            if spread == 0:
                pad = max(0.005, y_max * 0.02)
            lower = max(0.0, y_min - pad)
            upper = min(1.0, y_max + pad)
            if upper - lower < 0.01:
                mid = 0.5 * (upper + lower)
                lower = max(0.0, mid - 0.005)
                upper = min(1.0, mid + 0.005)
            ax.set_ylim(lower, upper)

    ax.grid(axis='y', alpha=0.25, linestyle='--')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_missing_maps(missing_map, labels: list[str], out_path: Path) -> None:
    sensor_dim = detect_sensor_dim(missing_map)
    x_axis, y_axis = map_plot_axes(missing_map)
    plot_kwargs = {'cmap': 'magma'}
    if x_axis and y_axis:
        plot_kwargs.update({'x': x_axis, 'y': y_axis})

    if sensor_dim is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        missing_map.astype('float32').plot(ax=ax, **plot_kwargs)
        ax.set_title('Missing fraction map (across time)')
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return

    sensor_count = missing_map.sizes[sensor_dim]
    fig, axes = plt.subplots(1, sensor_count, figsize=(6 * sensor_count, 5), squeeze=False)

    for idx in range(sensor_count):
        ax = axes[0, idx]
        panel = missing_map.isel({sensor_dim: idx})
        panel.astype('float32').plot(
            ax=ax,
            add_colorbar=(idx == sensor_count - 1),
            **plot_kwargs,
        )
        title = labels[idx] if idx < len(labels) else f'sensor {idx}'
        ax.set_title(title)

    fig.suptitle('Missing fraction maps by sensor (mean over time)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    ds = open_dataset(
        args.dataset,
        time_chunk=args.time_chunk,
        lat_chunk=args.lat_chunk,
        lon_chunk=args.lon_chunk,
    )
    var_name = infer_primary_variable(ds, args.variable)
    da = ds[var_name]
    da_with_time, time_dim, _ = with_time_coordinate(da, ds)
    if time_dim is None:
        raise ValueError('Could not find a time-like dimension in the selected variable.')

    sensor_dim = detect_sensor_dim(da_with_time)
    y_dim, x_dim = detect_spatial_dims(da_with_time)
    is_missing = da_with_time.isnull().astype('float32')
    if args.max_time_steps > 0:
        is_missing = is_missing.isel({time_dim: slice(0, args.max_time_steps)})

    fig_dir = ensure_dir(args.fig_dir)
    report_dir = ensure_dir(args.report_dir)

    labels = sensor_labels(ds)

    time_reduce_dims = [dim for dim in is_missing.dims if dim != time_dim]
    missing_by_time = is_missing.mean(dim=time_reduce_dims).compute()
    plot_missing_by_time(
        missing_by_time,
        fig_dir / '01_missing_by_time.png',
        full_y_range=args.full_y_range,
    )

    missing_map = is_missing
    if time_dim in missing_map.dims:
        missing_map = missing_map.mean(dim=time_dim)

    keep_dims = [dim for dim in (sensor_dim, y_dim, x_dim) if dim is not None]
    for dim in tuple(missing_map.dims):
        if dim not in keep_dims:
            missing_map = missing_map.mean(dim=dim)
    missing_map = missing_map.compute()
    plot_missing_maps(missing_map, labels, fig_dir / '01_missing_map_by_sensor.png')

    report_lines = []
    report_lines.append('Missing data summary')
    report_lines.append('')
    report_lines.append(f'Variable: {var_name}')
    if args.max_time_steps > 0:
        report_lines.append(f'Analyzed first {args.max_time_steps} time steps')
    else:
        report_lines.append('Analyzed full time series')
    report_lines.append(f'Overall missing fraction: {float(is_missing.mean().compute()):.6f}')

    if sensor_dim is not None:
        sensor_reduce_dims = [dim for dim in is_missing.dims if dim != sensor_dim]
        missing_by_sensor = is_missing.mean(dim=sensor_reduce_dims).compute()
        csv_path = report_dir / '01_missing_by_sensor.csv'
        missing_by_sensor.to_pandas().to_csv(csv_path, header=['missing_fraction'])
        report_lines.append(f'Missing by sensor CSV: {csv_path}')

        for idx, value in enumerate(missing_by_sensor.values):
            label = labels[idx] if idx < len(labels) else str(idx)
            report_lines.append(f'  - {label}: {float(value):.6f}')

    report_lines.append(f'Figure: {fig_dir / "01_missing_by_time.png"}')
    report_lines.append(f'Figure: {fig_dir / "01_missing_map_by_sensor.png"}')

    report_path = report_dir / '01_missing_summary.txt'
    report_path.write_text('\n'.join(report_lines), encoding='utf-8')

    print('\n'.join(report_lines))


if __name__ == '__main__':
    main()

