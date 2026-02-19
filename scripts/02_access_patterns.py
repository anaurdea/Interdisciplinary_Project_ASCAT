from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ascat_utils import (
    detect_sensor_dim,
    detect_spatial_dims,
    ensure_dir,
    infer_primary_variable,
    map_plot_axes,
    nearest_grid_index,
    open_dataset,
    select_nearest_time_slice,
    sensor_labels,
    subset_by_bbox,
    validate_sensor_indices,
    with_time_coordinate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Demonstrate time series, time slice, and volume access for ASCAT SSM.'
    )
    parser.add_argument('--dataset', default='Dataset/ASCAT_SSM_EASE2_25.nc', help='Path to NetCDF dataset')
    parser.add_argument('--variable', default=None, help='Optional data variable name')
    parser.add_argument('--lat', type=float, default=50.0, help='Latitude for time series point')
    parser.add_argument('--lon', type=float, default=8.0, help='Longitude for time series point')
    parser.add_argument('--date', default='2019-07-15', help='Date for time-slice map (YYYY-MM-DD)')
    parser.add_argument('--start-date', default='2019-01-01', help='Subset start date')
    parser.add_argument('--end-date', default='2019-12-31', help='Subset end date')
    parser.add_argument('--lat-min', type=float, default=40.0)
    parser.add_argument('--lat-max', type=float, default=60.0)
    parser.add_argument('--lon-min', type=float, default=-10.0)
    parser.add_argument('--lon-max', type=float, default=20.0)
    parser.add_argument('--input-sensors', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--target-sensor', type=int, default=2)
    parser.add_argument('--time-chunk', type=int, default=256)
    parser.add_argument('--lat-chunk', type=int, default=200)
    parser.add_argument('--lon-chunk', type=int, default=200)
    parser.add_argument(
        '--timeseries-max-points',
        type=int,
        default=300,
        help='Max points to draw in timeseries plots (decimates if longer). Use 0 for all points.',
    )
    parser.add_argument(
        '--map-step',
        type=int,
        default=4,
        help='Downsample map plot grid by this stride for faster rendering. Use 1 for full resolution.',
    )
    parser.add_argument('--fig-dir', default='outputs/figures')
    parser.add_argument('--report-dir', default='outputs/reports')
    return parser.parse_args()


def summarize_volume(da, label: str) -> dict:
    return {
        'label': label,
        'shape': {dim: int(size) for dim, size in da.sizes.items()},
        'mean': float(da.mean().compute()),
        'std': float(da.std().compute()),
        'missing_fraction': float(da.isnull().mean().compute()),
    }


def decimate_timeseries(da, time_dim: str, max_points: int):
    if max_points <= 0:
        return da
    n = int(da.sizes.get(time_dim, 0))
    if n <= max_points:
        return da
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    idx = np.unique(idx)
    return da.isel({time_dim: idx})


def decimate_map_grid(da, step: int):
    if step <= 1:
        return da
    y_dim, x_dim = detect_spatial_dims(da)
    if y_dim is None or x_dim is None:
        return da
    return da.isel({y_dim: slice(None, None, step), x_dim: slice(None, None, step)})


def subset_by_time_mask(da, time_dim: str, start_date: str, end_date: str):
    start = np.datetime64(start_date)
    end = np.datetime64(end_date)
    lo, hi = (start, end) if start <= end else (end, start)
    time_values = np.asarray(da[time_dim].values)
    if time_values.ndim != 1 or time_values.size == 0:
        raise ValueError('Time coordinate is empty or not one-dimensional.')

    indices = np.where((time_values >= lo) & (time_values <= hi))[0]
    if indices.size == 0:
        raise ValueError(
            f'No timestamps found between {lo} and {hi}. '
            'Try a broader range or check dataset time coverage.'
        )
    return da.isel({time_dim: indices})


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
    da, time_dim, _ = with_time_coordinate(da, ds)
    if time_dim is None:
        raise ValueError('Could not find a time-like dimension in the selected variable.')

    sensor_dim = detect_sensor_dim(da)
    if sensor_dim is None:
        raise ValueError('Expected a "sensor" dimension for input/target extraction.')

    input_sensor_indices = validate_sensor_indices(da, args.input_sensors)
    target_sensor_index = validate_sensor_indices(da, [args.target_sensor])[0]

    labels = sensor_labels(ds)
    input_label = '+'.join(
        labels[i] if i < len(labels) else f'sensor_{i}' for i in input_sensor_indices
    )
    target_label = labels[target_sensor_index] if target_sensor_index < len(labels) else f'sensor_{target_sensor_index}'

    fig_dir = ensure_dir(args.fig_dir)
    report_dir = ensure_dir(args.report_dir)

    # 1) Time series at a selected location.
    point_indexers = nearest_grid_index(da, args.lat, args.lon)
    point = da.isel(point_indexers)
    input_point = point.isel({sensor_dim: input_sensor_indices}).mean(dim=sensor_dim)
    target_point = point.isel({sensor_dim: target_sensor_index})
    input_point = decimate_timeseries(input_point, time_dim=time_dim, max_points=args.timeseries_max_points)
    target_point = decimate_timeseries(target_point, time_dim=time_dim, max_points=args.timeseries_max_points)

    fig_ts, ax_ts = plt.subplots(figsize=(12, 4))
    input_point.plot(ax=ax_ts, label=f'Input mean ({input_label})', linewidth=1.5)
    target_point.plot(ax=ax_ts, label=f'Target ({target_label})', linewidth=1.2, alpha=0.9)
    ax_ts.set_title(f'Time series at nearest grid point to lat={args.lat}, lon={args.lon}')
    ax_ts.set_ylabel(var_name)
    ax_ts.set_xlabel('Time')
    ax_ts.legend()
    fig_ts.tight_layout()
    ts_path = fig_dir / '02_timeseries_point.png'
    fig_ts.savefig(ts_path, dpi=160)
    plt.close(fig_ts)

    # 2) Time slice map.
    input_for_map = da.isel({sensor_dim: input_sensor_indices}).mean(dim=sensor_dim)
    target_for_map = da.isel({sensor_dim: target_sensor_index})
    input_slice, selected_time = select_nearest_time_slice(input_for_map, ds, args.date)
    target_slice, _ = select_nearest_time_slice(target_for_map, ds, args.date)
    input_slice = decimate_map_grid(input_slice, step=args.map_step)
    target_slice = decimate_map_grid(target_slice, step=args.map_step)
    x_axis, y_axis = map_plot_axes(input_slice)
    plot_kwargs = {'cmap': 'viridis'}
    if x_axis and y_axis:
        plot_kwargs.update({'x': x_axis, 'y': y_axis})

    fig_slice, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    input_slice.plot(ax=axes[0], **plot_kwargs)
    axes[0].set_title(f'Input mean ({input_label}) on {selected_time}')
    target_slice.plot(ax=axes[1], **plot_kwargs)
    axes[1].set_title(f'Target ({target_label}) on {selected_time}')
    slice_path = fig_dir / '02_timeslice_maps.png'
    fig_slice.savefig(slice_path, dpi=160)
    plt.close(fig_slice)

    # 3) Data volume subset + stats.
    subset = subset_by_time_mask(
        da,
        time_dim=time_dim,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    subset = subset_by_bbox(
        subset,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
    )

    input_volume = subset.isel({sensor_dim: input_sensor_indices}).mean(dim=sensor_dim)
    target_volume = subset.isel({sensor_dim: target_sensor_index})

    summary = {
        'variable': var_name,
        'input_sensor_indices': input_sensor_indices,
        'target_sensor_index': target_sensor_index,
        'input_label': input_label,
        'target_label': target_label,
        'timeseries_max_points': args.timeseries_max_points,
        'map_step': args.map_step,
        'subset_bounds': {
            'time': [args.start_date, args.end_date],
            'lat': [args.lat_min, args.lat_max],
            'lon': [args.lon_min, args.lon_max],
        },
        'input_volume_summary': summarize_volume(input_volume, 'input_volume'),
        'target_volume_summary': summarize_volume(target_volume, 'target_volume'),
    }

    summary_path = report_dir / '02_access_patterns_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('Saved:')
    print(f'  - {ts_path}')
    print(f'  - {slice_path}')
    print(f'  - {summary_path}')


if __name__ == '__main__':
    main()

