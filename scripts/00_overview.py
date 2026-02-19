from __future__ import annotations

import argparse
from pathlib import Path

from ascat_utils import (
    ensure_dir,
    format_dims,
    infer_primary_variable,
    missing_fraction_sample,
    open_dataset,
    sensor_labels,
    with_time_coordinate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Quick dataset overview for ASCAT SSM cube.')
    parser.add_argument('--dataset', default='Dataset/ASCAT_SSM_EASE2_25.nc', help='Path to NetCDF dataset')
    parser.add_argument('--variable', default=None, help='Optional data variable name')
    parser.add_argument('--time-chunk', type=int, default=256)
    parser.add_argument('--lat-chunk', type=int, default=200)
    parser.add_argument('--lon-chunk', type=int, default=200)
    parser.add_argument('--sample-time-steps', type=int, default=365)
    parser.add_argument('--report-dir', default='outputs/reports')
    return parser.parse_args()


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

    lines: list[str] = []
    lines.append('ASCAT dataset overview')
    lines.append('')
    lines.append(f'Dataset: {Path(args.dataset).resolve()}')
    lines.append(f'Dimensions: {format_dims(ds)}')
    lines.append(f'Coordinates: {list(ds.coords)}')
    lines.append(f'Data variables: {list(ds.data_vars)}')
    lines.append(f'Primary variable: {var_name}')
    lines.append(f'Primary variable dims: {da.dims}')
    lines.append(f'Primary variable dtype: {da.dtype}')

    if time_dim is not None:
        tmin = da_with_time[time_dim].min().compute().values
        tmax = da_with_time[time_dim].max().compute().values
        lines.append(f'Time range: {tmin} -> {tmax}')

    labels = sensor_labels(ds)
    if labels:
        lines.append(f'Sensor labels: {labels}')

    miss = missing_fraction_sample(da_with_time, sample_time_steps=args.sample_time_steps)
    lines.append(
        f'Missing fraction sample (first {args.sample_time_steps} steps or full if shorter): {miss:.4f}'
    )

    report_dir = ensure_dir(args.report_dir)
    report_path = report_dir / '00_dataset_overview.txt'
    report_path.write_text('\n'.join(lines), encoding='utf-8')

    print('\n'.join(lines))
    print(f'\nSaved report: {report_path}')


if __name__ == '__main__':
    main()

