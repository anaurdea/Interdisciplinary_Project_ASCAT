from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ascat_ml.cube_utils import (  # noqa: E402
    detect_sensor_dim,
    detect_spatial_dims,
    infer_primary_variable,
    open_dataset,
    with_time_coordinate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build a valid-sample index [time, y, x] for ASCAT input-output training pairs.'
    )
    parser.add_argument('--dataset', default='Dataset/ASCAT_SSM_EASE2_25.nc', help='Path to NetCDF dataset')
    parser.add_argument('--variable', default=None, help='Optional data variable name')
    parser.add_argument('--output-index', default='outputs/reports/train_index.npy')
    parser.add_argument('--output-meta', default='outputs/reports/train_index_meta.json')
    parser.add_argument('--input-channels', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--target-channel', type=int, default=2)
    parser.add_argument('--start-date', default=None, help='Optional start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=None, help='Optional end date (YYYY-MM-DD)')
    parser.add_argument('--lat-min', type=float, default=None)
    parser.add_argument('--lat-max', type=float, default=None)
    parser.add_argument('--lon-min', type=float, default=None)
    parser.add_argument('--lon-max', type=float, default=None)
    parser.add_argument('--time-stride', type=int, default=1)
    parser.add_argument('--spatial-step', type=int, default=1)
    parser.add_argument('--time-batch-size', type=int, default=48)
    parser.add_argument(
        '--allow-partial-inputs',
        action='store_true',
        help='Allow a sample if at least one input channel is finite. Default requires all input channels.',
    )
    parser.add_argument('--time-chunk', type=int, default=256)
    parser.add_argument('--lat-chunk', type=int, default=200)
    parser.add_argument('--lon-chunk', type=int, default=200)
    return parser.parse_args()


def select_time_indices(time_values: np.ndarray, start_date: str | None, end_date: str | None, stride: int) -> np.ndarray:
    mask = np.ones(time_values.shape[0], dtype=bool)

    if start_date is not None and end_date is not None:
        lo = np.datetime64(start_date)
        hi = np.datetime64(end_date)
        if lo > hi:
            lo, hi = hi, lo
        mask &= time_values >= lo
        mask &= time_values <= hi
    elif start_date is not None:
        lo = np.datetime64(start_date)
        mask &= time_values >= lo
    elif end_date is not None:
        hi = np.datetime64(end_date)
        mask &= time_values <= hi

    indices = np.where(mask)[0]
    if stride > 1:
        indices = indices[::stride]
    return indices.astype(np.int64)


def select_spatial_indices(
    da,
    y_dim: str,
    x_dim: str,
    *,
    lat_min: float | None,
    lat_max: float | None,
    lon_min: float | None,
    lon_max: float | None,
    spatial_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_size = da.sizes[y_dim]
    x_size = da.sizes[x_dim]
    y_indices = np.arange(y_size, dtype=np.int64)
    x_indices = np.arange(x_size, dtype=np.int64)

    has_bbox = None not in (lat_min, lat_max, lon_min, lon_max)
    if not has_bbox:
        if spatial_step > 1:
            y_indices = y_indices[::spatial_step]
            x_indices = x_indices[::spatial_step]
        return y_indices, x_indices

    if 'lat' not in da.coords or 'lon' not in da.coords:
        raise ValueError('lat/lon coordinates are required to use spatial bounding boxes.')

    lat_lo = min(float(lat_min), float(lat_max))
    lat_hi = max(float(lat_min), float(lat_max))
    lon_lo = min(float(lon_min), float(lon_max))
    lon_hi = max(float(lon_min), float(lon_max))

    lat = np.asarray(da['lat'].values)
    lon = np.asarray(da['lon'].values)

    if lat.ndim == 2 and lon.ndim == 2:
        mask = (
            (lat >= lat_lo)
            & (lat <= lat_hi)
            & (lon >= lon_lo)
            & (lon <= lon_hi)
            & np.isfinite(lat)
            & np.isfinite(lon)
        )
        yy, xx = np.where(mask)
        if yy.size == 0:
            raise ValueError('Spatial bounding box did not match any grid cells.')
        y_indices = np.arange(int(yy.min()), int(yy.max()) + 1, spatial_step, dtype=np.int64)
        x_indices = np.arange(int(xx.min()), int(xx.max()) + 1, spatial_step, dtype=np.int64)
        return y_indices, x_indices

    if lat.ndim == 1 and lon.ndim == 1:
        y_mask = (lat >= lat_lo) & (lat <= lat_hi) & np.isfinite(lat)
        x_mask = (lon >= lon_lo) & (lon <= lon_hi) & np.isfinite(lon)
        y_found = np.where(y_mask)[0]
        x_found = np.where(x_mask)[0]
        if y_found.size == 0 or x_found.size == 0:
            raise ValueError('Spatial bounding box did not match any grid cells.')
        y_indices = np.arange(int(y_found.min()), int(y_found.max()) + 1, spatial_step, dtype=np.int64)
        x_indices = np.arange(int(x_found.min()), int(x_found.max()) + 1, spatial_step, dtype=np.int64)
        return y_indices, x_indices

    raise ValueError('Unsupported lat/lon coordinate shape for spatial selection.')


def main() -> None:
    args = parse_args()
    if args.time_stride < 1:
        raise ValueError('--time-stride must be >= 1')
    if args.spatial_step < 1:
        raise ValueError('--spatial-step must be >= 1')
    if args.time_batch_size < 1:
        raise ValueError('--time-batch-size must be >= 1')

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
        raise ValueError('Could not detect time dimension.')

    sensor_dim = detect_sensor_dim(da)
    if sensor_dim is None:
        raise ValueError('Could not detect sensor/channel dimension.')

    y_dim, x_dim = detect_spatial_dims(da)
    if y_dim is None or x_dim is None:
        raise ValueError('Could not detect spatial dimensions.')

    input_channels = [int(i) for i in args.input_channels]
    target_channel = int(args.target_channel)
    max_channel = max(input_channels + [target_channel])
    if max_channel >= da.sizes[sensor_dim]:
        raise IndexError(
            f'Channel index {max_channel} is out of bounds for "{sensor_dim}" size {da.sizes[sensor_dim]}.'
        )

    time_values = np.asarray(da[time_dim].values)
    time_indices = select_time_indices(
        time_values=time_values,
        start_date=args.start_date,
        end_date=args.end_date,
        stride=args.time_stride,
    )
    if time_indices.size == 0:
        raise ValueError('No timestamps selected. Check date filters and stride.')

    y_indices, x_indices = select_spatial_indices(
        da,
        y_dim=y_dim,
        x_dim=x_dim,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        spatial_step=args.spatial_step,
    )

    rows: list[np.ndarray] = []
    require_all_inputs = not args.allow_partial_inputs

    for start in range(0, time_indices.size, args.time_batch_size):
        stop = min(start + args.time_batch_size, time_indices.size)
        t_batch = time_indices[start:stop]
        chunk = da.isel(
            {
                time_dim: t_batch,
                y_dim: y_indices,
                x_dim: x_indices,
            }
        )

        input_cube = np.asarray(
            chunk.isel({sensor_dim: input_channels}).transpose(sensor_dim, time_dim, y_dim, x_dim).values,
            dtype=np.float32,
        )
        target_cube = np.asarray(
            chunk.isel({sensor_dim: target_channel}).transpose(time_dim, y_dim, x_dim).values,
            dtype=np.float32,
        )

        if require_all_inputs:
            valid_input = np.isfinite(input_cube).all(axis=0)
        else:
            valid_input = np.isfinite(input_cube).any(axis=0)
        valid_target = np.isfinite(target_cube)
        valid = valid_input & valid_target

        t_local, y_local, x_local = np.where(valid)
        if t_local.size == 0:
            continue

        mapped = np.column_stack(
            [
                t_batch[t_local],
                y_indices[y_local],
                x_indices[x_local],
            ]
        ).astype(np.int32)
        rows.append(mapped)

        print(
            f'Processed time batch [{start}:{stop}) -> valid samples in batch: {mapped.shape[0]}',
            flush=True,
        )

    index = np.concatenate(rows, axis=0) if rows else np.empty((0, 3), dtype=np.int32)

    output_index = Path(args.output_index)
    output_meta = Path(args.output_meta)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_index, index)

    meta = {
        'dataset': str(Path(args.dataset).resolve()),
        'variable': var_name,
        'index_columns': [time_dim, y_dim, x_dim],
        'sensor_dim': sensor_dim,
        'input_channels': input_channels,
        'target_channel': target_channel,
        'allow_partial_inputs': bool(args.allow_partial_inputs),
        'start_date': args.start_date,
        'end_date': args.end_date,
        'lat_min': args.lat_min,
        'lat_max': args.lat_max,
        'lon_min': args.lon_min,
        'lon_max': args.lon_max,
        'time_stride': args.time_stride,
        'spatial_step': args.spatial_step,
        'time_batch_size': args.time_batch_size,
        'time_candidates': int(time_indices.size),
        'y_candidates': int(y_indices.size),
        'x_candidates': int(x_indices.size),
        'num_samples': int(index.shape[0]),
    }
    output_meta.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print('\nSaved index and metadata:')
    print(f'  - {output_index}')
    print(f'  - {output_meta}')
    print(f'  - samples: {index.shape[0]}')


if __name__ == '__main__':
    main()
