from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ascat_utils import (
    detect_sensor_dim,
    infer_primary_variable,
    open_dataset,
    select_nearest_time_slice,
    validate_sensor_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize ASCAT time slices with EOMaps.')
    parser.add_argument('--dataset', default='Dataset/ASCAT_SSM_EASE2_25.nc', help='Path to NetCDF dataset')
    parser.add_argument('--variable', default=None, help='Optional data variable name')
    parser.add_argument('--date', default='2019-07-15', help='Date for the map (YYYY-MM-DD)')
    parser.add_argument('--sensor-index', type=int, default=2, help='Sensor index to visualize')
    parser.add_argument('--time-chunk', type=int, default=256)
    parser.add_argument('--lat-chunk', type=int, default=200)
    parser.add_argument('--lon-chunk', type=int, default=200)
    parser.add_argument(
        '--no-shared-scale',
        action='store_true',
        help='Disable shared color scaling across all sensors for the selected timestamp.',
    )
    parser.add_argument(
        '--low-quantile',
        type=float,
        default=0.02,
        help='Lower quantile for robust shared color scaling (default: 0.02).',
    )
    parser.add_argument(
        '--high-quantile',
        type=float,
        default=0.98,
        help='Upper quantile for robust shared color scaling (default: 0.98).',
    )
    parser.add_argument('--save-png', default='outputs/figures/03_eomaps_timeslice.png')
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Skip opening the interactive EOMaps window and only save the figure.',
    )
    return parser.parse_args()


def try_save_figure(map_obj, out_path: Path) -> bool:
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


def set_map_title(map_obj, title: str) -> None:
    # Avoid EOmaps text transforms that can fail in some backends.
    for setter in (
        lambda: map_obj.ax.set_title(title),
        lambda: map_obj.f.suptitle(title),
    ):
        try:
            setter()
            return
        except Exception:
            continue


def robust_limits(
    values: np.ndarray,
    low_q: float,
    high_q: float,
) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None

    low_q = min(max(low_q, 0.0), 1.0)
    high_q = min(max(high_q, 0.0), 1.0)
    if low_q >= high_q:
        low_q, high_q = 0.0, 1.0

    vmin = float(np.quantile(finite, low_q))
    vmax = float(np.quantile(finite, high_q))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    return vmin, vmax


def print_sensor_stats(cube_at_time, sensor_dim: str, ds) -> None:
    label_coord = ds.coords.get('spacecraft')
    has_spacecraft = label_coord is not None and label_coord.dims == (sensor_dim,)

    print('Per-sensor data summary at selected timestamp:')
    for sensor_index in range(cube_at_time.sizes[sensor_dim]):
        sensor_slice = cube_at_time.isel({sensor_dim: sensor_index}).values
        finite = np.isfinite(sensor_slice)
        finite_count = int(finite.sum())
        total_count = int(sensor_slice.size)
        frac = (finite_count / total_count) if total_count else 0.0
        if finite_count > 0:
            vals = sensor_slice[finite]
            mean = float(vals.mean())
            std = float(vals.std())
        else:
            mean = float('nan')
            std = float('nan')
        if has_spacecraft:
            label = str(label_coord.values[sensor_index])
            print(
                f'  sensor={sensor_index} ({label}) | finite={frac:.4%} | '
                f'mean={mean:.3f} | std={std:.3f}'
            )
        else:
            print(
                f'  sensor={sensor_index} | finite={frac:.4%} | '
                f'mean={mean:.3f} | std={std:.3f}'
            )


def main() -> None:
    try:
        from eomaps import Maps
    except ImportError as exc:
        raise SystemExit('EOMaps is not installed. Run: pip install -r requirements.txt') from exc

    args = parse_args()

    ds = open_dataset(
        args.dataset,
        time_chunk=args.time_chunk,
        lat_chunk=args.lat_chunk,
        lon_chunk=args.lon_chunk,
    )
    var_name = infer_primary_variable(ds, args.variable)
    da = ds[var_name]

    sensor_dim = detect_sensor_dim(da)
    vmin, vmax = None, None
    if sensor_dim is not None:
        sensor_index = validate_sensor_indices(da, [args.sensor_index])[0]
        cube_at_time, selected_time = select_nearest_time_slice(da, ds, args.date)
        if not args.no_shared_scale:
            vmin, vmax = robust_limits(
                cube_at_time.values,
                low_q=args.low_quantile,
                high_q=args.high_quantile,
            )
        print_sensor_stats(cube_at_time, sensor_dim, ds)
        slice_da = cube_at_time.isel({sensor_dim: sensor_index})
        title = f'{var_name} | sensor={sensor_index} | time={selected_time}'
    else:
        slice_da, selected_time = select_nearest_time_slice(da, ds, args.date)
        title = f'{var_name} | time={selected_time}'
        vmin, vmax = robust_limits(
            slice_da.values,
            low_q=args.low_quantile,
            high_q=args.high_quantile,
        )

    m = Maps(crs=4326)
    m.set_data(
        slice_da.values,
        x=slice_da['lon'].values,
        y=slice_da['lat'].values,
        crs=4326,
    )
    plot_kwargs = {'cmap': 'viridis'}
    if vmin is not None and vmax is not None:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax
        print(f'Using color scale limits: vmin={vmin:.3f}, vmax={vmax:.3f}')
    m.plot_map(**plot_kwargs)
    m.add_feature.preset.coastline()
    m.add_colorbar(label=var_name)
    set_map_title(m, title)

    out_path = Path(args.save_png)
    if try_save_figure(m, out_path):
        print(f'Saved EOMaps figure: {out_path}')
    else:
        print('Could not auto-save EOMaps figure; interactive window will still open.')

    if not args.no_show:
        m.show()


if __name__ == '__main__':
    main()

