from __future__ import annotations

import argparse
from pathlib import Path

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
    if sensor_dim is not None:
        sensor_index = validate_sensor_indices(da, [args.sensor_index])[0]
        selected, selected_time = select_nearest_time_slice(
            da.isel({sensor_dim: sensor_index}),
            ds,
            args.date,
        )
        slice_da = selected
        title = f'{var_name} | sensor={sensor_index} | time={selected_time}'
    else:
        slice_da, selected_time = select_nearest_time_slice(da, ds, args.date)
        title = f'{var_name} | time={selected_time}'

    m = Maps(crs=4326)
    m.set_data(
        slice_da.values,
        x=slice_da['lon'].values,
        y=slice_da['lat'].values,
        crs=4326,
    )
    m.plot_map(cmap='viridis')
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

