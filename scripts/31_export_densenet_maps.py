from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
        description='Export DenseNet gap-filled maps from a saved checkpoint without retraining.'
    )
    parser.add_argument('--dataset', default='Dataset/ASCAT_SSM_EASE2_25.nc')
    parser.add_argument('--checkpoint', required=True, help='Path to saved DenseNet model checkpoint (.pt).')
    parser.add_argument(
        '--index',
        default=None,
        help='Optional index (.npy) used to select available timestamps. If omitted, full dataset timeline is used.',
    )
    parser.add_argument('--dates', nargs='*', default=None, help='Optional explicit map dates (YYYY-MM-DD).')
    parser.add_argument(
        '--step-days',
        type=int,
        default=30,
        help='When --dates is not provided, sample dates every N days.',
    )
    parser.add_argument(
        '--max-dates',
        type=int,
        default=12,
        help='Maximum number of auto-selected dates when --dates is not provided.',
    )
    parser.add_argument('--start-date', default=None, help='Optional start date filter (YYYY-MM-DD).')
    parser.add_argument('--end-date', default=None, help='Optional end date filter (YYYY-MM-DD).')
    parser.add_argument('--fig-dir', default='outputs/figures/maps')
    parser.add_argument('--prefix', default='31_densenet_gapfill')
    parser.add_argument('--save-eomaps', action='store_true', help='Also export EOMaps PNG per date.')
    parser.add_argument('--eomaps-prefix', default='31_densenet_gapfill_eomaps')
    parser.add_argument('--output-json', default='outputs/reports/31_densenet_map_exports.json')
    parser.add_argument('--time-chunk', type=int, default=256)
    parser.add_argument('--lat-chunk', type=int, default=200)
    parser.add_argument('--lon-chunk', type=int, default=200)
    parser.add_argument('--map-row-chunk', type=int, default=24)
    parser.add_argument('--map-batch-size', type=int, default=2048)
    return parser.parse_args()


def _filter_days(
    days: np.ndarray,
    start_date: str | None,
    end_date: str | None,
) -> np.ndarray:
    out = days
    if start_date:
        out = out[out >= np.datetime64(start_date, 'D')]
    if end_date:
        out = out[out <= np.datetime64(end_date, 'D')]
    return out


def _sample_days(days: np.ndarray, step_days: int, max_dates: int) -> list[str]:
    if days.size == 0:
        return []
    selected: list[np.datetime64] = [days[0]]
    for day in days[1:]:
        if (day - selected[-1]) >= np.timedelta64(max(1, step_days), 'D'):
            selected.append(day)
        if max_dates > 0 and len(selected) >= max_dates:
            break
    return [str(d.astype('datetime64[D]')) for d in selected]


def _select_dates(
    cube_time_values: np.ndarray,
    index_path: str | None,
    explicit_dates: list[str] | None,
    step_days: int,
    max_dates: int,
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
        timestamps = cube_time_values

    days = np.unique(np.asarray(timestamps, dtype='datetime64[D]'))
    days = _filter_days(days, start_date=start_date, end_date=end_date)
    return _sample_days(days, step_days=step_days, max_dates=max_dates)


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
        explicit_dates=args.dates,
        step_days=args.step_days,
        max_dates=args.max_dates,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if not dates:
        raise ValueError('No dates selected for export. Check filters or provide --dates.')

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, str | None]] = []
    for date in dates:
        safe_date = date.replace(':', '-')
        map_png = fig_dir / f'{args.prefix}_{safe_date}.png'
        eomaps_png = fig_dir / f'{args.eomaps_prefix}_{safe_date}.png'

        target_map, baseline_map, pred_map, gap_filled, lon_2d, lat_2d, selected_time = train_module.predict_map(
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
        train_module.plot_gapfill_maps(
            target_map=target_map,
            baseline_map=baseline_map,
            pred_map=pred_map,
            gap_filled=gap_filled,
            lon_2d=lon_2d,
            lat_2d=lat_2d,
            selected_time=selected_time,
            out_path=map_png,
        )

        eomaps_out: str | None = None
        if args.save_eomaps:
            ok = train_module.plot_gapfill_eomaps(
                gap_filled=gap_filled,
                lon_2d=lon_2d,
                lat_2d=lat_2d,
                selected_time=selected_time,
                out_path=eomaps_png,
            )
            if ok:
                eomaps_out = str(eomaps_png)

        outputs.append(
            {
                'requested_date': date,
                'selected_timestamp': selected_time,
                'map_png': str(map_png),
                'eomaps_png': eomaps_out,
            }
        )
        print(f'Exported map for {date} -> {map_png}')

    report = {
        'dataset': args.dataset,
        'checkpoint': args.checkpoint,
        'selected_dates': dates,
        'exports': outputs,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')

    cube.ds.close()
    print(f'Saved export report: {out_json}')


if __name__ == '__main__':
    main()
