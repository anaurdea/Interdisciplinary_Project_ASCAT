from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create a PyTorch DataLoader for ASCAT input/output vectors and print batch diagnostics.'
    )
    parser.add_argument('--dataset', default='Dataset/ASCAT_SSM_EASE2_25.nc')
    parser.add_argument('--index', default='outputs/reports/train_index.npy')
    parser.add_argument('--variable', default=None)
    parser.add_argument('--input-channels', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--target-channel', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-batches', type=int, default=2)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--drop-last', action='store_true')
    parser.add_argument('--ancillary-npz', default=None, help='Optional NPZ file with ancillary arrays')
    parser.add_argument(
        '--ancillary-keys',
        nargs='+',
        default=None,
        help='Optional key order for ancillary NPZ arrays',
    )
    parser.add_argument('--stats-json', default=None, help='Optional path to existing normalization stats JSON')
    parser.add_argument(
        '--fit-stats-samples',
        type=int,
        default=0,
        help='If > 0, fit normalization stats from random samples before DataLoader iteration.',
    )
    parser.add_argument('--save-stats-json', default='outputs/reports/normalization_stats.json')
    parser.add_argument('--include-input-std', action='store_true')
    parser.add_argument('--include-input-range', action='store_true')
    parser.add_argument('--include-channel-values', action='store_true')
    parser.add_argument('--no-time-features', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--time-chunk', type=int, default=256)
    parser.add_argument('--lat-chunk', type=int, default=200)
    parser.add_argument('--lon-chunk', type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ascat_ml.ancillary import EmptyAncillaryProvider, NPZAncillaryProvider
        from ascat_ml.dataloader import ASCATSampleDataset, create_dataloader, fit_standardization_stats
        from ascat_ml.preprocessing import FeatureConfig, StandardizationStats
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'Missing dependency for DataLoader pipeline. '
            'Install requirements and ensure torch is available: pip install -r requirements.txt'
        ) from exc

    ancillary_provider = EmptyAncillaryProvider()
    if args.ancillary_npz:
        ancillary_provider = NPZAncillaryProvider.from_npz(
            args.ancillary_npz,
            feature_order=args.ancillary_keys,
        )

    feature_config = FeatureConfig(
        input_channels=tuple(args.input_channels),
        include_input_std=bool(args.include_input_std),
        include_input_range=bool(args.include_input_range),
        include_channel_values=bool(args.include_channel_values),
        include_time_features=not args.no_time_features,
    )

    normalization = None
    if args.stats_json:
        stats_path = Path(args.stats_json)
        if stats_path.exists():
            normalization = StandardizationStats.from_json(stats_path)
        else:
            raise FileNotFoundError(f'Normalization stats file not found: {stats_path}')

    dataset = ASCATSampleDataset(
        dataset_path=args.dataset,
        index_path=args.index,
        variable=args.variable,
        feature_config=feature_config,
        target_channel=args.target_channel,
        normalization=normalization,
        ancillary_provider=ancillary_provider,
        return_metadata=False,
        time_chunk=args.time_chunk,
        lat_chunk=args.lat_chunk,
        lon_chunk=args.lon_chunk,
    )

    if normalization is not None:
        expected = len(dataset.feature_names)
        got = int(normalization.feature_mean.shape[0])
        if got != expected:
            raise ValueError(
                'Normalization stats are incompatible with current feature config: '
                f'stats has {got} features but current setup has {expected}. '
                'Refit stats with the same flags (e.g., --include-input-std/--include-input-range).'
            )
        if normalization.feature_names is not None and normalization.feature_names != dataset.feature_names:
            raise ValueError(
                'Normalization stats feature names do not match current feature names. '
                f'stats={normalization.feature_names}, current={dataset.feature_names}. '
                'Refit stats with the same feature flags.'
            )

    if args.fit_stats_samples > 0:
        stats = fit_standardization_stats(
            dataset,
            sample_size=args.fit_stats_samples,
            seed=args.seed,
        )
        stats_path = Path(args.save_stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_json(stats_path)
        dataset.normalization = stats
        print(f'Saved normalization stats: {stats_path}')

    loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )

    print('DataLoader ready')
    print(f'  samples: {len(dataset)}')
    print(f'  features: {dataset.feature_names}')
    print(f'  batch_size: {args.batch_size}')
    print(f'  num_workers: {args.num_workers}')

    for batch_idx, batch in enumerate(loader):
        x = batch['x']
        y = batch['y']
        x_mask = batch['x_mask']
        y_mask = batch['y_mask']
        print(
            f'batch={batch_idx} x={tuple(x.shape)} y={tuple(y.shape)} '
            f'x_mask_mean={x_mask.float().mean().item():.4f} y_mask_mean={y_mask.float().mean().item():.4f}'
        )
        if batch_idx + 1 >= args.max_batches:
            break

    dataset.close()


if __name__ == '__main__':
    main()
