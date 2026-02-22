from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
from ascat_ml.densenet_model import DenseNetRegressor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train and evaluate a DenseNet model for ASCAT soil-moisture gap filling.'
    )
    parser.add_argument('--dataset', default='Dataset/ASCAT_SSM_EASE2_25.nc')
    parser.add_argument('--index', default='outputs/reports/train_index.npy')
    parser.add_argument('--baseline-json', default='outputs/reports/20_baseline_results.json')
    parser.add_argument('--output-json', default='outputs/reports/30_densenet_results.json')
    parser.add_argument('--save-model', default='outputs/models/30_densenet_model.pt')
    parser.add_argument('--fig-dir', default='outputs/figures')
    parser.add_argument(
        '--eomaps-png',
        default='outputs/figures/30_densenet_gapfilled_eomaps.png',
        help='Optional EOMaps output for the DenseNet gap-filled map.',
    )
    parser.add_argument(
        '--no-eomaps',
        action='store_true',
        help='Disable EOMaps rendering for the gap-filled map.',
    )
    parser.add_argument('--variable', default=None)
    parser.add_argument('--input-channels', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--target-channel', type=int, default=2)
    parser.add_argument('--patch-size', type=int, default=9)
    parser.add_argument('--growth-rate', type=int, default=16)
    parser.add_argument('--block-layers', nargs=2, type=int, default=[4, 4])
    parser.add_argument('--init-features', type=int, default=32)
    parser.add_argument('--head-hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-mode', choices=['time', 'random'], default='time')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--cv-folds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--cv-epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--log-every', type=int, default=1, help='Print training status every N epochs.')
    parser.add_argument(
        '--require-cuda',
        action='store_true',
        help='Fail fast if CUDA is not available in the active Python environment.',
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=0, help='Optional cap for development runs.')
    parser.add_argument('--map-date', default='2019-07-15')
    parser.add_argument('--map-row-chunk', type=int, default=24)
    parser.add_argument('--map-batch-size', type=int, default=2048)
    parser.add_argument('--time-chunk', type=int, default=256)
    parser.add_argument('--lat-chunk', type=int, default=200)
    parser.add_argument('--lon-chunk', type=int, default=200)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {'rmse': rmse(y_true, y_pred), 'mae': mae(y_true, y_pred)}


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f'{hours:02d}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'


def summarize_metric_list(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {'rmse_mean': float('nan'), 'rmse_std': float('nan'), 'mae_mean': float('nan'), 'mae_std': float('nan')}
    rmse_vals = np.asarray([m['rmse'] for m in items], dtype=np.float64)
    mae_vals = np.asarray([m['mae'] for m in items], dtype=np.float64)
    return {
        'rmse_mean': float(rmse_vals.mean()),
        'rmse_std': float(rmse_vals.std()),
        'mae_mean': float(mae_vals.mean()),
        'mae_std': float(mae_vals.std()),
    }


def timestamp_features(timestamp: np.datetime64) -> np.ndarray:
    ts = np.datetime64(timestamp)
    day = int(np.datetime64(ts, 'D').astype(object).timetuple().tm_yday)
    sec_day = int((ts - np.datetime64(ts, 'D')) / np.timedelta64(1, 's'))
    hour = sec_day / 3600.0
    doy_angle = 2.0 * np.pi * day / 366.0
    hour_angle = 2.0 * np.pi * hour / 24.0
    return np.asarray(
        [np.sin(doy_angle), np.cos(doy_angle), np.sin(hour_angle), np.cos(hour_angle)],
        dtype=np.float32,
    )


def _ensure_non_empty_split(split: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx, val_idx, test_idx = split
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError('Train/val/test split produced an empty partition.')
    return split


def split_indices_random(n_samples: int, train_ratio: float, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_idx)

    n_train = max(1, int(round(n_samples * train_ratio)))
    n_val = max(1, int(round(n_samples * val_ratio)))
    n_train = min(n_train, n_samples - 2)
    n_val = min(n_val, n_samples - n_train - 1)
    n_test = n_samples - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_val = max(1, n_val - 1)

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train + n_val]
    test_idx = all_idx[n_train + n_val:]
    return _ensure_non_empty_split((train_idx, val_idx, test_idx))


def split_indices_by_time(
    time_idx: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_times = np.unique(time_idx)
    unique_times.sort()
    n_times = unique_times.size
    if n_times < 3:
        raise ValueError('Need at least 3 unique timestamps for time-based split.')

    n_train_t = max(1, int(round(n_times * train_ratio)))
    n_val_t = max(1, int(round(n_times * val_ratio)))
    n_train_t = min(n_train_t, n_times - 2)
    n_val_t = min(n_val_t, n_times - n_train_t - 1)

    train_times = unique_times[:n_train_t]
    val_times = unique_times[n_train_t:n_train_t + n_val_t]
    test_times = unique_times[n_train_t + n_val_t:]

    train_idx = np.where(np.isin(time_idx, train_times))[0]
    val_idx = np.where(np.isin(time_idx, val_times))[0]
    test_idx = np.where(np.isin(time_idx, test_times))[0]
    return _ensure_non_empty_split((train_idx, val_idx, test_idx))


def group_kfold_indices(
    sample_indices: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    group_values = groups[sample_indices]
    unique_groups = np.unique(group_values)
    if unique_groups.size < n_splits:
        n_splits = int(unique_groups.size)
    if n_splits < 2:
        return []

    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)
    folds = np.array_split(shuffled_groups, n_splits)

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_groups in folds:
        val_mask = np.isin(group_values, fold_groups)
        val_idx = sample_indices[val_mask]
        train_idx = sample_indices[~val_mask]
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        out.append((train_idx, val_idx))
    return out


@dataclass
class CubeContext:
    ds: xr.Dataset
    da: xr.DataArray
    sensor_dim: str
    time_dim: str
    y_dim: str
    x_dim: str
    time_values: np.ndarray


def open_cube_context(args: argparse.Namespace) -> CubeContext:
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

    return CubeContext(
        ds=ds,
        da=da,
        sensor_dim=sensor_dim,
        time_dim=time_dim,
        y_dim=y_dim,
        x_dim=x_dim,
        time_values=np.asarray(da[time_dim].values),
    )


class ASCATPatchDataset(Dataset):
    def __init__(
        self,
        cube: CubeContext,
        index_array: np.ndarray,
        rows: np.ndarray,
        *,
        input_channels: list[int],
        target_channel: int,
        patch_size: int,
        target_mean: float,
        target_std: float,
    ) -> None:
        self.cube = cube
        self.index_array = index_array
        self.rows = np.asarray(rows, dtype=np.int64)
        self.input_channels = [int(i) for i in input_channels]
        self.target_channel = int(target_channel)
        self.patch_size = int(patch_size)
        if self.patch_size % 2 == 0:
            raise ValueError('--patch-size must be odd.')
        self.pad = self.patch_size // 2
        self.target_mean = float(target_mean)
        self.target_std = float(target_std) if target_std > 1e-6 else 1.0

        self._cached_time: int | None = None
        self._cached_stack_padded: np.ndarray | None = None
        self._cached_target: np.ndarray | None = None
        self._cached_aux: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    def _load_time_slice(self, time_index: int) -> None:
        if self._cached_time == time_index:
            return
        da = self.cube.da
        inp = np.asarray(
            da.isel(
                {
                    self.cube.sensor_dim: self.input_channels,
                    self.cube.time_dim: int(time_index),
                }
            ).transpose(self.cube.sensor_dim, self.cube.y_dim, self.cube.x_dim).values,
            dtype=np.float32,
        )
        tgt = np.asarray(
            da.isel(
                {
                    self.cube.sensor_dim: self.target_channel,
                    self.cube.time_dim: int(time_index),
                }
            ).transpose(self.cube.y_dim, self.cube.x_dim).values,
            dtype=np.float32,
        )
        finite = np.isfinite(inp)
        inp_filled = np.where(finite, inp, 0.0).astype(np.float32)
        mask = finite.astype(np.float32)
        stack = np.concatenate([inp_filled, mask], axis=0).astype(np.float32)
        padded = np.pad(
            stack,
            ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),
            mode='constant',
            constant_values=0.0,
        )

        self._cached_time = int(time_index)
        self._cached_stack_padded = padded
        self._cached_target = tgt
        self._cached_aux = timestamp_features(self.cube.time_values[int(time_index)])

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = int(self.rows[item])
        t_idx, y_idx, x_idx = [int(v) for v in self.index_array[row]]
        self._load_time_slice(t_idx)
        assert self._cached_stack_padded is not None
        assert self._cached_target is not None
        assert self._cached_aux is not None

        patch = self._cached_stack_padded[:, y_idx:y_idx + self.patch_size, x_idx:x_idx + self.patch_size]
        target = float(self._cached_target[y_idx, x_idx])
        target_mask = float(np.isfinite(target))
        if not np.isfinite(target):
            target = self.target_mean

        y_norm = (target - self.target_mean) / self.target_std
        return {
            'x_img': torch.from_numpy(patch.astype(np.float32)),
            'x_aux': torch.from_numpy(self._cached_aux.astype(np.float32)),
            'y': torch.tensor(y_norm, dtype=torch.float32),
            'y_true': torch.tensor(target, dtype=torch.float32),
            'y_mask': torch.tensor(target_mask, dtype=torch.float32),
        }


def compute_train_target_stats(
    cube: CubeContext,
    index_array: np.ndarray,
    rows: np.ndarray,
    target_channel: int,
) -> tuple[float, float]:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        raise ValueError('No rows provided for training target statistics.')

    targets = np.empty((rows.size,), dtype=np.float32)
    time_col = index_array[:, 0].astype(np.int64)
    order = np.argsort(time_col[rows], kind='mergesort')
    sorted_rows = rows[order]

    cursor = 0
    while cursor < sorted_rows.size:
        row = int(sorted_rows[cursor])
        t = int(index_array[row, 0])
        end = cursor
        while end < sorted_rows.size and int(index_array[int(sorted_rows[end]), 0]) == t:
            end += 1

        batch_rows = sorted_rows[cursor:end]
        y_indices = index_array[batch_rows, 1].astype(np.int64)
        x_indices = index_array[batch_rows, 2].astype(np.int64)

        tgt_map = np.asarray(
            cube.da.isel(
                {
                    cube.sensor_dim: int(target_channel),
                    cube.time_dim: t,
                }
            ).transpose(cube.y_dim, cube.x_dim).values,
            dtype=np.float32,
        )
        vals = tgt_map[y_indices, x_indices]
        targets[order[cursor:end]] = vals
        cursor = end

    finite = targets[np.isfinite(targets)]
    if finite.size == 0:
        raise ValueError('No finite target values found in training split.')
    mean = float(finite.mean())
    std = float(finite.std())
    if std < 1e-6:
        std = 1.0
    return mean, std


def create_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    grad_clip: float,
    log_every: int,
) -> tuple[nn.Module, dict[str, list[float]]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=max(2, patience // 3),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float('inf')
    wait = 0
    history: dict[str, list[float]] = {'train_loss': [], 'val_loss': []}
    epoch_times: list[float] = []
    run_start = time.perf_counter()
    log_every = max(1, int(log_every))

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            x_img = batch['x_img'].to(device, non_blocking=True)
            x_aux = batch['x_aux'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            y_mask = batch['y_mask'].to(device, non_blocking=True)
            valid = y_mask > 0.5
            if not torch.any(valid):
                continue

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_img, x_aux)
            loss = criterion(pred[valid], y[valid])
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                x_img = batch['x_img'].to(device, non_blocking=True)
                x_aux = batch['x_aux'].to(device, non_blocking=True)
                y = batch['y'].to(device, non_blocking=True)
                y_mask = batch['y_mask'].to(device, non_blocking=True)
                valid = y_mask > 0.5
                if not torch.any(valid):
                    continue
                pred = model(x_img, x_aux)
                loss = criterion(pred[valid], y[valid])
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        epoch_sec = time.perf_counter() - epoch_start
        epoch_times.append(epoch_sec)
        elapsed = time.perf_counter() - run_start
        mean_epoch = float(np.mean(epoch_times)) if epoch_times else epoch_sec
        remaining_epochs = max(0, epochs - epoch)
        eta = remaining_epochs * mean_epoch

        should_log = (epoch == 1) or (epoch % log_every == 0) or (epoch == epochs) or (wait >= patience)
        if should_log:
            print(
                f'epoch={epoch:03d}/{epochs:03d} train_loss={train_loss:.6f} '
                f'val_loss={val_loss:.6f} wait={wait}/{patience} '
                f'epoch_time={format_seconds(epoch_sec)} elapsed={format_seconds(elapsed)} '
                f'eta~{format_seconds(eta)}',
                flush=True,
            )

        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_model(model: nn.Module, loader: DataLoader, target_mean: float, target_std: float) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x_img = batch['x_img'].to(device, non_blocking=True)
            x_aux = batch['x_aux'].to(device, non_blocking=True)
            y_true = batch['y_true'].cpu().numpy().astype(np.float32)
            y_mask = batch['y_mask'].cpu().numpy().astype(np.float32) > 0.5

            pred_norm = model(x_img, x_aux).cpu().numpy().astype(np.float32)
            pred = pred_norm * target_std + target_mean

            if y_mask.any():
                y_true_all.append(y_true[y_mask])
                y_pred_all.append(pred[y_mask])
    if not y_true_all:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


def predict_map(
    model: nn.Module,
    cube: CubeContext,
    *,
    input_channels: list[int],
    target_channel: int,
    patch_size: int,
    target_mean: float,
    target_std: float,
    map_date: str,
    row_chunk: int,
    infer_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    da = cube.da
    target = np.datetime64(map_date)
    t_values = cube.time_values
    t_idx = int(np.argmin(np.abs(t_values - target)))
    selected_time = str(np.datetime_as_string(t_values[t_idx], unit='s'))

    input_map = np.asarray(
        da.isel(
            {
                cube.sensor_dim: input_channels,
                cube.time_dim: t_idx,
            }
        ).transpose(cube.sensor_dim, cube.y_dim, cube.x_dim).values,
        dtype=np.float32,
    )
    lon_2d: np.ndarray | None = None
    lat_2d: np.ndarray | None = None
    if 'lon' in da.coords and 'lat' in da.coords:
        lon_coord = da['lon']
        lat_coord = da['lat']
        if lon_coord.ndim == 2 and lat_coord.ndim == 2:
            lon_2d = np.asarray(lon_coord.values, dtype=np.float32)
            lat_2d = np.asarray(lat_coord.values, dtype=np.float32)
        elif lon_coord.ndim == 1 and lat_coord.ndim == 1:
            lon_1d = np.asarray(lon_coord.values, dtype=np.float32)
            lat_1d = np.asarray(lat_coord.values, dtype=np.float32)
            lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    target_map = np.asarray(
        da.isel(
            {
                cube.sensor_dim: target_channel,
                cube.time_dim: t_idx,
            }
        ).transpose(cube.y_dim, cube.x_dim).values,
        dtype=np.float32,
    )
    finite_inputs = np.isfinite(input_map)
    denom = np.sum(finite_inputs, axis=0).astype(np.float32)
    numer = np.sum(np.where(finite_inputs, input_map, 0.0), axis=0).astype(np.float32)
    baseline_map = np.where(denom > 0, numer / np.maximum(denom, 1.0), np.nan).astype(np.float32)
    valid_domain = np.any(finite_inputs, axis=0) | np.isfinite(target_map)

    pad = patch_size // 2
    finite = finite_inputs
    filled = np.where(finite, input_map, 0.0).astype(np.float32)
    masks = finite.astype(np.float32)
    stack = np.concatenate([filled, masks], axis=0).astype(np.float32)
    padded = np.pad(stack, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0.0)

    aux = timestamp_features(t_values[t_idx]).astype(np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    c_in = stack.shape[0]
    h, w = target_map.shape
    pred_map = np.empty((h, w), dtype=np.float32)

    with torch.no_grad():
        for y0 in range(0, h, row_chunk):
            y1 = min(y0 + row_chunk, h)
            region = padded[:, y0:y1 + 2 * pad, :]
            region_t = torch.from_numpy(region).unsqueeze(0).to(device)
            patches = F.unfold(region_t, kernel_size=patch_size, stride=1)
            patches = patches.squeeze(0).transpose(0, 1).reshape(-1, c_in, patch_size, patch_size)

            n = patches.shape[0]
            pred_rows: list[np.ndarray] = []
            for start in range(0, n, infer_batch_size):
                stop = min(start + infer_batch_size, n)
                chunk = patches[start:stop]
                aux_chunk = torch.from_numpy(np.repeat(aux[None, :], stop - start, axis=0)).to(device)
                pred_norm = model(chunk, aux_chunk).cpu().numpy().astype(np.float32)
                pred = pred_norm * target_std + target_mean
                pred_rows.append(pred)

            pred_block = np.concatenate(pred_rows, axis=0).reshape(y1 - y0, w)
            pred_map[y0:y1, :] = pred_block.astype(np.float32)

    pred_map = np.where(valid_domain, pred_map, np.nan).astype(np.float32)
    gap_filled = np.where(np.isfinite(target_map), target_map, pred_map).astype(np.float32)
    gap_filled = np.where(valid_domain, gap_filled, np.nan).astype(np.float32)
    return target_map, baseline_map, pred_map, gap_filled, lon_2d, lat_2d, selected_time


def plot_metric_comparison(
    baseline_json_path: str | Path,
    densenet_metrics: dict[str, float],
    out_path: str | Path,
) -> None:
    labels: list[str] = []
    rmse_vals: list[float] = []
    mae_vals: list[float] = []

    path = Path(baseline_json_path)
    if path.exists():
        payload = json.loads(path.read_text(encoding='utf-8'))
        heldout = payload.get('heldout_test_metrics', {})
        for key in ('temporal_linear', 'spatial_idw', 'torch_mlp'):
            v = heldout.get(key)
            if isinstance(v, dict) and 'rmse' in v and 'mae' in v:
                labels.append(key)
                rmse_vals.append(float(v['rmse']))
                mae_vals.append(float(v['mae']))

    labels.append('densenet')
    rmse_vals.append(float(densenet_metrics['rmse']))
    mae_vals.append(float(densenet_metrics['mae']))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].bar(x, rmse_vals, color='#4C72B0')
    axes[0].set_xticks(x, labels, rotation=20, ha='right')
    axes[0].set_title('Held-out RMSE')
    axes[0].set_ylabel('RMSE')

    axes[1].bar(x, mae_vals, color='#55A868')
    axes[1].set_xticks(x, labels, rotation=20, ha='right')
    axes[1].set_title('Held-out MAE')
    axes[1].set_ylabel('MAE')

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_gapfill_maps(
    target_map: np.ndarray,
    baseline_map: np.ndarray,
    pred_map: np.ndarray,
    gap_filled: np.ndarray,
    lon_2d: np.ndarray | None,
    lat_2d: np.ndarray | None,
    selected_time: str,
    out_path: str | Path,
) -> None:
    finite_vals = np.concatenate(
        [
            target_map[np.isfinite(target_map)],
            baseline_map[np.isfinite(baseline_map)],
            pred_map[np.isfinite(pred_map)],
            gap_filled[np.isfinite(gap_filled)],
        ]
    )
    if finite_vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.quantile(finite_vals, 0.02))
        vmax = float(np.quantile(finite_vals, 0.98))
        if vmax <= vmin:
            vmax = vmin + 1.0

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='#d9d9d9')

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    panels = [
        (target_map, 'Observed target (Metop-C)'),
        (baseline_map, 'Baseline proxy (mean Metop-A/B)'),
        (pred_map, 'DenseNet prediction'),
        (gap_filled, 'DenseNet gap-filled map'),
    ]

    use_geo_coords = (
        lon_2d is not None
        and lat_2d is not None
        and lon_2d.shape == target_map.shape
        and lat_2d.shape == target_map.shape
    )

    for ax, (arr, title) in zip(axes.ravel(), panels):
        arr_plot = np.ma.masked_invalid(arr)
        if use_geo_coords:
            im = ax.pcolormesh(
                lon_2d,
                lat_2d,
                arr_plot,
                shading='auto',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        else:
            im = ax.imshow(arr_plot, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f'ASCAT gap filling map view at {selected_time}', fontsize=13)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


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


def plot_gapfill_eomaps(
    gap_filled: np.ndarray,
    lon_2d: np.ndarray | None,
    lat_2d: np.ndarray | None,
    selected_time: str,
    out_path: str | Path,
) -> bool:
    if lon_2d is None or lat_2d is None:
        print('Skipping EOMaps export: lon/lat coordinates not available.')
        return False

    try:
        from eomaps import Maps
    except ImportError:
        print('Skipping EOMaps export: EOMaps is not installed in the environment.')
        return False

    finite_vals = gap_filled[np.isfinite(gap_filled)]
    if finite_vals.size > 0:
        vmin = float(np.quantile(finite_vals, 0.02))
        vmax = float(np.quantile(finite_vals, 0.98))
        if vmax <= vmin:
            vmin = float(np.min(finite_vals))
            vmax = float(np.max(finite_vals))
    else:
        vmin, vmax = None, None

    m = Maps(crs=4326)
    m.set_data(
        gap_filled,
        x=lon_2d,
        y=lat_2d,
        crs=4326,
    )
    plot_kwargs = {'cmap': 'viridis'}
    if vmin is not None and vmax is not None:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax
    m.plot_map(**plot_kwargs)
    m.add_feature.preset.coastline()
    m.add_colorbar(label='SSM (gap-filled)')
    set_eomaps_title(m, f'DenseNet gap-filled map at {selected_time}')

    out = Path(out_path)
    if try_save_eomaps_figure(m, out):
        print(f'Saved EOMaps figure: {out}')
        return True
    print('EOMaps figure could not be saved (backend limitation).')
    return False


def build_model(args: argparse.Namespace) -> DenseNetRegressor:
    return DenseNetRegressor(
        in_channels=2 * len(args.input_channels),
        aux_dim=4,
        growth_rate=args.growth_rate,
        block_layers=(int(args.block_layers[0]), int(args.block_layers[1])),
        init_features=args.init_features,
        dropout=args.dropout,
        head_hidden=args.head_hidden,
    )


def run_training_eval(
    cube: CubeContext,
    index_array: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    args: argparse.Namespace,
    seed: int,
    epochs: int,
) -> tuple[DenseNetRegressor, dict[str, list[float]], dict[str, float], float, float]:
    set_seed(seed)
    target_mean, target_std = compute_train_target_stats(
        cube,
        index_array,
        train_rows,
        target_channel=args.target_channel,
    )

    ds_train = ASCATPatchDataset(
        cube,
        index_array,
        train_rows,
        input_channels=args.input_channels,
        target_channel=args.target_channel,
        patch_size=args.patch_size,
        target_mean=target_mean,
        target_std=target_std,
    )
    ds_val = ASCATPatchDataset(
        cube,
        index_array,
        val_rows,
        input_channels=args.input_channels,
        target_channel=args.target_channel,
        patch_size=args.patch_size,
        target_mean=target_mean,
        target_std=target_std,
    )
    ds_test = ASCATPatchDataset(
        cube,
        index_array,
        test_rows,
        input_channels=args.input_channels,
        target_channel=args.target_channel,
        patch_size=args.patch_size,
        target_mean=target_mean,
        target_std=target_std,
    )

    dl_train = create_loader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_val = create_loader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_test = create_loader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args)
    model, history = train_model(
        model,
        dl_train,
        dl_val,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=epochs,
        patience=args.patience,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
    )
    y_true, y_pred = evaluate_model(model, dl_test, target_mean=target_mean, target_std=target_std)
    if y_true.size == 0:
        metrics = {'rmse': float('nan'), 'mae': float('nan')}
    else:
        metrics = metric_dict(y_true, y_pred)
    return model, history, metrics, target_mean, target_std


def main() -> None:
    args = parse_args()
    if args.patch_size % 2 == 0:
        raise ValueError('--patch-size must be odd.')
    if args.log_every <= 0:
        raise ValueError('--log-every must be >= 1.')
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError('train_ratio and val_ratio must be > 0 and sum to < 1.')

    set_seed(args.seed)
    cuda_available = torch.cuda.is_available()
    runtime_device = 'cuda' if cuda_available else 'cpu'
    device_name = torch.cuda.get_device_name(0) if cuda_available else 'CPU'
    print(
        f'Runtime environment -> python={sys.executable} device={runtime_device} device_name={device_name}',
        flush=True,
    )
    if args.require_cuda and not cuda_available:
        raise RuntimeError('CUDA is not available, but --require-cuda was set.')

    index_array = np.load(args.index).astype(np.int64, copy=False)
    if index_array.ndim != 2 or index_array.shape[1] != 3:
        raise ValueError('Index file must have shape [N,3] with [time,y,x].')
    if args.max_samples > 0:
        index_array = index_array[: int(args.max_samples)]
    if index_array.shape[0] < 10:
        raise ValueError('Not enough samples in index to run training/evaluation.')

    time_col = index_array[:, 0]
    n_samples = index_array.shape[0]
    print(f'Indexed samples used: {n_samples}', flush=True)

    if args.split_mode == 'time':
        train_idx, val_idx, test_idx = split_indices_by_time(time_col, args.train_ratio, args.val_ratio)
    else:
        train_idx, val_idx, test_idx = split_indices_random(n_samples, args.train_ratio, args.val_ratio, args.seed)

    print(
        f'Split sizes -> train={train_idx.size}, val={val_idx.size}, test={test_idx.size}',
        flush=True,
    )

    cube = open_cube_context(args)
    model, history, heldout_metrics, target_mean, target_std = run_training_eval(
        cube,
        index_array=index_array,
        train_rows=train_idx,
        val_rows=val_idx,
        test_rows=test_idx,
        args=args,
        seed=args.seed,
        epochs=args.epochs,
    )

    dev_idx = np.concatenate([train_idx, val_idx], axis=0)
    cv_folds = group_kfold_indices(dev_idx, groups=time_col, n_splits=args.cv_folds, seed=args.seed)
    cv_metrics: list[dict[str, float]] = []
    for fold_i, (fold_train, fold_val) in enumerate(cv_folds):
        fold_start = time.perf_counter()
        print(f'CV fold {fold_i + 1}/{len(cv_folds)} (start)', flush=True)
        _, _, fold_metric, _, _ = run_training_eval(
            cube,
            index_array=index_array,
            train_rows=fold_train,
            val_rows=fold_val,
            test_rows=fold_val,
            args=args,
            seed=args.seed + fold_i + 1,
            epochs=args.cv_epochs,
        )
        cv_metrics.append(fold_metric)
        fold_elapsed = time.perf_counter() - fold_start
        print(
            f'CV fold {fold_i + 1}/{len(cv_folds)} (done) '
            f'rmse={fold_metric["rmse"]:.4f} mae={fold_metric["mae"]:.4f} '
            f'time={format_seconds(fold_elapsed)}',
            flush=True,
        )

    cv_summary = summarize_metric_list(cv_metrics)

    model_path = Path(args.save_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'config': vars(args),
            'target_mean': target_mean,
            'target_std': target_std,
            'input_channels': list(args.input_channels),
            'target_channel': int(args.target_channel),
            'patch_size': int(args.patch_size),
        },
        model_path,
    )

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_plot = fig_dir / '30_baseline_vs_densenet_metrics.png'
    map_plot = fig_dir / '30_densenet_gapfilled_map.png'

    plot_metric_comparison(
        baseline_json_path=args.baseline_json,
        densenet_metrics=heldout_metrics,
        out_path=metrics_plot,
    )

    target_map, baseline_map, pred_map, gap_filled_map, lon_2d, lat_2d, selected_time = predict_map(
        model,
        cube,
        input_channels=list(args.input_channels),
        target_channel=args.target_channel,
        patch_size=args.patch_size,
        target_mean=target_mean,
        target_std=target_std,
        map_date=args.map_date,
        row_chunk=args.map_row_chunk,
        infer_batch_size=args.map_batch_size,
    )
    plot_gapfill_maps(
        target_map=target_map,
        baseline_map=baseline_map,
        pred_map=pred_map,
        gap_filled=gap_filled_map,
        lon_2d=lon_2d,
        lat_2d=lat_2d,
        selected_time=selected_time,
        out_path=map_plot,
    )
    eomaps_plot = None
    if not args.no_eomaps:
        eomaps_path = Path(args.eomaps_png)
        if plot_gapfill_eomaps(
            gap_filled=gap_filled_map,
            lon_2d=lon_2d,
            lat_2d=lat_2d,
            selected_time=selected_time,
            out_path=eomaps_path,
        ):
            eomaps_plot = str(eomaps_path)

    baseline_payload = None
    baseline_path = Path(args.baseline_json)
    if baseline_path.exists():
        baseline_payload = json.loads(baseline_path.read_text(encoding='utf-8'))

    report = {
        'config': vars(args),
        'split_sizes': {
            'total': int(n_samples),
            'train': int(train_idx.size),
            'val': int(val_idx.size),
            'test': int(test_idx.size),
        },
        'heldout_test_metrics': {
            'densenet': heldout_metrics,
        },
        'cross_validation': {
            'n_folds': len(cv_folds),
            'fold_metrics': cv_metrics,
            'summary': cv_summary,
        },
        'target_normalization': {
            'mean': float(target_mean),
            'std': float(target_std),
        },
        'baseline_reference': baseline_payload.get('heldout_test_metrics') if isinstance(baseline_payload, dict) else None,
        'artifacts': {
            'model_checkpoint': str(model_path),
            'metrics_plot': str(metrics_plot),
            'gapfill_map_plot': str(map_plot),
            'gapfill_eomaps_plot': eomaps_plot,
            'map_timestamp': selected_time,
        },
        'history': history,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding='utf-8')

    cube.ds.close()

    print('\nDenseNet held-out test metrics:')
    print(f'  RMSE={heldout_metrics["rmse"]:.4f}')
    print(f'  MAE={heldout_metrics["mae"]:.4f}')
    print('\nCross-validation summary:')
    print(f'  RMSE mean/std={cv_summary["rmse_mean"]:.4f}/{cv_summary["rmse_std"]:.4f}')
    print(f'  MAE mean/std={cv_summary["mae_mean"]:.4f}/{cv_summary["mae_std"]:.4f}')
    print(f'\nSaved report: {output_json}')
    print(f'Saved model: {model_path}')
    extra = f' and {eomaps_plot}' if eomaps_plot else ''
    print(f'Saved figures: {metrics_plot} and {map_plot}{extra}')


if __name__ == '__main__':
    main()
