from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ascat_ml.dataloader import ASCATSampleDataset  # noqa: E402
from ascat_ml.preprocessing import FeatureConfig, StandardizationStats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run interpolation and PyTorch baselines with CV and held-out test metrics.'
    )
    parser.add_argument('--dataset', default='Dataset/ASCAT_SSM_EASE2_25.nc')
    parser.add_argument('--index', default='outputs/reports/train_index.npy')
    parser.add_argument('--output-json', default='outputs/reports/20_baseline_results.json')
    parser.add_argument('--variable', default=None)
    parser.add_argument('--input-channels', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--target-channel', type=int, default=2)
    parser.add_argument('--include-input-std', action='store_true')
    parser.add_argument('--include-input-range', action='store_true')
    parser.add_argument('--include-channel-values', action='store_true')
    parser.add_argument('--no-time-features', action='store_true')
    parser.add_argument('--split-mode', choices=['time', 'random'], default='time')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--cv-folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--idw-power', type=float, default=2.0)
    parser.add_argument('--idw-k', type=int, default=16)
    parser.add_argument('--run-kriging', action='store_true')
    parser.add_argument('--kriging-min-points', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=0, help='Optional cap for development runs.')
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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {'rmse': rmse(y_true, y_pred), 'mae': mae(y_true, y_pred)}


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


@dataclass
class SampleTable:
    x: np.ndarray
    y: np.ndarray
    time_idx: np.ndarray
    y_idx: np.ndarray
    x_idx: np.ndarray
    feature_names: list[str]

    def subset(self, indices: np.ndarray) -> 'SampleTable':
        return SampleTable(
            x=self.x[indices],
            y=self.y[indices],
            time_idx=self.time_idx[indices],
            y_idx=self.y_idx[indices],
            x_idx=self.x_idx[indices],
            feature_names=list(self.feature_names),
        )

    @property
    def size(self) -> int:
        return int(self.y.shape[0])


def load_sample_table(
    args: argparse.Namespace,
    feature_config: FeatureConfig,
) -> SampleTable:
    ds = ASCATSampleDataset(
        dataset_path=args.dataset,
        index_path=args.index,
        variable=args.variable,
        feature_config=feature_config,
        target_channel=args.target_channel,
        normalization=None,
        return_metadata=False,
        time_chunk=args.time_chunk,
        lat_chunk=args.lat_chunk,
        lon_chunk=args.lon_chunk,
    )

    n_total = len(ds)
    if args.max_samples > 0:
        n_total = min(n_total, int(args.max_samples))

    if n_total == 0:
        raise ValueError('Index is empty. Build a training index with at least one valid sample.')

    x = np.empty((n_total, len(ds.feature_names)), dtype=np.float32)
    y = np.empty((n_total,), dtype=np.float32)
    t_idx = np.empty((n_total,), dtype=np.int32)
    yi_idx = np.empty((n_total,), dtype=np.int32)
    xi_idx = np.empty((n_total,), dtype=np.int32)

    for i in range(n_total):
        x_i, y_i, (t_i, y_i_idx, x_i_idx) = ds.get_raw_sample(i)
        x[i] = x_i
        y[i] = y_i
        t_idx[i] = int(t_i)
        yi_idx[i] = int(y_i_idx)
        xi_idx[i] = int(x_i_idx)
        if (i + 1) % 1000 == 0 or i + 1 == n_total:
            print(f'Loaded samples: {i + 1}/{n_total}', flush=True)

    ds.close()
    finite_target = np.isfinite(y)
    if not finite_target.any():
        raise ValueError('No finite target values were found in the selected index.')

    x = x[finite_target]
    y = y[finite_target]
    t_idx = t_idx[finite_target]
    yi_idx = yi_idx[finite_target]
    xi_idx = xi_idx[finite_target]

    return SampleTable(
        x=x,
        y=y,
        time_idx=t_idx,
        y_idx=yi_idx,
        x_idx=xi_idx,
        feature_names=ds.feature_names,
    )


def _ensure_non_empty_split(split: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx, val_idx, test_idx = split
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            'Train/val/test split produced an empty partition. '
            'Adjust ratios or provide more indexed samples.'
        )
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

    train_mask = np.isin(time_idx, train_times)
    val_mask = np.isin(time_idx, val_times)
    test_mask = np.isin(time_idx, test_times)
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
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


def preprocess_features_and_target(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray | None,
    *,
    add_mask_features: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, StandardizationStats]:
    stats = StandardizationStats.from_samples(x_train, y_train)

    x_train_norm = stats.transform_features(x_train)
    x_eval_norm = stats.transform_features(x_eval)
    x_train_mask = np.isfinite(x_train_norm).astype(np.float32)
    x_eval_mask = np.isfinite(x_eval_norm).astype(np.float32)

    x_train_norm = np.where(np.isfinite(x_train_norm), x_train_norm, 0.0).astype(np.float32)
    x_eval_norm = np.where(np.isfinite(x_eval_norm), x_eval_norm, 0.0).astype(np.float32)

    if add_mask_features:
        x_train_out = np.concatenate([x_train_norm, x_train_mask], axis=1).astype(np.float32)
        x_eval_out = np.concatenate([x_eval_norm, x_eval_mask], axis=1).astype(np.float32)
    else:
        x_train_out = x_train_norm
        x_eval_out = x_eval_norm

    y_train_norm = stats.transform_target(y_train).astype(np.float32)
    y_eval_norm = stats.transform_target(y_eval).astype(np.float32) if y_eval is not None else None

    return x_train_out, y_train_norm, x_eval_out, y_eval_norm, stats


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError('depth must be >= 1')

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    tensor_ds = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(
        tensor_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def train_torch_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dim: int,
    depth: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    grad_clip: float,
    num_workers: int,
    seed: int,
) -> tuple[nn.Module, dict[str, list[float]]]:
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPRegressor(
        input_dim=x_train.shape[1],
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    ).to(device)

    train_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(x_val, y_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=max(2, patience // 3),
    )

    history: dict[str, list[float]] = {'train_loss': [], 'val_loss': []}
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float('inf')
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                val_loss = criterion(pred, yb)
                val_losses.append(float(val_loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f'epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}', flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_torch(model: nn.Module, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            chunk = torch.from_numpy(x[start:start + batch_size].astype(np.float32)).to(device)
            pred = model(chunk).detach().cpu().numpy().astype(np.float32)
            outputs.append(pred)
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)


def build_temporal_index(train_table: SampleTable) -> tuple[dict[tuple[int, int], tuple[np.ndarray, np.ndarray]], float]:
    global_mean = float(np.nanmean(train_table.y))
    keys = np.column_stack([train_table.y_idx, train_table.x_idx]).astype(np.int32)
    groups: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for i in range(train_table.size):
        key = (int(keys[i, 0]), int(keys[i, 1]))
        groups.setdefault(key, []).append((int(train_table.time_idx[i]), float(train_table.y[i])))

    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, values in groups.items():
        arr = np.asarray(values, dtype=np.float64)
        order = np.argsort(arr[:, 0])
        t_sorted = arr[order, 0].astype(np.int64)
        y_sorted = arr[order, 1].astype(np.float64)
        # Collapse duplicate timestamps by averaging.
        uniq_t, inv = np.unique(t_sorted, return_inverse=True)
        sums = np.zeros_like(uniq_t, dtype=np.float64)
        counts = np.zeros_like(uniq_t, dtype=np.float64)
        np.add.at(sums, inv, y_sorted)
        np.add.at(counts, inv, 1.0)
        out[key] = (uniq_t, sums / np.maximum(counts, 1.0))
    return out, global_mean


def predict_temporal_linear(
    eval_table: SampleTable,
    temporal_index: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    global_mean: float,
) -> np.ndarray:
    preds = np.empty((eval_table.size,), dtype=np.float32)
    for i in range(eval_table.size):
        key = (int(eval_table.y_idx[i]), int(eval_table.x_idx[i]))
        t = int(eval_table.time_idx[i])
        if key not in temporal_index:
            preds[i] = global_mean
            continue

        times, values = temporal_index[key]
        j = int(np.searchsorted(times, t))
        if j < times.size and times[j] == t:
            preds[i] = float(values[j])
        elif j == 0:
            preds[i] = float(values[0])
        elif j >= times.size:
            preds[i] = float(values[-1])
        else:
            t0 = float(times[j - 1])
            t1 = float(times[j])
            v0 = float(values[j - 1])
            v1 = float(values[j])
            if t1 == t0:
                preds[i] = v0
            else:
                alpha = (t - t0) / (t1 - t0)
                preds[i] = float(v0 + alpha * (v1 - v0))
    return preds


def build_time_cloud(train_table: SampleTable) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], float]:
    global_mean = float(np.nanmean(train_table.y))
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    unique_times = np.unique(train_table.time_idx)
    for t in unique_times:
        mask = train_table.time_idx == t
        out[int(t)] = (
            train_table.y_idx[mask].astype(np.float64),
            train_table.x_idx[mask].astype(np.float64),
            train_table.y[mask].astype(np.float64),
        )
    return out, global_mean


def _predict_idw_point(
    y: float,
    x: float,
    ys: np.ndarray,
    xs: np.ndarray,
    vals: np.ndarray,
    *,
    power: float,
    k: int,
) -> float:
    if vals.size == 0:
        return float('nan')
    dy = ys - y
    dx = xs - x
    d2 = dy * dy + dx * dx
    zero_mask = d2 == 0.0
    if zero_mask.any():
        return float(np.mean(vals[zero_mask]))
    if k > 0 and vals.size > k:
        idx = np.argpartition(d2, k)[:k]
        d2 = d2[idx]
        vals = vals[idx]
    weights = 1.0 / (np.power(d2, power * 0.5) + 1e-12)
    return float(np.sum(weights * vals) / np.sum(weights))


def predict_spatial_idw(
    eval_table: SampleTable,
    time_cloud: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    global_mean: float,
    *,
    power: float,
    k: int,
) -> np.ndarray:
    preds = np.empty((eval_table.size,), dtype=np.float32)
    for i in range(eval_table.size):
        t = int(eval_table.time_idx[i])
        if t not in time_cloud:
            preds[i] = global_mean
            continue
        ys, xs, vals = time_cloud[t]
        pred = _predict_idw_point(
            y=float(eval_table.y_idx[i]),
            x=float(eval_table.x_idx[i]),
            ys=ys,
            xs=xs,
            vals=vals,
            power=power,
            k=k,
        )
        preds[i] = float(pred if np.isfinite(pred) else global_mean)
    return preds


def predict_spatial_kriging(
    eval_table: SampleTable,
    time_cloud: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    global_mean: float,
    *,
    idw_power: float,
    idw_k: int,
    min_points: int,
) -> tuple[np.ndarray, dict[str, str | int]]:
    try:
        from pykrige.ok import OrdinaryKriging  # type: ignore
    except Exception:
        preds = predict_spatial_idw(
            eval_table,
            time_cloud,
            global_mean,
            power=idw_power,
            k=idw_k,
        )
        return preds, {'status': 'fallback_idw', 'reason': 'pykrige_not_installed'}

    cache: dict[int, object | None] = {}
    preds = np.empty((eval_table.size,), dtype=np.float32)
    built_models = 0
    for i in range(eval_table.size):
        t = int(eval_table.time_idx[i])
        if t not in cache:
            if t not in time_cloud:
                cache[t] = None
            else:
                ys, xs, vals = time_cloud[t]
                if vals.size < min_points:
                    cache[t] = None
                else:
                    try:
                        cache[t] = OrdinaryKriging(
                            xs,
                            ys,
                            vals,
                            variogram_model='linear',
                            verbose=False,
                            enable_plotting=False,
                        )
                        built_models += 1
                    except Exception:
                        cache[t] = None

        model = cache[t]
        if model is None:
            if t in time_cloud:
                ys, xs, vals = time_cloud[t]
                pred = _predict_idw_point(
                    y=float(eval_table.y_idx[i]),
                    x=float(eval_table.x_idx[i]),
                    ys=ys,
                    xs=xs,
                    vals=vals,
                    power=idw_power,
                    k=idw_k,
                )
                preds[i] = float(pred if np.isfinite(pred) else global_mean)
            else:
                preds[i] = global_mean
            continue

        try:
            z, _ = model.execute(
                'points',
                np.asarray([float(eval_table.x_idx[i])]),
                np.asarray([float(eval_table.y_idx[i])]),
            )
            preds[i] = float(np.asarray(z)[0])
        except Exception:
            ys, xs, vals = time_cloud[t]
            pred = _predict_idw_point(
                y=float(eval_table.y_idx[i]),
                x=float(eval_table.x_idx[i]),
                ys=ys,
                xs=xs,
                vals=vals,
                power=idw_power,
                k=idw_k,
            )
            preds[i] = float(pred if np.isfinite(pred) else global_mean)

    return preds, {'status': 'ok', 'models_built': built_models}


def run_torch_baseline(
    train_table: SampleTable,
    val_table: SampleTable,
    test_table: SampleTable,
    *,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    x_train_proc, y_train_proc, x_val_proc, y_val_proc, stats = preprocess_features_and_target(
        train_table.x,
        train_table.y,
        val_table.x,
        val_table.y,
        add_mask_features=True,
    )
    x_test_proc = stats.transform_features(test_table.x)
    x_test_mask = np.isfinite(x_test_proc).astype(np.float32)
    x_test_proc = np.where(np.isfinite(x_test_proc), x_test_proc, 0.0).astype(np.float32)
    x_test_proc = np.concatenate([x_test_proc, x_test_mask], axis=1).astype(np.float32)

    model, history = train_torch_regressor(
        x_train=x_train_proc,
        y_train=y_train_proc,
        x_val=x_val_proc,
        y_val=y_val_proc,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        seed=seed,
    )

    pred_test_norm = predict_torch(model, x_test_proc)
    pred_test = stats.inverse_transform_target(pred_test_norm).astype(np.float32)
    metrics = metric_dict(test_table.y.astype(np.float32), pred_test)
    return metrics, history


def run_cross_validation(
    table: SampleTable,
    dev_indices: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    folds = group_kfold_indices(
        sample_indices=dev_indices,
        groups=table.time_idx,
        n_splits=args.cv_folds,
        seed=args.seed,
    )
    if not folds:
        return {'n_folds': 0, 'temporal_linear': [], 'spatial_idw': [], 'torch_mlp': []}

    cv_temporal: list[dict[str, float]] = []
    cv_idw: list[dict[str, float]] = []
    cv_torch: list[dict[str, float]] = []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        train_t = table.subset(train_idx)
        val_t = table.subset(val_idx)

        temporal_index, global_mean = build_temporal_index(train_t)
        pred_lin = predict_temporal_linear(val_t, temporal_index, global_mean=global_mean)
        cv_temporal.append(metric_dict(val_t.y, pred_lin))

        time_cloud, global_mean_spatial = build_time_cloud(train_t)
        pred_idw = predict_spatial_idw(
            val_t,
            time_cloud,
            global_mean=global_mean_spatial,
            power=args.idw_power,
            k=args.idw_k,
        )
        cv_idw.append(metric_dict(val_t.y, pred_idw))

        torch_metrics, _ = run_torch_baseline(
            train_table=train_t,
            val_table=val_t,
            test_table=val_t,
            args=args,
            seed=args.seed + fold_i + 1,
        )
        cv_torch.append(torch_metrics)

    return {
        'n_folds': len(folds),
        'temporal_linear': cv_temporal,
        'temporal_linear_summary': summarize_metric_list(cv_temporal),
        'spatial_idw': cv_idw,
        'spatial_idw_summary': summarize_metric_list(cv_idw),
        'torch_mlp': cv_torch,
        'torch_mlp_summary': summarize_metric_list(cv_torch),
    }


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError('train_ratio and val_ratio must be > 0 and sum to < 1.')
    if args.cv_folds < 2:
        raise ValueError('--cv-folds must be >= 2')

    set_seed(args.seed)

    feature_config = FeatureConfig(
        input_channels=tuple(args.input_channels),
        include_input_std=bool(args.include_input_std),
        include_input_range=bool(args.include_input_range),
        include_channel_values=bool(args.include_channel_values),
        include_time_features=not args.no_time_features,
    )
    table = load_sample_table(args, feature_config)
    n_samples = table.size
    print(f'Total usable samples: {n_samples}', flush=True)
    print(f'Feature names: {table.feature_names}', flush=True)

    if args.split_mode == 'time':
        train_idx, val_idx, test_idx = split_indices_by_time(
            table.time_idx,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    else:
        train_idx, val_idx, test_idx = split_indices_random(
            n_samples=n_samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    train_table = table.subset(train_idx)
    val_table = table.subset(val_idx)
    test_table = table.subset(test_idx)

    print(
        f'Split sizes -> train={train_table.size}, val={val_table.size}, test={test_table.size}',
        flush=True,
    )

    # Interpolation baselines on held-out test.
    temporal_index, global_mean_temporal = build_temporal_index(train_table)
    pred_temporal = predict_temporal_linear(test_table, temporal_index, global_mean_temporal)
    temporal_metrics = metric_dict(test_table.y, pred_temporal)

    time_cloud, global_mean_spatial = build_time_cloud(train_table)
    pred_idw = predict_spatial_idw(
        test_table,
        time_cloud,
        global_mean_spatial,
        power=args.idw_power,
        k=args.idw_k,
    )
    idw_metrics = metric_dict(test_table.y, pred_idw)

    kriging_metrics: dict[str, float] | None = None
    kriging_info: dict[str, str | int] = {'status': 'skipped'}
    if args.run_kriging:
        pred_kriging, kriging_info = predict_spatial_kriging(
            test_table,
            time_cloud,
            global_mean_spatial,
            idw_power=args.idw_power,
            idw_k=args.idw_k,
            min_points=args.kriging_min_points,
        )
        kriging_metrics = metric_dict(test_table.y, pred_kriging)

    # PyTorch supervised baseline.
    torch_test_metrics, torch_history = run_torch_baseline(
        train_table=train_table,
        val_table=val_table,
        test_table=test_table,
        args=args,
        seed=args.seed,
    )

    # Cross-validation on development (train + val).
    dev_idx = np.concatenate([train_idx, val_idx], axis=0)
    cv_results = run_cross_validation(table, dev_indices=dev_idx, args=args)

    report = {
        'config': {
            'dataset': str(Path(args.dataset).resolve()),
            'index': str(Path(args.index).resolve()),
            'seed': args.seed,
            'split_mode': args.split_mode,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': 1.0 - args.train_ratio - args.val_ratio,
            'feature_config': {
                'input_channels': list(feature_config.input_channels),
                'include_input_std': feature_config.include_input_std,
                'include_input_range': feature_config.include_input_range,
                'include_channel_values': feature_config.include_channel_values,
                'include_time_features': feature_config.include_time_features,
            },
        },
        'split_sizes': {
            'total': n_samples,
            'train': int(train_table.size),
            'val': int(val_table.size),
            'test': int(test_table.size),
        },
        'heldout_test_metrics': {
            'temporal_linear': temporal_metrics,
            'spatial_idw': idw_metrics,
            'spatial_kriging': kriging_metrics,
            'torch_mlp': torch_test_metrics,
        },
        'kriging_info': kriging_info,
        'cross_validation': cv_results,
        'torch_history': torch_history,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print('\nHeld-out test metrics:')
    print(f'  temporal_linear: RMSE={temporal_metrics["rmse"]:.4f}, MAE={temporal_metrics["mae"]:.4f}')
    print(f'  spatial_idw: RMSE={idw_metrics["rmse"]:.4f}, MAE={idw_metrics["mae"]:.4f}')
    if kriging_metrics is not None:
        print(f'  spatial_kriging: RMSE={kriging_metrics["rmse"]:.4f}, MAE={kriging_metrics["mae"]:.4f}')
    else:
        print(f'  spatial_kriging: {kriging_info.get("status", "not_run")}')
    print(f'  torch_mlp: RMSE={torch_test_metrics["rmse"]:.4f}, MAE={torch_test_metrics["mae"]:.4f}')
    print(f'\nSaved report: {output_json}')


if __name__ == '__main__':
    main()
