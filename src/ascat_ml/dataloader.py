from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .ancillary import AncillaryProvider, EmptyAncillaryProvider
from .cube_utils import (
    detect_sensor_dim,
    detect_spatial_dims,
    infer_primary_variable,
    open_dataset,
    with_time_coordinate,
)
from .preprocessing import FeatureConfig, StandardizationStats, build_feature_names, build_feature_vector


class ASCATSampleDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        index_path: str | Path,
        *,
        variable: str | None = None,
        feature_config: FeatureConfig | None = None,
        target_channel: int = 2,
        normalization: StandardizationStats | None = None,
        ancillary_provider: AncillaryProvider | None = None,
        fill_value: float = 0.0,
        return_metadata: bool = False,
        time_chunk: int = 256,
        lat_chunk: int = 200,
        lon_chunk: int = 200,
    ) -> None:
        self.dataset_path = str(dataset_path)
        self.index_path = str(index_path)
        self.variable = variable
        self.feature_config = feature_config or FeatureConfig()
        self.target_channel = int(target_channel)
        self.normalization = normalization
        self.ancillary_provider = ancillary_provider or EmptyAncillaryProvider()
        self.fill_value = float(fill_value)
        self.return_metadata = bool(return_metadata)
        self.time_chunk = int(time_chunk)
        self.lat_chunk = int(lat_chunk)
        self.lon_chunk = int(lon_chunk)

        self.indices = np.load(self.index_path)
        if self.indices.ndim != 2 or self.indices.shape[1] != 3:
            raise ValueError('Index file must have shape [N, 3] with columns [time, y, x].')
        self.indices = self.indices.astype(np.int64, copy=False)

        self._ds = None
        self._var = None
        self._sensor_dim: str | None = None
        self._time_dim: str | None = None
        self._y_dim: str | None = None
        self._x_dim: str | None = None
        self._time_values: np.ndarray | None = None

        self._feature_names = build_feature_names(
            self.feature_config,
            ancillary_names=self.ancillary_provider.feature_names(),
        )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def _ensure_open(self) -> None:
        if self._var is not None:
            return

        ds = open_dataset(
            self.dataset_path,
            time_chunk=self.time_chunk,
            lat_chunk=self.lat_chunk,
            lon_chunk=self.lon_chunk,
        )
        var_name = infer_primary_variable(ds, self.variable)
        da = ds[var_name]
        da, time_dim, _ = with_time_coordinate(da, ds)

        sensor_dim = detect_sensor_dim(da)
        if sensor_dim is None:
            raise ValueError('Data variable has no sensor/channel dimension.')
        y_dim, x_dim = detect_spatial_dims(da)
        if time_dim is None or y_dim is None or x_dim is None:
            raise ValueError('Could not detect time/y/x dimensions for the selected variable.')

        max_input_channel = max(self.feature_config.input_channels)
        max_channel = max(max_input_channel, self.target_channel)
        if max_channel >= da.sizes[sensor_dim]:
            raise IndexError(
                f'Channel index {max_channel} is out of bounds for dimension "{sensor_dim}" '
                f'with size {da.sizes[sensor_dim]}.'
            )

        self._ds = ds
        self._var = da
        self._sensor_dim = sensor_dim
        self._time_dim = time_dim
        self._y_dim = y_dim
        self._x_dim = x_dim
        self._time_values = np.asarray(da[time_dim].values)

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
        self._ds = None
        self._var = None
        self._time_values = None

    def __del__(self) -> None:
        self.close()

    def get_raw_sample(self, position: int) -> tuple[np.ndarray, float, tuple[int, int, int]]:
        self._ensure_open()
        assert self._var is not None
        assert self._time_dim is not None
        assert self._y_dim is not None
        assert self._x_dim is not None
        assert self._time_values is not None
        assert self._sensor_dim is not None

        t_idx, y_idx, x_idx = [int(v) for v in self.indices[position]]
        pixel = self._var.isel(
            {
                self._time_dim: t_idx,
                self._y_dim: y_idx,
                self._x_dim: x_idx,
            }
        )
        channel_values = np.asarray(pixel.values, dtype=np.float32)
        if channel_values.ndim != 1:
            channel_values = np.asarray(channel_values).reshape(-1).astype(np.float32)

        timestamp = np.datetime64(self._time_values[t_idx])
        ancillary_values = self.ancillary_provider.get_features(t_idx, y_idx, x_idx)
        features = build_feature_vector(
            channel_values=channel_values,
            timestamp=timestamp,
            config=self.feature_config,
            ancillary_values=ancillary_values,
        )
        target = float(channel_values[self.target_channel])
        return features, target, (t_idx, y_idx, x_idx)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor | dict[str, int | str]]:
        raw_features, raw_target, (t_idx, y_idx, x_idx) = self.get_raw_sample(position)

        if self.normalization is not None:
            features = self.normalization.transform_features(raw_features)
            target_value = float(self.normalization.transform_target(raw_target))
        else:
            features = raw_features.astype(np.float32, copy=True)
            target_value = float(raw_target)

        x_mask = np.isfinite(features).astype(np.float32)
        y_mask = float(np.isfinite(target_value))

        features = np.where(np.isfinite(features), features, self.fill_value).astype(np.float32)
        if not np.isfinite(target_value):
            target_value = self.fill_value

        sample: dict[str, torch.Tensor | dict[str, int | str]] = {
            'x': torch.from_numpy(features),
            'y': torch.tensor(target_value, dtype=torch.float32),
            'x_mask': torch.from_numpy(x_mask),
            'y_mask': torch.tensor(y_mask, dtype=torch.float32),
        }

        if self.return_metadata:
            assert self._time_values is not None
            sample['meta'] = {
                'time_index': int(t_idx),
                'y_index': int(y_idx),
                'x_index': int(x_idx),
                'timestamp': str(np.datetime_as_string(self._time_values[t_idx], unit='s')),
            }
        return sample


def fit_standardization_stats(
    dataset: ASCATSampleDataset,
    *,
    sample_size: int = 50000,
    seed: int = 42,
) -> StandardizationStats:
    n_total = len(dataset)
    if n_total == 0:
        raise ValueError('Cannot fit normalization stats: dataset index is empty.')

    n_sample = min(int(sample_size), n_total)
    rng = np.random.default_rng(seed)
    positions = rng.choice(n_total, size=n_sample, replace=False)

    feature_rows: list[np.ndarray] = []
    target_rows: list[float] = []
    for pos in positions:
        x, y, _ = dataset.get_raw_sample(int(pos))
        feature_rows.append(x)
        target_rows.append(y)

    x_matrix = np.vstack(feature_rows).astype(np.float32)
    y_vector = np.asarray(target_rows, dtype=np.float32)
    valid_target = np.isfinite(y_vector)
    if not valid_target.any():
        raise ValueError('No finite targets were found for normalization fitting.')

    return StandardizationStats.from_samples(
        features=x_matrix[valid_target],
        targets=y_vector[valid_target],
        feature_names=dataset.feature_names,
    )


def create_dataloader(
    dataset: ASCATSampleDataset,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
