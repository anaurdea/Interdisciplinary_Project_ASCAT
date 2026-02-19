from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    input_channels: tuple[int, ...] = (0, 1)
    include_input_std: bool = True
    include_input_range: bool = True
    include_channel_values: bool = False
    include_time_features: bool = True


def build_feature_names(config: FeatureConfig, ancillary_names: Iterable[str] = ()) -> list[str]:
    names = ['input_mean']
    if config.include_input_std:
        names.append('input_std')
    if config.include_input_range:
        names.append('input_range')
    if config.include_channel_values:
        names.extend([f'input_channel_{idx}' for idx in config.input_channels])
    if config.include_time_features:
        names.extend(['doy_sin', 'doy_cos', 'hour_sin', 'hour_cos'])
    names.extend(list(ancillary_names))
    return names


def _safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(finite.mean())


def _safe_std(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(finite.std())


def _safe_range(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(finite.max() - finite.min())


def _timestamp_features(timestamp: np.datetime64) -> list[float]:
    ts = pd.Timestamp(timestamp)
    day_of_year = ts.dayofyear
    hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    doy_angle = 2.0 * np.pi * day_of_year / 366.0
    hour_angle = 2.0 * np.pi * hour / 24.0
    return [np.sin(doy_angle), np.cos(doy_angle), np.sin(hour_angle), np.cos(hour_angle)]


def build_feature_vector(
    channel_values: np.ndarray,
    timestamp: np.datetime64,
    config: FeatureConfig,
    ancillary_values: np.ndarray | None = None,
) -> np.ndarray:
    channels = np.asarray(channel_values, dtype=np.float32)
    selected_inputs = channels[list(config.input_channels)]
    features: list[float] = [_safe_mean(selected_inputs)]

    if config.include_input_std:
        features.append(_safe_std(selected_inputs))
    if config.include_input_range:
        features.append(_safe_range(selected_inputs))
    if config.include_channel_values:
        features.extend([float(v) for v in selected_inputs])
    if config.include_time_features:
        features.extend(_timestamp_features(timestamp))
    if ancillary_values is not None:
        features.extend([float(v) for v in np.asarray(ancillary_values)])

    return np.asarray(features, dtype=np.float32)


@dataclass
class StandardizationStats:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float
    feature_names: list[str] | None = None

    @classmethod
    def from_samples(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> 'StandardizationStats':
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError('Features must be a 2D array [samples, features].')
        if y.ndim != 1:
            raise ValueError('Targets must be a 1D array [samples].')
        if x.shape[0] != y.shape[0]:
            raise ValueError('Feature and target sample counts must match.')

        feature_mean = np.nanmean(x, axis=0).astype(np.float32)
        feature_std = np.nanstd(x, axis=0).astype(np.float32)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)

        target_mean = float(np.nanmean(y))
        target_std = float(np.nanstd(y))
        if target_std < 1e-6:
            target_std = 1.0

        return cls(
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
            feature_names=feature_names,
        )

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        expected = self.feature_mean.shape[0]

        if x.ndim == 1:
            if x.shape[0] != expected:
                raise ValueError(
                    'Feature vector size does not match normalization stats: '
                    f'got {x.shape[0]}, expected {expected}. '
                    'Refit stats with the same feature configuration used at inference.'
                )
            transformed = (x - self.feature_mean) / self.feature_std
            transformed[~np.isfinite(x)] = np.nan
            return transformed.astype(np.float32)

        if x.ndim == 2:
            if x.shape[1] != expected:
                raise ValueError(
                    'Feature matrix size does not match normalization stats: '
                    f'got {x.shape[1]}, expected {expected}. '
                    'Refit stats with the same feature configuration used at inference.'
                )
            transformed = (x - self.feature_mean[None, :]) / self.feature_std[None, :]
            transformed[~np.isfinite(x)] = np.nan
            return transformed.astype(np.float32)

        raise ValueError(f'Expected 1D or 2D features, got shape {x.shape}.')

    def transform_target(self, target: float | np.ndarray) -> np.ndarray:
        y = np.asarray(target, dtype=np.float32)
        transformed = (y - self.target_mean) / self.target_std
        return np.where(np.isfinite(y), transformed, np.nan).astype(np.float32)

    def inverse_transform_target(self, target: float | np.ndarray) -> np.ndarray:
        y = np.asarray(target, dtype=np.float32)
        return (y * self.target_std + self.target_mean).astype(np.float32)

    def to_json(self, path: str | Path) -> None:
        payload = {
            'feature_mean': self.feature_mean.tolist(),
            'feature_std': self.feature_std.tolist(),
            'target_mean': self.target_mean,
            'target_std': self.target_std,
            'feature_names': self.feature_names,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')

    @classmethod
    def from_json(cls, path: str | Path) -> 'StandardizationStats':
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(
            feature_mean=np.asarray(payload['feature_mean'], dtype=np.float32),
            feature_std=np.asarray(payload['feature_std'], dtype=np.float32),
            target_mean=float(payload['target_mean']),
            target_std=float(payload['target_std']),
            feature_names=payload.get('feature_names'),
        )
