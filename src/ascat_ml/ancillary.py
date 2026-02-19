from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class AncillaryProvider(Protocol):
    def feature_names(self) -> list[str]:
        ...

    def get_features(self, time_index: int, y_index: int, x_index: int) -> np.ndarray:
        ...


@dataclass
class EmptyAncillaryProvider:
    def feature_names(self) -> list[str]:
        return []

    def get_features(self, time_index: int, y_index: int, x_index: int) -> np.ndarray:
        return np.empty((0,), dtype=np.float32)


class NPZAncillaryProvider:
    def __init__(self, arrays: dict[str, np.ndarray], feature_order: list[str] | None = None) -> None:
        if not arrays:
            raise ValueError('No arrays provided for ancillary data.')
        self._arrays = arrays
        self._order = feature_order or sorted(arrays.keys())
        for key in self._order:
            if key not in arrays:
                raise KeyError(f'Feature "{key}" was not found in ancillary arrays.')
            if arrays[key].ndim not in (2, 3):
                raise ValueError(f'Ancillary feature "{key}" must be 2D [y,x] or 3D [time,y,x].')

    @classmethod
    def from_npz(cls, npz_path: str | Path, feature_order: list[str] | None = None) -> 'NPZAncillaryProvider':
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f'Ancillary NPZ file not found: {path}')
        loaded = np.load(path, allow_pickle=False)
        arrays = {name: loaded[name].astype(np.float32, copy=False) for name in loaded.files}
        return cls(arrays=arrays, feature_order=feature_order)

    def feature_names(self) -> list[str]:
        return list(self._order)

    def get_features(self, time_index: int, y_index: int, x_index: int) -> np.ndarray:
        values: list[float] = []
        for key in self._order:
            arr = self._arrays[key]
            if arr.ndim == 2:
                values.append(float(arr[y_index, x_index]))
            else:
                values.append(float(arr[time_index, y_index, x_index]))
        return np.asarray(values, dtype=np.float32)
