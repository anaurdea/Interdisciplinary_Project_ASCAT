from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr

DEFAULT_DATASET_PATH = Path('Dataset/ASCAT_SSM_EASE2_25.nc')
DEFAULT_CHUNKS = {'time': 256, 'lat': 200, 'lon': 200}


def open_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    time_chunk: int = DEFAULT_CHUNKS['time'],
    lat_chunk: int = DEFAULT_CHUNKS['lat'],
    lon_chunk: int = DEFAULT_CHUNKS['lon'],
) -> xr.Dataset:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset not found: {path}')

    ds = xr.open_dataset(path)
    chunks: dict[str, int] = {}

    if 'time' in ds.dims:
        chunks['time'] = time_chunk
    elif 'timestamp' in ds.dims:
        chunks['timestamp'] = time_chunk

    if 'lat' in ds.dims:
        chunks['lat'] = lat_chunk
    elif 'y' in ds.dims:
        lat_coord = ds.coords.get('lat')
        lon_coord = ds.coords.get('lon')
        has_2d_geo = (
            lat_coord is not None
            and lon_coord is not None
            and lat_coord.ndim == 2
            and lon_coord.ndim == 2
        )
        if not has_2d_geo:
            chunks['y'] = lat_chunk

    if 'lon' in ds.dims:
        chunks['lon'] = lon_chunk
    elif 'x' in ds.dims:
        lat_coord = ds.coords.get('lat')
        lon_coord = ds.coords.get('lon')
        has_2d_geo = (
            lat_coord is not None
            and lon_coord is not None
            and lat_coord.ndim == 2
            and lon_coord.ndim == 2
        )
        if not has_2d_geo:
            chunks['x'] = lon_chunk

    return ds.chunk(chunks) if chunks else ds


def infer_primary_variable(ds: xr.Dataset, variable: str | None = None) -> str:
    if variable:
        if variable not in ds.data_vars:
            raise KeyError(
                f'Variable "{variable}" not found. Available: {list(ds.data_vars)}'
            )
        return variable

    preferred_dims = {'time', 'y', 'x'}
    for name, da in ds.data_vars.items():
        if preferred_dims.issubset(set(da.dims)):
            return name

    if not ds.data_vars:
        raise ValueError('Dataset has no data variables.')

    return next(iter(ds.data_vars))


def sensor_labels(ds: xr.Dataset) -> list[str]:
    sensor_dim = infer_sensor_dim_from_dataset(ds)
    if sensor_dim is None:
        return []

    label_coord: xr.DataArray | None = None
    for candidate in ('sensor', 'spacecraft'):
        if candidate in ds.coords and ds[candidate].dims == (sensor_dim,):
            label_coord = ds[candidate]
            break

    if label_coord is None:
        for coord_name, coord in ds.coords.items():
            if coord.dims == (sensor_dim,):
                label_coord = coord
                break

    if label_coord is None:
        return [str(i) for i in range(ds.sizes[sensor_dim])]

    labels: list[str] = []
    for value in label_coord.values:
        if hasattr(value, 'decode'):
            labels.append(value.decode('utf-8'))
        else:
            labels.append(str(value))
    return labels


def format_dims(ds: xr.Dataset) -> str:
    return ', '.join(f'{name}={size}' for name, size in ds.sizes.items())


def missing_fraction_sample(da: xr.DataArray, sample_time_steps: int = 365) -> float:
    sample = da
    time_dim = detect_time_dim(da)
    if time_dim is not None:
        sample = da.isel({time_dim: slice(0, sample_time_steps)})
    return float(sample.isnull().mean().compute())


def reduce_over_dims(da: xr.DataArray, keep_dims: Sequence[str]) -> xr.DataArray:
    dims_to_reduce = [dim for dim in da.dims if dim not in keep_dims]
    if not dims_to_reduce:
        return da
    return da.mean(dim=dims_to_reduce)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def available_sensor_indices(da: xr.DataArray) -> list[int]:
    sensor_dim = detect_sensor_dim(da)
    if sensor_dim is None:
        return []
    return list(range(da.sizes[sensor_dim]))


def validate_sensor_indices(da: xr.DataArray, indices: Iterable[int]) -> list[int]:
    available = set(available_sensor_indices(da))
    clean = [int(i) for i in indices]
    missing = [i for i in clean if i not in available]
    if missing:
        raise IndexError(
            f'Sensor index out of bounds: {missing}. Available: {sorted(available)}'
        )
    return clean


def detect_sensor_dim(da: xr.DataArray) -> str | None:
    for name in ('sensor', 'channel'):
        if name in da.dims:
            return name
    return None


def infer_sensor_dim_from_dataset(ds: xr.Dataset) -> str | None:
    for name in ('sensor', 'channel'):
        if name in ds.dims:
            return name
    return None


def detect_time_dim(da: xr.DataArray) -> str | None:
    for name in ('time', 'timestamp'):
        if name in da.dims:
            return name
    return None


def detect_time_coord(ds: xr.Dataset, time_dim: str | None) -> str | None:
    if time_dim is None:
        return None
    if time_dim in ds.coords and ds[time_dim].dims == (time_dim,):
        return time_dim
    for name in ('timestamp', 'time'):
        if name in ds.coords and ds[name].dims == (time_dim,):
            return name
    for name, coord in ds.coords.items():
        if coord.dims == (time_dim,) and np.issubdtype(coord.dtype, np.datetime64):
            return name
    return None


def with_time_coordinate(
    da: xr.DataArray,
    ds: xr.Dataset,
) -> tuple[xr.DataArray, str | None, str | None]:
    time_dim = detect_time_dim(da)
    if time_dim is None:
        return da, None, None

    time_coord = detect_time_coord(ds, time_dim)
    if time_coord and time_coord != time_dim:
        da = da.assign_coords({time_dim: ds[time_coord]})
    return da, time_dim, time_coord


def detect_spatial_dims(da: xr.DataArray) -> tuple[str | None, str | None]:
    if 'lat' in da.dims and 'lon' in da.dims:
        return 'lat', 'lon'
    if 'y' in da.dims and 'x' in da.dims:
        return 'y', 'x'
    if 'lat' in da.coords and da['lat'].ndim == 2:
        y_dim, x_dim = da['lat'].dims
        return y_dim, x_dim
    return None, None


def map_plot_axes(da: xr.DataArray) -> tuple[str | None, str | None]:
    y_dim, x_dim = detect_spatial_dims(da)
    if y_dim is None or x_dim is None:
        return None, None

    if 'lon' in da.coords and 'lat' in da.coords:
        lon = da['lon']
        lat = da['lat']
        if lon.ndim == 2 and lat.ndim == 2 and lon.dims == (y_dim, x_dim) and lat.dims == (y_dim, x_dim):
            return 'lon', 'lat'
        if lon.ndim == 1 and lat.ndim == 1:
            return 'lon', 'lat'

    return x_dim, y_dim


def nearest_grid_index(da: xr.DataArray, lat_value: float, lon_value: float) -> dict[str, int]:
    if 'lat' not in da.coords or 'lon' not in da.coords:
        raise ValueError('DataArray does not include lat/lon coordinates.')

    lat_coord = da['lat']
    lon_coord = da['lon']

    if lat_coord.ndim == 2 and lon_coord.ndim == 2:
        y_dim, x_dim = lat_coord.dims
        lat_arr = lat_coord.values
        lon_arr = lon_coord.values
        distance = (lat_arr - lat_value) ** 2 + (lon_arr - lon_value) ** 2
        distance = np.where(np.isfinite(distance), distance, np.inf)
        if not np.isfinite(distance).any():
            raise ValueError('No finite lat/lon values found in dataset.')
        flat_index = int(np.argmin(distance))
        iy, ix = np.unravel_index(flat_index, distance.shape)
        return {y_dim: int(iy), x_dim: int(ix)}

    if lat_coord.ndim == 1 and lon_coord.ndim == 1:
        lat_idx = int(np.argmin(np.abs(lat_coord.values - lat_value)))
        lon_idx = int(np.argmin(np.abs(lon_coord.values - lon_value)))
        return {lat_coord.dims[0]: lat_idx, lon_coord.dims[0]: lon_idx}

    raise ValueError('Unsupported lat/lon coordinate shape for nearest point selection.')


def select_nearest_time_slice(
    da: xr.DataArray,
    ds: xr.Dataset,
    target_time: str,
) -> tuple[xr.DataArray, np.datetime64]:
    da_with_time, time_dim, _ = with_time_coordinate(da, ds)
    if time_dim is None:
        raise ValueError('DataArray has no time-like dimension.')

    coord_values = np.asarray(da_with_time[time_dim].values)
    if coord_values.ndim != 1 or coord_values.size == 0:
        raise ValueError('Time coordinate is empty or not one-dimensional.')

    target = np.datetime64(target_time)
    diffs = np.abs(coord_values - target)
    index = int(np.argmin(diffs))
    return da_with_time.isel({time_dim: index}), coord_values[index]


def subset_by_bbox(
    da: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.DataArray:
    if 'lat' not in da.coords or 'lon' not in da.coords:
        raise ValueError('DataArray does not include lat/lon coordinates.')

    lat_coord = da['lat']
    lon_coord = da['lon']

    lat_lo = min(lat_min, lat_max)
    lat_hi = max(lat_min, lat_max)
    lon_lo = min(lon_min, lon_max)
    lon_hi = max(lon_min, lon_max)

    if lat_coord.ndim == 2 and lon_coord.ndim == 2:
        mask = (
            (lat_coord.values >= lat_lo)
            & (lat_coord.values <= lat_hi)
            & (lon_coord.values >= lon_lo)
            & (lon_coord.values <= lon_hi)
        )
        if not mask.any():
            raise ValueError('Bounding box did not match any grid cells.')
        yy, xx = np.where(mask)
        y_dim, x_dim = lat_coord.dims
        return da.isel(
            {
                y_dim: slice(int(yy.min()), int(yy.max()) + 1),
                x_dim: slice(int(xx.min()), int(xx.max()) + 1),
            }
        )

    if lat_coord.ndim == 1 and lon_coord.ndim == 1:
        lat_vals = lat_coord.values
        lon_vals = lon_coord.values
        lat_mask = (lat_vals >= lat_lo) & (lat_vals <= lat_hi)
        lon_mask = (lon_vals >= lon_lo) & (lon_vals <= lon_hi)
        if not lat_mask.any() or not lon_mask.any():
            raise ValueError('Bounding box did not match any grid cells.')
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        return da.isel(
            {
                lat_coord.dims[0]: slice(int(lat_idx.min()), int(lat_idx.max()) + 1),
                lon_coord.dims[0]: slice(int(lon_idx.min()), int(lon_idx.max()) + 1),
            }
        )

    raise ValueError('Unsupported lat/lon coordinate shape for bounding-box subset.')

