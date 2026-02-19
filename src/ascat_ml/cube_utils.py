from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def open_dataset(
    dataset_path: str | Path,
    *,
    time_chunk: int = 256,
    lat_chunk: int = 200,
    lon_chunk: int = 200,
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
            raise KeyError(f'Variable "{variable}" not found. Available: {list(ds.data_vars)}')
        return variable

    preferred_dims = {'time', 'y', 'x'}
    for name, da in ds.data_vars.items():
        if preferred_dims.issubset(set(da.dims)):
            return name

    if not ds.data_vars:
        raise ValueError('Dataset has no data variables.')

    return next(iter(ds.data_vars))


def detect_sensor_dim(da: xr.DataArray) -> str | None:
    for name in ('sensor', 'channel'):
        if name in da.dims:
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
