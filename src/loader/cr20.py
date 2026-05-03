"""20CRv3 reanalysis loader via NOAA PSL THREDDS / OPeNDAP."""
from __future__ import annotations

import contextlib
import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta, timezone


_BASE_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/20thC_ReanV3/prsSI"


@contextlib.contextmanager
def _suppress_hdf5_diag():
    """Redirect stderr at OS level to silence HDF5 diagnostic output from netCDF4."""
    old_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)


def snap_to_3h(dt: datetime) -> datetime:
    """Round a datetime to the nearest 3-hour mark (00, 03, 06, ..., 21 UTC)."""
    hh = int(round(dt.hour / 3) * 3) % 24
    return dt.replace(hour=hh, minute=0, second=0, microsecond=0)


def jst_to_utc(dt_jst: datetime) -> datetime:
    """Convert a JST datetime (timezone-naive or tz-aware) to UTC (naive)."""
    if dt_jst.tzinfo is None:
        dt_jst = dt_jst.replace(tzinfo=timezone(timedelta(hours=9)))
    return dt_jst.astimezone(timezone.utc).replace(tzinfo=None)


def load_20cr(lat0: float, lon0: float, t_use: datetime) -> pd.DataFrame:
    """
    Load a single-point atmospheric profile from 20CRv3 via OPeNDAP.

    Parameters
    ----------
    lat0, lon0 : target latitude/longitude [degrees]
    t_use : target datetime (UTC, naive); will be snapped to nearest 3-h mark

    Returns
    -------
    DataFrame with columns:
        Pressure_hPa, Altitude_m, WindSpeed_mps, WindDirection_deg, Temperature_K
    sorted by Altitude_m ascending.
    """
    t_use = snap_to_3h(t_use)
    year = t_use.year

    urls = {
        "u":   f"{_BASE_URL}/uwnd.{year}.nc",
        "v":   f"{_BASE_URL}/vwnd.{year}.nc",
        "air": f"{_BASE_URL}/air.{year}.nc",
        "hgt": f"{_BASE_URL}/hgt.{year}.nc",
    }

    with _suppress_hdf5_diag():
        dsu = xr.open_dataset(urls["u"],   engine="netcdf4")
        dsv = xr.open_dataset(urls["v"],   engine="netcdf4")
        dst = xr.open_dataset(urls["air"], engine="netcdf4")
        dsh = xr.open_dataset(urls["hgt"], engine="netcdf4")

    def _sel(ds):
        da = ds[list(ds.data_vars)[0]]
        return (
            da.sel(time=t_use, method="nearest")
              .sel(lat=lat0, lon=lon0, method="nearest")
        )

    pt_u = _sel(dsu)
    pt_v = _sel(dsv)
    pt_T = _sel(dst)
    pt_h = _sel(dsh)

    plev_name = "level" if "level" in pt_u.coords else list(pt_u.coords)[0]
    pressure = np.ravel(pt_u[plev_name].values).astype(float)
    u = np.ravel(pt_u.values)
    v = np.ravel(pt_v.values)
    T = np.ravel(pt_T.values)
    altitude = np.ravel(pt_h.values)  # 20CRv3 hgt is already in metres

    wind_speed = np.sqrt(u ** 2 + v ** 2)
    wind_dir = (np.degrees(np.arctan2(-u, -v)) + 360) % 360

    df = pd.DataFrame({
        "Pressure_hPa":      pressure,
        "Altitude_m":        altitude,
        "WindSpeed_mps":     wind_speed,
        "WindDirection_deg": wind_dir,
        "Temperature_K":     T,
    }).sort_values("Altitude_m").reset_index(drop=True)

    return df
