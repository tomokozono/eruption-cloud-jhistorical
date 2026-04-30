"""ERA5 reanalysis loader: NetCDF reader and CDS API downloader."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Optional


_G0 = 9.80665  # standard gravity for geopotential conversion


def load_era5_nc(ncfile: str, lat0: float, lon0: float) -> pd.DataFrame:
    """
    Read an ERA5 pressure-level NetCDF file and return an atmospheric profile.

    Parameters
    ----------
    ncfile : path to NetCDF file
    lat0, lon0 : target latitude/longitude [degrees]

    Returns
    -------
    DataFrame with columns:
        Pressure_hPa, Altitude_m, WindSpeed_mps, WindDirection_deg, Temperature_K
    sorted by Altitude_m ascending.
    """
    ds = xr.open_dataset(ncfile)
    pt = ds.sel(latitude=lat0, longitude=lon0, method="nearest")

    # Drop extra dimensions that ERA5 sometimes includes
    for dim in ("valid_time", "time", "expver", "number"):
        if dim in pt.dims:
            pt = pt.isel({dim: 0}, drop=True)
    pt = pt.squeeze(drop=True)

    plev_name = "pressure_level" if "pressure_level" in pt.coords else "level"

    pressure = np.ravel(pt[plev_name].values).astype(float)
    u = np.ravel(pt["u"].values)
    v = np.ravel(pt["v"].values)
    T = np.ravel(pt["t"].values)
    altitude = np.ravel(pt["z"].values) / _G0  # geopotential [m²/s²] → height [m]

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


def download_era5(
    ncfile: str,
    year: str,
    month: str,
    day: str,
    time: str,
    area: List[float],
    pressure_levels: Optional[List[int]] = None,
) -> None:
    """
    Download ERA5 pressure-level data via the CDS API.

    Parameters
    ----------
    ncfile : output file path
    year, month, day : date strings (e.g. "1962", "06", "29")
    time : hour string (e.g. "18:00")
    area : bounding box [north, west, south, east]
    pressure_levels : list of pressure levels in hPa (default: standard set)
    """
    import cdsapi  # imported lazily; only needed when downloading

    if pressure_levels is None:
        pressure_levels = [
            1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700,
            650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50,
        ]

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "u_component_of_wind",
                "v_component_of_wind",
                "temperature",
                "geopotential",
            ],
            "pressure_level": [str(p) for p in pressure_levels],
            "year": year,
            "month": month,
            "day": day,
            "time": [time],
            "area": area,
            "format": "netcdf",
        },
        ncfile,
    )
