"""Atmospheric profile preparation from reanalysis DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Callable, Tuple


def build_profile(
    df: pd.DataFrame,
    vent_height: float,
) -> Tuple[pd.DataFrame, Callable, Callable, Callable]:
    """
    Prepare an atmospheric profile relative to the vent altitude.

    Adds a z_rel_m column (altitude above vent), inserts a z=0 anchor point
    by linear interpolation if one is missing, and builds interpolating
    functions with fixed-endpoint extrapolation.

    Parameters
    ----------
    df : DataFrame with columns
        Altitude_m, WindSpeed_mps, Temperature_K, Pressure_hPa, WindDirection_deg
    vent_height : vent altitude above sea level [m]

    Returns
    -------
    df_rel : DataFrame with z_rel_m added, sorted, deduplicated
    v_func : z_rel [m] → wind speed [m/s]
    tempa_func : z_rel [m] → temperature [K]
    p_func : z_rel [m] → pressure [Pa]
    """
    df = df.copy()
    df["z_rel_m"] = df["Altitude_m"] - vent_height
    df = df.sort_values("z_rel_m").drop_duplicates(subset="z_rel_m").reset_index(drop=True)

    # Insert z=0 anchor by linear interpolation between the two surrounding levels
    if not np.any(np.isclose(df["z_rel_m"].values, 0.0, atol=1e-6)):
        df_neg = df[df["z_rel_m"] < 0]
        df_pos = df[df["z_rel_m"] > 0]
        if len(df_neg) > 0 and len(df_pos) > 0:
            a = df_neg.iloc[df_neg["z_rel_m"].values.argmax()]
            b = df_pos.iloc[df_pos["z_rel_m"].values.argmin()]
            za, zb = float(a["z_rel_m"]), float(b["z_rel_m"])
            w = -za / (zb - za)

            def _lin(xa, xb):
                return float(xa + (xb - xa) * w)

            anchor = {
                "Pressure_hPa":      _lin(a["Pressure_hPa"],      b["Pressure_hPa"]),
                "Altitude_m":        vent_height,
                "WindSpeed_mps":     _lin(a["WindSpeed_mps"],      b["WindSpeed_mps"]),
                "WindDirection_deg": _lin(a["WindDirection_deg"],  b["WindDirection_deg"]),
                "Temperature_K":     _lin(a["Temperature_K"],      b["Temperature_K"]),
                "z_rel_m":           0.0,
            }
            df = (
                pd.concat([df, pd.DataFrame([anchor])], ignore_index=True)
                .sort_values("z_rel_m")
                .reset_index(drop=True)
            )

    z = df["z_rel_m"].values

    v_interp = interp1d(
        z, df["WindSpeed_mps"].values,
        bounds_error=False,
        fill_value=(float(df["WindSpeed_mps"].iloc[0]), float(df["WindSpeed_mps"].iloc[-1])),
    )
    T_interp = interp1d(
        z, df["Temperature_K"].values,
        bounds_error=False,
        fill_value=(float(df["Temperature_K"].iloc[0]), float(df["Temperature_K"].iloc[-1])),
    )
    p_interp = interp1d(
        z, df["Pressure_hPa"].values * 100.0,
        bounds_error=False,
        fill_value=(
            float(df["Pressure_hPa"].iloc[0]) * 100.0,
            float(df["Pressure_hPa"].iloc[-1]) * 100.0,
        ),
    )

    return (
        df,
        lambda z: float(v_interp(z)),
        lambda z: float(T_interp(z)),
        lambda z: float(p_interp(z)),
    )
