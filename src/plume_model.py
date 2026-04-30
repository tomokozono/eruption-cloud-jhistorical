"""Woodhouse et al. (2013) 1-D integral eruption cloud model."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class PlumeParams:
    """Eruption source and model parameters (Woodhouse 2013 defaults)."""
    T0: float = 1273.0      # vent temperature [K]
    n0: float = 0.03        # initial water vapour mass fraction
    theta0: float = np.pi / 2  # launch angle [rad] (π/2 = vertical)
    ke: float = 0.06        # radial entrainment coefficient
    kw: float = 0.20        # wind entrainment coefficient
    g: float = 9.8          # gravitational acceleration [m/s²]
    rga: float = 285.0      # gas constant, dry air [J/(kg·K)]
    rgv: float = 462.0      # gas constant, water vapour [J/(kg·K)]
    cpa: float = 1000.0     # specific heat, air [J/(kg·K)]
    cpm: float = 1000.0     # specific heat, magma/solid [J/(kg·K)]
    rhol: float = 2500.0    # solid particle density [kg/m³]


def run_plume(
    r0: float,
    u0: float,
    params: PlumeParams,
    v_func: Callable[[float], float],
    tempa_func: Callable[[float], float],
    p_func: Callable[[float], float],
    ds_step: float = 10.0,
    nstep: int = 10_000,
    z_stop: float = 30_000.0,
) -> dict:
    """
    Integrate the Woodhouse (2013) 1-D plume ODEs with RK4.

    State vector y = [y1, y2, y3, y4]:
        y1 = Q/π          (scaled mass flux)
        y2 = Q·u_plume/π  (scaled momentum flux)
        y3 = Q·H/π        (scaled energy flux)
        y4 = θ            (plume angle from horizontal [rad])

    Parameters
    ----------
    r0 : vent radius [m]
    u0 : vent exit velocity [m/s]
    params : PlumeParams
    v_func, tempa_func, p_func : atmospheric profile callables
        each maps z_rel [m] → wind speed [m/s], temperature [K], pressure [Pa]
    ds_step : arc-length step size [m]
    nstep : maximum number of steps
    z_stop : altitude ceiling (above vent) [m]

    Returns
    -------
    dict with keys:
        z  : np.ndarray, altitude above vent [m]
        u  : np.ndarray, plume velocity [m/s]
        x  : np.ndarray, horizontal distance [m]
        Q  : float, initial mass flux [kg/s]
    """
    pr = params
    nv = pr.n0
    pa = p_func(0.0)

    def _na(z, y1, Q):
        return 1.0 - Q / (np.pi * y1) if z > 0 else 0.0

    def _rg(z, y1, Q):
        na_ = _na(z, y1, Q)
        return (na_ * pr.rga + nv * (1 - na_) * pr.rgv) / (na_ + nv * (1 - na_))

    def _cp(z, y1, Q):
        na_ = _na(z, y1, Q)
        return na_ * pr.cpa + (1 - na_) * pr.cpm

    def _temp(z, y1, y2, y3, Q):
        return (y3 / y1 - 0.5 * (y2 / y1) ** 2 - pr.g * z) / _cp(z, y1, Q)

    def _rho(z, y1, y2, y3, Q):
        na_ = _na(z, y1, Q)
        gas = (na_ + nv * (1 - na_)) * _rg(z, y1, Q) * _temp(z, y1, y2, y3, Q) / p_func(z)
        solid = (1 - na_) * (1 - nv) / pr.rhol
        return 1.0 / (gas + solid)

    def _r(z, y1, y2, y3, Q):
        return np.sqrt(y1 / _rho(z, y1, y2, y3, Q) / (y2 / y1))

    def _rhoa(z):
        return p_func(z) / (pr.rga * tempa_func(z))

    def _uke(z, y1, y2, y4, Q):
        u_pl = y2 / y1
        return pr.ke * abs(u_pl - v_func(z) * np.cos(y4)) + pr.kw * abs(v_func(z) * np.sin(y4))

    def f_vec(s, z, y, Q):
        y1, y2, y3, y4 = y
        r_ = _r(z, y1, y2, y3, Q)
        rho_ = _rho(z, y1, y2, y3, Q)
        rhoa_ = _rhoa(z)
        uke_ = _uke(z, y1, y2, y4, Q)
        f1 = 2 * uke_ * r_ * rhoa_
        f2 = r_ ** 2 * (rhoa_ - rho_) * pr.g * np.sin(y4) + v_func(z) * np.cos(y4) * f1
        f3 = (pr.cpa * tempa_func(z) + pr.g * z) * f1
        f4 = (r_ ** 2 * (rhoa_ - rho_) * pr.g * np.cos(y4) - v_func(z) * np.sin(y4) * f1) / y2
        return np.array([f1, f2, f3, f4])

    # Initial conditions
    rho0 = 1.0 / (nv * pr.rgv * pr.T0 / pa + (1 - nv) / pr.rhol)
    Q = rho0 * u0 * np.pi * r0 ** 2
    y = np.array([Q / np.pi, Q / np.pi * u0, Q / np.pi * (pr.cpm * pr.T0 + 0.5 * u0 ** 2), pr.theta0])

    s, z, x = 0.0, 0.0, 0.0
    z_list, u_list, x_list = [], [], []

    for _ in range(nstep):
        y4 = y[3]
        k1 = ds_step * f_vec(s, z, y, Q)
        k2 = ds_step * f_vec(s + ds_step / 2, z + np.sin(y4) * ds_step / 2, y + k1 / 2, Q)
        k3 = ds_step * f_vec(s + ds_step / 2, z + np.sin(y4) * ds_step / 2, y + k2 / 2, Q)
        k4 = ds_step * f_vec(s + ds_step, z + np.sin(y4) * ds_step, y + k3, Q)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        z += ds_step * np.sin(y4)
        x += ds_step * np.cos(y4)
        s += ds_step

        z_list.append(z)
        u_list.append(y[1] / y[0])
        x_list.append(x)

        if np.degrees(y[3]) <= 0 or z > z_stop:
            break

    return {
        "z": np.array(z_list),
        "u": np.array(u_list),
        "x": np.array(x_list),
        "Q": Q,
    }


def solve_u0_for_target(
    r0: float,
    params: PlumeParams,
    z_target: float,
    v_func: Callable,
    tempa_func: Callable,
    p_func: Callable,
    u0_hi: float = 400.0,
    u0_floor: float = 5.0,
    down_factor: float = 0.85,
    tol_z: float = 50.0,
    max_bisect: int = 35,
    z_stop: float = 30_000.0,
) -> Tuple[float, float, float, bool]:
    """
    Find u0 such that run_plume reaches z_target (i.e., z_max ≈ z_target).

    Strategy: scan u0 downward from u0_hi until the plume no longer reaches
    z_target, then bisect to locate the boundary.

    Returns
    -------
    (u0, Q, z_max, success)
        u0      : exit velocity [m/s]
        Q       : mass flux [kg/s]
        z_max   : achieved plume height [m]
        success : True if |z_max - z_target| <= tol_z
    """
    def _run(u0):
        res = run_plume(r0, u0, params, v_func, tempa_func, p_func, z_stop=z_stop)
        z_max = float(np.max(res["z"])) if len(res["z"]) > 0 else 0.0
        return z_max, res["Q"]

    def _reaches(z):
        return np.isfinite(z) and z >= z_target

    # Verify the upper bound reaches z_target
    z_hi, Q_hi = _run(u0_hi)
    if not _reaches(z_hi):
        return np.nan, np.nan, z_hi, False

    # Scan downward to bracket the threshold
    u_good, Q_good = u0_hi, Q_hi
    u_bad = None

    u = u0_hi
    while True:
        u_next = u * down_factor
        if u_next < u0_floor:
            z_fl, Q_fl = _run(u0_floor)
            if _reaches(z_fl):
                # Even u0_floor reaches z_target; return it as-is
                return u0_floor, Q_fl, z_fl, True
            u_bad = u0_floor
            break
        z_next, Q_next = _run(u_next)
        if _reaches(z_next):
            u_good, Q_good = u_next, Q_next
            u = u_next
        else:
            u_bad = u_next
            break

    # After the downward scan:
    #   u_good > u_bad  (u_good is the lowest u0 that still reaches; u_bad went one step lower)
    # Bisect [u_bad, u_good] to find the u0 where z_max ≈ z_target.
    lo_bad = u_bad    # doesn't reach z_target (lower u0)
    hi_good = u_good  # reaches z_target (higher u0)

    best_z, _ = _run(hi_good)
    best_u, best_Q, best_z_val = hi_good, Q_good, best_z

    for _ in range(max_bisect):
        mid = 0.5 * (lo_bad + hi_good)
        z_mid, Q_mid = _run(mid)

        if _reaches(z_mid):
            hi_good = mid
            best_u, best_Q, best_z_val = mid, Q_mid, z_mid
        else:
            lo_bad = mid

        if np.isfinite(best_z_val) and abs(best_z_val - z_target) <= tol_z:
            return best_u, best_Q, best_z_val, True
        if abs(hi_good - lo_bad) < 1e-3:
            break

    converged = np.isfinite(best_z_val) and abs(best_z_val - z_target) <= tol_z
    return best_u, best_Q, best_z_val, converged
