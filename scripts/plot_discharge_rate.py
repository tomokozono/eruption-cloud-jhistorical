#!/usr/bin/env python3
"""
Discharge rate summary: Q at observed plume height for each eruption.

For each eruption and each u0 (100, 150, 200 m/s), sweeps r0 to build a
Q–H curve using per-eruption T0 and n0, then interpolates Q at z_target.
Results are plotted as a categorical dot plot (eruption on Y, Q on X).

Usage:
    uv run python scripts/plot_discharge_rate.py
    uv run python scripts/plot_discharge_rate.py --output-dir output
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atmos_profile import build_profile
from loader.cr20 import load_20cr
from loader.era5 import load_era5_nc
from plume_model import PlumeParams, run_plume

R0_MIN  = 10.0
R0_MAX  = 250.0
R0_STEP = 5.0
U0_LIST = [100, 150, 200]

U0_STYLES = {
    100: dict(color="tab:blue",   marker="o", label="$u_0$ = 100 m/s"),
    150: dict(color="tab:orange", marker="s", label="$u_0$ = 150 m/s"),
    200: dict(color="tab:green",  marker="^", label="$u_0$ = 200 m/s"),
}

ERUPTION_ORDER = ["Sakurajima1914", "Komagatake1929", "Tokachi1962", "Usu1977"]


def load_atmosphere(cfg: dict):
    t_use = datetime.strptime(cfg["analysis_utc"], "%Y-%m-%d %H:%M")
    if cfg["reanalysis"] == "era5":
        df = load_era5_nc(str(ROOT / cfg["era5_ncfile"]), cfg["lat"], cfg["lon"])
    elif "cr20_csvfile" in cfg:
        df = pd.read_csv(ROOT / cfg["cr20_csvfile"])
    else:
        df = load_20cr(cfg["lat"], cfg["lon"], t_use)
    return build_profile(df, cfg["vent_height_m"])


def trim_curve(Q_vals, z_vals):
    z_peak, n = 0.0, 0
    for q, z in zip(Q_vals, z_vals):
        if not np.isfinite(z) or z < 500:
            break
        if z_peak > 2000 and z < z_peak * 0.5:
            break
        z_peak = max(z_peak, z)
        n += 1
    return np.array(Q_vals[:n]), np.array(z_vals[:n])


def interp_Q_at_z(Q_arr, z_arr, z_target):
    if len(z_arr) < 2:
        return None
    sort_idx = np.argsort(z_arr)
    z_s = z_arr[sort_idx]
    Q_s = Q_arr[sort_idx]
    if z_target < z_s[0] or z_target > z_s[-1]:
        return None
    return float(np.interp(z_target, z_s, Q_s))


def compute_Q_at_target(cfg: dict, u0: float) -> float | None:
    T0 = float(cfg["forward"]["T0_K"])
    n0 = float(cfg["forward"]["n0"])
    z_target = float(cfg["qdet"]["z_target_m"])

    df_rel, v_func, tempa_func, p_func = load_atmosphere(cfg)
    z_stop = float(df_rel["z_rel_m"].max())
    params = PlumeParams(T0=T0, n0=n0)

    r0_arr = np.arange(R0_MIN, R0_MAX + R0_STEP * 0.5, R0_STEP)
    Q_vals, z_vals = [], []
    for r0 in r0_arr:
        res = run_plume(r0, u0, params, v_func, tempa_func, p_func, z_stop=z_stop)
        z_max = float(np.max(res["z"])) if len(res["z"]) > 0 else np.nan
        Q_vals.append(res["Q"])
        z_vals.append(z_max)

    Q_arr, z_arr = trim_curve(Q_vals, z_vals)
    return interp_Q_at_z(Q_arr, z_arr, z_target)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    with open(ROOT / "eruptions" / "catalog.yaml") as f:
        catalog = yaml.safe_load(f)["eruptions"]

    # --- compute ----------------------------------------------------------
    results = {}   # results[key][u0] = Q or None
    for key in ERUPTION_ORDER:
        cfg = catalog[key]
        results[key] = {}
        for u0 in U0_LIST:
            Q = compute_Q_at_target(cfg, u0)
            results[key][u0] = Q
            status = f"{Q:.2e} kg/s" if Q is not None else "not reached"
            print(f"  [{key}] u0={u0} m/s → Q = {status}")

    # --- plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))

    y_positions = {key: i for i, key in enumerate(reversed(ERUPTION_ORDER))}
    y_labels    = {key: catalog[key].get("name_en", key) for key in ERUPTION_ORDER}

    offset = {100: -0.15, 150: 0.0, 200: 0.15}   # vertical jitter per u0

    for u0 in U0_LIST:
        sty = U0_STYLES[u0]
        xs, ys = [], []
        for key in ERUPTION_ORDER:
            Q = results[key][u0]
            if Q is not None:
                xs.append(Q)
                ys.append(y_positions[key] + offset[u0])
        ax.scatter(xs, ys, color=sty["color"], marker=sty["marker"],
                   s=80, zorder=5, label=sty["label"])

    ax.set_xscale("log")
    ax.set_xlabel("Discharge rate  (kg/s)", fontsize=12)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([y_labels[k] for k in reversed(ERUPTION_ORDER)],
                       fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(1e6, 1e8)
    ax.grid(True, axis="x", alpha=0.4, which="both")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_title("Estimated discharge rate at observed plume height", fontsize=11)

    plt.tight_layout()

    out_dir = ROOT / args.output_dir / "discharge_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".eps"):
        out = out_dir / f"discharge_rate{suffix}"
        kw = {"dpi": 150} if suffix == ".png" else {}
        plt.savefig(out, **kw)
        print(f"→ saved {out.relative_to(ROOT)}")
    plt.close()


if __name__ == "__main__":
    main()
