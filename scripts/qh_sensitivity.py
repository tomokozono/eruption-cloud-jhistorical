#!/usr/bin/env python3
"""
Q–H sensitivity: eruption column height vs mass flux for varying r0.

Fix u0=100 m/s and T0 per eruption (from catalog qdet settings), then sweep
r0 from 10 to 200 m.  One figure per n0 value (0.03, 0.04, 0.05); each figure
shows all four eruptions (distinguished by line style) with curves colored by r0.
A black dot marks the observed plume height (z_target_m) on each curve when the
curve reaches that height.

Usage:
    uv run python scripts/qh_sensitivity.py
    uv run python scripts/qh_sensitivity.py --output-dir output
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atmos_profile import build_profile
from loader.cr20 import load_20cr
from loader.era5 import load_era5_nc
from plume_model import PlumeParams, run_plume

R0_MIN  = 10.0
R0_MAX  = 250.0
R0_STEP = 5.0
N0_LIST  = [0.03, 0.04, 0.05]
U0_LIST  = list(range(100, 201, 20))   # [100, 120, 140, 160, 180, 200]


def load_atmosphere(cfg: dict):
    t_use = datetime.strptime(cfg["analysis_utc"], "%Y-%m-%d %H:%M")
    if cfg["reanalysis"] == "era5":
        df = load_era5_nc(str(ROOT / cfg["era5_ncfile"]), cfg["lat"], cfg["lon"])
    elif "cr20_csvfile" in cfg:
        df = pd.read_csv(ROOT / cfg["cr20_csvfile"])
    else:
        df = load_20cr(cfg["lat"], cfg["lon"], t_use)
    return build_profile(df, cfg["vent_height_m"])


def make_segments(x, y):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def trim_curve(Q_vals, z_vals, r0_arr):
    """Keep only the initial consecutive run of valid, non-crashing points."""
    z_peak = 0.0
    n = 0
    for q, z in zip(Q_vals, z_vals):
        if not np.isfinite(z) or z < 500:
            break
        if z_peak > 2000 and z < z_peak * 0.5:
            break
        z_peak = max(z_peak, z)
        n += 1
    return np.array(Q_vals[:n]), np.array(z_vals[:n]), r0_arr[:n]


def interp_Q_at_z(Q_arr, z_arr, z_target):
    """Interpolate Q at z=z_target on a (Q, z) curve. Returns None if out of range."""
    if len(z_arr) < 2:
        return None
    sort_idx = np.argsort(z_arr)
    z_sorted = z_arr[sort_idx]
    Q_sorted = Q_arr[sort_idx]
    if z_target < z_sorted[0] or z_target > z_sorted[-1]:
        return None
    return float(np.interp(z_target, z_sorted, Q_sorted))


def make_figure(catalog: dict, n0: float, u0: float, output_path: Path):
    r0_arr = np.arange(R0_MIN, R0_MAX + R0_STEP * 0.5, R0_STEP)
    norm = mcolors.Normalize(vmin=R0_MIN, vmax=R0_MAX)
    cmap = cm.plasma
    linestyles = ["-", "--", "-.", ":"]
    eruption_keys = list(catalog.keys())

    fig, ax = plt.subplots(figsize=(9, 6))
    legend_handles = []
    all_Q, all_z = [], []

    for idx, key in enumerate(eruption_keys):
        cfg = catalog[key]
        T0 = float(cfg["qdet"]["T0_grid_K"][0])
        z_target = float(cfg["qdet"]["z_target_m"])
        name_ja = cfg.get("name_ja", key)
        ls = linestyles[idx % len(linestyles)]

        print(f"  [{key}] T0={T0} K  n0={n0}  u0={u0} m/s")

        df_rel, v_func, tempa_func, p_func = load_atmosphere(cfg)
        z_stop = float(df_rel["z_rel_m"].max())
        params = PlumeParams(T0=T0, n0=n0)

        Q_vals, z_vals = [], []
        for r0 in r0_arr:
            res = run_plume(r0, u0, params, v_func, tempa_func, p_func, z_stop=z_stop)
            z_max = float(np.max(res["z"])) if len(res["z"]) > 0 else np.nan
            Q_vals.append(res["Q"])
            z_vals.append(z_max)

        Q_arr, z_arr, r0_plot = trim_curve(Q_vals, z_vals, r0_arr)

        if len(Q_arr) >= 2:
            all_Q.extend(Q_arr.tolist())
            all_z.extend(z_arr.tolist())

            segs = make_segments(Q_arr, z_arr)
            r0_mid = 0.5 * (r0_plot[:-1] + r0_plot[1:])
            lc = LineCollection(segs, cmap=cmap, norm=norm, linestyle=ls,
                                linewidth=2.5, alpha=0.9)
            lc.set_array(r0_mid)
            ax.add_collection(lc)

            # Black dot at observed plume height
            Q_dot = interp_Q_at_z(Q_arr, z_arr, z_target)
            if Q_dot is not None:
                ax.scatter([Q_dot], [z_target], color="black", s=60, zorder=5,
                           clip_on=True)

        handle = Line2D(
            [0], [0], color="gray", linestyle=ls, linewidth=2,
            label=f"{key}  $T_0$={int(T0)} K",
        )
        legend_handles.append(handle)

    ax.set_xscale("log")
    ax.set_xlim(1e5, 1e8)
    ax.set_ylim(0, 25000)

    ax.set_xlabel("Mass flux  $Q$  [kg/s]", fontsize=12)
    ax.set_ylabel("Column height above vent  [m]", fontsize=12)
    ax.set_title(
        f"Column height vs mass flux\n"
        f"($n_0$ = {n0},  $u_0$ = {int(u0)} m/s,  "
        f"$r_0$ = {int(R0_MIN)}–{int(R0_MAX)} m)\n"
        f"black dot: observed plume height",
        fontsize=11,
    )
    ax.grid(True, alpha=0.4, which="both")
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("$r_0$  [m]", fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  → saved {output_path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Root output directory (default: output/)",
    )
    args = parser.parse_args()

    with open(ROOT / "eruptions" / "catalog.yaml") as f:
        catalog = yaml.safe_load(f)["eruptions"]

    out_dir = ROOT / args.output_dir / "qh_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(N0_LIST) * len(U0_LIST)
    count = 0
    for n0 in N0_LIST:
        n0_str = f"{n0:.2f}".replace(".", "")
        for u0 in U0_LIST:
            count += 1
            print(f"\n[{count}/{total}] n0={n0}  u0={u0} m/s")
            out = out_dir / f"qh_sensitivity_n0{n0_str}_u0{u0:03d}.png"
            make_figure(catalog, n0, u0, out)


if __name__ == "__main__":
    main()
