#!/usr/bin/env python3
"""
Q–H sensitivity using per-eruption n0 (from catalog forward.n0).

Sweeps u0 = 100, 150, 200 m/s.  One figure per u0; each figure shows all
four eruptions with their individual n0 values colored by r0.
A black dot marks the observed plume height on each curve when reachable.

Usage:
    uv run python scripts/qh_sensitivity_each.py
    uv run python scripts/qh_sensitivity_each.py --output-dir output
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
U0_LIST = [100, 150, 200]


def load_atmosphere(cfg: dict):
    t_use = datetime.strptime(cfg["analysis_utc"], "%Y-%m-%d %H:%M")
    if cfg["reanalysis"] == "era5":
        df = load_era5_nc(str(ROOT / cfg["era5_ncfile"]), cfg["lat"], cfg["lon"])
    elif "cr20_csvfile" in cfg:
        df = pd.read_csv(ROOT / cfg["cr20_csvfile"])
    else:
        df = load_20cr(cfg["lat"], cfg["lon"], t_use)
    return build_profile(df, cfg["vent_height_m"])


def plot_gradient_line(ax, Q_arr, z_arr, r0_plot, cmap, norm, linestyle,
                       linewidth=2.5, alpha=0.9, n_chunks=10):
    n = len(Q_arr)
    if n < 2:
        return
    chunk_size = max(2, n // n_chunks)
    for start in range(0, n - 1, chunk_size):
        end = min(start + chunk_size + 1, n)
        r0_mid = float(np.mean(r0_plot[start:end]))
        ax.plot(Q_arr[start:end], z_arr[start:end],
                color=cmap(norm(r0_mid)), linestyle=linestyle,
                linewidth=linewidth, alpha=alpha, solid_capstyle="butt")


def trim_curve(Q_vals, z_vals, r0_arr):
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
    if len(z_arr) < 2:
        return None
    sort_idx = np.argsort(z_arr)
    z_sorted = z_arr[sort_idx]
    Q_sorted = Q_arr[sort_idx]
    if z_target < z_sorted[0] or z_target > z_sorted[-1]:
        return None
    return float(np.interp(z_target, z_sorted, Q_sorted))


def make_figure(catalog: dict, u0: float, output_path: Path):
    r0_arr = np.arange(R0_MIN, R0_MAX + R0_STEP * 0.5, R0_STEP)
    norm = mcolors.Normalize(vmin=R0_MIN, vmax=R0_MAX)
    cmap = cm.plasma
    linestyles = ["-", "--", "-.", ":"]
    eruption_keys = list(catalog.keys())

    fig, ax = plt.subplots(figsize=(9, 6))
    legend_handles = []

    for idx, key in enumerate(eruption_keys):
        cfg = catalog[key]
        T0  = float(cfg["forward"]["T0_K"])
        n0  = float(cfg["forward"]["n0"])
        z_target = float(cfg["qdet"]["z_target_m"])
        name_en  = cfg.get("name_en", key)
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
            plot_gradient_line(ax, Q_arr, z_arr, r0_plot, cmap, norm, ls)

            Q_dot = interp_Q_at_z(Q_arr, z_arr, z_target)
            if Q_dot is not None:
                ax.scatter([Q_dot], [z_target], color="black", s=60, zorder=5,
                           clip_on=True)

        handle = Line2D(
            [0], [0], color="gray", linestyle=ls, linewidth=2,
            label=f"{name_en}  $T_0$={int(T0)} K,  $n_0$={n0}",
        )
        legend_handles.append(handle)

    ax.set_xscale("log")
    ax.set_xlim(1e5, 1e8)
    ax.set_ylim(0, 25000)
    ax.set_xlabel("Discharge rate  $Q$  [kg/s]", fontsize=12)
    ax.set_ylabel("Column height above vent  [m]", fontsize=12)
    ax.set_title(
        f"Column height vs discharge rate\n"
        f"($u_0$ = {int(u0)} m/s,  "
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
    out_eps = output_path.with_suffix(".eps")
    plt.savefig(out_eps)
    plt.close()
    print(f"  → saved {output_path.relative_to(ROOT)}")
    print(f"  → saved {out_eps.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", default="output",
                        help="Root output directory (default: output/)")
    args = parser.parse_args()

    with open(ROOT / "eruptions" / "catalog.yaml") as f:
        catalog = yaml.safe_load(f)["eruptions"]

    out_dir = ROOT / args.output_dir / "qh_sensitivity_each"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(U0_LIST)
    for count, u0 in enumerate(U0_LIST, 1):
        print(f"\n[{count}/{total}] u0={u0} m/s")
        out = out_dir / f"qh_sensitivity_each_u0{u0:03d}.png"
        make_figure(catalog, u0, out)


if __name__ == "__main__":
    main()
