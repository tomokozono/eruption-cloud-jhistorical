#!/usr/bin/env python3
"""
Discharge rate summary: Q at observed plume height for each eruption.

For each eruption and each u0 (100, 150, 200 m/s), sweeps r0 to build a
Q–H curve using per-eruption T0 and n0, then interpolates Q and r0 at
z_target.  Marker shape encodes u0; dot color encodes r0 (plasma colormap).

Usage:
    uv run python scripts/plot_discharge_rate.py
    uv run python scripts/plot_discharge_rate.py --output-dir output
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

U0_MARKERS = {100: "o", 150: "s", 200: "^"}

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


def trim_curve(Q_vals, z_vals, r0_arr):
    z_peak, n = 0.0, 0
    for q, z in zip(Q_vals, z_vals):
        if not np.isfinite(z) or z < 500:
            break
        if z_peak > 2000 and z < z_peak * 0.5:
            break
        z_peak = max(z_peak, z)
        n += 1
    return np.array(Q_vals[:n]), np.array(z_vals[:n]), r0_arr[:n]


def interp_at_z(Q_arr, z_arr, r0_arr, z_target):
    """Return (Q, r0) interpolated at z_target, or (None, None)."""
    if len(z_arr) < 2:
        return None, None
    sort_idx = np.argsort(z_arr)
    z_s  = z_arr[sort_idx]
    Q_s  = Q_arr[sort_idx]
    r0_s = r0_arr[sort_idx]
    if z_target < z_s[0] or z_target > z_s[-1]:
        return None, None
    Q  = float(np.interp(z_target, z_s, Q_s))
    r0 = float(np.interp(z_target, z_s, r0_s))
    return Q, r0


def compute_at_target(cfg: dict, u0: float, ke: float, kw: float):
    T0 = float(cfg["forward"]["T0_K"])
    n0 = float(cfg["forward"]["n0"])
    z_target = float(cfg["qdet"]["z_target_m"])

    df_rel, v_func, tempa_func, p_func = load_atmosphere(cfg)
    z_stop = float(df_rel["z_rel_m"].max())
    params = PlumeParams(T0=T0, n0=n0, ke=ke, kw=kw)

    r0_arr = np.arange(R0_MIN, R0_MAX + R0_STEP * 0.5, R0_STEP)
    Q_vals, z_vals = [], []
    for r0 in r0_arr:
        res = run_plume(r0, u0, params, v_func, tempa_func, p_func, z_stop=z_stop)
        z_max = float(np.max(res["z"])) if len(res["z"]) > 0 else np.nan
        Q_vals.append(res["Q"])
        z_vals.append(z_max)

    Q_arr, z_arr, r0_trim = trim_curve(Q_vals, z_vals, r0_arr)
    return interp_at_z(Q_arr, z_arr, r0_trim, z_target)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--ke", type=float, default=0.06,
                        help="Radial entrainment coefficient (default: 0.06)")
    parser.add_argument("--kw", type=float, default=0.20,
                        help="Wind entrainment coefficient (default: 0.20)")
    parser.add_argument("--errorbar", action="store_true",
                        help="Add error bars from ke=0.05/kw=0.1 (min) to ke=0.06/kw=0.3 (max)")
    args = parser.parse_args()

    ke, kw = args.ke, args.kw
    subdir = f"ke{int(ke*1000):03d}_kw{int(kw*1000):03d}"

    with open(ROOT / "eruptions" / "catalog.yaml") as f:
        catalog = yaml.safe_load(f)["eruptions"]

    # --- compute ----------------------------------------------------------
    results = {}   # results[key][u0] = (Q, r0) or (None, None)
    for key in ERUPTION_ORDER:
        cfg = catalog[key]
        results[key] = {}
        for u0 in U0_LIST:
            Q, r0 = compute_at_target(cfg, u0, ke, kw)
            results[key][u0] = (Q, r0)
            if Q is not None:
                print(f"  [{key}] u0={u0} m/s → Q={Q:.2e} kg/s  r0={r0:.1f} m")
            else:
                print(f"  [{key}] u0={u0} m/s → not reached")

    # error bar bounds (ke=0.05/kw=0.1 = min, ke=0.06/kw=0.3 = max)
    if args.errorbar:
        print("\n--- computing error bar bounds ---")
        results_min, results_max = {}, {}
        for key in ERUPTION_ORDER:
            cfg = catalog[key]
            results_min[key], results_max[key] = {}, {}
            for u0 in U0_LIST:
                Q_min, _ = compute_at_target(cfg, u0, ke=0.05, kw=0.1)
                Q_max, _ = compute_at_target(cfg, u0, ke=0.06, kw=0.3)
                results_min[key][u0] = Q_min
                results_max[key][u0] = Q_max
                print(f"  [{key}] u0={u0}: Q_min={Q_min:.2e}  Q_max={Q_max:.2e}")

    # --- plot -------------------------------------------------------------
    cmap = cm.plasma
    norm = mcolors.Normalize(vmin=R0_MIN, vmax=R0_MAX)

    fig, ax = plt.subplots(figsize=(7, 4))

    y_positions = {key: i for i, key in enumerate(reversed(ERUPTION_ORDER))}
    y_labels    = {key: catalog[key].get("name_en", key) for key in ERUPTION_ORDER}

    offsets = {100: 0.15, 150: 0.0, 200: -0.15}

    for u0 in U0_LIST:
        marker = U0_MARKERS[u0]
        for key in ERUPTION_ORDER:
            Q, r0 = results[key][u0]
            if Q is None:
                continue
            y = y_positions[key] + offsets[u0]
            color = cmap(norm(r0))
            # error bars
            if args.errorbar:
                Q_min = results_min[key][u0]
                Q_max = results_max[key][u0]
                if Q_min is not None and Q_max is not None:
                    ax.plot([Q_min, Q_max], [y, y], color="gray",
                            linewidth=1.2, zorder=4)
                    ax.plot([Q_min, Q_min], [y - 0.06, y + 0.06], color="gray",
                            linewidth=1.2, zorder=4)
                    ax.plot([Q_max, Q_max], [y - 0.06, y + 0.06], color="gray",
                            linewidth=1.2, zorder=4)
            ax.scatter([Q], [y], color=color, marker=marker, s=50,
                       edgecolors="gray", linewidths=0.5, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Discharge rate  (kg/s)", fontsize=12)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([y_labels[k] for k in reversed(ERUPTION_ORDER)],
                       fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(1e6, 1e8)
    ax.grid(True, axis="x", alpha=0.4, which="both")
    eb_note = ("\nerror bar: $k_e$=0.05/$k_w$=0.1 (min) – $k_e$=0.06/$k_w$=0.3 (max)"
               if args.errorbar else "")
    ax.set_title(
        f"Estimated discharge rate at observed plume height\n"
        f"($k_e$ = {ke},  $k_w$ = {kw}){eb_note}",
        fontsize=11,
    )

    # legend: marker shape for u0
    legend_handles = [
        Line2D([0], [0], marker=U0_MARKERS[u0], color="gray", linestyle="none",
               markersize=8, label=f"$u_0$ = {u0} m/s")
        for u0 in U0_LIST
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="lower right")

    # colorbar: r0
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("$r_0$  [m]", fontsize=11)

    plt.tight_layout()

    if args.errorbar:
        out_dir = ROOT / args.output_dir / "discharge_rate"
        stem = "discharge_rate"
    else:
        out_dir = ROOT / args.output_dir / "discharge_rate" / subdir
        stem = "discharge_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".eps"):
        out = out_dir / f"{stem}{suffix}"
        save_kw = {"dpi": 150} if suffix == ".png" else {}
        plt.savefig(out, **save_kw)
        print(f"→ saved {out.relative_to(ROOT)}")
    plt.close()


if __name__ == "__main__":
    main()
