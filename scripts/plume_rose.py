#!/usr/bin/env python3
"""
Plume rose diagram: horizontal trajectory of the plume colored by height.

Uses the same deterministic solution as forward_qdet_u150 (u0=150 m/s,
ke=0.06, kw=0.2, per-eruption T0/n0, r0 that matches z_target).

At each height step, the plume is assumed to travel downwind.
Wind direction is "from" direction, so plume direction = wind_dir + 180°.
The incremental horizontal displacement at step i uses the wind direction
interpolated at that height, giving a 2-D trajectory in (East, North) space.

Usage:
    uv run python scripts/plume_rose.py
    uv run python scripts/plume_rose.py --eruption Tokachi1962
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
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atmos_profile import build_profile
from loader.cr20 import load_20cr
from loader.era5 import load_era5_nc
from plume_model import PlumeParams, run_plume

U0      = 150.0
KE      = 0.06
KW      = 0.20
R0_MIN  = 10.0
R0_MAX  = 250.0
R0_STEP = 5.0

ERUPTION_ORDER = ["Sakurajima1914", "Komagatake1929", "Tokachi1962", "Usu1977"]

# Per-eruption circle settings (step and max in metres)
CIRCLE_CONFIGS = {
    "Sakurajima1914": {"step": 2000.0, "max": 8000.0},
    "default":        {"step": 1000.0, "max": 4000.0},
}


def load_atmosphere(cfg: dict):
    t_use = datetime.strptime(cfg["analysis_utc"], "%Y-%m-%d %H:%M")
    if cfg["reanalysis"] == "era5":
        df = load_era5_nc(str(ROOT / cfg["era5_ncfile"]), cfg["lat"], cfg["lon"])
    elif "cr20_csvfile" in cfg:
        df = pd.read_csv(ROOT / cfg["cr20_csvfile"])
    else:
        df = load_20cr(cfg["lat"], cfg["lon"], t_use)
    return build_profile(df, cfg["vent_height_m"])


def find_r0_at_target(cfg, v_func, tempa_func, p_func, z_stop, params):
    z_target = float(cfg["qdet"]["z_target_m"])
    r0_arr   = np.arange(R0_MIN, R0_MAX + R0_STEP * 0.5, R0_STEP)
    Q_vals, z_vals = [], []
    for r0 in r0_arr:
        res = run_plume(r0, U0, params, v_func, tempa_func, p_func, z_stop=z_stop)
        z_max = float(np.max(res["z"])) if len(res["z"]) > 0 else np.nan
        Q_vals.append(res["Q"])
        z_vals.append(z_max)
    z_peak, n = 0.0, 0
    for z in z_vals:
        if not np.isfinite(z) or z < 500:
            break
        if z_peak > 2000 and z < z_peak * 0.5:
            break
        z_peak = max(z_peak, z)
        n += 1
    r0_trim = r0_arr[:n]
    z_trim  = np.array(z_vals[:n])
    if n < 2 or z_target < z_trim[0] or z_target > z_trim[-1]:
        return None, None
    sort_idx = np.argsort(z_trim)
    r0_det = float(np.interp(z_target, z_trim[sort_idx], r0_trim[sort_idx]))
    Q_det  = float(np.interp(z_target, z_trim[sort_idx],
                             np.array(Q_vals[:n])[sort_idx]))
    return r0_det, Q_det


def plume_trajectory_2d(res: dict, df_rel) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (East, North) plume trajectory from horizontal distances and wind directions.

    Wind direction is the FROM direction.  Plume moves downwind = wind_dir + 180°.
    """
    z_arr = res["z"]
    x_arr = res["x"]

    # Interpolate wind direction (FROM) at each plume height
    wd = np.interp(z_arr, df_rel["z_rel_m"].values, df_rel["WindDirection_deg"].values)

    # Incremental horizontal distance between steps
    dx = np.diff(x_arr)

    # Wind direction at midpoints
    wd_mid = 0.5 * (wd[:-1] + wd[1:])

    # Plume direction = downwind = from-direction + 180°
    plume_dir_rad = np.radians((wd_mid + 180.0) % 360.0)

    # East (+x) and North (+y) increments
    east_inc  = np.sin(plume_dir_rad) * dx
    north_inc = np.cos(plume_dir_rad) * dx

    east  = np.concatenate([[0.0], np.cumsum(east_inc)])
    north = np.concatenate([[0.0], np.cumsum(north_inc)])
    return east, north


def make_rose(cfg: dict, eruption_key: str, output_dir: Path):
    name_en  = cfg.get("name_en", eruption_key)
    T0       = float(cfg["forward"]["T0_K"])
    n0       = float(cfg["forward"]["n0"])
    z_target = float(cfg["qdet"]["z_target_m"])

    print(f"[{eruption_key}] {name_en}")

    df_rel, v_func, tempa_func, p_func = load_atmosphere(cfg)
    z_stop = float(df_rel["z_rel_m"].max())
    params = PlumeParams(T0=T0, n0=n0, ke=KE, kw=KW)

    r0_det, Q_det = find_r0_at_target(cfg, v_func, tempa_func, p_func, z_stop, params)
    if r0_det is None:
        print(f"  WARNING: z_target not reached — skipping")
        return None

    res = run_plume(r0_det, U0, params, v_func, tempa_func, p_func, z_stop=z_stop)
    print(f"  r0_det={r0_det:.1f} m  Q_det={Q_det:.2e} kg/s")

    east, north = plume_trajectory_2d(res, df_rel)
    z_arr = res["z"]

    # --- plot ----------------------------------------------------------------
    cc = CIRCLE_CONFIGS.get(eruption_key, CIRCLE_CONFIGS["default"])
    CIRCLE_STEP = cc["step"]
    CIRCLE_MAX  = cc["max"]
    max_plot    = CIRCLE_MAX * 1.15

    fig, ax = plt.subplots(figsize=(6, 6))

    # Concentric distance circles (fixed 2 km intervals up to 8 km)
    for r in np.arange(CIRCLE_STEP, CIRCLE_MAX + 1.0, CIRCLE_STEP):
        circle = plt.Circle((0, 0), r, fill=False, color="black",
                             linewidth=0.7, zorder=1)
        ax.add_patch(circle)
        ax.text(0, r, f"{r/1000:.0f} km", ha="center", va="bottom",
                fontsize=7, color="gray")

    # Compass labels — just outside the outermost circle
    pad = CIRCLE_MAX * 0.08
    ax.text(0,  CIRCLE_MAX + pad, "N", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0, -CIRCLE_MAX - pad, "S", ha="center", va="top",    fontsize=10, fontweight="bold")
    ax.text( CIRCLE_MAX + pad, 0, "E", ha="left",   va="center", fontsize=10, fontweight="bold")
    ax.text(-CIRCLE_MAX - pad, 0, "W", ha="right",  va="center", fontsize=10, fontweight="bold")

    # Plume trajectory colored by height
    cmap_z = cm.plasma
    norm_z = mcolors.Normalize(vmin=0, vmax=15000)
    points  = np.array([east, north]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap_z, norm=norm_z, linewidth=2.5, zorder=3)
    lc.set_array(0.5 * (z_arr[:-1] + z_arr[1:]))
    ax.add_collection(lc)

    # Volcano at origin
    ax.scatter([0], [0], color="black", s=60, zorder=5)

    ax.set_xlim(-max_plot, max_plot)
    ax.set_ylim(-max_plot, max_plot)
    ax.set_aspect("equal")
    ax.axis("off")

    cbar = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Height above vent  [m]", fontsize=10)

    ax.set_title(
        f"{name_en}\n"
        f"Plume trajectory  ($u_0$={int(U0)} m/s,  $r_0$={r0_det:.0f} m,  "
        f"$Q$={Q_det:.2e} kg/s)\n"
        f"$T_0$={int(T0)} K,  $n_0$={n0},  $k_e$={KE},  $k_w$={KW}",
        fontsize=10,
    )

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"plume_rose_{eruption_key}.png"
    out_eps = output_dir / f"plume_rose_{eruption_key}.eps"
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_eps)
    plt.close()
    print(f"  → saved {out_png.relative_to(ROOT)}")
    print(f"  → saved {out_eps.relative_to(ROOT)}")
    return out_png


def merge_figures(png_paths: list[Path], out_root: Path):
    images = [Image.open(p) for p in png_paths]
    gap     = 30
    total_w = sum(img.width for img in images) + gap * (len(images) - 1)
    max_h   = max(img.height for img in images)
    merged  = Image.new("RGB", (total_w, max_h), color=(255, 255, 255))
    x = 0
    for img in images:
        merged.paste(img, (x, 0))
        x += img.width + gap
    out = out_root / "plume_rose_merged.png"
    merged.save(out, dpi=(150, 150))
    print(f"\n→ merged: {out.relative_to(ROOT)}  ({merged.width}×{merged.height})")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--eruption", default="all",
                        help='Eruption key or "all" (default: all)')
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    with open(ROOT / "eruptions" / "catalog.yaml") as f:
        catalog = yaml.safe_load(f)["eruptions"]

    eruptions = ERUPTION_ORDER if args.eruption == "all" else [args.eruption]

    png_paths = []
    for key in eruptions:
        if key not in catalog:
            raise SystemExit(f"Unknown eruption '{key}'. Available: {list(catalog)}")
        out_dir = ROOT / args.output_dir / key / "plume_rose"
        png = make_rose(catalog[key], key, out_dir)
        if png:
            png_paths.append(png)

    if args.eruption == "all" and len(png_paths) > 1:
        merge_figures(png_paths, ROOT / args.output_dir)


if __name__ == "__main__":
    main()
