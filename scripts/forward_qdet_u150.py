#!/usr/bin/env python3
"""
Forward simulation at the deterministic (r0, u0=150 m/s) solution.

For each eruption, sweeps r0 to find the value that reproduces the observed
plume height (z_target_m) at u0=150 m/s, ke=0.06, kw=0.2 with per-eruption
T0 and n0.  Plots a forward3-style figure (1 row × 3 cols) for that solution.
Figures are merged horizontally at the end in chronological eruption order.

Usage:
    uv run python scripts/forward_qdet_u150.py
    uv run python scripts/forward_qdet_u150.py --eruption Tokachi1962
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atmos_profile import build_profile
from loader.cr20 import load_20cr
from loader.era5 import load_era5_nc
from plume_model import PlumeParams, run_plume

U0   = 150.0
KE   = 0.06
KW   = 0.20
R0_MIN  = 10.0
R0_MAX  = 250.0
R0_STEP = 5.0

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


def find_r0_at_target(cfg: dict, v_func, tempa_func, p_func, z_stop: float,
                       params: PlumeParams) -> tuple[float | None, float | None]:
    """Sweep r0 and interpolate the value that achieves z_target."""
    z_target = float(cfg["qdet"]["z_target_m"])
    r0_arr = np.arange(R0_MIN, R0_MAX + R0_STEP * 0.5, R0_STEP)

    Q_vals, z_vals = [], []
    for r0 in r0_arr:
        res = run_plume(r0, U0, params, v_func, tempa_func, p_func, z_stop=z_stop)
        z_max = float(np.max(res["z"])) if len(res["z"]) > 0 else np.nan
        Q_vals.append(res["Q"])
        z_vals.append(z_max)

    # trim to valid ascending portion
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


def run_forward_qdet(cfg: dict, eruption_key: str, output_dir: Path):
    t_label  = cfg["analysis_utc"] + " UTC"
    reanalysis = cfg["reanalysis"].upper()
    name_en  = cfg.get("name_en", eruption_key)
    T0       = float(cfg["forward"]["T0_K"])
    n0       = float(cfg["forward"]["n0"])
    z_target = float(cfg["qdet"]["z_target_m"])

    print(f"[{eruption_key}] {name_en}")
    print(f"  T0={T0} K  n0={n0}  u0={U0} m/s  ke={KE}  kw={KW}")

    df_rel, v_func, tempa_func, p_func = load_atmosphere(cfg)
    z_stop = float(df_rel["z_rel_m"].max())
    params = PlumeParams(T0=T0, n0=n0, ke=KE, kw=KW)

    r0_det, Q_det = find_r0_at_target(cfg, v_func, tempa_func, p_func, z_stop, params)
    if r0_det is None:
        print(f"  WARNING: z_target={z_target:.0f} m not reached — skipping")
        return None

    print(f"  → r0_det={r0_det:.1f} m  Q_det={Q_det:.2e} kg/s")

    res = run_plume(r0_det, U0, params, v_func, tempa_func, p_func, z_stop=z_stop)

    # --- plot ----------------------------------------------------------------
    x_max = 10_000 if cfg["reanalysis"] == "20cr" else 5_000
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

    axs[0].plot(res["u"], res["z"], color="tab:orange", linewidth=2)
    axs[1].plot(res["x"], res["z"], color="tab:orange", linewidth=2)

    for col in (0, 1):
        axs[col].axhline(z_target, color="black", linewidth=1.2, linestyle="--",
                         label=f"observed  {z_target/1000:.1f} km")

    axs[0].set_ylabel("Height above vent (m)")
    axs[0].set_xlabel("Velocity (m/s)")
    axs[0].set_title(
        f"$T_0$={int(T0)} K,  $n_0$={n0}\n"
        f"Velocity profile\n"
        f"$r_0$={r0_det:.1f} m,  $u_0$={int(U0)} m/s,  $Q$={Q_det:.2e} kg/s"
    )
    axs[0].legend(fontsize=8)
    axs[0].set_ylim(0, 25_000)
    axs[0].grid(True)

    axs[1].set_title("Horizontal distance")
    axs[1].set_xlabel("Distance (m)")
    axs[1].set_xlim(0, x_max)
    axs[1].set_ylim(0, 25_000)
    axs[1].grid(True)

    ax_wind = axs[2]
    ax_dir  = ax_wind.twiny()
    ax_wind.plot(df_rel["WindSpeed_mps"], df_rel["z_rel_m"], "b")
    ax_wind.set_xlabel("Wind speed (m/s)", color="b")
    ax_wind.tick_params(axis="x", colors="b")
    ax_wind.set_xlim(0, 80)
    ax_wind.set_title(f"{reanalysis} wind profile\n{eruption_key}  {t_label}")
    ax_wind.set_ylim(0, 25_000)
    ax_wind.grid(True)

    ax_dir.plot(df_rel["WindDirection_deg"], df_rel["z_rel_m"], "r")
    ax_dir.set_xlabel("Wind direction (deg from N)", color="r")
    ax_dir.tick_params(axis="x", colors="r")
    ax_dir.set_xlim(0, 360)

    # eruption name label (suptitle-style text above center panel)
    axs[1].text(0.5, 1.12, name_en, transform=axs[1].transAxes,
                fontsize=13, fontweight="bold", ha="center", va="bottom")

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"plume_forward_qdet_u150_{eruption_key}.png"
    out_eps = output_dir / f"plume_forward_qdet_u150_{eruption_key}.eps"
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_eps)
    plt.close()
    print(f"  → saved {out_png.relative_to(ROOT)}")
    print(f"  → saved {out_eps.relative_to(ROOT)}")
    return out_png


def merge_figures(png_paths: list[Path], out_root: Path):
    """Merge PNGs horizontally with a small gap."""
    images = [Image.open(p) for p in png_paths]
    gap = 30
    total_w = sum(img.width for img in images) + gap * (len(images) - 1)
    max_h   = max(img.height for img in images)
    merged  = Image.new("RGB", (total_w, max_h), color=(255, 255, 255))
    x = 0
    for img in images:
        merged.paste(img, (x, 0))
        x += img.width + gap
    out = out_root / "forward_qdet_u150_merged.png"
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
        out_dir = ROOT / args.output_dir / key / "forward_qdet_u150"
        png = run_forward_qdet(catalog[key], key, out_dir)
        if png:
            png_paths.append(png)

    if args.eruption == "all" and len(png_paths) > 1:
        merge_figures(png_paths, ROOT / args.output_dir)


if __name__ == "__main__":
    main()
