#!/usr/bin/env python3
"""
Forward simulation (v3): 1-row figure with 4 curves per panel.

Fixed combinations: u0 = 100, 200 m/s  ×  r0 = 50, 100 m.
Color encodes u0; linestyle encodes r0.
Layout: [velocity profile | horizontal distance | wind profile]

Usage:
    uv run python scripts/forward3.py --eruption Tokachi1962
    uv run python scripts/forward3.py --eruption all
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
from loader.era5 import download_era5, load_era5_nc
from plume_model import PlumeParams, run_plume

U0_LIST = [100, 200]
R0_LIST = [50, 100]

# color → u0, linestyle → r0
U0_COLORS = {100: "tab:green", 200: "tab:orange"}
R0_STYLES  = {50: "-", 100: "--"}


def load_atmosphere(cfg: dict):
    t_use = datetime.strptime(cfg["analysis_utc"], "%Y-%m-%d %H:%M")
    if cfg["reanalysis"] == "era5":
        ncfile = ROOT / cfg["era5_ncfile"]
        if not ncfile.exists():
            print(f"NC file not found: {ncfile}")
            print("Downloading from CDS API …")
            dl = cfg.get("era5_download", {})
            download_era5(
                str(ncfile),
                year=str(t_use.year),
                month=f"{t_use.month:02d}",
                day=f"{t_use.day:02d}",
                time=t_use.strftime("%H:%M"),
                area=dl.get("area", [90, -180, -90, 180]),
            )
        df = load_era5_nc(str(ncfile), cfg["lat"], cfg["lon"])
    else:
        csvfile = ROOT / cfg["cr20_csvfile"] if "cr20_csvfile" in cfg else None
        if csvfile and csvfile.exists():
            df = pd.read_csv(csvfile)
        else:
            print("Fetching 20CR from OPeNDAP …")
            df = load_20cr(cfg["lat"], cfg["lon"], t_use)
            if csvfile:
                csvfile.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csvfile, index=False)
                print(f"  → saved {csvfile.relative_to(ROOT)}")
    return build_profile(df, cfg["vent_height_m"])


def run_forward3(cfg: dict, eruption_key: str, output_dir: Path):
    t_label = cfg["analysis_utc"] + " UTC"
    reanalysis = cfg["reanalysis"].upper()
    name_ja = cfg.get("name_ja", eruption_key)
    name_en = cfg.get("name_en", eruption_key)
    T0 = float(cfg["forward"]["T0_K"])
    n0 = float(cfg["forward"]["n0"])
    z_target = float(cfg["qdet"]["z_target_m"])

    print(f"[{eruption_key}] {name_ja}  ({reanalysis}  {t_label})")
    print(f"  u0_list={U0_LIST}  r0_list={R0_LIST}")
    print(f"  T0={T0} K  n0={n0}  z_target={z_target:.0f} m")

    df_rel, v_func, tempa_func, p_func = load_atmosphere(cfg)
    z_stop = float(df_rel["z_rel_m"].max())
    params = PlumeParams(T0=T0, n0=n0)

    # --- run model --------------------------------------------------------
    results: dict[int, dict[int, dict]] = {}
    for r0 in R0_LIST:
        results[r0] = {}
        for u0 in U0_LIST:
            res = run_plume(r0, u0, params, v_func, tempa_func, p_func, z_stop=z_stop)
            results[r0][u0] = res
            z_max = float(np.max(res["z"])) if len(res["z"]) > 0 else 0.0
            print(f"    r0={r0:>3d} m  u0={u0:>3d} m/s  Q={res['Q']:.2e} kg/s  "
                  f"z_max={z_max:.0f} m")

    # --- plot -------------------------------------------------------------
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)
    x_max = 10_000 if cfg["reanalysis"] == "20cr" else 5_000

    for r0 in R0_LIST:
        ls = R0_STYLES[r0]
        for u0 in U0_LIST:
            res = results[r0][u0]
            c = U0_COLORS[u0]
            label = f"u0={u0} m/s, r0={r0} m  (Q={res['Q']:.2e} kg/s)"
            axs[0].plot(res["u"], res["z"], color=c, linestyle=ls, label=label)
            axs[1].plot(res["x"], res["z"], color=c, linestyle=ls, label=label)

    # Observed plume height
    for col in (0, 1):
        axs[col].axhline(z_target, color="black", linewidth=1.2, linestyle=":",
                         label=f"observed  {z_target/1000:.1f} km")

    axs[0].set_ylabel("Height above vent (m)")
    axs[0].set_xlabel("Velocity (m/s)")
    axs[0].set_title(f"$T_0$={int(T0)} K,  $n_0$={n0}\nVelocity profile")
    axs[0].legend(fontsize=8)
    axs[0].set_ylim(0, 25_000)
    axs[0].grid(True)

    axs[1].set_title("Horizontal distance")
    axs[1].set_xlabel("Distance (m)")
    axs[1].set_xlim(0, x_max)
    axs[1].set_ylim(0, 25_000)
    axs[1].grid(True)

    # Wind profile (right panel)
    ax_wind = axs[2]
    ax_dir = ax_wind.twiny()
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

    plt.tight_layout()

    out_png = output_dir / f"plume_forward3_{eruption_key}.png"
    out_eps = output_dir / f"plume_forward3_{eruption_key}.eps"
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_eps)
    plt.close()
    print(f"  → saved {out_png.relative_to(ROOT)}")
    print(f"  → saved {out_eps.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eruption", required=True,
                        help='Eruption key (e.g. Tokachi1962) or "all"')
    parser.add_argument("--output-dir", default="output",
                        help="Root output directory (default: output/)")
    args = parser.parse_args()

    with open(ROOT / "eruptions" / "catalog.yaml") as f:
        catalog = yaml.safe_load(f)["eruptions"]

    eruptions = list(catalog.keys()) if args.eruption == "all" else [args.eruption]
    for key in eruptions:
        if key not in catalog:
            raise SystemExit(f"Unknown eruption '{key}'. Available: {list(catalog)}")
        out_dir = ROOT / args.output_dir / key / "forward3"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_forward3(catalog[key], key, out_dir)


if __name__ == "__main__":
    main()
