#!/usr/bin/env python3
"""
Forward simulation: run Woodhouse (2013) plume model for multiple (r0, u0) pairs.

Usage:
    uv run python scripts/forward.py --eruption Tokachi1962
    uv run python scripts/forward.py --eruption all
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


def load_catalog(eruption: str) -> dict:
    with open(ROOT / "eruptions" / "catalog.yaml") as f:
        cat = yaml.safe_load(f)
    if eruption not in cat["eruptions"]:
        raise SystemExit(f"Unknown eruption '{eruption}'. Available: {list(cat['eruptions'])}")
    return cat["eruptions"][eruption]


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
            print(f"Fetching 20CR from OPeNDAP …")
            df = load_20cr(cfg["lat"], cfg["lon"], t_use)
            if csvfile:
                csvfile.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csvfile, index=False)
                print(f"  → saved {csvfile.relative_to(ROOT)}")

    return build_profile(df, cfg["vent_height_m"])


def run_forward(cfg: dict, eruption_key: str, output_dir: Path):
    t_label = cfg["analysis_utc"] + " UTC"
    reanalysis = cfg["reanalysis"].upper()
    name_ja = cfg.get("name_ja", eruption_key)

    r0_list = cfg["forward"]["r0_list"]
    u0_list = cfg["forward"]["u0_list"]
    T0 = float(cfg["forward"]["T0_K"])
    n0 = float(cfg["forward"]["n0"])

    print(f"[{eruption_key}] {name_ja}  ({reanalysis}  {t_label})")
    print(f"  r0_list = {r0_list}")
    print(f"  u0_list = {u0_list}")
    print(f"  T0={T0} K  n0={n0}")

    df_rel, v_func, tempa_func, p_func = load_atmosphere(cfg)
    z_stop = float(df_rel["z_rel_m"].max())
    params = PlumeParams(T0=T0, n0=n0)

    # --- run model --------------------------------------------------------
    results: dict[float, dict[int, dict]] = {}
    for r0 in r0_list:
        results[r0] = {}
        for u0 in u0_list:
            res = run_plume(r0, u0, params, v_func, tempa_func, p_func, z_stop=z_stop)
            results[r0][u0] = res
            z_max = float(np.max(res["z"])) if len(res["z"]) > 0 else 0.0
            print(f"    r0={r0:>5.0f} m  u0={u0:>3d} m/s  Q={res['Q']:.2e} kg/s  "
                  f"z_max={z_max:.0f} m")

    # --- plot -------------------------------------------------------------
    n_rows = len(r0_list)
    colors = ["tab:blue", "tab:green", "tab:red", "tab:orange", "tab:purple"]

    fig, axs = plt.subplots(nrows=n_rows, ncols=3, figsize=(15, 4.5 * n_rows), sharey=True)
    if n_rows == 1:
        axs = axs[np.newaxis, :]

    x_max = 10_000 if cfg["reanalysis"] == "20cr" else 5_000

    for i, r0 in enumerate(r0_list):
        for j, u0 in enumerate(u0_list):
            res = results[r0][u0]
            label = f"u0={u0} m/s  Q={res['Q']:.2e} kg/s"
            c = colors[j % len(colors)]
            axs[i, 0].plot(res["u"], res["z"], label=label, color=c)
            axs[i, 1].plot(res["x"], res["z"], color=c)

        axs[i, 0].set_ylabel("Height above vent (m)")
        axs[i, 0].set_xlabel("Velocity (m/s)")
        axs[i, 0].set_title(f"Velocity profile  r0={int(r0)} m")
        axs[i, 0].legend(fontsize=8)
        axs[i, 0].grid(True)

        axs[i, 1].set_title("Horizontal distance")
        axs[i, 1].set_xlabel("Distance (m)")
        axs[i, 1].set_xlim(0, x_max)
        axs[i, 1].grid(True)

    # wind profile in top-right panel
    ax_wind = axs[0, 2]
    ax_dir = ax_wind.twiny()
    ax_wind.plot(df_rel["WindSpeed_mps"], df_rel["z_rel_m"], "b")
    ax_wind.set_xlabel("Wind speed (m/s)", color="b")
    ax_wind.tick_params(axis="x", colors="b")
    ax_wind.set_xlim(0, 80)
    ax_wind.set_title(f"{reanalysis} wind profile\n{eruption_key}  {t_label}")
    ax_wind.grid(True)

    ax_dir.plot(df_rel["WindDirection_deg"], df_rel["z_rel_m"], "r")
    ax_dir.set_xlabel("Wind direction (deg from N)", color="r")
    ax_dir.tick_params(axis="x", colors="r")
    ax_dir.set_xlim(0, 360)

    for i in range(1, n_rows):
        axs[i, 2].axis("off")

    for i in range(n_rows):
        for j in range(2):
            axs[i, j].set_ylim(0, 30_000)

    plt.tight_layout()

    out_png = output_dir / f"plume_forward_{eruption_key}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  → saved {out_png.relative_to(ROOT)}")


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
        out_dir = ROOT / args.output_dir / key / "forward"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_forward(catalog[key], key, out_dir)


if __name__ == "__main__":
    main()
