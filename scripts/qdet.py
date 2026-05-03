#!/usr/bin/env python3
"""
Q determination: find mass flux Q for each (r0, T0, n0) that matches
the observed plume height (z_target) using bisection on u0.

Usage:
    uv run python scripts/qdet.py --eruption Tokachi1962
    uv run python scripts/qdet.py --eruption all
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atmos_profile import build_profile
from loader.cr20 import load_20cr
from loader.era5 import download_era5, load_era5_nc
from plume_model import PlumeParams, solve_u0_for_target


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


def run_qdet(cfg: dict, eruption_key: str, output_dir: Path):
    t_label = cfg["analysis_utc"] + " UTC"
    reanalysis = cfg["reanalysis"].upper()

    qcfg = cfg["qdet"]
    z_target = float(qcfg["z_target_m"])
    r0_grid = qcfg["r0_grid"]
    T0_grid = qcfg["T0_grid_K"]
    n0_grid = qcfg["n0_grid"]

    print(f"[{eruption_key}]  ({reanalysis}  {t_label})")
    print(f"  z_target = {z_target:.0f} m")
    print(f"  r0_grid  = {r0_grid}")
    print(f"  T0_grid  = {T0_grid}")
    print(f"  n0_grid  = {n0_grid}")
    total = len(r0_grid) * len(T0_grid) * len(n0_grid)
    print(f"  {total} combinations …")

    df_rel, v_func, tempa_func, p_func = load_atmosphere(cfg)
    z_stop = float(df_rel["z_rel_m"].max())

    rows = []
    for r0 in r0_grid:
        for T0 in T0_grid:
            for n0 in n0_grid:
                params = PlumeParams(T0=float(T0), n0=float(n0))
                u0_sol, Q_sol, z_sol, ok = solve_u0_for_target(
                    r0=float(r0),
                    params=params,
                    z_target=z_target,
                    v_func=v_func,
                    tempa_func=tempa_func,
                    p_func=p_func,
                    z_stop=z_stop,
                )
                rows.append({
                    "r0_m":   r0,
                    "T0_K":   T0,
                    "n0":     n0,
                    "u0_mps": u0_sol,
                    "Q_kgps": Q_sol,
                    "zmax_m": z_sol,
                    "ok":     ok,
                })

    df = pd.DataFrame(rows)
    df["dz_m"] = (df["zmax_m"] - z_target).abs()

    t_str = datetime.strptime(cfg["analysis_utc"], "%Y-%m-%d %H:%M").strftime("%Y%m%d_%HUTC")
    fname = f"{eruption_key.lower()}_qdet_z{int(z_target)}m_{t_str}.csv"
    out_csv = output_dir / fname
    df.to_csv(out_csv, index=False)
    print(f"  → saved {out_csv.relative_to(ROOT)}")

    # Print top results
    ok_df = df[df["ok"]].sort_values("dz_m")
    if ok_df.empty:
        print("  No converged solutions found.")
    else:
        print(f"\n  Top 10 solutions (sorted by |z_max - z_target|):")
        cols = ["r0_m", "T0_K", "n0", "u0_mps", "Q_kgps", "zmax_m", "dz_m"]
        print(ok_df[cols].head(10).to_string(index=False,
              float_format=lambda x: f"{x:.4g}"))


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
        out_dir = ROOT / args.output_dir / key / "qdet"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_qdet(catalog[key], key, out_dir)


if __name__ == "__main__":
    main()
