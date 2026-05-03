#!/usr/bin/env python3
"""
Meteorological data fetch test.

For each eruption in the catalog, attempts to load atmospheric data and
reports success/failure, elapsed time, and any error messages.

- ERA5  : reads local NetCDF file under data/; with --test-download, temporarily
          moves the file aside and re-downloads it via CDS API to test the full flow.
- 20CR  : connects to NOAA PSL OPeNDAP (configurable timeout)

Usage:
    uv run python scripts/fetch_data.py
    uv run python scripts/fetch_data.py --timeout 60
    uv run python scripts/fetch_data.py --test-download
"""
import argparse
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loader.cr20 import load_20cr
from loader.era5 import download_era5, load_era5_nc


def _load_era5_local(cfg: dict):
    ncfile = ROOT / cfg["era5_ncfile"]
    if not ncfile.exists():
        raise FileNotFoundError(f"NC file not found: {ncfile.relative_to(ROOT)}")
    return load_era5_nc(str(ncfile), cfg["lat"], cfg["lon"])


def _download_and_load_era5(cfg: dict):
    t_use = datetime.strptime(cfg["analysis_utc"], "%Y-%m-%d %H:%M")
    ncfile = ROOT / cfg["era5_ncfile"]
    dl = cfg.get("era5_download", {})
    download_era5(
        str(ncfile),
        year=str(t_use.year),
        month=f"{t_use.month:02d}",
        day=f"{t_use.day:02d}",
        time=t_use.strftime("%H:%M"),
        area=dl.get("area", [90, -180, -90, 180]),
    )
    return load_era5_nc(str(ncfile), cfg["lat"], cfg["lon"])


def _load_20cr(cfg: dict):
    t_use = datetime.strptime(cfg["analysis_utc"], "%Y-%m-%d %H:%M")
    return load_20cr(cfg["lat"], cfg["lon"], t_use)


def fetch_with_timeout(fn, timeout: int):
    """Run fn() in a daemon thread, return (df, elapsed) or raise on error/timeout."""
    result = [None]
    error = [None]

    def target():
        try:
            result[0] = fn()
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=target, daemon=True)
    t0 = time.perf_counter()
    t.start()
    t.join(timeout=timeout)
    elapsed = time.perf_counter() - t0

    if t.is_alive():
        raise TimeoutError(f"timed out after {timeout}s")
    if error[0] is not None:
        raise error[0]
    return result[0], elapsed


def test_era5_download(key: str, cfg: dict, timeout: int) -> dict:
    """Temporarily move the NC file aside, download via CDS API, then restore on failure."""
    ncfile = ROOT / cfg["era5_ncfile"]
    backup = ncfile.with_suffix(".nc.bak")

    # Move existing file aside
    ncfile.rename(backup)
    print(f"  (moved {ncfile.name} to .bak, downloading from CDS API ...)", flush=True)

    try:
        df, elapsed = fetch_with_timeout(
            lambda: _download_and_load_era5(cfg), timeout
        )
        backup.unlink()  # download succeeded — remove backup
        print(f"  download OK  ({len(df)} levels, {elapsed:.1f}s)")
        return {"key": key, "ok": True, "elapsed": elapsed, "error": None}

    except Exception as e:
        # Restore the original file so the repo is not left broken
        if backup.exists():
            backup.rename(ncfile)
            print(f"  (restored original {ncfile.name})")
        err = str(e)
        print(f"  download ERROR  {err}")
        return {"key": key, "ok": False, "elapsed": None, "error": err}


def test_eruption(key: str, cfg: dict, timeout: int, test_download: bool) -> dict:
    reanalysis = cfg["reanalysis"].upper()
    name_ja = cfg.get("name_ja", key)
    print(f"[{key}] {name_ja}  ({reanalysis})", end="", flush=True)

    try:
        if cfg["reanalysis"] == "era5":
            if test_download:
                print()
                return test_era5_download(key, cfg, timeout)
            else:
                print(end="  ... ", flush=True)
                df, elapsed = fetch_with_timeout(lambda: _load_era5_local(cfg), timeout)
        else:
            print(end="  ... ", flush=True)
            df, elapsed = fetch_with_timeout(lambda: _load_20cr(cfg), timeout)

        print(f"OK  ({len(df)} levels, {elapsed:.1f}s)")
        return {"key": key, "ok": True, "elapsed": elapsed, "error": None}

    except TimeoutError as e:
        print(f"TIMEOUT  ({e})")
        return {"key": key, "ok": False, "elapsed": timeout, "error": str(e)}
    except Exception as e:
        print(f"ERROR  {e}")
        return {"key": key, "ok": False, "elapsed": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Timeout in seconds (default: 600; ERA5 download can take several minutes)",
    )
    parser.add_argument(
        "--test-download", action="store_true",
        help="For ERA5 eruptions: move the local NC file aside and re-download via CDS API",
    )
    args = parser.parse_args()

    with open(ROOT / "eruptions" / "catalog.yaml") as f:
        catalog = yaml.safe_load(f)["eruptions"]

    mode = "download test" if args.test_download else "local file"
    print(f"Fetching data for {len(catalog)} eruptions")
    print(f"  ERA5: {mode}  |  20CR timeout: {args.timeout}s\n")

    results = []
    for key, cfg in catalog.items():
        results.append(test_eruption(key, cfg, args.timeout, args.test_download))

    n_ok = sum(r["ok"] for r in results)
    n_total = len(results)

    print(f"\n{'─' * 50}")
    print(f"Result: {n_ok}/{n_total} OK")
    if n_ok < n_total:
        print("Failed:")
        for r in results:
            if not r["ok"]:
                print(f"  {r['key']}  →  {r['error']}")

    # daemon threads may still be running (timed-out connections); force exit.
    sys.exit(0 if n_ok == n_total else 1)


if __name__ == "__main__":
    main()
