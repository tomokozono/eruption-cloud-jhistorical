#!/usr/bin/env python3
"""
Merge forward3 figures horizontally in chronological eruption order.

Reads output/<key>/forward3/plume_forward3_<key>.png for each eruption
and concatenates them left-to-right, saving to output/forward3_merged.png.

Usage:
    uv run python scripts/merge_forward3.py
    uv run python scripts/merge_forward3.py --output-dir output
"""
import argparse
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).parent.parent

ERUPTION_ORDER = ["Sakurajima1914", "Komagatake1929", "Tokachi1962", "Usu1977"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", default="output",
                        help="Root output directory (default: output/)")
    args = parser.parse_args()

    out_root = ROOT / args.output_dir

    images = []
    for key in ERUPTION_ORDER:
        png = out_root / key / "forward3" / f"plume_forward3_{key}.png"
        if not png.exists():
            raise SystemExit(f"Missing: {png}\nRun forward3.py first.")
        img = Image.open(png)
        images.append(img)
        print(f"  loaded {png.relative_to(ROOT)}  ({img.width}×{img.height})")

    gap = 30  # pixels between figures
    total_w = sum(img.width for img in images) + gap * (len(images) - 1)
    max_h   = max(img.height for img in images)

    merged = Image.new("RGB", (total_w, max_h), color=(255, 255, 255))
    x = 0
    for img in images:
        merged.paste(img, (x, 0))
        x += img.width + gap

    out_path = out_root / "forward3_merged.png"
    merged.save(out_path, dpi=(150, 150))
    print(f"\n→ saved {out_path.relative_to(ROOT)}  ({merged.width}×{merged.height})")


if __name__ == "__main__":
    main()
