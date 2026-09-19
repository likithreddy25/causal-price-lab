"""Merges per-category real datasets (each produced by
real_pipeline/build_dataset.py) into one combined parquet for a
multi-category PSM run.

This is a separate, restartable step rather than one multi-category call to
build_dataset.py's own build(), because the real Amazon-Reviews-2023
category files vary enormously in size (Office_Products' raw metadata is
~120MB; Home_and_Kitchen's is closer to 1GB with 3.7M+ rows before
filtering), and downloading + embedding several of them back-to-back in one
long process risks losing everything to a single stall, OOM, or network
blip. Building one category at a time, verifying each parquet
independently, and merging afterward is the practical way to do this on a
laptop rather than a well-resourced server.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Per-category parquet files produced by build_dataset.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/real_dataset_combined.parquet"),
    )
    args = parser.parse_args()

    frames = [pd.read_parquet(p) for p in args.inputs]
    combined = pd.concat(frames, ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out)

    print(f"Merged {len(args.inputs)} category file(s) into {len(combined)} products -> {args.out}")
    print(combined["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
