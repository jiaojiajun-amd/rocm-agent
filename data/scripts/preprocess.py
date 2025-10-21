#!/usr/bin/env python3
"""
Download SWE-bench datasets from Hugging Face and save as Parquet files.

- SWE-bench/SWE-bench_Verified test split -> test.parquet
- SWE-bench/SWE-bench train split -> train.parquet

Requirements:
  pip install datasets pyarrow

Usage:
  python save_swebench_to_parquet.py --outdir ./data
"""

import argparse
from pathlib import Path

from datasets import load_dataset


def save_parquet(dataset_name: str, split: str, out_path: Path) -> None:
    print(f"Loading {dataset_name} split='{split}' ...")
    ds = load_dataset(dataset_name, split=split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_path} ...")
    ds.to_parquet(str(out_path))
    print(f"Done: {out_path} (rows={len(ds)})")


def main():
    parser = argparse.ArgumentParser(description="Save SWE-bench Hugging Face datasets to Parquet.")
    parser.add_argument("--outdir", type=Path, default=Path("./data"), help="Output directory for parquet files.")
    args = parser.parse_args()

    # SWE-bench_Verified test split -> test.parquet
    save_parquet(
        dataset_name="SWE-bench/SWE-bench_Lite",
        split="dev",
        out_path=args.outdir / "test.parquet",
    )

    # SWE-bench train split -> train.parquet
    save_parquet(
        dataset_name="SWE-bench/SWE-bench_Lite",
        split="test",
        out_path=args.outdir / "train.parquet",
    )


if __name__ == "__main__":
    main()
