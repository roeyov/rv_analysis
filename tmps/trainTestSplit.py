#!/usr/bin/env python3
"""
Add a per-row train/test boolean flag to each Parquet file in a directory tree.

Usage:
    python add_train_test_flag.py \
        --data_dir /path/to/samples \
        --pattern 'noise_analysis_*' \
        --train_frac 0.8 \
        --seed 42 \
        [--out_dir /path/to/output]

If --out_dir is omitted, each file is overwritten in place.
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np

def main(data_dir, pattern, train_frac, seed, out_dir=None):
    # discover all .parquet files under data_dir/<pattern>/
    search = os.path.join(data_dir, pattern, '*.parquet')
    files = sorted(glob.glob(search))
    if not files:
        print(f"No files found matching {search}")
        return

    # rng for reproducibility
    rng = np.random.default_rng(seed)

    for file_path in files:
        # load the file
        df = pd.read_parquet(file_path)

        # per-row random mask: True for train, False for test
        mask = rng.random(size=len(df)) < train_frac
        df['is_train'] = mask

        # determine where to write
        if out_dir:
            rel_dir = os.path.relpath(os.path.dirname(file_path), data_dir)
            dst_dir = os.path.join(out_dir, rel_dir)
            os.makedirs(dst_dir, exist_ok=True)
            dest_path = os.path.join(dst_dir, os.path.basename(file_path))
        else:
            dest_path = file_path

        # write back
        df.to_parquet(dest_path, index=False)
        n_train = int(mask.sum())
        n_test  = len(df) - n_train
        print(f"Written {dest_path}: {n_train} train rows, {n_test} test rows")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Add per-row train/test flags to Parquet files")
    p.add_argument("--data_dir", required=True,
                   help="Root directory containing subfolders of Parquet files")
    p.add_argument("--pattern", default="*",
                   help="Subdirectory glob pattern (e.g. 'noise_analysis_*')")
    p.add_argument("--train_frac", type=float, default=0.8,
                   help="Fraction of rows to mark as training")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--out_dir", default=None,
                   help="If provided, write flagged files here (preserves subdirs); otherwise overwrite")
    args = p.parse_args()

    main(
        data_dir  = args.data_dir,
        pattern   = args.pattern,
        train_frac= args.train_frac,
        seed      = args.seed,
        out_dir   = args.out_dir
    )
