#!/usr/bin/env python3
"""
cloud.collect_results — Download and merge cloud pipeline results.

Each cloud worker uploads:
    SIMuLaTioN_NNNNNN/
        SIMuLaTioN_NNNNNN_CCF_RVs.csv   (simulated RV data)
        truth_row.csv                     (one-row truth table)
        pipeline_output/                  (lmfit_summary.csv, etc.)

This script:
    1. Downloads everything from GCS
    2. Merges all truth_row.csv into a combined orbital_params_truth.csv
    3. Merges all lmfit_summary.csv into a single DataFrame
    4. Joins results with truth for analysis

Usage:
    python -m cloud.collect_results --run-id 20240101_120000 --bucket binary-pipeline-sims
"""

import argparse
import glob
import os
import subprocess
import sys

import pandas as pd


BUCKET = "binary-pipeline-sims"
LOCAL_OUTPUT = "/tmp/cloud_results"


def download_results(bucket, gcs_prefix, local_dir):
    """Download all results from GCS recursively."""
    gcs_path = f"gs://{bucket}/{gcs_prefix}/output/"
    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading results from {gcs_path} ...")
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", gcs_path, local_dir],
        check=True,
    )
    print(f"Downloaded to {local_dir}")


def merge_truth_rows(results_dir):
    """Find all per-task truth_row.csv files and merge into combined truth table."""
    pattern = os.path.join(results_dir, "**", "truth_row.csv")
    truth_files = glob.glob(pattern, recursive=True)

    if not truth_files:
        print("No truth_row.csv files found!")
        return None

    print(f"Found {len(truth_files)} truth_row.csv files")

    dfs = []
    for path in sorted(truth_files):
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: failed to read {path}: {e}")

    if not dfs:
        return None

    truth_df = pd.concat(dfs, ignore_index=True)
    truth_df.sort_values("sim_id", inplace=True)
    print(f"Merged truth table: {len(truth_df)} simulations")
    return truth_df


def merge_lmfit_summaries(results_dir):
    """Find all lmfit_summary.csv files and merge into one DataFrame."""
    pattern = os.path.join(results_dir, "**", "lmfit_summary.csv")
    csv_files = glob.glob(pattern, recursive=True)

    if not csv_files:
        print("No lmfit_summary.csv files found!")
        return None

    print(f"Found {len(csv_files)} lmfit_summary.csv files")

    dfs = []
    for csv_path in sorted(csv_files):
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: failed to read {csv_path}: {e}")

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    print(f"Merged results: {len(merged)} total rows from {len(dfs)} files")
    return merged


def join_with_truth(merged_df, truth_df):
    """Join pipeline results with truth table on simulation ID."""
    # Extract sim_id from star_name (e.g., "BLOeM_SIM_000003" -> 3)
    # The pipeline renames SIMuLaTioN_NNNNNN to BLOeM_SIM_NNNNNN internally
    merged_df["sim_id"] = (
        merged_df["star_name"]
        .str.extract(r"_(\d+)$", expand=False)
        .astype(float)
    )

    # Join on sim_id
    joined = merged_df.merge(
        truth_df,
        on="sim_id",
        how="left",
        suffixes=("_fit", "_true"),
    )

    return joined


def main():
    parser = argparse.ArgumentParser(description="Collect cloud pipeline results.")
    parser.add_argument("--run-id", required=True, help="Run ID from submission")
    parser.add_argument("--bucket", default=BUCKET, help=f"GCS bucket (default: {BUCKET})")
    parser.add_argument("--output-dir", default=LOCAL_OUTPUT, help="Local output directory")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing local files")

    args = parser.parse_args()
    gcs_prefix = f"runs/{args.run_id}"
    local_dir = os.path.join(args.output_dir, args.run_id)

    if not args.skip_download:
        download_results(args.bucket, gcs_prefix, local_dir)

    # Find the output directory (gsutil -m cp -r may nest it)
    results_dir = os.path.join(local_dir, "output")
    if not os.path.isdir(results_dir):
        # Try flat structure
        results_dir = local_dir

    # 1) Merge all per-task truth rows into combined truth table
    truth_df = merge_truth_rows(results_dir)
    if truth_df is not None:
        truth_path = os.path.join(local_dir, "orbital_params_truth.csv")
        truth_df.to_csv(truth_path, index=False)
        print(f"Combined truth table: {truth_path}")

    # 2) Merge all lmfit results
    merged_df = merge_lmfit_summaries(results_dir)

    if merged_df is None:
        print("No results to merge.")
        sys.exit(1)

    # Save merged results
    merged_path = os.path.join(local_dir, "merged_lmfit_results.csv")
    merged_df.to_csv(merged_path, index=False)
    print(f"\nMerged results: {merged_path}")

    # 3) Join with truth
    if truth_df is not None:
        joined = join_with_truth(merged_df, truth_df)
        joined_path = os.path.join(local_dir, "results_with_truth.csv")
        joined.to_csv(joined_path, index=False)
        print(f"Results + truth: {joined_path}")
        print(f"\nTruth table:  {len(truth_df)} simulations")
        print(f"Results:      {len(merged_df)} rows ({merged_df['star_name'].nunique()} unique stars)")

        # Quick summary of best-row results
        best_mask = merged_df.get("is_best", pd.Series(dtype=bool)).fillna(False).astype(bool)
        n_best = best_mask.sum()
        if n_best > 0:
            print(f"Best rows:    {n_best}")
    else:
        print("Warning: no truth rows found, skipping join.")

    print(f"\nReady for analysis:")
    print(f"  python -m cloud.analyze_results --run-id {args.run_id}")


if __name__ == "__main__":
    main()
