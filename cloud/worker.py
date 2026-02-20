"""
cloud.worker — Container entry point: generate one simulation + run pipeline.

Called inside a Google Cloud Batch task.  Each task generates its own
simulation on the fly (no pre-upload), runs the pipeline evaluator,
and uploads **three** things to GCS:
    1. The simulated RV CSV      (input to the pipeline)
    2. A one-row truth CSV       (true orbital parameters)
    3. The lmfit output directory (pipeline results)

Environment variables (set by the batch job script):
    TASK_INDEX   : int — BATCH_TASK_INDEX (0 .. n_tasks-1)
    BASE_SEED    : int — base random seed from orchestrator
    N_FIELDS     : int — number of BLOeM MJD fields (8)
    GCS_OUTPUT_DIR : gs:// prefix for uploading everything
    CONFIG_PATH  : (optional) path to params YAML, default /app/params_cloud.yaml

Seed & field logic:
    seed      = BASE_SEED + TASK_INDEX   (unique per task)
    field_idx = TASK_INDEX % N_FIELDS    (cycles 0-7, 0-7, …)
"""

import os
import sys

import numpy as np
import pandas as pd

from google.cloud import storage

from simulations.common import BLOEM_MJD_ARRAYS
from simulations.create_binary_simulations import (
    sample_orbital_params,
    generate_binary_rv_at_mjds,
)


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def upload_file(local_path, gcs_path):
    """Upload a single local file to gs://bucket/blob."""
    parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name, blob_name = parts[0], parts[1]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"  Uploaded {local_path} -> {gcs_path}")


def upload_directory(local_dir, gcs_prefix):
    """Recursively upload local_dir/* to gs://bucket/prefix/..."""
    parts = gcs_prefix.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _dirs, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            blob_name = f"{prefix}/{rel_path}" if prefix else rel_path
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
    print(f"  Uploaded dir {local_dir} -> {gcs_prefix}")


# ---------------------------------------------------------------------------
# Simulation generation (one system)
# ---------------------------------------------------------------------------

def generate_one_simulation(task_index, base_seed, n_fields):
    """
    Generate a single binary RV simulation.

    Returns
    -------
    local_csv : str
        Path to the written *_CCF_RVs.csv file.
    truth_csv : str
        Path to the written truth_row.csv file.
    """
    seed = base_seed + task_index
    field_idx = task_index % n_fields
    rng = np.random.default_rng(seed)

    mjds = np.asarray(BLOEM_MJD_ARRAYS[field_idx], dtype=float)

    # Sample orbital parameters (unique per seed)
    orb = sample_orbital_params(rng=rng)

    # Generate noisy Keplerian RVs at the field's MJD times
    rvs, sigmas = generate_binary_rv_at_mjds(mjds, orb)

    if rvs is None:
        print(f"WARNING: Kepler solver failed for task {task_index} (seed={seed})")
        return None, None

    # --- Write pipeline-compatible CSV ---
    input_dir = "/app/input"
    os.makedirs(input_dir, exist_ok=True)

    sim_name = f"SIMuLaTioN_{task_index:06d}"
    csv_filename = f"{sim_name}_CCF_RVs.csv"
    local_csv = os.path.join(input_dir, csv_filename)

    df = pd.DataFrame({
        "Mean RV": rvs,
        "Mean RVsig": sigmas,
        "MJD": mjds,
        "SNR_PPL": 100.0 * np.ones_like(rvs),
        "SNR": 100.0 * np.ones_like(rvs),
    })
    df.to_csv(local_csv, index=False)

    # --- Write one-row truth CSV ---
    truth_row = pd.DataFrame([{
        "sim_id": task_index,
        "filename": csv_filename,
        "field_idx": field_idx,
        "seed": seed,
        "n_obs": len(mjds),
        "T0": orb["t0"],
        "Period": orb["period"],
        "Eccentricity": orb["ecc"],
        "OMEGA_rad": orb["omega"],
        "OMEGA_deg": np.degrees(orb["omega"]),
        "K1": orb["k1"],
        "K2": orb["k2"],
        "GAMMA": orb["gamma"],
        "Mass1": orb["m1"],
        "MassRatio": orb["q"],
        "Inclination_rad": orb["inc"],
        "Inclination_deg": np.degrees(orb["inc"]),
    }])
    truth_csv = os.path.join(input_dir, "truth_row.csv")
    truth_row.to_csv(truth_csv, index=False)

    print(f"Generated simulation: {csv_filename}")
    print(f"  seed={seed}, field={field_idx}, P={orb['period']:.2f}d, "
          f"e={orb['ecc']:.3f}, K1={orb['k1']:.1f}km/s")

    return local_csv, truth_csv


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

def main():
    task_index = int(os.environ.get("TASK_INDEX", os.environ.get("BATCH_TASK_INDEX", "0")))
    base_seed = int(os.environ.get("BASE_SEED", "42"))
    n_fields = int(os.environ.get("N_FIELDS", str(len(BLOEM_MJD_ARRAYS))))
    gcs_output = os.environ.get("GCS_OUTPUT_DIR")
    config_path = os.environ.get("CONFIG_PATH", "/app/params_cloud.yaml")

    if not gcs_output:
        print("ERROR: GCS_OUTPUT_DIR must be set", file=sys.stderr)
        sys.exit(1)

    sim_name = f"SIMuLaTioN_{task_index:06d}"
    task_gcs = f"{gcs_output}/{sim_name}"

    print(f"=== Worker task {task_index} ===")
    print(f"  base_seed={base_seed}, n_fields={n_fields}")
    print(f"  output -> {task_gcs}")

    # 1) Generate simulation on the fly
    local_csv, truth_csv = generate_one_simulation(task_index, base_seed, n_fields)
    if local_csv is None:
        print("Simulation generation failed (Kepler solver). Exiting.")
        sys.exit(1)

    # 2) Run pipeline
    from pipeline.evaluator import main_single

    output_dir = "/app/output/"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running pipeline on {os.path.basename(local_csv)} ...")
    main_single(local_csv, output_dir, use_fwhm=True, json_param_file=config_path)

    # 3) Upload everything to GCS
    print("Uploading results ...")

    # a) The simulated RV CSV (input data)
    upload_file(local_csv, f"{task_gcs}/{os.path.basename(local_csv)}")

    # b) The truth row
    upload_file(truth_csv, f"{task_gcs}/truth_row.csv")

    # c) The pipeline output directory
    upload_directory(output_dir, f"{task_gcs}/pipeline_output")

    print(f"=== Worker task {task_index} done ===")


if __name__ == "__main__":
    main()
