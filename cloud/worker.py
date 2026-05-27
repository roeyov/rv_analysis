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
    RUN_ID       : str — unique run identifier (timestamp or user-supplied)
    N_FIELDS     : int — number of BLOeM MJD fields (8)
    GCS_OUTPUT_DIR : gs:// prefix for uploading everything
    CONFIG_PATH  : (optional) path to params YAML, default /app/configs/params_cloud.yaml

Seed & field logic:
    seed      = sha256(run_id + task_index) % 2^31  (unique per run × task)
    field_idx = TASK_INDEX % N_FIELDS               (cycles 0-7, 0-7, …)
"""

import hashlib
import os
import sys

import numpy as np
import pandas as pd

try:
    from google.cloud import storage
except ImportError:
    storage = None  # Not needed for local runs

from simulations.common import BLOEM_MJD_ARRAYS
from simulations.create_binary_simulations import (
    sample_orbital_params,
    generate_binary_rv_at_mjds,
    is_roche_valid,
    build_mass_row,
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

def generate_one_simulation(task_index, run_id, n_fields, input_dir="/app/input"):
    """
    Generate a single binary RV simulation.

    Returns
    -------
    local_csv : str or None
        Path to the written *_CCF_RVs.csv file.
    truth_csv : str or None
        Path to the written truth_row.csv file.
    massdf : pd.DataFrame or None
        One-row DataFrame in mass_bloem.csv format for Roche lobe constraints.
    """
    seed = int(hashlib.sha256(f"{run_id}_{task_index}".encode()).hexdigest(), 16) % (2**31)
    field_idx = task_index % n_fields
    rng = np.random.default_rng(seed)

    mjds = np.asarray(BLOEM_MJD_ARRAYS[field_idx], dtype=float)

    # Sample orbital parameters (unique per seed)
    orb = sample_orbital_params(rng=rng)

    # Validate Roche lobe constraint
    if not is_roche_valid(orb["m1"], orb["r_phys"], orb["period"],
                          orb["k1"], orb["ecc"], orb["inc"]):
        print(f"WARNING: Roche lobe violation for task {task_index} (seed={seed}), "
              f"P={orb['period']:.2f}d, R={orb['r_phys']:.1f}Rsun")
        return None, None, None

    # Generate noisy Keplerian RVs at the field's MJD times
    rvs, sigmas = generate_binary_rv_at_mjds(mjds, orb)

    if rvs is None:
        print(f"WARNING: Kepler solver failed for task {task_index} (seed={seed})")
        return None, None, None

    # --- Write pipeline-compatible CSV ---
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
        "run_id": run_id,
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
        "R_star": orb["r_phys"],
    }])
    truth_csv = os.path.join(input_dir, "truth_row.csv")
    truth_row.to_csv(truth_csv, index=False)

    # --- Build mass DataFrame for pipeline Roche lobe constraints ---
    sim_id_str = f"BLOeM_SIM_{task_index:06d}"
    massdf = pd.DataFrame([build_mass_row(sim_id_str, orb["m1"], orb["r_phys"])])

    print(f"Generated simulation: {csv_filename}")
    print(f"  {'Seed':>12}: {seed}")
    print(f"  {'Field':>12}: {field_idx}  ({len(mjds)} obs)")
    print(f"  {'Period':>12}: {orb['period']:.4f} d")
    print(f"  {'Ecc':>12}: {orb['ecc']:.4f}")
    print(f"  {'omega':>12}: {orb['omega']:.4f} rad")
    print(f"  {'T0':>12}: {orb['t0']:.4f}")
    print(f"  {'K1':>12}: {orb['k1']:.2f} km/s")
    print(f"  {'K2':>12}: {orb['k2']:.2f} km/s")
    print(f"  {'gamma':>12}: {orb['gamma']:.2f} km/s")
    print(f"  {'M1':>12}: {orb['m1']:.2f} Msun")
    print(f"  {'q':>12}: {orb['q']:.4f}")
    print(f"  {'inc':>12}: {np.degrees(orb['inc']):.2f} deg")
    print(f"  {'R_star':>12}: {orb['r_phys']:.2f} Rsun")

    return local_csv, truth_csv, massdf


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

def main():
    task_index = int(os.environ.get("TASK_INDEX", os.environ.get("BATCH_TASK_INDEX", "0")))
    run_id = os.environ.get("RUN_ID", "unknown")
    n_fields = int(os.environ.get("N_FIELDS", str(len(BLOEM_MJD_ARRAYS))))
    gcs_output = os.environ.get("GCS_OUTPUT_DIR")
    config_path = os.environ.get("CONFIG_PATH", "/app/configs/params_cloud.yaml")

    if not gcs_output:
        print("ERROR: GCS_OUTPUT_DIR must be set", file=sys.stderr)
        sys.exit(1)

    sim_name = f"SIMuLaTioN_{task_index:06d}"
    task_gcs = f"{gcs_output}/{sim_name}"

    print(f"=== Worker task {task_index} ===")
    print(f"  run_id={run_id}, n_fields={n_fields}")
    print(f"  output -> {task_gcs}")

    # 1) Generate simulation on the fly
    local_csv, truth_csv, massdf = generate_one_simulation(task_index, run_id, n_fields)
    if local_csv is None:
        print("Simulation generation failed (Roche/Kepler). Exiting.")
        sys.exit(1)

    # 2) Run pipeline
    from pipeline.evaluator import main_single

    output_dir = "/app/output/"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running pipeline on {os.path.basename(local_csv)} ...")
    main_single(local_csv, output_dir, massdf=massdf, use_fwhm=True,
                json_param_file=config_path)

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
