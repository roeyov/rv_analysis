#!/usr/bin/env python3
"""
cloud.run_cloud_batch — Orchestrate cloud simulation runs on Google Cloud Batch.

Architecture (revised):
    - No local simulation generation or upload
    - Each cloud task generates its own simulation on the fly
    - Seed derived from run_id + task_index (unique per run, reproducible)
    - Workers upload results + truth rows + simulated RV CSV to GCS

Usage:
    python -m cloud.run_cloud_batch \\
        --n-per-field 1 \\
        --project binary-pipeline \\
        --region us-central1 \\
        --bucket binary-pipeline-sims \\
        --run-id my-run-001

Cost estimate (e2-highcpu-16 SPOT, ~19 min median):
    ~$0.056/task
    N=1   ->   8 tasks -> ~$0.45
    N=10  ->  80 tasks -> ~$4.50
    N=100 -> 800 tasks -> ~$45
    N=1000-> 8000 tasks-> ~$450
"""

import argparse
import json
import os
import subprocess
import sys
import time

from simulations.common import BLOEM_MJD_ARRAYS
from cloud.worker import generate_one_simulation
from cloud.collect_results import merge_truth_rows, merge_lmfit_summaries, join_with_truth


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

PROJECT_ID = "binary-pipeline"
REGION = "us-central1"
BUCKET = "binary-pipeline-sims"
IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/binary-pipeline/worker"
N_PER_FIELD = 1
N_FIELDS = len(BLOEM_MJD_ARRAYS)         # 8
MAX_PARALLEL = 16     # max concurrent tasks in batch job
MACHINE_TYPE = "e2-highcpu-16"  # 16 vCPU, 16 GB — SPOT for LS permutation MP


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------

LOCAL_OUTPUT = "/tmp/cloud_results"


def run_local(n_tasks, n_fields, run_id, config_path):
    """Run all tasks locally and sequentially, mirroring cloud directory layout."""
    from pipeline.evaluator import main_single

    run_dir = os.path.join(LOCAL_OUTPUT, run_id)
    output_base = os.path.join(run_dir, "output")

    failed = []
    for task_index in range(n_tasks):
        sim_name = f"SIMuLaTioN_{task_index:06d}"
        sim_dir = os.path.join(output_base, sim_name)
        os.makedirs(sim_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  Task {task_index + 1}/{n_tasks}  ({sim_name})")
        print(f"  field={task_index % n_fields}")
        print(f"{'=' * 60}")

        # 1) Generate simulation — write CSV + truth into sim_dir
        local_csv, truth_csv, massdf = generate_one_simulation(
            task_index, run_id, n_fields, input_dir=sim_dir,
        )
        if local_csv is None:
            print(f"Skipping {sim_name} (generation failed)")
            failed.append(task_index)
            continue

        # 2) Run pipeline — output into sim_dir/pipeline_output/
        pipeline_out = os.path.join(sim_dir, "pipeline_output")
        os.makedirs(pipeline_out, exist_ok=True)

        try:
            main_single(local_csv, pipeline_out, massdf=massdf,
                        use_fwhm=True, json_param_file=config_path)
        except Exception as e:
            print(f"ERROR running pipeline for {sim_name}: {e}")
            failed.append(task_index)

    # 3) Merge results (same logic as collect_results.py)
    print(f"\n{'=' * 60}")
    print("  Merging results")
    print(f"{'=' * 60}")

    truth_df = merge_truth_rows(output_base)
    if truth_df is not None:
        truth_path = os.path.join(run_dir, "orbital_params_truth.csv")
        truth_df.to_csv(truth_path, index=False)
        print(f"Combined truth table: {truth_path}")

    merged_df = merge_lmfit_summaries(output_base)
    if merged_df is not None:
        merged_path = os.path.join(run_dir, "merged_lmfit_results.csv")
        merged_df.to_csv(merged_path, index=False)
        print(f"Merged results: {merged_path}")

        if truth_df is not None:
            joined = join_with_truth(merged_df, truth_df)
            joined_path = os.path.join(run_dir, "results_with_truth.csv")
            joined.to_csv(joined_path, index=False)
            print(f"Results + truth: {joined_path}")
    else:
        print("No lmfit results to merge.")

    print(f"\nLocal run complete: {run_dir}")
    if failed:
        print(f"  Failed tasks: {failed}")
    print(f"  Succeeded: {n_tasks - len(failed)}/{n_tasks}")

    return run_dir


# ---------------------------------------------------------------------------
# Submit Google Cloud Batch job
# ---------------------------------------------------------------------------

def submit_batch_job(n_tasks, bucket, gcs_prefix, image, machine_type, run_id,
                     n_fields,
                     project=PROJECT_ID, region=REGION, max_parallel=MAX_PARALLEL):
    """Submit a Cloud Batch job where each task generates + processes one simulation."""
    job_id = f"sim-{run_id}"

    gcs_output = f"gs://{bucket}/{gcs_prefix}/output"

    # Each task gets BATCH_TASK_INDEX automatically from Cloud Batch.
    # The worker uses: seed = sha256(run_id + task_index) % 2^31
    #                  field_idx = BATCH_TASK_INDEX % N_FIELDS
    task_script = f"""#!/bin/bash
set -e
export TASK_INDEX=$BATCH_TASK_INDEX
export RUN_ID="{run_id}"
export N_FIELDS={n_fields}
export GCS_OUTPUT_DIR="{gcs_output}"
export CONFIG_PATH="/app/configs/params_cloud.yaml"
echo "Task $TASK_INDEX: run_id={run_id}, field=$((TASK_INDEX % N_FIELDS))"
python -m cloud.worker
"""

    job_spec = {
        "taskGroups": [{
            "taskSpec": {
                "runnables": [{
                    "container": {
                        "imageUri": image,
                        "entrypoint": "/bin/bash",
                        "commands": ["-c", task_script],
                    }
                }],
                "computeResource": {
                    "cpuMilli": 16000,     # 16 vCPU
                    "memoryMib": 16384,    # 16 GB
                },
                "maxRunDuration": "1800s",  # 30 min timeout per task
                "maxRetryCount": 1,
            },
            "taskCount": str(n_tasks),
            "parallelism": str(min(n_tasks, max_parallel)),
        }],
        "allocationPolicy": {
            "instances": [{
                "policy": {
                    "machineType": machine_type,
                    "provisioningModel": "SPOT",
                }
            }],
            "location": {
                "allowedLocations": [
                    f"zones/{region}-a",
                    f"zones/{region}-b",
                    f"zones/{region}-c",
                ]
            }
        },
        "logsPolicy": {
            "destination": "CLOUD_LOGGING"
        },
    }

    job_json_path = f"/tmp/batch_job_{run_id}.json"
    with open(job_json_path, "w") as f:
        json.dump(job_spec, f, indent=2)

    print(f"\nSubmitting Cloud Batch job: {job_id}")
    print(f"  Tasks:       {n_tasks}")
    print(f"  Parallelism: {min(n_tasks, max_parallel)}")
    print(f"  Machine:     {machine_type} (SPOT)")
    print(f"  Image:       {image}")
    print(f"  Output:      {gcs_output}")
    print(f"  Run ID:      {run_id}")
    print(f"  Fields:      {n_fields}")

    cmd = [
        "gcloud", "batch", "jobs", "submit", job_id,
        f"--project={project}",
        f"--location={region}",
        f"--config={job_json_path}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR submitting job:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"\nJob submitted successfully!")
    print(f"Monitor: gcloud batch jobs describe {job_id} --project={project} --location={region}")
    print(f"Logs:    gcloud batch jobs logs {job_id} --project={project} --location={region}")

    return job_id


# ---------------------------------------------------------------------------
# Build & push Docker image
# ---------------------------------------------------------------------------

def build_and_push_image(project_id, region=REGION):
    """Build and push Docker image to Artifact Registry."""
    ar_repo = f"{region}-docker.pkg.dev/{project_id}/binary-pipeline"
    image = f"{ar_repo}/worker"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Ensure Artifact Registry repo exists (idempotent)
    subprocess.run([
        "gcloud", "artifacts", "repositories", "create", "binary-pipeline",
        f"--project={project_id}", f"--location={region}",
        "--repository-format=docker",
    ], check=False)  # OK if already exists

    # Configure docker auth for Artifact Registry
    subprocess.run([
        "gcloud", "auth", "configure-docker", f"{region}-docker.pkg.dev", "--quiet",
    ], check=True)

    # Build for linux/amd64 (Cloud Batch VMs are x86_64)
    print(f"Building Docker image: {image}")
    subprocess.run([
        "docker", "build", "--platform", "linux/amd64", "-t", image, repo_root,
    ], check=True)

    print("Pushing to Artifact Registry...")
    subprocess.run(["docker", "push", image], check=True)

    return image


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Submit simulation pipeline jobs to Google Cloud Batch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time: build image, create bucket, submit 8 tasks (N=1)
  python -m cloud.run_cloud_batch --build --create-bucket --n-per-field 1

  # Larger run (800 tasks):
  python -m cloud.run_cloud_batch --n-per-field 100

  # Dry-run (print what would be submitted):
  python -m cloud.run_cloud_batch --dry-run --n-per-field 1
        """,
    )
    parser.add_argument("--n-per-field", type=int, default=N_PER_FIELD,
                        help=f"Simulations per BLOeM field (default: {N_PER_FIELD}). "
                             f"Total tasks = N * {N_FIELDS} fields")
    parser.add_argument("--project", default=PROJECT_ID,
                        help=f"GCP project ID (default: {PROJECT_ID})")
    parser.add_argument("--region", default=REGION,
                        help=f"GCP region (default: {REGION})")
    parser.add_argument("--bucket", default=BUCKET,
                        help=f"GCS bucket (default: {BUCKET})")
    parser.add_argument("--machine-type", default=MACHINE_TYPE,
                        help=f"VM machine type (default: {MACHINE_TYPE})")
    parser.add_argument("--max-parallel", type=int, default=MAX_PARALLEL,
                        help=f"Max parallel tasks (default: {MAX_PARALLEL})")
    parser.add_argument("--build", action="store_true",
                        help="Build and push Docker image before submitting")
    parser.add_argument("--create-bucket", action="store_true",
                        help="Create GCS bucket if it doesn't exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be submitted, don't actually submit")
    parser.add_argument("--local-run", action="store_true",
                        help="Run all tasks locally and sequentially "
                             "(output in /tmp/cloud_results/<run-id>/)")
    parser.add_argument("--run-id", default=None,
                        help="Run ID (default: timestamp)")

    args = parser.parse_args()

    project = args.project
    region = args.region
    bucket = args.bucket
    image = f"{region}-docker.pkg.dev/{project}/binary-pipeline/worker"
    max_parallel = args.max_parallel

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    gcs_prefix = f"runs/{run_id}"

    n_tasks = args.n_per_field * N_FIELDS
    print("=" * 60)
    print("  Cloud Simulation Pipeline")
    print(f"  Project:     {project}")
    print(f"  Region:      {region}")
    print(f"  Bucket:      {bucket}")
    print(f"  Run ID:      {run_id}")
    print(f"  N per field: {args.n_per_field}")
    print(f"  N fields:    {N_FIELDS}")
    print(f"  Total tasks: {n_tasks}")
    print(f"  Machine:     {args.machine_type} (SPOT)")
    print("=" * 60)

    # Show task->field mapping for first few tasks
    print("\nTask assignment (first 16):")
    for i in range(min(n_tasks, 16)):
        print(f"  task {i:3d} -> field {i % N_FIELDS}")
    if n_tasks > 16:
        print(f"  ... ({n_tasks - 16} more)")

    if args.dry_run:
        est_cost = n_tasks * 0.056
        print(f"\n[DRY RUN] Would submit {n_tasks} tasks")
        print(f"  Estimated cost: ~${est_cost:.2f} (SPOT pricing)")
        print(f"  Output: gs://{bucket}/{gcs_prefix}/output/")
        return

    # --- Local execution path ---
    if args.local_run:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(repo_root, "configs", "params_cloud.yaml")
        run_local(n_tasks, N_FIELDS, run_id, config_path)
        return

    # --- Cloud execution path ---

    # Step 0: Optionally build Docker image
    if args.build:
        image = build_and_push_image(project, region)

    # Step 0b: Optionally create bucket
    if args.create_bucket:
        subprocess.run(
            ["gsutil", "mb", "-p", project, "-l", region, f"gs://{bucket}"],
            check=False,  # OK if already exists
        )

    # Submit Cloud Batch job (workers generate their own simulations)
    job_id = submit_batch_job(
        n_tasks, bucket, gcs_prefix, image, args.machine_type, run_id,
        n_fields=N_FIELDS,
        project=project, region=region, max_parallel=max_parallel,
    )

    # Save run metadata locally
    meta = {
        "run_id": run_id,
        "job_id": job_id,
        "project": project,
        "region": region,
        "bucket": bucket,
        "gcs_prefix": gcs_prefix,
        "n_per_field": args.n_per_field,
        "n_fields": N_FIELDS,
        "n_tasks": n_tasks,
        "run_id_seed": run_id,
        "machine_type": args.machine_type,
    }
    meta_path = f"/tmp/cloud_run_{run_id}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nRun metadata saved: {meta_path}")
    print(f"\nNext steps:")
    print(f"  1. Monitor:  gcloud batch jobs describe sim-{run_id} --project={project} --location={region}")
    print(f"  2. Collect:  python -m cloud.collect_results --run-id {run_id} --bucket {bucket}")
    print(f"  3. Analyze:  python -m cloud.analyze_results --run-id {run_id}")


if __name__ == "__main__":
    main()
