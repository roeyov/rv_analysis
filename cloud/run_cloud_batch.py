#!/usr/bin/env python3
"""
cloud.run_cloud_batch — Orchestrate cloud simulation runs on Google Cloud Batch.

Architecture (revised):
    - No local simulation generation or upload
    - Each cloud task generates its own simulation on the fly
    - Task index determines unique seed + which BLOeM field to use
    - Workers upload results + truth rows + simulated RV CSV to GCS

Usage:
    python -m cloud.run_cloud_batch \\
        --n-per-field 1 \\
        --project binary-pipeline \\
        --region us-central1 \\
        --bucket binary-pipeline-sims \\
        --seed 42

Cost estimate (e2-standard-4 SPOT, ~5 min each):
    ~$0.02/task
    N=1  ->  8 tasks  -> ~$0.16
    N=10 -> 80 tasks  -> ~$1.60
    N=100-> 800 tasks -> ~$16
"""

import argparse
import json
import os
import subprocess
import sys
import time

from simulations.common import BLOEM_MJD_ARRAYS


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

PROJECT_ID = "binary-pipeline"
REGION = "us-central1"
BUCKET = "binary-pipeline-sims"
IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/binary-pipeline/worker"
N_PER_FIELD = 1
N_FIELDS = len(BLOEM_MJD_ARRAYS)         # 8
SEED = 42
MAX_PARALLEL = 16     # max concurrent tasks in batch job
MACHINE_TYPE = "e2-standard-4"  # 4 vCPU, 16 GB — enough for LS permutation MP


# ---------------------------------------------------------------------------
# Submit Google Cloud Batch job
# ---------------------------------------------------------------------------

def submit_batch_job(n_tasks, bucket, gcs_prefix, image, machine_type, run_id,
                     base_seed, n_fields,
                     project=PROJECT_ID, region=REGION, max_parallel=MAX_PARALLEL):
    """Submit a Cloud Batch job where each task generates + processes one simulation."""
    job_id = f"sim-{run_id}"

    gcs_output = f"gs://{bucket}/{gcs_prefix}/output"

    # Each task gets BATCH_TASK_INDEX automatically from Cloud Batch.
    # The worker uses: seed = BASE_SEED + BATCH_TASK_INDEX
    #                  field_idx = BATCH_TASK_INDEX % N_FIELDS
    task_script = f"""#!/bin/bash
set -e
export TASK_INDEX=$BATCH_TASK_INDEX
export BASE_SEED={base_seed}
export N_FIELDS={n_fields}
export GCS_OUTPUT_DIR="{gcs_output}"
export CONFIG_PATH="/app/params_cloud.yaml"
echo "Task $TASK_INDEX: seed=$((BASE_SEED + TASK_INDEX)), field=$((TASK_INDEX % N_FIELDS))"
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
                    "cpuMilli": 4000,      # 4 vCPU
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
    print(f"  Base seed:   {base_seed}")
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
  python -m cloud.run_cloud_batch --n-per-field 100 --seed 123

  # Dry-run (print what would be submitted):
  python -m cloud.run_cloud_batch --dry-run --n-per-field 1
        """,
    )
    parser.add_argument("--n-per-field", type=int, default=N_PER_FIELD,
                        help=f"Simulations per BLOeM field (default: {N_PER_FIELD}). "
                             f"Total tasks = N * {N_FIELDS} fields")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Base random seed (default: {SEED})")
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
    print(f"  Base seed:   {args.seed}")
    print(f"  Machine:     {args.machine_type} (SPOT)")
    print("=" * 60)

    # Show task->field->seed mapping for first few tasks
    print("\nTask assignment (first 16):")
    for i in range(min(n_tasks, 16)):
        print(f"  task {i:3d} -> field {i % N_FIELDS}, seed {args.seed + i}")
    if n_tasks > 16:
        print(f"  ... ({n_tasks - 16} more)")

    if args.dry_run:
        est_cost = n_tasks * 0.02
        print(f"\n[DRY RUN] Would submit {n_tasks} tasks")
        print(f"  Estimated cost: ~${est_cost:.2f} (SPOT pricing)")
        print(f"  Output: gs://{bucket}/{gcs_prefix}/output/")
        return

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
        base_seed=args.seed, n_fields=N_FIELDS,
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
        "base_seed": args.seed,
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
