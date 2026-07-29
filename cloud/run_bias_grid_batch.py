"""
cloud.run_bias_grid_batch — Google Cloud Batch launcher for the bias grid.

Submits ``simulations.bias_grid`` as a Cloud Batch job: the grid is split
into chunks of ``--chunk-size`` consecutive grid points, one Batch task per
chunk, one task per VM (``parallel_grid`` + explicit ``n_workers`` inside
the container). Each task runs

    python -m simulations.bias_grid --config /app/configs/<cfg>.yaml \
        --grid-start $((BATCH_TASK_INDEX * C)) --grid-end $((... + C))

and writes its ``task_I_J/`` directory (checkpoint CSV + cubes + det
shards) onto a GCS bucket mounted via the Batch gcsfuse volume, under
``gs://<bucket>/bias_runs/<run_id>/output/``. Per-point seeding is
``default_rng(seed + step * 137)`` on the *global* step index, so any
chunking reproduces a single-box run; det-shard resume makes Spot
preemption retries idempotent (see simulations/bias_grid_lib/engine.py).

IMPORTANT — chunk size is part of a run's identity. Task dirs are named by
their grid range (task_0_64, task_64_128, ...), so ALL submissions that
share a run-id MUST use the same --chunk-size: mixing sizes creates
overlapping task dirs that double-count cells at aggregation. The launcher
refuses to submit when existing task dirs in GCS don't align with the
current chunk grid. Partial (calibration) runs of the SAME run-id are safe
via --task-count-override; small throwaway calibrations of a different
shape belong under a different run-id.

Aggregation is NOT done in the cloud: rsync the run's ``output/`` locally
(or to astro3) and run ``python -m simulations.bias_grid --aggregate <dir>``.
Never aggregate against the mounted bucket — aggregate_tasks deletes the
per-task dirs after merging.

Usage:
    # build + push image, then submit (SPOT, c3d):
    python -m cloud.run_bias_grid_batch --run-id ext-rlofoff-001 --build

    # calibration: first task only, SAME chunk size — its points are
    # skipped via shard resume when the full job is submitted later with
    # the same run-id:
    python -m cloud.run_bias_grid_batch --run-id ext-rlofoff-001 \
        --task-count-override 1

    # throwaway micro-benchmark (~16 pts) on a candidate machine — note
    # the DIFFERENT run-id because the chunk size differs:
    python -m cloud.run_bias_grid_batch --run-id cal-c4d-001 \
        --machine-type c4d-highcpu-16 --chunk-size 16 --task-count-override 1

    # inspect the job spec without submitting:
    python -m cloud.run_bias_grid_batch --run-id test --dry-run

    # progress of a submitted run:
    python -m cloud.run_bias_grid_batch --run-id ext-rlofoff-001 --progress
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

import yaml

# --- Defaults (all overridable via CLI) ------------------------------------
PROJECT_ID = "binary-pipeline"
REGION = "us-central1"
BUCKET = "binary-pipeline-sims"
AR_REPO = "binary-pipeline"          # Artifact Registry repository name
IMAGE_NAME = "bias-grid"
# c3d-standard-16 (4 GB/vCPU) is the safe calibration default; drop to
# c3d-highcpu-16 (2 GB/vCPU) once measured worker RSS confirms headroom.
MACHINE_TYPE = "c3d-standard-16"
DEFAULT_CONFIG = "configs/params_bias_cloud.yaml"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data files the Dockerfile bakes into /app/data (staged by
# cloud/stage_data.sh into <repo>/cloud_data/).
STAGED_DATA_FILES = [
    "tables/sb1_solutions.tex",
    "tables/sb2_solutions.tex",
    "tables/ostar_catalog.csv",
    "mass_bloem.csv",
]
STAGED_DATA_DIRS = ["rv_csvs", "sb2_per_star"]


def sh(cmd, capture=False, check=True):
    """Run a shell command list, streaming output unless capture=True."""
    print("  $ %s" % " ".join(cmd))
    if capture:
        res = subprocess.run(cmd, check=check, text=True,
                             capture_output=True)
        return res.stdout.strip()
    subprocess.run(cmd, check=check)
    return None


def git_sha():
    """Short git SHA of the repo, with -dirty suffix if the tree differs."""
    sha = subprocess.run(
        ["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
        check=True, text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", REPO_ROOT, "status", "--porcelain"],
        check=True, text=True, capture_output=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def parse_vcpus(machine_type):
    """vCPU count from a machine type name (last dash-separated integer)."""
    m = re.search(r"-(\d+)$", machine_type)
    if not m:
        raise SystemExit(
            "Cannot parse vCPU count from machine type %r (expected a "
            "trailing -<n> like c3d-highcpu-16)" % machine_type)
    return int(m.group(1))


def load_yaml(config_path):
    with open(config_path) as fh:
        return yaml.safe_load(fh) or {}


def read_total(cfg, config_path):
    """(total_points, preset_name) from the YAML's bias_grid.preset."""
    from simulations.bias_config import GRID_PRESETS

    bg = cfg.get("bias_grid") or {}
    preset_name = bg.get("preset")
    if preset_name not in GRID_PRESETS:
        raise SystemExit(
            "bias_grid.preset %r in %s not in GRID_PRESETS (valid: %s)"
            % (preset_name, config_path, ", ".join(sorted(GRID_PRESETS))))
    preset = GRID_PRESETS[preset_name]
    total = 1
    for ax in ("pi", "kappa", "eta", "fbin"):
        total *= len(preset[ax])
    return total, preset_name


def validate_for_submit(cfg, config_path):
    """Container-critical knobs the engine cannot check itself.

    Only enforced on submission — --progress needs none of this.
    """
    bg = cfg.get("bias_grid") or {}
    if bg.get("n_workers") is None:
        raise SystemExit(
            "bias_grid.n_workers is null in %s — inside a container "
            "os.cpu_count() misreads cgroup limits; set it explicitly to "
            "the machine's vCPU count." % config_path)
    if not bg.get("parallel_grid"):
        raise SystemExit(
            "bias_grid.parallel_grid must be true for cloud tasks "
            "(one multi-point task per VM) in %s" % config_path)
    if cfg.get("fast_detect_backend") != "numpy":
        print("WARNING: fast_detect_backend=%r — Linux VMs only run the "
              "numpy backend (mlx is Apple-only)."
              % cfg.get("fast_detect_backend"))
    if bg.get("output_dir") != "/mnt/gcs/output":
        print("WARNING: bias_grid.output_dir=%r — expected /mnt/gcs/output "
              "(the run_id lives in the GCS mount's remotePath)."
              % bg.get("output_dir"))
    return bg


def container_config_path(config_abs):
    """Container path of the config — must live inside the repo.

    The image is built from the repo root (COPY . . -> /app), so only
    repo-internal configs exist in the container; anything else would
    submit a job whose every task dies on config-not-found.
    """
    rel = os.path.relpath(config_abs, REPO_ROOT)
    if rel.startswith(".."):
        raise SystemExit(
            "--config %s is outside the repo (%s) — it is not baked into "
            "the image. Put the config under configs/ and retry."
            % (config_abs, REPO_ROOT))
    return "/app/%s" % rel


def check_staged_data():
    """Verify cloud/stage_data.sh has populated <repo>/cloud_data/."""
    data_root = os.path.join(REPO_ROOT, "cloud_data")
    missing = [f for f in STAGED_DATA_FILES
               if not os.path.exists(os.path.join(data_root, f))]
    missing += [d for d in STAGED_DATA_DIRS
                if not os.path.isdir(os.path.join(data_root, d))]
    if missing:
        raise SystemExit(
            "cloud_data/ is missing %s — run ./cloud/stage_data.sh first "
            "(stages tables, mass catalog, RV CSVs and SB2 files into the "
            "Docker build context)." % ", ".join(missing))


def image_uri(project, region, tag):
    return "%s-docker.pkg.dev/%s/%s/%s:%s" % (
        region, project, AR_REPO, IMAGE_NAME, tag)


def build_and_push_image(project, region, tag):
    """Build the worker image (linux/amd64) and push to Artifact Registry."""
    check_staged_data()
    repo_check = subprocess.run(
        ["gcloud", "artifacts", "repositories", "describe", AR_REPO,
         "--project", project, "--location", region],
        capture_output=True, text=True)
    if repo_check.returncode != 0:
        print("Creating Artifact Registry repo %r..." % AR_REPO)
        sh(["gcloud", "artifacts", "repositories", "create", AR_REPO,
            "--repository-format=docker", "--project", project,
            "--location", region])
    uri = image_uri(project, region, tag)
    sh(["docker", "build", "--platform", "linux/amd64", "-t", uri, REPO_ROOT])
    sh(["docker", "push", uri])
    return uri


def check_existing_task_dirs(bucket, run_id, chunk, total):
    """Refuse to submit if GCS task dirs don't align with this chunk grid.

    Mixing chunk sizes under one run-id creates overlapping task_I_J dirs;
    aggregate_tasks merges them additively and silently double-counts the
    overlapped cells. Best-effort: an empty/inaccessible prefix passes.
    """
    res = subprocess.run(
        ["gcloud", "storage", "ls",
         "gs://%s/bias_runs/%s/output/" % (bucket, run_id)],
        capture_output=True, text=True)
    if res.returncode != 0:
        return  # no bucket / first submission / no auth here — nothing to check
    bad = []
    for line in res.stdout.splitlines():
        m = re.search(r"/task_(\d+)_(\d+)/?$", line.strip())
        if not m:
            continue
        start, end = int(m.group(1)), int(m.group(2))
        if start % chunk != 0 or end != min(start + chunk, total):
            bad.append("task_%d_%d" % (start, end))
    if bad:
        raise SystemExit(
            "Existing task dirs in gs://%s/bias_runs/%s/output/ don't align "
            "with --chunk-size %d: %s.\nMixing chunk sizes in one run-id "
            "double-counts cells at aggregation. Resubmit with the original "
            "chunk size, or use a fresh run-id."
            % (bucket, run_id, chunk, ", ".join(bad)))


def make_task_script(config_container_path, chunk, total):
    """Bash script each Batch task runs: task index -> grid range."""
    return "\n".join([
        "#!/bin/bash",
        "set -e",
        "START=$((BATCH_TASK_INDEX * %d))" % chunk,
        "END=$((START + %d))" % chunk,
        "if [ $END -gt %d ]; then END=%d; fi" % (total, total),
        'echo "Task $BATCH_TASK_INDEX: grid points [$START, $END) of %d"'
        % total,
        "exec python -m simulations.bias_grid --config %s "
        "--grid-start $START --grid-end $END" % config_container_path,
    ])


def make_job_spec(image, task_script, n_tasks, parallelism, machine_type,
                  provisioning, region, bucket, run_id, vcpus, memory_mib,
                  max_run_s, max_retries):
    """Cloud Batch job spec (JSON dict for `gcloud batch jobs submit`)."""
    return {
        "taskGroups": [{
            "taskSpec": {
                "runnables": [{
                    "container": {
                        "imageUri": image,
                        "entrypoint": "/bin/bash",
                        "commands": ["-c", task_script],
                        # Re-map the host gcsfuse mount into the container.
                        "volumes": ["/mnt/disks/gcs:/mnt/gcs:rw"],
                    },
                }],
                "volumes": [{
                    "gcs": {"remotePath": "%s/bias_runs/%s" % (bucket,
                                                               run_id)},
                    "mountPath": "/mnt/disks/gcs",
                }],
                "computeResource": {
                    # Full-VM task: claim every vCPU so Batch schedules
                    # exactly one task per VM.
                    "cpuMilli": vcpus * 1000,
                    "memoryMib": memory_mib,
                },
                "maxRunDuration": "%ds" % max_run_s,
                "maxRetryCount": max_retries,
            },
            "taskCount": str(n_tasks),
            "parallelism": str(min(n_tasks, parallelism)),
        }],
        "allocationPolicy": {
            "instances": [{
                "policy": {
                    "machineType": machine_type,
                    "provisioningModel": provisioning,
                },
            }],
            "location": {"allowedLocations": ["regions/%s" % region]},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }


def submit(job_id, spec, project, region, dry_run, progress_hint):
    spec_path = os.path.join(tempfile.gettempdir(),
                             "batch_job_%s.json" % job_id)
    with open(spec_path, "w") as fh:
        json.dump(spec, fh, indent=2)
    print("Job spec written to %s" % spec_path)
    if dry_run:
        print("--dry-run: not submitting. Spec:")
        print(json.dumps(spec, indent=2))
        return
    sh(["gcloud", "batch", "jobs", "submit", job_id,
        "--project", project, "--location", region,
        "--config", spec_path])
    print("Submitted job %s." % job_id)
    print("Monitor:  gcloud batch jobs describe %s --project %s "
          "--location %s" % (job_id, project, region))
    print("Progress: %s" % progress_hint)


def list_run_jobs(project, region, run_id):
    """All Batch jobs of this run (job ids bias-<run-id>-<stamp>)."""
    res = subprocess.run(
        ["gcloud", "batch", "jobs", "list", "--project", project,
         "--location", region,
         "--format", "value(name,status.state)"],
        capture_output=True, text=True)
    jobs = []
    if res.returncode == 0:
        prefix = "/jobs/bias-%s-" % run_id
        for line in res.stdout.splitlines():
            parts = line.split()
            if parts and prefix in parts[0]:
                jobs.append((parts[0].rsplit("/jobs/", 1)[1],
                             parts[1] if len(parts) > 1 else "?"))
    return jobs


def print_progress(project, region, bucket, run_id, total):
    """Completed grid points (GCS shard count) + Batch job/task states."""
    shards = subprocess.run(
        ["gcloud", "storage", "ls",
         "gs://%s/bias_runs/%s/output/task_*/det_shards/*.npz"
         % (bucket, run_id)],
        capture_output=True, text=True)
    n_shards = (len(shards.stdout.splitlines())
                if shards.returncode == 0 else 0)
    print("Grid points completed (det shards in GCS): %d / %d (%.1f%%)"
          % (n_shards, total, 100.0 * n_shards / max(total, 1)))
    jobs = list_run_jobs(project, region, run_id)
    if not jobs:
        print("(no Batch jobs found matching bias-%s-*)" % run_id)
    for job_id, state in jobs:
        tasks = subprocess.run(
            ["gcloud", "batch", "tasks", "list", "--job", job_id,
             "--project", project, "--location", region,
             "--format", "value(status.state)"],
            capture_output=True, text=True)
        counts = {}
        for line in tasks.stdout.splitlines():
            counts[line] = counts.get(line, 0) + 1
        detail = (", ".join("%s=%d" % kv for kv in sorted(counts.items()))
                  or "no tasks listed")
        print("Job %s [%s]: %s" % (job_id, state, detail))


def positive_int(value):
    iv = int(value)
    if iv < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return iv


def main():
    p = argparse.ArgumentParser(
        description="Submit the bias grid to Google Cloud Batch "
                    "(one chunk of grid points per task, one task per VM).")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="Bias YAML inside the repo (baked into the image "
                        "under /app/). Default: %s" % DEFAULT_CONFIG)
    p.add_argument("--run-id", required=True,
                   help="Run identifier: GCS prefix bias_runs/<run-id>/ and "
                        "Batch job ids bias-<run-id>-<stamp>. Lowercase "
                        "letters, digits, hyphens; no trailing hyphen; "
                        "max 40 chars.")
    p.add_argument("--project", default=PROJECT_ID)
    p.add_argument("--region", default=REGION)
    p.add_argument("--bucket", default=BUCKET)
    p.add_argument("--machine-type", default=MACHINE_TYPE)
    p.add_argument("--provisioning", default="SPOT",
                   choices=["SPOT", "STANDARD"])
    p.add_argument("--chunk-size", type=positive_int, default=64,
                   help="Grid points per Batch task (default 64). Part of "
                        "the run's identity — every submission of one "
                        "run-id must use the same value.")
    p.add_argument("--parallelism", type=positive_int, default=32,
                   help="Max concurrent tasks (= VMs; default 32). Gated by "
                        "the regional PREEMPTIBLE_CPUS quota.")
    p.add_argument("--minutes-per-point", type=float, default=60.0,
                   help="Expected minutes per grid point per worker, for "
                        "maxRunDuration sizing (default 60 = astro3 "
                        "rlof-off rate; calibrate!).")
    p.add_argument("--safety-factor", type=float, default=1.5)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--memory-mib", type=int, default=None,
                   help="Task memoryMib. Default: (vCPUs * per-vCPU MiB of "
                        "the machine) - 2048 headroom for the Batch agent.")
    p.add_argument("--task-count-override", type=positive_int, default=None,
                   help="Submit only the first N tasks (calibration). Safe "
                        "to resubmit the full job with the same run-id AND "
                        "the same --chunk-size later: completed points are "
                        "skipped via shard resume.")
    p.add_argument("--build", action="store_true",
                   help="docker build + push the image before submitting.")
    p.add_argument("--image-tag", default=None,
                   help="Image tag to run (default: current git SHA).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--progress", action="store_true",
                   help="Print run progress and exit (no submission).")
    args = p.parse_args()

    # Batch job ids must match [a-z]([a-z0-9-]*[a-z0-9])? and stay <= 63
    # chars; "bias-" + run-id (<=40) + "-MMDD-HHMMSS" fits.
    if not re.fullmatch(r"[a-z]([a-z0-9-]{0,38}[a-z0-9])?", args.run_id):
        raise SystemExit(
            "--run-id must be 1-40 chars of [a-z0-9-], starting with a "
            "letter and not ending with a hyphen, got %r" % args.run_id)

    config_abs = (args.config if os.path.isabs(args.config)
                  else os.path.join(REPO_ROOT, args.config))
    cfg = load_yaml(config_abs)
    total, preset_name = read_total(cfg, config_abs)

    if args.progress:
        print_progress(args.project, args.region, args.bucket,
                       args.run_id, total)
        return

    bg = validate_for_submit(cfg, config_abs)
    config_container = container_config_path(config_abs)

    vcpus = parse_vcpus(args.machine_type)
    n_workers = int(bg["n_workers"])
    if n_workers > vcpus:
        raise SystemExit(
            "bias_grid.n_workers=%d exceeds %s's %d vCPUs — fix the YAML "
            "or pick a bigger machine." % (n_workers, args.machine_type,
                                           vcpus))
    if n_workers < vcpus:
        print("WARNING: n_workers=%d < %d vCPUs — %d vCPUs will idle."
              % (n_workers, vcpus, vcpus - n_workers))

    if args.memory_mib is not None:
        memory_mib = args.memory_mib
    else:
        # Per-vCPU MiB by machine flavor; leave 2 GiB for the Batch agent
        # + gcsfuse on the host.
        per_vcpu = {"highcpu": 2048, "standard": 4096, "highmem": 8192}
        flavor = next((f for f in per_vcpu if "-%s-" % f in args.machine_type),
                      "standard")
        memory_mib = vcpus * per_vcpu[flavor] - 2048

    n_tasks_full = math.ceil(total / args.chunk_size)
    n_tasks = (min(args.task_count_override, n_tasks_full)
               if args.task_count_override else n_tasks_full)
    pts_this_job = min(n_tasks * args.chunk_size, total)
    minutes_per_task = (math.ceil(args.chunk_size / n_workers)
                        * args.minutes_per_point * args.safety_factor)
    max_run_s = int(minutes_per_task * 60)

    if not args.dry_run:
        check_existing_task_dirs(args.bucket, args.run_id,
                                 args.chunk_size, total)

    tag = args.image_tag or git_sha()
    uri = image_uri(args.project, args.region, tag)
    if args.build:
        uri = build_and_push_image(args.project, args.region, tag)
    if tag.endswith("-dirty"):
        print("WARNING: image tag %s — uncommitted changes in the repo."
              % tag)

    task_script = make_task_script(config_container, args.chunk_size, total)
    # Unique job id per submission: Batch job ids cannot be reused while an
    # earlier job with the same id exists, and resubmission under one
    # run-id is the documented calibration/recovery flow.
    job_id = "bias-%s-%s" % (args.run_id, time.strftime("%m%d-%H%M%S"))
    spec = make_job_spec(uri, task_script, n_tasks, args.parallelism,
                         args.machine_type, args.provisioning, args.region,
                         args.bucket, args.run_id, vcpus, memory_mib,
                         max_run_s, args.max_retries)

    print("=" * 64)
    print("  Bias grid on Cloud Batch")
    print("  Preset: %s -> %d grid points" % (preset_name, total))
    print("  This job: %d task(s) x %d pts = %d pts%s"
          % (n_tasks, args.chunk_size, pts_this_job,
             " (PARTIAL — calibration)" if n_tasks < n_tasks_full else ""))
    print("  Machine: %s (%s), %d workers/VM, %d MiB"
          % (args.machine_type, args.provisioning, n_workers, memory_mib))
    print("  Parallelism: %d VMs (%d vCPUs) — check PREEMPTIBLE_CPUS quota"
          % (min(n_tasks, args.parallelism),
             min(n_tasks, args.parallelism) * vcpus))
    print("  maxRunDuration: %.1f h/task attempt" % (minutes_per_task / 60))
    print("  Image: %s" % uri)
    print("  Job id: %s" % job_id)
    print("  Output: gs://%s/bias_runs/%s/output/" % (args.bucket,
                                                      args.run_id))
    print("=" * 64)

    progress_hint = ("python -m cloud.run_bias_grid_batch --run-id %s "
                     "--config %s --progress" % (args.run_id, args.config))
    submit(job_id, spec, args.project, args.region, args.dry_run,
           progress_hint)


if __name__ == "__main__":
    main()
