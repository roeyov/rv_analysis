FROM python:3.11-slim

# System dependencies for matplotlib (headless) and scipy/numba
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Headless matplotlib + mandatory thread pinning: pipeline_batched runs one
# single-threaded worker per vCPU; un-pinned BLAS/numba oversubscribes the VM
# (see fast_detect/BENCHMARK_RESULTS.md). Numba cache is baked at build time.
ENV MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMBA_CACHE_DIR=/opt/numba_cache

WORKDIR /app

# Install Python dependencies (constraints pin the numeric stack to the
# versions validated in the tau_binary env)
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy pipeline code + staged input data (cloud/stage_data.sh -> cloud_data/),
# relocating the data to the /app/data paths used by configs/params_bias_cloud*.yaml.
# Fails fast if stage_data.sh was not run.
COPY . .
RUN mv /app/cloud_data /app/data

# Build-time import smoke test: catches broken pins before any VM spends
# money. (No JIT warmup: numba's disk cache keys on the build host's CPU
# identity, so a cache baked under Rosetta/QEMU emulation would never hit
# on the cloud VMs. Runtime JIT is seconds per worker vs multi-hour tasks;
# NUMBA_CACHE_DIR still lets a task's pool workers share the compile.)
RUN mkdir -p /opt/numba_cache && python -c "\
import simulations.bias_grid_lib.physics, fast_detect.detector, \
simulations.bias_grid, cloud.run_bias_grid_batch"

# Cloud Batch overrides the entrypoint with the per-task grid-range script
# (cloud/run_bias_grid_batch.py); this default is for manual smoke tests.
ENTRYPOINT ["python", "-m", "simulations.bias_grid"]
