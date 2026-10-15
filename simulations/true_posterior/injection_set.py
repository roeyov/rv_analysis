"""
simulations.true_posterior.injection_set — the star-labelled,
theta-independent injection set.

For every catalogue star and every injection, the driver:

1. draws (logP, q, e) from the reference density g
   (``population.ReferenceDensity``) instead of from one theta;
2. draws the nuisances and noise exactly as the bias grid does
   (``injection._star_binary_rvs``: p_circ rule, isotropic i, omega, T0,
   30% M1/R1 scatter, K1, Keplerian RVs on the field's MJD array, Gaussian
   noise at the star's rv_err);
3. runs the grid's detector: ``pipeline_batched`` (fast_detect.detect_batch
   over the field's pooled curves) or ``rv_threshold`` (the pairwise ΔRV
   cut).

It also runs noise-only curves ("singles") per star, which give the
false-positive rate beta_s.

Each record keeps the star index, so the likelihood can reweight
per-star detection probabilities to any
(pi, kappa, eta) with weights p(z | theta) / g(z). Detection decisions
are column-independent inside ``detect_batch`` (per-curve periodograms,
fits and permutation shuffles), so pooling a field's curves into one call
does not change any single curve's statistics.

Work unit: one task per (field, chunk). A task holds ``chunk`` binaries and
``chunk_single`` singles for every star in that field, and writes
``shards/f<field>_c<chunk>.npz``. Existing shards are skipped, so a relaunch
resumes. Seeds are SeedSequence([seed, star_index, chunk]) for the draws
and SeedSequence([seed, 10**6 + field, chunk]) for the detector.

Usage (astro3; export OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1 first):
    python -u -m simulations.true_posterior.injection_set \
        --config configs/params_bias_astro3_ed_hoc_rlof_check_ext3.yaml \
        --out-dir /jonathan_storage/roeyovadia/bias_grid_results/tp_injset_pb \
        --n-per-star 100000 --chunk 250 --n-single-per-star 10000 \
        --n-workers 90
Local smoke (macOS: keep --n-workers 1):
    python -m simulations.true_posterior.injection_set \
        --config configs/params_bias_local_tier4_smoke.yaml \
        --out-dir $SCRIPTS_OUT/simulation_pipeline/true_posterior/injset_smoke \
        --n-per-star 400 --chunk 200 --n-single-per-star 200 --stars 1-023,2-086
"""

import argparse
import copy
import datetime as _dt
import glob
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from simulations.bias_config import DEFAULT_BIAS_CFG
from simulations.bias_grid_lib.constants import KILL_DRV
from simulations.bias_grid_lib.injection import _star_binary_rvs
from simulations.common import BLOEM_MJD_ARRAYS
from simulations.true_posterior.population import ReferenceDensity

SHARD_DIR = "shards"
MANIFEST = "manifest.yaml"
STARS_CSV = "stars.csv"
DETECTORS = ("pipeline_batched", "rv_threshold")


# ---------------------------------------------------------------------------
# Configuration (mirrors simulations/bias_grid.py for the keys that matter)
# ---------------------------------------------------------------------------

def load_run_config(config_path):
    """(args_dict, cfg, bg, detect_method) from a bias YAML.

    ``cfg`` receives the same sampling bounds, RLOF-gate switch and
    periodogram block that bias_grid.main() copies into its cfg. Those are
    the only cfg keys the draws and detect_batch read.
    """
    from pipeline.config import load_args
    args_dict = load_args(config_path)
    bg = args_dict.get("bias_grid", {}) or {}

    def get(key):
        v = bg.get(key)
        return DEFAULT_BIAS_CFG.get(key) if v is None else v

    cfg = copy.deepcopy(DEFAULT_BIAS_CFG)
    for k in ("log_p_min", "log_p_max", "q_min", "q_max", "e_max", "p_circ"):
        cfg[k] = float(get(k))
    cfg["apply_rlof_gate"] = bool(get("apply_rlof_gate"))
    cfg["periodogram_params"] = dict(args_dict.get("periodogram_params") or {})
    detect_method = get("detect_method")
    if detect_method not in DETECTORS:
        raise ValueError("injection_set supports detect_method in %s, got %r"
                         % (DETECTORS, detect_method))
    return args_dict, cfg, bg, detect_method


def load_stars(bg):
    """The bias grid's injection sample (catalogue order)."""
    from simulations.bias_grid_lib.star_loading import load_star_properties

    def get(key):
        v = bg.get(key)
        return DEFAULT_BIAS_CFG.get(key) if v is None else v
    df = load_star_properties(get("mass_file"), get("rv_dir"),
                              sb2_analysis_dir=get("sb2_analysis_dir"),
                              ostar_catalog=get("ostar_catalog"))
    df = df.reset_index(drop=True)
    df.insert(0, "star_index", np.arange(len(df), dtype=int))
    return df


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _det_branch(info, det):
    """0 = not detected, 1 = LS branch, 2 = PDC branch."""
    ls = np.asarray(info.get("detected_ls", np.zeros_like(det)), dtype=bool)
    out = np.zeros(det.shape, dtype=np.int8)
    out[det & ls] = 1
    out[det & ~ls] = 2
    return out


def _run_task(task):
    """Draw + detect one (field, chunk). Writes its shard; returns a summary."""
    (field, chunk, star_rows, reference, bounds, cfg, args_dict,
     detect_method, n_bin, n_single, seed, out_path) = task
    if os.path.exists(out_path):
        return {"field": field, "chunk": chunk, "skipped": True}
    t0 = time.time()
    ref = ReferenceDensity(bounds, reference)
    mjds = np.asarray(BLOEM_MJD_ARRAYS[field], dtype=np.float64)
    n_ep = len(mjds)

    cols = {k: [] for k in ("star", "logP", "q", "e_draw", "circ_comp", "e",
                            "K1", "M1", "incl", "rv_err")}
    rv_rows, s_rows, s_star, s_err = [], [], [], []
    for row in star_rows:
        s = int(row["star_index"])
        rng = np.random.default_rng(np.random.SeedSequence([seed, s, chunk]))
        rv_err = float(row["rv_err"])
        logP, q, e_draw, circ = ref.sample(n_bin, rng)
        e = e_draw.copy()
        (logP, e, K1, q, rv_obs, M1_j, _R1_j, incl, _om, _t0) = \
            _star_binary_rvs(mjds, n_ep, rv_err, float(row["Mspec"]),
                             float(row["R_star"]), float(row["gamma"]),
                             logP, q, e, cfg, rng)
        for k, v in (("star", np.full(n_bin, s, dtype=np.int16)),
                     ("logP", logP), ("q", q), ("e_draw", e_draw),
                     ("circ_comp", circ), ("e", e),
                     ("K1", K1.astype(np.float32)),
                     ("M1", M1_j.astype(np.float32)),
                     ("incl", incl.astype(np.float32)),
                     ("rv_err", np.full(n_bin, rv_err, dtype=np.float32))):
            cols[k].append(v)
        rv_rows.append(rv_obs)
        if n_single > 0:
            s_rows.append(float(row["gamma"])
                          + rng.normal(0.0, rv_err, (n_single, n_ep)))
            s_star.append(np.full(n_single, s, dtype=np.int16))
            s_err.append(np.full(n_single, rv_err))

    rec = {k: np.concatenate(v) for k, v in cols.items()}
    rv_all = np.concatenate(rv_rows, axis=0)
    err_all = rec["rv_err"].astype(np.float64)

    if detect_method == "pipeline_batched":
        from fast_detect.detector import detect_batch
        backend = args_dict.get("fast_detect_backend", "numpy")
        if cfg.get("apply_rlof_gate", True):
            raise ValueError("injection_set requires apply_rlof_gate: false "
                             "(the likelihood's P_det(x) cancellation assumes "
                             "detection depends only on the RV curve)")
        det_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 10 ** 6 + field, chunk]))
        det, info = detect_batch(mjds, rv_all.T, err_all, None, cfg,
                                 backend=backend, rng=det_rng)
        det = np.asarray(det, dtype=bool)
        kill = np.asarray(info["kill_flag"], dtype=np.int8)
        branch = _det_branch(info, det)
        p_fit = np.where(branch == 1, info["P_fit_ls"], info["P_fit_pdc"])
        p_fit = np.where(det, p_fit, np.nan).astype(np.float32)
        s_det = np.zeros(0, dtype=bool)
        if s_rows:
            s_all = np.concatenate(s_rows, axis=0)
            s_det, _ = detect_batch(mjds, s_all.T, np.concatenate(s_err),
                                    None, cfg, backend=backend, rng=det_rng)
            s_det = np.asarray(s_det, dtype=bool)
    else:  # rv_threshold: same cut as _worker_star_injections_vectorized
        drv = float(args_dict.get("drv_threshold", 20.0))
        sig = float(args_dict.get("significance_threshold", 4.0))
        eff = np.maximum(drv, sig * err_all * np.sqrt(2.0))
        det = (rv_all.max(axis=1) - rv_all.min(axis=1)) > eff
        kill = np.where(det, 0, KILL_DRV).astype(np.int8)
        branch = det.astype(np.int8) * 3
        p_fit = np.full(det.shape, np.nan, dtype=np.float32)
        s_det = np.zeros(0, dtype=bool)
        if s_rows:
            s_all = np.concatenate(s_rows, axis=0)
            s_e = np.concatenate(s_err)
            s_eff = np.maximum(drv, sig * s_e * np.sqrt(2.0))
            s_det = (s_all.max(axis=1) - s_all.min(axis=1)) > s_eff

    rec.pop("rv_err")
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, det=det, kill_flag=kill, det_branch=branch, P_fit=p_fit,
             single_star=(np.concatenate(s_star) if s_star
                          else np.zeros(0, np.int16)),
             single_det=s_det, field=np.array(field), chunk=np.array(chunk),
             **rec)
    os.replace(tmp, out_path)
    return {"field": field, "chunk": chunk, "skipped": False,
            "n_bin": int(det.size), "n_det": int(det.sum()),
            "n_single": int(s_det.size), "n_fp": int(s_det.sum()),
            "seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_tasks(stars, reference, bounds, cfg, args_dict, detect_method,
                n_per_star, chunk, n_single_per_star, seed, out_dir):
    if n_per_star % chunk:
        raise ValueError("--n-per-star must be a multiple of --chunk")
    n_chunks = n_per_star // chunk
    if n_single_per_star % n_chunks:
        raise ValueError("--n-single-per-star must be a multiple of "
                         "n_per_star / chunk = %d" % n_chunks)
    n_single = n_single_per_star // n_chunks
    tasks = []
    for field, grp in stars.groupby("field", sort=True):
        rows = grp[["star_index", "rv_err", "Mspec", "R_star",
                    "gamma"]].to_dict("records")
        for c in range(n_chunks):
            out_path = os.path.join(out_dir, SHARD_DIR,
                                    "f%d_c%05d.npz" % (int(field), c))
            tasks.append((int(field), c, rows, reference, bounds, cfg,
                          args_dict, detect_method, chunk, n_single, seed,
                          out_path))
    return tasks


def _git_hash():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", here, "rev-parse",
                                        "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 — provenance only
        return "unknown"


def write_manifest(out_dir, **fields):
    path = os.path.join(out_dir, MANIFEST)
    with open(path, "w") as fh:
        yaml.safe_dump(fields, fh, sort_keys=False)
    return path


def run(config_path, out_dir, n_per_star, chunk, n_single_per_star, seed,
        n_workers, stars_subset=None, reference=None):
    args_dict, cfg, bg, detect_method = load_run_config(config_path)
    stars = load_stars(bg)
    if stars_subset:
        keep = set(stars_subset)
        unknown = keep - set(stars["ID"])
        if unknown:
            raise ValueError("unknown star IDs: %s" % sorted(unknown))
        stars = stars[stars["ID"].isin(keep)]
    bounds = {k: cfg[k] for k in ("log_p_min", "log_p_max", "q_min", "q_max",
                                  "e_max")}
    ref = ReferenceDensity(bounds, reference)
    os.makedirs(os.path.join(out_dir, SHARD_DIR), exist_ok=True)

    man_path = os.path.join(out_dir, MANIFEST)
    identity = {"config_path": os.path.abspath(config_path),
                "detect_method": detect_method, "seed": int(seed),
                "n_per_star": int(n_per_star), "chunk": int(chunk),
                "n_single_per_star": int(n_single_per_star),
                "bounds": bounds, "p_circ": float(cfg["p_circ"]),
                "apply_rlof_gate": bool(cfg["apply_rlof_gate"]),
                "periodogram_params": cfg["periodogram_params"],
                "drv_threshold": args_dict.get("drv_threshold"),
                "significance_threshold": args_dict.get(
                    "significance_threshold"),
                "reference": ref.to_dict(),
                "star_ids": list(map(str, stars["ID"]))}
    if os.path.exists(man_path):
        with open(man_path) as fh:
            old = yaml.safe_load(fh)
        clash = {k: (old.get(k), v) for k, v in identity.items()
                 if old.get(k) != v}
        if clash:
            raise ValueError("out-dir holds a different injection set "
                             "(mismatched: %s); use a fresh --out-dir"
                             % sorted(clash))
    write_manifest(out_dir, **identity, status="running",
                   started=_dt.datetime.now().isoformat(timespec="seconds"),
                   git=_git_hash(), host=platform.node(),
                   numpy=np.__version__)
    stars.to_csv(os.path.join(out_dir, STARS_CSV), index=False)

    tasks = build_tasks(stars, ref.spec, bounds, cfg, args_dict,
                        detect_method, n_per_star, chunk, n_single_per_star,
                        seed, out_dir)
    todo = [t for t in tasks if not os.path.exists(t[-1])]
    print("[injset] %s: %d stars, %d fields, %d tasks (%d to do), "
          "%d binaries + %d singles per star"
          % (detect_method, len(stars), stars["field"].nunique(), len(tasks),
             len(todo), n_per_star, n_single_per_star), flush=True)

    t0 = time.time()
    done = 0
    n_det = n_bin = n_fp = n_sng = 0

    def report(res):
        nonlocal done, n_det, n_bin, n_fp, n_sng
        done += 1
        if not res.get("skipped"):
            n_det += res["n_det"]; n_bin += res["n_bin"]
            n_fp += res["n_fp"]; n_sng += res["n_single"]
        if done % max(1, len(todo) // 50) == 0 or done == len(todo):
            el = time.time() - t0
            print("[injset] %d/%d tasks  %.0f s  det %.4f  fp %.5f  "
                  "(%.0f binaries/s)"
                  % (done, len(todo), el, n_det / max(n_bin, 1),
                     n_fp / max(n_sng, 1), n_bin / max(el, 1e-9)),
                  flush=True)

    if n_workers <= 1:
        for t in todo:
            report(_run_task(t))
    else:
        method = "forkserver" if sys.platform.startswith("linux") else "spawn"
        ctx = mp.get_context(method)
        with ctx.Pool(n_workers, maxtasksperchild=500) as pool:
            for res in pool.imap_unordered(_run_task, todo, chunksize=1):
                report(res)

    missing = [t[-1] for t in tasks if not os.path.exists(t[-1])]
    status = "complete" if not missing else "incomplete"
    write_manifest(out_dir, **identity, status=status,
                   finished=_dt.datetime.now().isoformat(timespec="seconds"),
                   git=_git_hash(), host=platform.node(),
                   numpy=np.__version__, n_tasks=len(tasks),
                   n_missing=len(missing))
    print("[injset] %s: %d/%d shards present -> %s"
          % (status, len(tasks) - len(missing), len(tasks), out_dir),
          flush=True)
    return status


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_injection_set(out_dir, require_complete=True):
    """Concatenate all shards into one dict of per-injection arrays.

    Returns ``(data, stars, manifest)``. ``data`` holds the binary arrays
    (star, logP, q, e_draw, circ_comp, e, K1, M1, incl, det, kill_flag,
    det_branch, P_fit) and the singles (single_star, single_det).
    """
    with open(os.path.join(out_dir, MANIFEST)) as fh:
        manifest = yaml.safe_load(fh)
    if require_complete and manifest.get("status") != "complete":
        raise RuntimeError("injection set at %s is %r"
                           % (out_dir, manifest.get("status")))
    stars = pd.read_csv(os.path.join(out_dir, STARS_CSV))
    paths = sorted(glob.glob(os.path.join(out_dir, SHARD_DIR, "f*_c*.npz")))
    if not paths:
        raise FileNotFoundError("no shards under %s" % out_dir)
    keys = ("star", "logP", "q", "e_draw", "circ_comp", "e", "K1", "M1",
            "incl", "det", "kill_flag", "det_branch", "P_fit",
            "single_star", "single_det")
    parts = {k: [] for k in keys}
    for p in paths:
        with np.load(p) as z:
            for k in keys:
                parts[k].append(z[k])
    data = {k: np.concatenate(v) for k, v in parts.items()}
    return data, stars, manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-star", type=int, required=True,
                    help="binary injections per star")
    ap.add_argument("--chunk", type=int, default=250,
                    help="binaries per star per task (default 250)")
    ap.add_argument("--n-single-per-star", type=int, default=0,
                    help="noise-only curves per star (FP rate)")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--stars", default=None,
                    help="comma-separated star IDs (smoke subsets)")
    args = ap.parse_args(argv)
    subset = ([s.strip() for s in args.stars.split(",") if s.strip()]
              if args.stars else None)
    status = run(args.config, args.out_dir, args.n_per_star, args.chunk,
                 args.n_single_per_star, args.seed, args.n_workers,
                 stars_subset=subset)
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
