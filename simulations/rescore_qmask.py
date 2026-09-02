"""Offline q-masked re-score of a pipeline_batched bias-grid run.

Re-scores an existing run's det shards with all injected binaries below
``--q-cut`` reclassified as SINGLES — exactly as if, at fixed realization
count, the model had binary fraction f_bin × F(q ≥ q_cut | κ) with q
drawn on [q_cut, q_max]. This is EXACT for ``pipeline_batched`` runs
with the detection-side RLOF gate off: every injected binary appears in
the shard's det+nondet arrays with its q (``n_rlof ≡ 0``), so

    n_det'      = Σ det  q ≥ q_cut
    p_det'      = n_det' / n_physical      (denominator UNCHANGED)

and the CDF tests run on the masked detected arrays. Keeping the
original n_physical as denominator is what makes the f_bin × F node
relabeling exact: dropping the masked binaries from the denominator
instead would put the node at the odds-form fraction
f_bin·F / (1 − f_bin·(1−F)) and silently skew the comparison against a
q_min = q_cut injection run (caught in adversarial review, 2026-09-02).
Because the same noise realizations are reused, the comparison against
the full-domain cubes isolates the q-domain restriction with zero
Monte-Carlo noise — the cleanest possible statement of the κ↔q_min
degeneracy on the shared κ range.

Coordinate semantics: the grid nodes keep their FULL-DOMAIN labels. A
node (π, κ, η, f_bin) scored under the mask represents a q ≥ q_cut model
whose domain-restricted binary fraction is EXACTLY
f_bin × F(q ≥ q_cut | κ); derive that per-draw downstream
(posterior_slices.py) when comparing to a q_min = q_cut injection run.

Obs-side inputs are read from the run's own cube (the frozen copy of the
tables at launch time), and scoring knobs from its run_config.yaml — NOT
from DEFAULT_BIAS_CFG — so the re-score is faithful to the run even if
the repo defaults or the tex tables have moved since.

Usage (astro3):
  python -m simulations.rescore_qmask \
      --run-dir /jonathan_storage/roeyovadia/bias_grid_results/ed_hoc_rlof_check_ext3 \
      --q-cut 0.1 \
      --output-dir /jonathan_storage/roeyovadia/bias_grid_results/ext3_qmask01 \
      --n-workers 8
"""

import argparse
import os
import sys

import numpy as np
import yaml

from simulations.bias_grid_lib.constants import (
    _DET_SHARDS_DIR, split_variant_tag,
)
from simulations.bias_grid_lib.cutoffs import _build_variant_inputs
from simulations.bias_grid_lib.scoring import (
    _compute_scores, _make_scoring_ctx,
)
from simulations.bias_grid_lib.statistics import _ALL_TESTS

# Globals installed by the pool initializer (fork/forkserver-safe).
_CTXS = None
_SHARD_DIR = None
_Q_CUT = None


def _load_run_inputs(run_dir):
    """Frozen obs arrays from the cube + scoring knobs from run_config."""
    cube_path = None
    for name in ("grid_cubes.npz", "checkpoint_cubes.npz"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            cube_path = p
            break
    if cube_path is None:
        raise FileNotFoundError("no cube npz in %s" % run_dir)
    cubes = np.load(cube_path, allow_pickle=True)

    rc_path = os.path.join(run_dir, "run_config.yaml")
    with open(rc_path) as fh:
        rc = yaml.safe_load(fh)
    bg = rc.get("bias_grid", {}) or {}
    bias_cfg = rc.get("bias_cfg", {}) or {}

    obs = {
        "logP": np.asarray(cubes["obs_logP"], dtype=float),
        "e_value": np.asarray(cubes["obs_e_value"], dtype=float),
        "e_is_upper_limit": np.asarray(cubes["obs_e_is_upper_limit"],
                                       dtype=bool),
        "K1": np.asarray(cubes["obs_K1"], dtype=float),
        "n_catalog_total": int(cubes["obs_n_catalog_total"]),
        "n_catalog_nonsingle": int(cubes["obs_n_catalog_nonsingle"]),
    }
    tags = [str(t) for t in np.atleast_1d(cubes["variants"])]
    grids = {g: np.asarray(cubes[g], dtype=float)
             for g in ("pi_grid", "kappa_grid", "eta_grid", "fbin_grid")}

    knobs = {
        "logP_cutoff_scope": bg.get("logP_cutoff_scope", "exclude"),
        "smooth_sigma": float(bg.get("logP_cutoff_smooth_sigma", 0.15)),
        "manual_value": bg.get("logP_cutoff_value"),
        "e_mask_mode": bg.get("e_mask_mode", "lucy_all"),
        "envelope_period_days": float(bg.get("envelope_period_days", 5.0)),
        "binomial_mode": bg.get("binomial_mode", "catalog"),
        "n_stars_sample": int(bg.get("n_stars_sample")
                              or bias_cfg.get("n_stars_sample") or 134),
        "scored_tests": list(bg.get("scored_tests")
                             or ["ks", "ad", "cvm", "wass"]),
    }
    return cubes, obs, tags, grids, knobs


def _build_ctxs(obs, tags, knobs):
    ctxs = {}
    meta = {}
    active_pvalue_tests = {k: _ALL_TESTS[k] for k in knobs["scored_tests"]
                           if k in _ALL_TESTS}
    score_wass = "wass" in knobs["scored_tests"]
    for tag in tags:
        em, cm, lucy = split_variant_tag(tag)
        if lucy is None:
            raise ValueError("legacy 2-part variant tag %r unsupported" % tag)
        vi = _build_variant_inputs(
            em, cm, lucy, knobs["logP_cutoff_scope"],
            obs["logP"], obs["e_value"], obs["e_is_upper_limit"], obs["K1"],
            knobs["n_stars_sample"],
            obs["n_catalog_total"], obs["n_catalog_nonsingle"],
            smooth_sigma=knobs["smooth_sigma"],
            manual_value=knobs["manual_value"],
            e_mask_mode=knobs["e_mask_mode"],
            envelope_period_days=knobs["envelope_period_days"])
        ctx = _make_scoring_ctx(
            obs_logP=vi["obs_logP"], obs_e=vi["obs_e"], obs_K1=vi["obs_K1"],
            clip_range=vi["clip_range"], e_score_mode=em,
            obs_e_cont=vi["obs_e_cont"],
            n_obs_circ=vi["n_obs_circ"] or 0,
            n_obs_e_total=vi["n_obs_e_total"] or 0,
            N_det_obs=vi["N_det_obs"], N_stars=vi["N_stars"],
            sim_logP_floor=vi["sim_logP_floor"],
            binomial_mode=knobs["binomial_mode"])
        ctx["pvalue_tests"] = active_pvalue_tests
        ctx["score_wass"] = score_wass
        ctxs[tag] = ctx
        meta[tag] = {
            "logP_cutoff": vi["logP_cutoff"],
            "N_stars": vi["N_stars"],
            "N_det_obs": vi["N_det_obs"],
            "e_score_mode": em,
            "logP_cutoff_mode": cm,
            "apply_lucy_sweeny_e": bool(lucy),
            "wass_sigma": ctx["wass_sigma"],
        }
    return ctxs, meta


def _init_worker(shard_dir, ctxs, q_cut):
    global _CTXS, _SHARD_DIR, _Q_CUT
    _CTXS = ctxs
    _SHARD_DIR = shard_dir
    _Q_CUT = q_cut


def _score_one(step):
    path = os.path.join(_SHARD_DIR, "step_%06d.npz" % step)
    with np.load(path) as d:
        if "n_physical" not in d.files or "q_nondet" not in d.files:
            raise ValueError(
                "shard %d predates the counts/nondet schema — q-mask "
                "re-scoring needs n_physical + *_nondet arrays" % step)
        if int(d["n_rlof"]) != 0:
            raise ValueError(
                "shard %d has n_rlof=%d — q-mask exactness assumes the "
                "detection-side RLOF gate was off" % (step, int(d["n_rlof"])))
        q_det = np.asarray(d["q"], dtype=float)
        q_non = np.asarray(d["q_nondet"], dtype=float)
        n_physical = int(d["n_physical"])
        logP_det = np.asarray(d["logP"], dtype=float)
        e_det = np.asarray(d["e"], dtype=float)
        K1_det = np.asarray(d["K1"], dtype=float)
        ijkl = tuple(int(x) for x in np.asarray(d["ijkl"]).ravel()[:4])
    n_singles = n_physical - len(q_det) - len(q_non)
    if n_singles < 0:
        raise ValueError("shard %d: n_singles=%d < 0" % (step, n_singles))

    keep = q_det >= _Q_CUT
    n_det = int(keep.sum())
    # Denominator stays the ORIGINAL n_physical: masked q < q_cut binaries
    # are reclassified as singles, so the node is exactly a q >= q_cut
    # model with binary fraction f_bin * F(q >= q_cut | kappa). See the
    # module docstring — do NOT shrink the denominator.
    res = {
        "logP_det": logP_det[keep],
        "e_det": e_det[keep],
        "K1_det": K1_det[keep],
        "n_physical": n_physical,
        "p_det": (n_det / n_physical) if n_physical > 0 else 0.0,
    }
    scores = {tag: _compute_scores(res, ctx) for tag, ctx in _CTXS.items()}
    return step, ijkl, res["p_det"], scores


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--q-cut", type=float, default=0.1)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N shards (smoke test)")
    args = ap.parse_args()

    cubes, obs, tags, grids, knobs = _load_run_inputs(args.run_dir)
    ctxs, meta = _build_ctxs(obs, tags, knobs)
    print("run: %s" % args.run_dir)
    print("variants: %s | tests: %s | q_cut: %g"
          % (tags, knobs["scored_tests"], args.q_cut))

    shard_dir = os.path.join(args.run_dir, _DET_SHARDS_DIR)
    steps = sorted(
        int(f[5:-4]) for f in os.listdir(shard_dir)
        if f.startswith("step_") and f.endswith(".npz"))
    if args.limit:
        steps = steps[:args.limit]
    print("%d shards to re-score" % len(steps))

    shape = tuple(len(grids[g]) for g in
                  ("pi_grid", "kappa_grid", "eta_grid", "fbin_grid"))
    pdet_cube = np.full(shape, np.nan, dtype=np.float32)
    out_cubes = {}
    score_keys = None

    def _consume(step, ijkl, p_det, scores):
        nonlocal score_keys
        pdet_cube[ijkl] = p_det
        if score_keys is None:
            score_keys = {
                tag: [k for k in sc
                      if k.startswith("log_gmf_") or "_p_" in k]
                for tag, sc in scores.items()}
            for tag, keys in score_keys.items():
                for k in keys:
                    out_cubes[(tag, k)] = np.full(shape, np.nan,
                                                  dtype=np.float32)
        for tag, sc in scores.items():
            for k in score_keys[tag]:
                v = sc.get(k)
                out_cubes[(tag, k)][ijkl] = (np.nan if v is None
                                             else np.float32(v))

    if args.n_workers > 1:
        import multiprocessing as mp
        ctx_mp = mp.get_context(
            "fork" if sys.platform.startswith("linux") else "spawn")
        with ctx_mp.Pool(args.n_workers, initializer=_init_worker,
                         initargs=(shard_dir, ctxs, args.q_cut)) as pool:
            from tqdm import tqdm
            for step, ijkl, p_det, scores in tqdm(
                    pool.imap_unordered(_score_one, steps, chunksize=64),
                    total=len(steps), unit="pt", mininterval=2.0):
                _consume(step, ijkl, p_det, scores)
    else:
        _init_worker(shard_dir, ctxs, args.q_cut)
        from tqdm import tqdm
        for step in tqdm(steps, unit="pt", mininterval=2.0):
            _consume(*_score_one(step))

    os.makedirs(args.output_dir, exist_ok=True)
    save_kw = {
        "cube_schema_version": np.int64(7),
        "variants": np.array(tags),
        "pdet_cube": pdet_cube,
        "q_cut": np.float64(args.q_cut),
        "qmask_source_run": np.str_(os.path.abspath(args.run_dir)),
    }
    for g, arr in grids.items():
        save_kw[g] = arr
    # Copy the frozen obs block through for downstream readers.
    for k in cubes.files:
        if k.startswith("obs_") or k in ("sb1_tex", "sb2_tex",
                                         "ostar_catalog"):
            save_kw[k] = cubes[k]
    for tag, m in meta.items():
        for mk, mv in m.items():
            if mk == "wass_sigma":
                for ch, val in mv.items():
                    save_kw["v__%s__wass_sigma_%s" % (tag, ch)] = \
                        np.float64(val)
            else:
                save_kw["v__%s__%s" % (tag, mk)] = np.asarray(mv)
    for (tag, k), cube in out_cubes.items():
        # Match production naming: log_gmf_<t> -> gmf_<t>_cube,
        # <t>_p_<par> -> <t>_<par>_cube.
        if k.startswith("log_gmf_"):
            key = "v__%s__gmf_%s_cube" % (tag, k[len("log_gmf_"):])
        else:
            t, par = k.split("_p_", 1)
            key = "v__%s__%s_%s_cube" % (tag, t, par)
        save_kw[key] = cube

    out_path = os.path.join(args.output_dir, "grid_cubes.npz")
    np.savez_compressed(out_path, **save_kw)
    with open(os.path.join(args.output_dir, "rescore_qmask.yaml"), "w") as fh:
        yaml.safe_dump({
            "source_run": os.path.abspath(args.run_dir),
            "q_cut": args.q_cut,
            "n_shards": len(steps),
            "variants": tags,
            "knobs": {k: (v if not isinstance(v, np.floating) else float(v))
                      for k, v in knobs.items()},
            "note": ("grid nodes keep FULL-DOMAIN f_bin labels; "
                     "f_bin(q>=q_cut) = f_bin * F(q>=q_cut | kappa) "
                     "per draw downstream"),
        }, fh, sort_keys=False)
    print("wrote %s" % out_path)

    for (tag, k), cube in sorted(out_cubes.items()):
        if k.startswith("log_gmf_"):
            im = np.unravel_index(np.nanargmax(cube), cube.shape)
            print("  %s %s argmax at (pi=%.3f, kappa=%.3f, eta=%.3f, "
                  "fbin=%.3f)"
                  % (tag, k, grids["pi_grid"][im[0]],
                     grids["kappa_grid"][im[1]], grids["eta_grid"][im[2]],
                     grids["fbin_grid"][im[3]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
