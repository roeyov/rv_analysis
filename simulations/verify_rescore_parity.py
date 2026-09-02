"""Acceptance checks for offline bias-grid re-scores (2026-09 campaign).

Compares a re-score output dir against its reference run (the dir whose
``det_shards/`` the re-score symlinked) and enforces the mode-specific
expectations agreed before any science variant ran:

  parity        — unchanged settings must reproduce the reference exactly:
                  every ``v__*`` cube, ``pdet_cube``, and the global hists
                  equal (np.array_equal, allclose(rtol=1e-6) fallback);
                  identical joint argmax.
  manual_parity — like ``parity`` but the test dir's variant tag differs
                  (``*__manual__*`` vs the reference ``*__numerical__*``):
                  cubes are compared across the tag pair. Validates the
                  manual-cutoff path against the elbow it reproduces.
  env           — envelope e-mask variant: the logP and K1 p-value cubes
                  must be BYTE-identical to the reference (the mask may only
                  touch the e channel); the e/gmf cubes are reported, not
                  compared.
  binom         — binomial_mode=none variant: on a random sample of cells,
                  log_gmf_ref − log_gmf_none == log binom.pmf(N_det_obs,
                  N_stars, p_det) to float32 tolerance; logP/e/K1 p-value
                  cubes byte-identical.

Exit code 0 = all checks pass; 1 = any failure (the chain script aborts).

Usage:
  python -m simulations.verify_rescore_parity \
      --ref  /path/to/ed_hoc_rlof_check_ext3 \
      --test /path/to/ext3_rescore_parity \
      --mode parity
"""

import argparse
import os
import sys

import numpy as np
from scipy.stats import binom


def _load_cubes(run_dir):
    for name in ("grid_cubes.npz", "checkpoint_cubes.npz"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return np.load(path, allow_pickle=True), path
    raise FileNotFoundError("no grid_cubes.npz / checkpoint_cubes.npz in %s"
                            % run_dir)


def _load_hists(run_dir):
    for name in ("grid_hists.npz", "checkpoint_hists.npz"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return np.load(path, allow_pickle=True), path
    return None, None


def _variant_tags(cubes):
    if "variants" in cubes.files:
        return [str(t) for t in np.atleast_1d(cubes["variants"])]
    return []


def _fail(msg):
    print("FAIL: %s" % msg)
    return 1


def _ok(msg):
    print("  ok: %s" % msg)
    return 0


def _cmp_arrays(name, a, b, rtol=1e-6, atol=0.0, strict=False):
    """Equality check; returns 0 on pass.

    strict=True (the parity gates' default): bit-identity or fail — the
    re-score runs the same code on the same shards in the same env, so
    ANY numeric drift is a real behavioral change in the scoring path,
    even when it is far below allclose tolerance (adversarial-review
    finding, 2026-09-02). strict=False keeps the NaN/-inf-aware allclose
    fallback for cross-tag comparisons where tiny representation
    differences are expected.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return _fail("%s shape %s != %s" % (name, a.shape, b.shape))
    same_special = (np.array_equal(np.isnan(a), np.isnan(b))
                    and np.array_equal(np.isposinf(a), np.isposinf(b))
                    and np.array_equal(np.isneginf(a), np.isneginf(b)))
    finite = np.isfinite(a) & np.isfinite(b)
    if same_special and np.array_equal(a[finite], b[finite]):
        return _ok("%s exactly equal" % name)
    d = (np.max(np.abs(a[finite] - b[finite]))
         if finite.any() else float("nan"))
    if strict:
        return _fail("%s not bit-identical (max finite |Δ| = %.3g) — "
                     "real drift in the scoring path; pass --allow-close "
                     "ONLY if the drift is understood and accepted"
                     % (name, d))
    if same_special and np.allclose(a[finite], b[finite],
                                    rtol=rtol, atol=atol):
        return _ok("%s allclose (max |Δ| = %.3g)" % (name, d))
    nbad = int(np.sum(~np.isclose(a, b, rtol=rtol, atol=atol,
                                  equal_nan=True)))
    return _fail("%s differs in %d/%d cells" % (name, nbad, a.size))


def _report_shift(tag, cubes_ref, cubes_test, key_ref, key_test):
    """Print the argmax shift between two gmf cubes (informational)."""
    ref = cubes_ref[key_ref]
    test = cubes_test[key_test]
    im_r = np.unravel_index(np.nanargmax(ref), ref.shape)
    im_t = np.unravel_index(np.nanargmax(test), test.shape)
    print("  info[%s]: argmax ref=%s test=%s%s"
          % (tag, im_r, im_t, "" if im_r == im_t else "  << MOVED"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True, help="reference run dir")
    ap.add_argument("--test", required=True, help="re-score dir to verify")
    ap.add_argument("--mode", required=True,
                    choices=["parity", "manual_parity", "env", "binom"])
    ap.add_argument("--n-sample", type=int, default=1000,
                    help="binom mode: number of random cells to check")
    ap.add_argument("--allow-close", action="store_true",
                    help="parity modes: downgrade the bit-identity "
                         "requirement to allclose(rtol=1e-6) — use ONLY "
                         "when a sub-tolerance drift is understood")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cubes_ref, p_ref = _load_cubes(args.ref)
    cubes_test, p_test = _load_cubes(args.test)
    print("ref:  %s" % p_ref)
    print("test: %s" % p_test)

    tags_ref = _variant_tags(cubes_ref)
    tags_test = _variant_tags(cubes_test)
    if len(tags_ref) != 1 or len(tags_test) != 1:
        print("FAIL: expected single-variant cubes (ref=%s test=%s)"
              % (tags_ref, tags_test))
        return 1
    tag_r, tag_t = tags_ref[0], tags_test[0]
    print("variants: ref=%s test=%s" % (tag_r, tag_t))

    failures = 0

    # Grid axes must always match.
    for g in ("pi_grid", "kappa_grid", "eta_grid", "fbin_grid"):
        failures += _cmp_arrays(g, cubes_ref[g], cubes_test[g])

    def vkeys(cubes, tag):
        pref = "v__%s__" % tag
        return sorted(k for k in cubes.files
                      if k.startswith(pref) and k.endswith("_cube"))

    keys_r = vkeys(cubes_ref, tag_r)
    keys_t = vkeys(cubes_test, tag_t)
    # Map test keys onto ref keys by suffix (tag may differ in manual_parity).
    suf_r = {k[len("v__%s__" % tag_r):]: k for k in keys_r}
    suf_t = {k[len("v__%s__" % tag_t):]: k for k in keys_t}
    if set(suf_r) != set(suf_t):
        failures += _fail("cube key sets differ: only-ref=%s only-test=%s"
                          % (sorted(set(suf_r) - set(suf_t)),
                             sorted(set(suf_t) - set(suf_r))))
        return 1

    if args.mode in ("parity", "manual_parity"):
        strict = not args.allow_close
        failures += _cmp_arrays("pdet_cube", cubes_ref["pdet_cube"],
                                cubes_test["pdet_cube"], strict=strict)
        for suf in sorted(suf_r):
            failures += _cmp_arrays(suf, cubes_ref[suf_r[suf]],
                                    cubes_test[suf_t[suf]], strict=strict)
        # Joint argmax must not move for any gmf cube.
        for suf in sorted(suf_r):
            if not suf.startswith("gmf_"):
                continue
            ref = cubes_ref[suf_r[suf]]
            test = cubes_test[suf_t[suf]]
            if (np.unravel_index(np.nanargmax(ref), ref.shape)
                    != np.unravel_index(np.nanargmax(test), test.shape)):
                failures += _fail("%s argmax moved" % suf)
            else:
                failures += _ok("%s argmax identical" % suf)
        h_ref, hp = _load_hists(args.ref)
        h_test, _ = _load_hists(args.test)
        if h_ref is not None and h_test is not None:
            for k in h_ref.files:
                if k not in h_test.files:
                    failures += _fail("hist key %s missing in test" % k)
                    continue
                failures += _cmp_arrays("hist:%s" % k, h_ref[k],
                                        h_test[k], strict=strict)
        else:
            print("  info: hists not compared (missing on one side)")

    elif args.mode == "env":
        # The envelope mask may only touch the e channel: logP and K1
        # p-value cubes must be byte-identical.
        failures += _cmp_arrays("pdet_cube", cubes_ref["pdet_cube"],
                                cubes_test["pdet_cube"])
        for suf in sorted(suf_r):
            if "_logP_cube" in suf or "_K1_cube" in suf:
                a, b = cubes_ref[suf_r[suf]], cubes_test[suf_t[suf]]
                if np.array_equal(a, b):
                    failures += _ok("%s byte-identical" % suf)
                else:
                    failures += _fail("%s changed — mask leaked out of the "
                                      "e channel" % suf)
            elif "_e_cube" in suf or suf.startswith("gmf_"):
                _report_shift(suf, cubes_ref, cubes_test,
                              suf_r[suf], suf_t[suf])

    elif args.mode == "binom":
        # CDF p-value cubes byte-identical; gmf differs by exactly
        # log binom.pmf(N_det_obs, N_stars, p_det).
        for suf in sorted(suf_r):
            if suf.startswith("gmf_"):
                continue
            a, b = cubes_ref[suf_r[suf]], cubes_test[suf_t[suf]]
            if np.array_equal(a, b):
                failures += _ok("%s byte-identical" % suf)
            else:
                failures += _fail("%s changed — binomial_mode must not "
                                  "touch CDF p-values" % suf)
        n_det = int(cubes_ref["v__%s__N_det_obs" % tag_r])
        n_stars = int(cubes_ref["v__%s__N_stars" % tag_r])
        pdet = np.asarray(cubes_ref["pdet_cube"], dtype=np.float64)
        rng = np.random.default_rng(args.seed)
        gmf_sufs = [s for s in suf_r if s.startswith("gmf_")]
        for suf in gmf_sufs:
            ref = np.asarray(cubes_ref[suf_r[suf]], dtype=np.float64)
            test = np.asarray(cubes_test[suf_t[suf]], dtype=np.float64)
            flat = np.flatnonzero(np.isfinite(ref) & np.isfinite(test))
            if len(flat) == 0:
                failures += _fail("%s: no finite cells to compare" % suf)
                continue
            pick = rng.choice(flat, size=min(args.n_sample, len(flat)),
                              replace=False)
            idx = np.unravel_index(pick, ref.shape)
            with np.errstate(divide="ignore"):
                logpb = np.log(binom.pmf(n_det, n_stars, pdet[idx]))
            delta = ref[idx] - test[idx]
            bad = ~np.isclose(delta, logpb, rtol=1e-5, atol=1e-3)
            if bad.any():
                worst = np.max(np.abs(delta - logpb))
                failures += _fail(
                    "%s: identity log_gmf_ref - log_gmf_none = log p_binom "
                    "violated in %d/%d sampled cells (max |Δ| = %.3g)"
                    % (suf, int(bad.sum()), len(pick), worst))
            else:
                failures += _ok(
                    "%s: binomial identity holds on %d sampled cells"
                    % (suf, len(pick)))

    if failures:
        print("RESULT: FAIL (%d check(s) failed)" % failures)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
