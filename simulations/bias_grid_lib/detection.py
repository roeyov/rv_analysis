"""Pluggable detection backends used by the injection workers.

The grid search injects a synthetic binary and asks one of these callables
"detected or not?". The full-pipeline backend re-uses the production
periodogram + lmfit stack; the fast RV-threshold backend collapses
detection to a pair-wise delta-RV cutoff for closure/smoke runs.
"""


def _detect_full_pipeline(MJDs, rv_obs, rv_err, args_dict):
    """Original: full periodogram + lmfit + BICc pipeline."""
    from pipeline.evaluator import detect_from_arrays
    return detect_from_arrays(MJDs, rv_obs, rv_err, args_dict, use_fwhm=True)


def _detect_rv_threshold(MJDs, rv_obs, rv_err, args_dict):
    """Fast: pairwise delta-RV significance threshold.

    When rv_err is constant (uniform errors), the pairwise sigma matrix
    reduces to a scalar: rv_err_val * sqrt(2).  The two conditions
    (|drv| > drv_thresh  AND  significance > sign_thresh) collapse to
    a single threshold on max(rv) - min(rv).
    """
    drv_thresh = args_dict.get("drv_threshold", 20.0)
    sign_thresh = args_dict.get("significance_threshold", 4.0)

    # Pre-computed threshold may be cached in args_dict by the caller
    eff_thresh = args_dict.get("_rv_eff_threshold")
    if eff_thresh is not None:
        detected = bool((rv_obs.max() - rv_obs.min()) > eff_thresh)
    else:
        # Fallback: compute from rv_err (handles non-uniform case)
        from orbital.statistics import binary_rv_threshold
        detected = bool(binary_rv_threshold(rv_obs, rv_err,
                                            drv_tresh=drv_thresh,
                                            sign_threshold=sign_thresh))

    info = {"reason": "", "detection_method": "rv_threshold"}
    return detected, info


DETECTION_METHODS = {
    "pipeline": _detect_full_pipeline,
    "rv_threshold": _detect_rv_threshold,
}
