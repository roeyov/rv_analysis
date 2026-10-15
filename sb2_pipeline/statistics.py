"""
sb2_pipeline.statistics — information criteria for the joint fit and the
K1 < K2 naming convention.

The IC formulas mirror orbital.statistics.calculate_statistical_flags with
two-component sums and N = 2 * n_epochs (both RV series count as data).
"""

import numpy as np

from utils.constants import (
    PERIOD, GAMMA, K1_STR, K2_STR, OMEGA, ECC, T,
    LN_SIGMA_JITTER, LN_SIGMA_JITTER_1, LN_SIGMA_JITTER_2,
)
from orbital.fitting import _safe_errors
from orbital.kepler import nus1
from simulations.common import RV12

from sb2_pipeline.constants import (
    SB2_TS1, SB2_TS2, SB2_RV1, SB2_ERR1, SB2_RV2, SB2_ERR2, GAMMA1, GAMMA2,
    RV1, ERR1, RV2, ERR2, FALLBACK1, FALLBACK2, USE1, USE2,
)


def sb2_statistical_flags(data, mini_results, is_null=False):
    """Joint-model information criteria (llh/aic/bic/bicc/ev_bic/redchi)."""
    ts1 = np.asarray(data[SB2_TS1], float)
    ts2 = np.asarray(data[SB2_TS2], float)
    rv1 = np.asarray(data[SB2_RV1], float)
    rv2 = np.asarray(data[SB2_RV2], float)
    err1 = _safe_errors(data[SB2_ERR1])
    err2 = _safe_errors(data[SB2_ERR2])

    p = mini_results.params
    if LN_SIGMA_JITTER in p:            # shared-jitter mode
        sj1 = sj2 = float(np.exp(p[LN_SIGMA_JITTER].value))
    else:
        sj1 = float(np.exp(p[LN_SIGMA_JITTER_1].value))
        sj2 = float(np.exp(p[LN_SIGMA_JITTER_2].value))

    if is_null:
        v1 = float(p[GAMMA1].value)
        v2 = float(p[GAMMA2].value)
    else:
        P = float(p[PERIOD].value)
        T0 = float(p[T].value)
        ecc = float(p[ECC].value)
        gam = float(p[GAMMA].value)
        K1 = float(p[K1_STR].value)
        K2 = float(p[K2_STR].value)
        om = float(p[OMEGA].value)
        v1, _ = RV12(nus1(ts1, P, T0, ecc), gam, K1, K2, om, ecc)
        _, v2 = RV12(nus1(ts2, P, T0, ecc), gam, K1, K2, om, ecc)

    sig1 = err1 ** 2 + sj1 ** 2
    sig2 = err2 ** 2 + sj2 ** 2
    calc_chisqr = float(np.sum((rv1 - v1) ** 2 / sig1)
                        + np.sum((rv2 - v2) ** 2 / sig2))

    ndata = rv1.size + rv2.size
    nvarys = mini_results.nvarys
    dof = max(ndata - nvarys, 1)
    redchi = calc_chisqr / dof

    llh = calc_chisqr + float(np.sum(np.log(sig1)) + np.sum(np.log(sig2)))
    aic = 2 * nvarys + llh
    bic = nvarys * np.log(ndata) + llh
    bicc = nvarys * np.log(ndata) * (ndata / (ndata - nvarys - 2)) + llh
    ev_bic = nvarys * np.log(ndata) + llh * (1 - 1 / ndata)

    return {
        "llh": llh,
        "chisqr": calc_chisqr,
        "aic": aic,
        "bic": bic,
        "bicc": bicc,
        "ev_bic": ev_bic,
        "redchi": redchi,
        "ndata": ndata,
    }


def enforce_k_convention_row(row):
    """
    Relabel a summary row to K1 < K2 (component 1 = primary).

    Swapping K1 <-> K2 with omega -> omega + pi leaves the per-component model
    curves attached to the *other* data series, so any code consuming a
    k_swapped row must also swap the data columns (apply_k_swap_to_data).
    The jitters are per-data-series nuisance parameters and swap with their
    series; likewise the circ_* component columns. The circular comparison
    has omega pinned (no omega column), so its relabelling is a T0 shift of
    half a period instead. Joint statistics are label-invariant and untouched.
    """
    k1 = row.get(f"{K1_STR}_value", np.nan)
    k2 = row.get(f"{K2_STR}_value", np.nan)
    row["k_swapped"] = False
    if not (np.isfinite(k1) and np.isfinite(k2) and k1 > k2):
        return row

    for name1, name2 in ((K1_STR, K2_STR),
                         (LN_SIGMA_JITTER_1, LN_SIGMA_JITTER_2)):
        for suffix in ("init", "value", "vary", "stderr"):
            a, b = f"{name1}_{suffix}", f"{name2}_{suffix}"
            if a in row or b in row:
                row[a], row[b] = row.get(b, np.nan), row.get(a, np.nan)

    om = row.get(f"{OMEGA}_value", np.nan)
    if np.isfinite(om):
        row[f"{OMEGA}_value"] = (om + np.pi) % (2 * np.pi)

    for a, b in (("circ_K1_value", "circ_K2_value"),
                 ("circ_ln_sigmaJ1_value", "circ_ln_sigmaJ2_value")):
        if a in row or b in row:
            row[a], row[b] = row.get(b, np.nan), row.get(a, np.nan)

    circ_P = row.get("circ_Period_value", np.nan)
    circ_T0 = row.get("circ_T0_value", np.nan)
    if np.isfinite(circ_P) and np.isfinite(circ_T0):
        row["circ_T0_value"] = circ_T0 + circ_P / 2.0

    row["k_swapped"] = True
    return row


def apply_k_swap_to_data(df):
    """Swap the component columns of a corrected per-epoch table (rv1 <-> rv2)."""
    df = df.copy()
    pairs = [(RV1, RV2), (ERR1, ERR2)]
    for a, b in ((FALLBACK1, FALLBACK2), (USE1, USE2)):
        if a in df.columns and b in df.columns:
            pairs.append((a, b))
    for a, b in pairs:
        df[[a, b]] = df[[b, a]].to_numpy()
    return df
