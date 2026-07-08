"""Batched mass_flag_peri (Roche-lobe overflow at periastron of the fitted orbit).

The detection gate rejects a solution if ``mass_flag_peri == 1`` — the fitted
period is shorter than the minimum Roche-stable period for that star (given its
mass/radius ± catalog errors, and the fitted K1, e). We REUSE the production
``utils.roche_lobe.compute_min_period_row`` so the flag is bit-identical to the
pipeline, looping over injections (it is a cheap per-system root-find over the
mass ratio). Apply it only to candidate survivors to keep the loop short.
"""

import numpy as np
from utils.roche_lobe import compute_min_period_row


def mass_flag_peri_batch(mass_rows, K1, e, P, i_deg=90.0):
    """True where the fitted orbit triggers Roche-lobe overflow at periastron.

    mass_rows : sequence of per-injection star rows (Mspec, R_star + ±errors).
    K1, e, P  : per-injection fitted values (arrays of equal length).
    Returns bool array (True = flagged -> reject the detection).
    """
    n = len(K1)
    flag = np.zeros(n, dtype=bool)
    for i in range(n):
        try:
            row = compute_min_period_row(mass_rows[i], P_fit=float(P[i]),
                                         K1=float(K1[i]), e=float(e[i]), i_deg=i_deg)
            flag[i] = bool(row["mass_flag_peri"])
        except Exception:
            # Degenerate fit (e.g. K1->0 makes the Roche root-find ill-posed):
            # cannot establish an RLOF violation, so do not reject on this gate.
            flag[i] = False
    return flag
