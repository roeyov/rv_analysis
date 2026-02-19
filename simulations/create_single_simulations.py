"""
simulations.createSingleSimulations — Generate single-star (noise-only) RV
simulations using explicit BLOeM MJD timing arrays.

Output format matches Stage 2 pipeline input:
    filename: SIMuLaTioN_NNNNNN_CCF_RVs.csv
    columns:  Mean RV, Mean RVsig, MJD, SNR_PPL, SNR
"""

import os

import numpy as np
import pandas as pd

from simulations.common import (
    BLOEM_MJD_ARRAYS as var,
    SIGMA_SHAPE, SIGMA_LOC, SIGMA_SCALE,
    simulate_system_refined,
    sample_gamma,
)

N = 5
output_path = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/simulations_light/"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

saved_systems = 0
digits_in_max_real = len(str(N * len(var)))
for i in var:
    # Drop NaNs in case some fields have fewer observations than others
    current_mjds = i
    tmp_rvs = np.zeros_like(current_mjds)

    for j in range(N):
        # 1. Sample errors and noise from your lognormal fit
        cur_rvs, cur_rv_sig = simulate_system_refined(tmp_rvs, SIGMA_SHAPE, SIGMA_LOC, SIGMA_SCALE)

        # 2. Add the systemic velocity (gamma)
        cur_rvs_w_gamma = cur_rvs + sample_gamma()

        # 3. Create DataFrame with specific column names
        df = pd.DataFrame({
            "Mean RV": cur_rvs_w_gamma,
            "Mean RVsig": cur_rv_sig,
            "MJD": current_mjds,
            'SNR_PPL': 100 * np.ones_like(cur_rvs_w_gamma),
            'SNR': 100 * np.ones_like(cur_rvs_w_gamma),
        })

        # 4. Save to CSV
        filename = f"SIMuLaTioN_{saved_systems:0{digits_in_max_real}d}_CCF_RVs.csv"
        out = os.path.join(output_path, filename)
        df.to_csv(out, index=False)

        saved_systems += 1
