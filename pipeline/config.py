"""
pipeline.config — Parameter loading (YAML or JSON) and search-region manipulation.
"""

import json
import os

import numpy as np
import pandas as pd

from utils.constants import (
    LMFIT_PARAMS, SEARCH_REGION, INIT_VAL, MIN_VAL, MAX_VAL, VARY,
    PERIODOGRAM_PARAMS, PERI_MIN_PERIOD, PERI_MAX_PERIOD, MANUAL_CANDIDATES,
)

# Resolve default config relative to the repository root (two levels up).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Prefer params.yaml; fall back to params.json for backward compat.
PARAM_FILE = os.path.join(_REPO_ROOT, "params.yaml")
if not os.path.exists(PARAM_FILE):
    PARAM_FILE = os.path.join(_REPO_ROOT, "params.json")

# Keep old name as alias so existing imports still work.
JSON_PARAM_FILE = PARAM_FILE


def load_args(config_path=None):
    """Load pipeline parameters from a YAML or JSON file (auto-detected)."""
    if config_path is None:
        config_path = PARAM_FILE
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, "r") as f:
        if ext in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(f)
        else:
            return json.load(f)


def _load_json_args(json_param_file=None):
    """Backward-compatible alias for :func:`load_args`."""
    return load_args(json_param_file)


def change_search_region_default(args_dict, field_name, init_val, min_val, max_val, vary):
    """Set the search-region bounds for one lmfit parameter in *args_dict*."""
    args_dict[LMFIT_PARAMS][SEARCH_REGION][field_name][INIT_VAL] = init_val
    args_dict[LMFIT_PARAMS][SEARCH_REGION][field_name][MIN_VAL] = min_val
    args_dict[LMFIT_PARAMS][SEARCH_REGION][field_name][MAX_VAL] = max_val
    args_dict[LMFIT_PARAMS][SEARCH_REGION][field_name][VARY] = vary


def _inject_manual_candidates(candidates_df: pd.DataFrame,
                              manual_periods: list[float],
                              *,
                              pmin: float,
                              pmax: float,
                              jitter: bool = True,
                              full_period_range: bool = True) -> pd.DataFrame:
    """Append manually specified candidate periods to the candidates DataFrame."""
    if not manual_periods:
        return candidates_df

    extra_rows = []
    for p in manual_periods:
        rec = {
            "method": "MANUAL",
            "period": float(p),
            "jitter": bool(jitter),
            # so your worker can use use_fwhm=True and these bounds
            "fwhm_per_low": float(pmin) if full_period_range else float(p * 0.9),
            "fwhm_per_high": float(pmax) if full_period_range else float(p * 1.1),
            "LS_power": np.nan, "LS_fap": np.nan, "LS_iter_fap": np.nan,
            "PDC_power": np.nan, "PDC_fap": np.nan, "PDC_iter_fap": np.nan,
        }
        extra_rows.append(rec)

    extra_df = pd.DataFrame(extra_rows)
    if candidates_df is None or candidates_df.empty:
        return extra_df
    return pd.concat([candidates_df, extra_df], ignore_index=True)
