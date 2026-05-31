"""bias_grid_lib — internal split of simulations.bias_grid.

Importers should keep using ``from simulations.bias_grid import …`` —
this package is the implementation layer behind that facade. Direct
imports from sub-modules (e.g. ``simulations.bias_grid_lib.physics``)
work too but are not part of the public contract.
"""

from simulations.bias_grid_lib.constants import (
    CUBE_SCHEMA_VERSION,
    G_CGS, MSUN, RSUN, DAY, KM, TWOPI,
    _DET_SHARDS_DIR,
    _E_SCORE_MODES, _LOGP_CUTOFF_MODES, _LOGP_CUTOFF_SCOPES,
    _HIST_BINS, _HIST_NBINS, _HIST_PAIRS,
    _trapz,
)
from simulations.bias_grid_lib.logging_utils import logger, setup_logging
from simulations.bias_grid_lib.physics import (
    compute_K1, compute_K1_batch, kepler_E, kepler_E_batch,
    powerlaw_draw, roche_lobe_check, rv_model_jit,
)
from simulations.bias_grid_lib.detection import (
    DETECTION_METHODS, _detect_full_pipeline, _detect_rv_threshold,
)
from simulations.bias_grid_lib.statistics import (
    _ALL_TESTS, _DIST_TESTS, _SCORED_TESTS,
    _ad_pvalue, _clip_to_range, _cvm_pvalue, _ks_pvalue,
    _mad, _safe_distance, _safe_pvalue, _wasserstein_distance,
)
from simulations.bias_grid_lib.cutoffs import (
    _compute_logP_cutoff, _numerical_logP_cutoff,
    _resolve_e_score_mode,
    _resolve_logP_cutoff_mode, _resolve_logP_cutoff_scope,
)
from simulations.bias_grid_lib.scoring import (
    _compute_scores, _make_scoring_ctx,
)
from simulations.bias_grid_lib.obs_io import (
    _load_catalog_counts, _parse_val_with_errors, load_observed_from_tex,
)
from simulations.bias_grid_lib.injection import (
    _worker_star_injections, _worker_star_injections_vectorized,
    detect_single_star, inject_and_detect,
)
from simulations.bias_grid_lib.checkpointing import (
    _det_shard_path, _hists_from_shard,
    _load_det_shard, _save_checkpoint, _save_det_index, _save_det_shard,
)
from simulations.bias_grid_lib.parallel import (
    _init_grid_worker, _init_resume_worker,
    _resume_score_worker, _worker_grid_point,
)
from simulations.bias_grid_lib.star_loading import (
    load_observed_star_properties, load_star_properties,
)
from simulations.bias_grid_lib.plotting import (
    format_grid_summary, plot_grid_results,
)
from simulations.bias_grid_lib.aggregation import aggregate_tasks
from simulations.bias_grid_lib.engine import GridSearchEngine
