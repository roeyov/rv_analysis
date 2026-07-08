"""fast_detect — batched, backend-pluggable re-implementation of the BLOeM
binary-detection criterion, for accelerating ``detect_method`` in the bias grid.

The original production stack (``pipeline/``, ``period_search/``, ``orbital/``,
``PDC/``, ``utils/roche_lobe.py``) is left untouched and serves as the validation
oracle. This package reproduces the *same* detection decision — the
``filter_expression`` gate (prob_bicc > 0.5 & mass_flag_peri == 0 &
iter_fap < 0.2) — but evaluated over a whole population of injected RV curves at
once via batched linear algebra, on either a NumPy/BLAS backend (astro3 + Mac
CPU) or an MLX/Metal backend (Mac GPU).

See the package modules:
  backends    — array-module abstraction (numpy | mlx)
  batched_ls  — batched floating-mean Lomb-Scargle power / peak period
  batched_perm — batched permutation null -> iter_fap
  batched_fit — batched fixed-period Keplerian fit + null -> prob_bicc
  rlof        — batched mass_flag_peri (Roche-lobe overflow at periastron)
  detector    — detect_batch(): the full filter_expression gate
"""
