"""simulations.true_posterior — likelihood-based posterior for the bias grid.

Replaces the Fisher-combined CvM "pseudo-posterior" of simulations.bias_grid
with a selection-corrected, per-star population likelihood over
(pi, kappa, eta, f_bin). The forward model is the same: the power-law
population, the pipeline_batched detector, and the noise and nuisance
draws. What changes is the inference:

* one theta-independent, star-labelled injection set per detector
  (injection_set.py), reweighted to any population parameters;
* the per-star likelihood (likelihood.py): Bernoulli count terms for
  every catalogue star, plus the intrinsic density of each detected
  star's (logP, e, K1), divided by its detection probability above the
  logP cut;
* the per-star observed outcomes (obs_catalog.py).

Design and validation plan:
~/.claude/plans/what-are-the-reasons-cozy-hopper.md
"""
