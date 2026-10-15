"""
simulations.closure_anchor — point a mock grid run's binomial term at the
mock survey's own detection count.

Under ``logP_cutoff_scope: exclude`` the binomial term of bias_grid is
anchored on the O-star catalogue counts (real data: 77 non-single of 134).
Mock surveys (bias_grid_closure_generate) write no catalogue. The closure
and coverage drivers therefore used to score every mock against the REAL
count of 77, whatever the mock actually detected. At the kappa = -1 truth,
for example, mocks detected 46-75 systems, so the binomial pulled f_bin
toward the real-data value. This module fixes that: it reads the mock's
detection summary and writes the ``binomial_n_total`` /
``binomial_n_nonsingle`` override keys into the per-seed YAML.

The anchor is what an observer of the mock would count as non-single:
detected binaries plus false-positive detections of single stars. The real
catalogue count is formed the same way.

    python -m simulations.closure_anchor \
        --seed-yaml <seed_dir>/params_seed.yaml \
        --truth-yaml <seed_dir>/truth_params.yaml
"""

import argparse
import sys

import yaml


def mock_anchor(truth_yaml):
    """(n_total, n_nonsingle) of a generated mock survey."""
    with open(truth_yaml) as fh:
        meta = yaml.safe_load(fh)
    summary = meta.get("summary") or {}
    try:
        n_total = int(summary["n_stars"])
        n_detected = int(summary["n_detected"])
    except KeyError as exc:
        raise KeyError("%s has no summary.%s — regenerate the mock with the "
                       "current bias_grid_closure_generate"
                       % (truth_yaml, exc.args[0]))
    n_fp = int(summary.get("n_false_positive") or 0)
    if n_fp < 0:
        raise ValueError("%s: n_false_positive=%d (unmeasured) — cannot form "
                         "the observed count" % (truth_yaml, n_fp))
    n_nonsingle = n_detected + n_fp
    if not 0 <= n_nonsingle <= n_total:
        raise ValueError("%s: need 0 <= n_nonsingle (%d) <= n_total (%d)"
                         % (truth_yaml, n_nonsingle, n_total))
    return n_total, n_nonsingle


def patch_seed_yaml(seed_yaml, truth_yaml):
    """Write the mock's binomial anchor into ``seed_yaml`` (in place)."""
    n_total, n_nonsingle = mock_anchor(truth_yaml)
    with open(seed_yaml) as fh:
        cfg = yaml.safe_load(fh)
    bg = cfg.get("bias_grid") or {}
    bg["binomial_n_total"] = n_total
    bg["binomial_n_nonsingle"] = n_nonsingle
    cfg["bias_grid"] = bg
    with open(seed_yaml, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return n_total, n_nonsingle


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seed-yaml", required=True)
    ap.add_argument("--truth-yaml", required=True)
    args = ap.parse_args(argv)
    n_total, n_nonsingle = patch_seed_yaml(args.seed_yaml, args.truth_yaml)
    print("binomial anchor: N_stars=%d N_det_obs=%d -> %s"
          % (n_total, n_nonsingle, args.seed_yaml))
    return 0


if __name__ == "__main__":
    sys.exit(main())
