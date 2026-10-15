"""
simulations.true_posterior.regen_mocks — deterministically regenerate a
Tier-4 mock campaign so every mock records its false-positive star IDs.

Mocks generated before 2026-09-29 stored only a false-positive count. The
generator is deterministic given (truth, seed, config, detector), so
re-running it reproduces the same catalogue, now with
``summary.false_positive_ids``. The driver writes each regenerated mock to
``<out>/<tag>/seed_<n>/`` and checks it against the original:
  * the same detected stars and roles, and the same n_detected and
    n_false_positive (required);
  * synthetic_obs.csv byte-identical (reported). On the original host it
    should be; across macOS/Linux, values differ at ~1 ULP.

    python -m simulations.true_posterior.regen_mocks \
        --campaign-dir <tier4 dir> --out-dir <new dir>
"""

import argparse
import filecmp
import glob
import os
import sys

import pandas as pd
import yaml

from simulations import bias_grid_closure_generate as gen


def regen_one(seed_dir, out_dir):
    with open(os.path.join(seed_dir, "truth_params.yaml")) as fh:
        tp = yaml.safe_load(fh)
    with open(os.path.join(seed_dir, "mock_meta.yaml")) as fh:
        mm = yaml.safe_load(fh)
    t = tp["truth"]
    argv = ["--truth-pi", str(t["pi"]), "--truth-kappa", str(t["kappa"]),
            "--truth-eta", str(t["eta"]), "--truth-fbin", str(t["fbin"]),
            "--seed", str(tp["seed"]),
            "--n-inject", str(tp.get("n_inject_per_star", 1)),
            "--sb2-q-threshold", str(tp.get("sb2_q_threshold", 0.5)),
            "--detect-method", mm["detect_method"],
            "--config", tp["config"], "--output-dir", out_dir]
    gen.main(argv)
    a = pd.read_csv(os.path.join(seed_dir, "synthetic_obs.csv"))
    b = pd.read_csv(os.path.join(out_dir, "synthetic_obs.csv"))
    with open(os.path.join(out_dir, "truth_params.yaml")) as fh:
        tp2 = yaml.safe_load(fh)
    same_sets = (list(a["star_id"]) == list(b["star_id"])
                 and list(a["role"]) == list(b["role"]))
    s1, s2 = tp["summary"], tp2["summary"]
    same_counts = (s1["n_detected"] == s2["n_detected"]
                   and s1["n_false_positive"] == s2["n_false_positive"])
    identical = filecmp.cmp(os.path.join(seed_dir, "synthetic_obs.csv"),
                            os.path.join(out_dir, "synthetic_obs.csv"),
                            shallow=False)
    return {"ok": bool(same_sets and same_counts), "identical": identical,
            "n_detected": s2["n_detected"],
            "fp_ids": s2.get("false_positive_ids", [])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    seeds = sorted(glob.glob(os.path.join(a.campaign_dir, "*", "seed_*")))
    bad = 0
    for s in seeds:
        tag = os.path.basename(os.path.dirname(s))
        out = os.path.join(a.out_dir, tag, os.path.basename(s))
        r = regen_one(s, out)
        bad += (not r["ok"])
        print("[regen] %s/%s ok=%s identical=%s n_det=%d fp=%s"
              % (tag, os.path.basename(s), r["ok"], r["identical"],
                 r["n_detected"], r["fp_ids"]), flush=True)
    print("[regen] %d mocks, %d mismatched" % (len(seeds), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
