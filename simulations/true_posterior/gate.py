"""
simulations.true_posterior.gate — acceptance gate of a finished posterior
run.

A run is rejected when its grid truncates the posterior: an outer node of
the pi, kappa, eta, f_tot or f_gt marginal, the top node of f_lt, or the
overflow bin of log10 rho holds more than ``--edge-max`` (default 1e-4) of
the marginal mass. f_lt = 0 and rho = 0 are physical boundaries and are
reported only, and so is the top node of f_tot or f_gt when that node is
the physical limit f = 1 (a grid that stops below 1 is gated as before; the
mass at the limit is recorded and, above ``--edge-max``, reported as a
warning). A run is also rejected when a grid leaves the exponent span
of the injection set's reference mixture, outside which the importance
weights are unbounded (checked when the run's injection set is readable).

A posterior array with a NaN or without a finite entry rejects the run,
and so does an edge mass that is not a finite number.

A 68% half-width below ``--min-steps`` grid steps (default 2) is reported
as a warning and does not reject.

Writes ``<run>/gate.yaml`` and an empty ``ACCEPTED`` or ``REJECTED`` file;
the exit code is 0 or 1.

    python -m simulations.true_posterior.gate <run dir> [--edge-max 1e-4]
"""

import argparse
import os
import sys

import numpy as np
import yaml


# upper limits set by the model, not by the grid: no star fraction exceeds 1
PHYSICAL_UPPER = {"f_tot": 1.0, "f_gt": 1.0, "fbin": 1.0}


def _edges(post):
    tot = float(np.sum(post))
    return float(post[0] / tot), float(post[-1] / tot)


def gate(run_dir, edge_max=1e-4, min_steps=2.0):
    with open(os.path.join(run_dir, "posterior_summary.yaml")) as fh:
        summ = yaml.safe_load(fh)
    with open(os.path.join(run_dir, "run_posterior.yaml")) as fh:
        meta = yaml.safe_load(fh)
    z = np.load(os.path.join(run_dir, "grid_cubes.npz"), allow_pickle=True)
    key = [k for k in z.files if k.endswith("__gmf_lik_cube")][0]
    split = "likmixsplit" in key
    cube = np.asarray(z[key], float)
    axes, reject, warn = {}, [], []

    def finite_check(label, arr):
        arr = np.asarray(arr, float)
        if np.isnan(arr).any() or not np.isfinite(arr).any():
            reject.append("%s: log-posterior has a NaN or no finite entry"
                          % label)

    finite_check("cube", cube)
    with np.errstate(invalid="ignore"):
        prob = np.exp(cube - np.nanmax(cube))

    def add(name, grid, post, gate_lo=True, gate_hi=True, summ_key=None):
        lo, hi = _edges(post)
        g = np.asarray(grid, float)
        step = float(np.median(np.diff(g))) if g.size > 1 else float("nan")
        s = summ.get(summ_key or name, {})
        hw = (0.5 * (s["p84"] - s["p16"]) if "p16" in s and "p84" in s
              else float("nan"))
        rec = {"grid": [float(g[0]), float(g[-1]), int(g.size)],
               "step": step, "edge_mass_lo": lo, "edge_mass_hi": hi,
               "p50": float(s["p50"]) if "p50" in s else float("nan"),
               "hw68": float(hw), "hw68_over_step": float(hw / step)}
        phys = PHYSICAL_UPPER.get(name)
        at_limit = phys is not None and g.size > 0 and g[-1] >= phys - 1e-9
        rec["upper_edge_is_physical_limit"] = bool(at_limit)
        axes[name] = rec
        # written so that a non-finite edge mass rejects
        if gate_lo and not (lo <= edge_max):
            reject.append("%s: lower edge node holds %.2e" % (name, lo))
        if gate_hi and at_limit:
            if not np.isfinite(hi):
                reject.append("%s: upper edge mass is not finite" % name)
            elif hi > edge_max:
                warn.append("%s: %.2e of the mass lies at the physical "
                            "limit %g" % (name, hi, phys))
        elif gate_hi and not (hi <= edge_max):
            reject.append("%s: upper edge node holds %.2e" % (name, hi))
        if np.isfinite(hw) and hw / step < min_steps:
            warn.append("%s: 68%% half-width is %.1f grid steps"
                        % (name, hw / step))

    f_name = "f_tot" if split else "fbin"
    for ax, (name, gk) in enumerate((("pi", "pi_grid"),
                                     ("kappa", "kappa_grid"),
                                     ("eta", "eta_grid"),
                                     (f_name, "fbin_grid"))):
        post = prob.sum(axis=tuple(a for a in range(4) if a != ax))
        add(name, z[gk], post)
    if split:
        t = np.asarray(z["lnpost_fg_fl"], float)
        finite_check("lnpost_fg_fl", t)
        with np.errstate(invalid="ignore"):
            p2 = np.exp(t - np.max(t))
        add("f_gt", z["fg_grid"], p2.sum(axis=1))
        add("f_lt", z["fl_grid"], p2.sum(axis=0), gate_lo=False)
        r = np.asarray(z["lnpost_rho"], float)
        finite_check("lnpost_rho", r)
        with np.errstate(invalid="ignore"):
            pr = np.exp(r - np.max(r))
        under, over = _edges(pr)
        axes["log10_rho"] = {"underflow_mass": under, "overflow_mass": over,
                             "p50": float(summ["log10_rho"]["p50"])}
        if not (over <= edge_max):
            reject.append("log10_rho: overflow bin holds %.2e" % over)

    # reference span of the injection set (bounded weights inside it only)
    span = None
    man_path = os.path.join(str(meta.get("injset") or ""), "manifest.yaml")
    if os.path.exists(man_path):
        with open(man_path) as fh:
            ref = yaml.safe_load(fh)["reference"]
        span = {}
        for name, rk in (("pi", "logP"), ("kappa", "q"), ("eta", "e")):
            lo, hi = min(ref[rk]["alphas"]), max(ref[rk]["alphas"])
            span[name] = [float(lo), float(hi)]
            g = axes[name]["grid"]
            if g[0] < lo - 1e-9 or g[1] > hi + 1e-9:
                reject.append("%s grid [%g, %g] leaves the reference span "
                              "[%g, %g]" % (name, g[0], g[1], lo, hi))
    out = {"run_dir": os.path.abspath(run_dir), "accepted": not reject,
           "edge_max": float(edge_max), "min_steps": float(min_steps),
           "rejections": reject, "warnings": warn,
           "reference_span": span,
           "settings": {k: meta.get(k) for k in
                        ("model", "q_min", "k1_floor", "split_f", "pi_grid",
                         "kappa_grid", "eta_grid", "obs_csv")},
           "obs_summary": meta.get("obs_summary"), "axes": axes}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run_dir")
    ap.add_argument("--edge-max", type=float, default=1e-4)
    ap.add_argument("--min-steps", type=float, default=2.0)
    a = ap.parse_args(argv)
    res = gate(a.run_dir, a.edge_max, a.min_steps)
    with open(os.path.join(a.run_dir, "gate.yaml"), "w") as fh:
        yaml.safe_dump(res, fh, sort_keys=False)
    for name in ("ACCEPTED", "REJECTED"):
        p = os.path.join(a.run_dir, name)
        if os.path.exists(p):
            os.remove(p)
    open(os.path.join(a.run_dir, "ACCEPTED" if res["accepted"]
                      else "REJECTED"), "w").close()
    for name, r in res["axes"].items():
        if "grid" in r:
            print("  %-9s p50 %+.4f  hw68 %.4f (%.1f steps)  edge mass "
                  "lo %.1e hi %.1e" % (name, r["p50"], r["hw68"],
                                       r["hw68_over_step"],
                                       r["edge_mass_lo"], r["edge_mass_hi"]))
        else:
            print("  %-9s p50 %+.4f  underflow %.1e overflow %.1e"
                  % (name, r["p50"], r["underflow_mass"],
                     r["overflow_mass"]))
    for w in res["warnings"]:
        print("  warning: %s" % w)
    for r in res["rejections"]:
        print("  REJECT: %s" % r)
    print("[gate] %s: %s" % (a.run_dir, "ACCEPTED" if res["accepted"]
                             else "REJECTED"))
    return 0 if res["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
