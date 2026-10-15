"""
simulations.true_posterior.real_variants — the true posterior on the real
BLOeM data: the headline run plus likelihood analogues of the 2026-09 CvM
variant campaign.

Every variant reuses one loaded injection set. The cut-independent
tensors (den, num_all) are computed once per grid; variants that change
the cut, the q domain or the grid recompute what they need.

  main          4-080 B, logP* = numerical elbow (sigma 0.15), no errors
  errors        + per-system K1 / e measurement errors
  v4080AaAb     4-080 represented by its Aa,Ab orbit
  weakK1_k10    drop the six K1 < 10 km/s orbits      (CvM weakK1_k10)
  weakK1_c1     drop the four K1/sigma < 5 orbits     (CvM weakK1_c1)
  no_inner      drop the higher-order inner orbits    (CvM no_inner)
  pstar_2d/5d   logP* = log10(2) / log10(5)           (CvM pstar_*)
  nocut         no period cut (logP* = log_p_min)     (mix only)
  sig010/020    elbow with smoothing sigma 0.10/0.20  (CvM sig0*)
  env5          c2 = 0: long-period LS systems are censored power-law
                members only                           (CvM env5)
  binom_none    shape-only likelihood (no count terms) (CvM binom_none)
  binom_n72     only stars with orbital solutions count as detected
                                                       (CvM binom_n72)
  qmin01        q domain [0.1, 1], kappa grid to -2.44 (CvM qmin01)
  rv10          rv_threshold (drv 10) detector (needs --rv-injset)

Results go to <out-dir>/<variant>/ (grid_cubes.npz etc.) and to
<out-dir>/variant_summary.csv with p16/p50/p84 per axis.

    python -u -m simulations.true_posterior.real_variants \
        --injset <pb injset> --config <bias yaml with table paths> \
        --tables <paper tables dir> --out-dir <dir> [--rv-injset <dir>] \
        [--only main,errors] [--n-workers 16] \
        [--below-shape k1e --below-pcirc-as-circular] [--c-nodes 0,...,1] \
        [--pi-grid=lo,hi,n --kappa-grid=... --eta-grid=...]

The likelihood settings --below-shape, --below-pcirc-as-circular and
--c-nodes apply to every variant, so each differs from the headline fit
only by its own switch; so do the grid overrides (default: make_grids). --print-options prints each variant's run()
options without loading anything.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from simulations.bias_grid_lib.cutoffs import _compute_logP_cutoff
from simulations.bias_grid_lib.obs_io import load_observed_from_tex
from simulations.true_posterior.injection_set import load_injection_set
from simulations.true_posterior.likelihood import InjectionSet
from simulations.true_posterior.obs_catalog import (
    _bias_paths, build_star_table, summarize)
from simulations.true_posterior.run_posterior import (make_grids,
                                                      parse_c_nodes, run)

WEAK_K10 = ["1-068", "2-087", "3-078", "2-035", "8-028", "6-098"]
WEAK_C1 = ["1-068", "3-078", "4-074", "6-091"]
NO_INNER = ["3-042", "3-075", "4-043 (inner)", "4-080 Aa,Ab", "4-080 B",
            "4-110"]


def variant_specs(elbow):
    """name -> dict(cut, choose, exclude, solutions_only, run_kwargs)."""
    base = {"cut": elbow[0.15], "choose": {"4-080": "4-080 B"},
            "exclude": [], "solutions_only": False, "run": {}}

    def v(**kw):
        d = {k: (dict(val) if isinstance(val, dict) else
                 list(val) if isinstance(val, list) else val)
             for k, val in base.items()}
        run_kw = kw.pop("run", {})
        d.update(kw)
        d["run"] = dict(run_kw)
        return d
    return {
        "main": v(),
        "errors": v(run={"use_errors": True}),
        "v4080AaAb": v(choose={"4-080": "4-080 Aa,Ab"}),
        "weakK1_k10": v(exclude=WEAK_K10),
        "weakK1_c1": v(exclude=WEAK_C1),
        "no_inner": v(exclude=NO_INNER, choose={}),
        # 1-072 (P = 2.206 d, e = 0.033) lies below p_circ = 2.26 d, where
        # the model forces e = 0; treat such orbits as circular here.
        "pstar_2d": v(cut=float(np.log10(2.0)),
                      run={"below_pcirc_ecc": "as_circular"}),
        "pstar_5d": v(cut=float(np.log10(5.0))),
        # no period cut at all: every solved orbit enters the shape term
        # (the model's period floor, log_p_min = 0.046, is the lowest
        # admissible cut; 7-081 at 1.112 d sits just above it). Meaningful
        # for the one-component model only: with no support below the cut
        # the split model has no short-period component.
        "nocut": v(cut=0.046, run={"below_pcirc_ecc": "as_circular"}),
        "sig010": v(cut=elbow[0.10]),
        "sig020": v(cut=elbow[0.20]),
        "env5": v(run={"c2_zero": True}),
        "binom_none": v(run={"count": False}),
        "binom_n72": v(solutions_only=True),
        "qmin01": v(run={"q_min": 0.1, "kappa_grid": "-2.44,1.80,43"}),
        "rv10": v(run={"_rv": True}),
    }


def run_options(spec_run, model, below_shape=None, as_circular=False,
                c_nodes=None, grids=None):
    """run() keywords of one variant: its own switches plus the global
    likelihood settings given on the command line (the same for every
    variant, so that each differs from the headline fit only by its own
    switch)."""
    kw = dict(spec_run)
    if below_shape is not None:
        if model != "split":
            raise ValueError("--below-shape is a split-model option")
        kw["below_shape"] = below_shape
    if as_circular:
        kw["below_pcirc_ecc"] = "as_circular"
    if c_nodes is not None:
        kw["c_grid"] = np.asarray(c_nodes, float)
    for k, v in (grids or {}).items():
        if v is not None:
            kw.setdefault(k, v)
    return kw


def build_table(paths, tables, spec):
    df = build_star_table(
        paths["ostar_catalog"], paths["sb1_tex"], paths["sb2_tex"],
        spec["cut"], star_props=None, choose=spec["choose"],
        exclude_ids=spec["exclude"] or None,
        errors_csvs=[os.path.join(tables, "sb1_solutions.csv"),
                     os.path.join(tables, "sb2_solutions.csv")])
    if spec["solutions_only"]:
        m = (df["outcome"] == "DO") & df["reason"].isin(
            ["no_solution", "P_lower_limit", "no_row"])
        df.loc[m, "outcome"] = "ND"
        df.loc[m, "reason"] = "solutions_only"
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--injset", required=True, help="pipeline_batched set")
    ap.add_argument("--rv-injset", default=None, help="rv_threshold set")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tables", required=True,
                    help="dir with sb1/sb2_solutions.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--model", choices=["mix", "split"], default="mix")
    ap.add_argument("--below-shape", choices=["k1", "k1e"], default=None,
                    help="as run_posterior --below-shape, for every variant "
                         "(split model)")
    ap.add_argument("--below-pcirc-as-circular", action="store_true",
                    help="as run_posterior --below-pcirc-as-circular, for "
                         "every variant")
    ap.add_argument("--c-nodes", default=None, metavar="c0,c1,...",
                    help="as run_posterior --c-nodes, for every variant")
    ap.add_argument("--pi-grid", default=None,
                    help="lo,hi,n for every variant (default: make_grids)")
    ap.add_argument("--kappa-grid", default=None, help="lo,hi,n")
    ap.add_argument("--eta-grid", default=None, help="lo,hi,n")
    ap.add_argument("--print-options", action="store_true",
                    help="print each variant's run() options and exit")
    a = ap.parse_args(argv)
    if a.below_shape is not None and a.model != "split":
        ap.error("--below-shape needs --model split")
    c_nodes = parse_c_nodes(a.c_nodes)
    gspec = {"pi_grid": a.pi_grid, "kappa_grid": a.kappa_grid,
             "eta_grid": a.eta_grid}
    if a.print_options:
        fake = {0.10: 0.5216, 0.15: 0.5266, 0.20: 0.5505}
        specs = variant_specs(fake)
        names = list(specs) if not a.only else a.only.split(",")
        for name in names:
            kw = run_options(specs[name]["run"], a.model, a.below_shape,
                             a.below_pcirc_as_circular, c_nodes, gspec)
            print("%-11s %s" % (name, {k: (list(map(float, v))
                                           if isinstance(v, np.ndarray)
                                           else v) for k, v in kw.items()}))
        return 0

    paths = _bias_paths(a.config)
    obs = load_observed_from_tex(paths["sb1_tex"], paths["sb2_tex"])
    elbow = {s: _compute_logP_cutoff(obs["logP"], "numerical",
                                     smooth_sigma=s)
             for s in (0.10, 0.15, 0.20)}
    specs = variant_specs(elbow)
    names = list(specs) if not a.only else a.only.split(",")
    if "rv10" in names and not a.rv_injset:
        names.remove("rv10")
    if a.model == "split":
        # c2_zero / shape-only / no-cut are mix-model switches
        names = [n for n in names if n not in ("env5", "binom_none", "nocut")]
    print("[real] elbows %s; variants %s" % (elbow, names), flush=True)

    t0 = time.time()
    data, stars, man = load_injection_set(a.injset)
    inj = InjectionSet(data, stars, man)
    del data
    grids = make_grids(None, a.pi_grid, a.kappa_grid, a.eta_grid)
    base = inj.alpha_tensors(grids[0], grids[1], grids[2], cut=None,
                             groups=("den", "num_all"))
    print("[real] injection set + shared tensors: %.0f s"
          % (time.time() - t0), flush=True)
    inj_rv = None

    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for name in names:
        spec = specs[name]
        df = build_table(paths, a.tables, spec)
        vdir = os.path.join(a.out_dir, name)
        os.makedirs(vdir, exist_ok=True)
        obs_csv = os.path.join(vdir, "obs_catalog.csv")
        df.to_csv(obs_csv, index=False)
        kw = run_options(spec["run"], a.model, a.below_shape,
                         a.below_pcirc_as_circular, c_nodes, gspec)
        use_inj, use_base = inj, base
        if kw.pop("_rv", False):
            if inj_rv is None:
                d2, s2, m2 = load_injection_set(a.rv_injset)
                inj_rv = InjectionSet(d2, s2, m2)
            use_inj, use_base = inj_rv, None
        # the shared tensors hold the command-line grids; a variant that sets
        # its own q domain or kappa grid recomputes them
        if "q_min" in spec["run"] or "kappa_grid" in spec["run"]:
            use_base = None
        t1 = time.time()
        summ = run(None, obs_csv, vdir, inj_cache=use_inj,
                   base_tensors=use_base, n_workers=a.n_workers,
                   log=lambda *x: None, model=a.model, **kw)
        if a.model == "split":
            summ = dict(summ)
            summ["fbin"] = summ["f_tot"]
        rec = {"variant": name, "cut": spec["cut"],
               **{"n_%s" % k: v for k, v in summarize(df).items()
                  if k in ("DX", "D", "ND")}}
        extra = (("f_gt", "f_lt", "log10_rho") if a.model == "split"
                 else ())
        for ax in ("pi", "kappa", "eta", "fbin", "c1", "c2") + extra:
            for q in ("p16", "p50", "p84"):
                rec["%s_%s" % (ax, q)] = summ[ax][q]
        rows.append(rec)
        print("[real] %-11s %4.0f s  pi %+.3f  kappa %+.3f  eta %+.3f  "
              "fbin %.3f  (c1 %.2f c2 %.2f)%s"
              % (name, time.time() - t1, summ["pi"]["p50"],
                 summ["kappa"]["p50"], summ["eta"]["p50"],
                 summ["fbin"]["p50"], summ["c1"]["p50"], summ["c2"]["p50"],
                 ("  f_gt %.3f f_lt %.3f log10rho %+.2f"
                  % (summ["f_gt"]["p50"], summ["f_lt"]["p50"],
                     summ["log10_rho"]["p50"]))
                 if a.model == "split" else ""),
              flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out_dir,
                                               "variant_summary.csv"),
                                  index=False)
    print("[real] done in %.0f s" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
