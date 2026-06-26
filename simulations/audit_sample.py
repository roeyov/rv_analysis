"""Audit the bias_grid injection sample against the O-star catalog.

Reads the same YAML config as ``simulations.bias_grid`` (the ``bias_grid``
section), runs ``load_star_properties`` to enumerate the resolvable
sample, then cross-checks against ``ostar_catalog.csv``.

Usage
-----
    python -m simulations.audit_sample --config configs/params_bias.yaml
"""

import argparse
import os
import sys

import pandas as pd

from pipeline.config import load_args
from simulations.bias_config import DEFAULT_BIAS_CFG
from simulations.bias_grid import load_star_properties


def _bg(bg_cfg, key):
    if key in bg_cfg and bg_cfg[key] is not None:
        return bg_cfg[key]
    return DEFAULT_BIAS_CFG.get(key)


def _print_section(title):
    print()
    print(title)
    print("=" * len(title))


def main():
    parser = argparse.ArgumentParser(
        description="Audit which O-stars resolve for the bias_grid "
                    "injection sample and compare against ostar_catalog.csv.",
    )
    parser.add_argument(
        "--config", default="configs/params_bias.yaml",
        help="Path to pipeline config YAML. Default: configs/params_bias.yaml",
    )
    cli = parser.parse_args()

    args_dict = load_args(cli.config)
    bg_cfg = args_dict.get("bias_grid", {}) or {}

    mass_file = _bg(bg_cfg, "mass_file")
    rv_dir = _bg(bg_cfg, "rv_dir")
    sb2_analysis_dir = _bg(bg_cfg, "sb2_analysis_dir")
    ostar_catalog = _bg(bg_cfg, "ostar_catalog")

    _print_section("Sample audit")
    print("config:            %s" % cli.config)
    print("mass_file:         %s" % mass_file)
    print("rv_dir:            %s" % rv_dir)
    print("sb2_analysis_dir:  %s" % sb2_analysis_dir)
    print("ostar_catalog:     %s" % ostar_catalog)

    # 1. Catalogs
    massdf = pd.read_csv(mass_file)
    print()
    print("Mass catalog (%s): N=%d" % (os.path.basename(mass_file), len(massdf)))

    if not os.path.isfile(ostar_catalog):
        print("ERROR: ostar_catalog not found at %s" % ostar_catalog)
        sys.exit(1)
    cat = pd.read_csv(ostar_catalog)
    cat["_key"] = cat["BLOeM ID"].astype(str).str.replace("BLOeM_", "", regex=False).str.strip()
    cat["_status"] = cat["Binary status"].astype(str).str.strip()
    print("O-star catalog:              N=%d" % len(cat))
    status_counts = cat["_status"].value_counts(dropna=False).to_dict()
    print("   Binary status breakdown:")
    for status in sorted(status_counts.keys()):
        print("     %-22s %d" % (status + ":", status_counts[status]))

    # 2. Resolve injection sample using the production loader.
    star_df = load_star_properties(
        mass_file, rv_dir,
        sb2_analysis_dir=sb2_analysis_dir,
        ostar_catalog=ostar_catalog,
    )
    resolved_ids = set(star_df["ID"].astype(str))
    source_counts = star_df["source"].value_counts().to_dict()

    _print_section("Resolved for injection")
    print("   via CCF_RVs.csv:                 %d" % source_counts.get("ccf", 0))
    print("   via rv_final_for_mcmc.csv:       %d" % source_counts.get("sb2_final", 0))
    print("   via rv_corrected.csv:            %d" % source_counts.get("sb2_corrected", 0))
    print("   via rv_extracted.csv:            %d" % source_counts.get("sb2_extracted", 0))
    print("   total resolved:                  %d" % len(star_df))

    if "mass_source" in star_df.columns:
        mass_counts = star_df["mass_source"].value_counts().to_dict()
        print()
        print("   Mass / radius source:")
        print("     mass_bloem.csv:        %d" % mass_counts.get("mass_bloem", 0))
        print("     nearest spectral type: %d" % mass_counts.get("spectral_type", 0))
        print("     sample median:         %d" % mass_counts.get("median", 0))

    # 3. Resolved breakdown by Binary status (from catalog).
    cat_by_key = dict(zip(cat["_key"], cat["_status"]))
    status_of_resolved = star_df["ID"].astype(str).map(
        lambda s: cat_by_key.get(s, "<not in catalog>"))
    print()
    print("   Resolved breakdown by Binary status:")
    status_total = cat["_status"].value_counts().to_dict()
    for status in sorted(set(list(status_total.keys()) + ["<not in catalog>"])):
        resolved_n = int((status_of_resolved == status).sum())
        total_n = int(status_total.get(status, 0))
        print("     %-22s %d / %d" % (status + ":", resolved_n, total_n))

    # 4. Diff: missing / extra
    catalog_keys = set(cat["_key"])
    missing = sorted(catalog_keys - resolved_ids)
    extra = sorted(resolved_ids - catalog_keys)

    _print_section("Missing (in catalog but unresolved)")
    if missing:
        for sid in missing:
            print("  %-12s  %s" % (sid, cat_by_key.get(sid, "?")))
    else:
        print("  (none)")

    _print_section("Extra (resolved but not in catalog)")
    if extra:
        for sid in extra:
            print("  %s" % sid)
    else:
        print("  (none)")

    print()
    print("n_stars_sample (binomial denominator): %d" % len(star_df))


if __name__ == "__main__":
    main()
