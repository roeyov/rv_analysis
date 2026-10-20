"""
spectroscopy.select_bdwarf_sample — pick a CCF batch from the BLOeM B-dwarf
SB1 list and record which stars are nebular.

Input list: CSV without header, columns ``star, cls`` (e.g.
``BLOeM_1-002,SB1``). Every star is annotated from the Comprehensive
catalogue, the v sin i table, the mass table, the Crowther per-star templates
and the DR5 FITS epochs, then:

* nebular = ``Nebular(DSSred)`` in {neb, inside, edge} OR a ``neb`` tag in
  ``Spectral type (new)`` (``Nebular(DSSred) == 'neb'`` reproduces the O-star
  ``ostars_sb1_neb.txt``);
* the clean pool = not nebular, >= ``--min-epochs`` epochs, has a template and
  a mass row, not a Gaia H-alpha emitter;
* the batch = the ``--n`` slowest rotators of the pool, sorted by
  (vsini, star) so ties are broken by name and the selection is reproducible.

Outputs (in ``--out-dir``):
    <prefix>_slow<n>.txt     one BLOeM_X-YYY per line (CCF path_to_list_of_objects)
    <prefix>_neb.txt         the nebular stars
    <prefix>_selection.csv   every star with its flags and selected / excluded reason

Usage:
    python -m spectroscopy.select_bdwarf_sample \
        --input ~/Downloads/Bdwarf_classification_sb1.csv --n 30
"""

import argparse
import glob
import os
import re

import pandas as pd

DATA = "/Users/roeyovadia/Documents/Data/BLOeM_Data"
SCRIPTS_OUT = os.environ.get(
    "SCRIPTS_OUT", "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut")

NEB_DSS_VALUES = ("neb", "inside", "edge")
STAR_RE = re.compile(r"(BLOeM_\d-\d{3})")


def _star_ids(names):
    """BLOeM_X-YYY ids found in a list of file names."""
    out = []
    for n in names:
        m = STAR_RE.search(os.path.basename(n))
        if m:
            out.append(m.group(1))
    return out


def annotate(stars, catalogue, vsini, mass, templates_dir, fits_dir):
    """One row per star with every quantity the selection uses."""
    cat = pd.read_excel(catalogue, sheet_name="comprehensive_catalog")
    missing = sorted(set(stars.star) - set(cat.BLOeMID))
    if missing:
        raise SystemExit(f"not in the catalogue: {missing}")
    df = stars.merge(
        cat[["BLOeMID", "Spectral type (new)", "Nebular(DSSred)",
             "Ha-emitters (Gaia)", "Comments"]],
        left_on="star", right_on="BLOeMID", how="left").drop(columns="BLOeMID")

    v = pd.read_csv(vsini)
    v["star"] = "BLOeM_" + v["ID"].astype(str)
    df = df.merge(v[["star", "vsini", "evsini"]], on="star", how="left")

    mass_ids = set(pd.read_csv(mass, encoding="utf-8-sig")["ID"])
    tmpl_ids = set(_star_ids(os.listdir(templates_dir)))
    fits = glob.glob(os.path.join(fits_dir, "**", "*Combined.fits"),
                     recursive=True)
    n_ep = pd.Series(_star_ids(fits)).value_counts()

    df["n_epochs"] = df.star.map(n_ep).fillna(0).astype(int)
    df["has_template"] = df.star.isin(tmpl_ids)
    df["has_mass"] = df.star.isin(mass_ids)
    df["neb_dss"] = df["Nebular(DSSred)"].isin(NEB_DSS_VALUES)
    df["neb_spt"] = df["Spectral type (new)"].str.contains("neb", na=False)
    df["nebular"] = df.neb_dss | df.neb_spt
    df["ha_emitter"] = df["Ha-emitters (Gaia)"].eq("yes")
    return df


def select(df, n, min_epochs):
    """Mark the clean pool and the n slowest rotators in it."""
    reasons = []
    for _, r in df.iterrows():
        why = []
        if r.nebular:
            why.append("nebular")
        if r.n_epochs < min_epochs:
            why.append(f"epochs<{min_epochs}")
        if not r.has_template:
            why.append("no_template")
        if not r.has_mass:
            why.append("no_mass")
        if r.ha_emitter:
            why.append("ha_emitter")
        if pd.isna(r.vsini):
            why.append("no_vsini")
        reasons.append(";".join(why))
    df = df.assign(excluded=reasons)
    pool = df[df.excluded == ""].sort_values(["vsini", "star"])
    if len(pool) < n:
        raise SystemExit(f"only {len(pool)} stars in the clean pool, asked for {n}")
    chosen = set(pool.star.iloc[:n])
    df["selected"] = df.star.isin(chosen)
    df.loc[(df.excluded == "") & ~df.selected, "excluded"] = "not_in_slowest_n"
    return df, pool.iloc[:n]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Select the slowest-rotating clean B dwarfs for a CCF batch "
                    "and list the nebular ones.")
    ap.add_argument("--input", required=True,
                    help="CSV without header: star, cls")
    ap.add_argument("--catalogue",
                    default=os.path.join(DATA, "Comprehensive catalogue.xlsx"))
    ap.add_argument("--vsini",
                    default="/Users/roeyovadia/Documents/master_table_vsini_v1.csv")
    ap.add_argument("--mass", default=os.path.join(DATA, "mass_bloem.csv"))
    ap.add_argument("--templates",
                    default=os.path.join(DATA, "Crowther_templates",
                                         "per_star_templates"))
    ap.add_argument("--fits-dir", default=os.path.join(DATA, "BLOeM_DR5.0_Combined"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--min-epochs", type=int, default=25)
    ap.add_argument("--out-dir",
                    default=os.path.join(SCRIPTS_OUT, "CCF", "bdwarfs_dr5"))
    ap.add_argument("--prefix", default="bdwarfs_sb1")
    args = ap.parse_args(argv)

    stars = pd.read_csv(args.input, header=None, names=["star", "cls"])
    if stars.star.duplicated().any():
        raise SystemExit(f"duplicate stars: {stars.star[stars.star.duplicated()].tolist()}")
    df = annotate(stars, args.catalogue, args.vsini, args.mass,
                  args.templates, args.fits_dir)
    df, chosen = select(df, args.n, args.min_epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    batch_path = os.path.join(args.out_dir, f"{args.prefix}_slow{args.n}.txt")
    neb_path = os.path.join(args.out_dir, f"{args.prefix}_neb.txt")
    table_path = os.path.join(args.out_dir, f"{args.prefix}_selection.csv")
    with open(batch_path, "w") as f:
        f.write("\n".join(chosen.star) + "\n")
    with open(neb_path, "w") as f:
        f.write("\n".join(sorted(df.star[df.nebular])) + "\n")
    df.to_csv(table_path, index=False)

    pool_n = int((df.selected | (df.excluded == "not_in_slowest_n")).sum())
    print(f"{len(df)} stars | nebular {int(df.nebular.sum())} | clean pool {pool_n} "
          f"| selected {len(chosen)} (vsini {chosen.vsini.min():g}-"
          f"{chosen.vsini.max():g} km/s)")
    print(f"  {batch_path}\n  {neb_path}\n  {table_path}")


if __name__ == "__main__":
    main()
