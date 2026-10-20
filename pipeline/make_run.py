"""
pipeline.make_run — set up a run folder for the SB1 chain on a list of stars,
with every config path pointing at YOUR data and output directories.

Copies the repo's template configs for the chosen setup (line set + fit
settings), rewrites only their path keys, checks that every star has the
inputs the chain needs, and writes

    <out-dir>/CCF/<run>/
        stars.txt               the star list (BLOeM_X-YYY, one per line)
        configs/ccf_pass1.yaml  CCF pass 1 (model templates, v sin i broadening)
        configs/ccf_pass2.yaml  CCF pass 2 (template = pass-1 co-added spectrum)
        configs/params.yaml     fit + MCMC (base_dir = pass2/)
        run.env                 path of the RV corrections table
        input_check.csv         per-star input check

Then run the chain with ``./run_sample.sh <out-dir>/CCF/<run>``.

Expected data layout (override any piece with its own option):

    <data-dir>/BLOeM_DR5.0_Combined/           DR5 *_Combined.fits (searched recursively)
    <data-dir>/Crowther_templates/per_star_templates/   BLOeM_X-YYY_*.csv model templates
    <data-dir>/master_table_vsini_v1.csv       ID, vsini, evsini (ID without BLOeM_)
    <data-dir>/mass_bloem.csv                  ID, Mspec, R_star (+ errors)
    <data-dir>/RVcorrectionsDR5.csv            config, MJD, rv_correction, Error

Usage:
    python -m pipeline.make_run --run my_bdwarfs --stars ~/my_stars.txt \
        --data-dir ~/BLOeM/Data --out-dir ~/BLOeM/scriptsOut
"""

import argparse
import glob
import os
import re

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# setup -> (ccf pass 1 template, ccf pass 2 template, params template), repo-relative
SETUPS = {
    "bdwarfs": ("spectroscopy/ccf_input_first_bdwarfs.yaml",
                "spectroscopy/ccf_input_second_bdwarfs.yaml",
                "configs/params_bdwarfs.yaml"),
    "bdwarfs_neb": ("spectroscopy/ccf_input_first_bdwarfs_neb.yaml",
                    "spectroscopy/ccf_input_second_bdwarfs_neb.yaml",
                    "configs/params_bdwarfs.yaml"),
    "ostars": ("spectroscopy/ccf_input_first.yaml",
               "spectroscopy/ccf_input_second.yaml",
               "configs/params.yaml"),
}
STAR_RE = re.compile(r"BLOeM_\d-\d{3}")


def _set_key(text, key, value, indent=""):
    """Replace the value of one YAML key (top-level or given indent), keep comments."""
    pat = re.compile(rf"^{indent}{re.escape(key)}:.*$", re.M)
    if not pat.search(text):
        raise SystemExit(f"key '{indent}{key}' not found in template")
    return pat.sub(lambda _: f"{indent}{key}: {value}", text, count=1)


def _q(path):
    return '"' + path + '"'


def read_stars(path):
    stars = []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok or tok[0].startswith("#"):
                continue
            s = tok[0] if tok[0].startswith("BLOeM_") else f"BLOeM_{tok[0]}"
            if not STAR_RE.fullmatch(s):
                raise SystemExit(f"{path}: cannot read star id '{tok[0]}' "
                                 "(expected BLOeM_X-YYY or X-YYY)")
            if s not in stars:
                stars.append(s)
    if not stars:
        raise SystemExit(f"{path}: no stars")
    return stars


def check_inputs(stars, p):
    """One row per star: epochs, template, v sin i, mass, RV-correction field."""
    for k in ("fits_dir", "templates"):
        if not os.path.isdir(p[k]):
            raise SystemExit(f"not a directory: {p[k]} (--{k.replace('_', '-')})")
    for k in ("vsini", "mass", "corrections"):
        if not os.path.isfile(p[k]):
            raise SystemExit(f"file not found: {p[k]} (--{k})")
    fits = glob.glob(os.path.join(p["fits_dir"], "**", "*Combined.fits"), recursive=True)
    n_ep = pd.Series([m.group(0) for f in fits
                      if (m := STAR_RE.search(os.path.basename(f)))]).value_counts()
    tmpl = {m.group(0) for f in os.listdir(p["templates"])
            if (m := STAR_RE.search(f)) and not f.endswith("_CCF_RVs.csv")}
    vs = pd.read_csv(p["vsini"])
    vs_ids = set("BLOeM_" + vs["ID"].astype(str))
    mass_ids = set(pd.read_csv(p["mass"], encoding="utf-8-sig")["ID"])
    corr = pd.read_csv(p["corrections"])
    corr_fields = set(corr["config"].str.extract(r"^F(\d+)_")[0].astype(int))
    t = pd.DataFrame({"star": stars})
    t["n_epochs"] = t.star.map(n_ep).fillna(0).astype(int)
    t["template"] = t.star.isin(tmpl)
    t["vsini"] = t.star.isin(vs_ids)
    t["mass"] = t.star.isin(mass_ids)
    t["rv_corr_field"] = t.star.str.extract(r"BLOeM_(\d)-")[0].astype(int).isin(corr_fields)
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Make a run folder (configs with your paths) for run_sample.sh.")
    ap.add_argument("--run", required=True, help="run name (folder under <out-dir>/CCF/)")
    ap.add_argument("--stars", required=True, help="text file, one BLOeM_X-YYY per line")
    ap.add_argument("--data-dir", required=True, help="BLOeM input data root")
    ap.add_argument("--out-dir", required=True, help="output root (like scriptsOut/)")
    ap.add_argument("--setup", choices=sorted(SETUPS), default="bdwarfs",
                    help="line set + fit settings to copy (default: bdwarfs)")
    ap.add_argument("--fits-dir")
    ap.add_argument("--templates")
    ap.add_argument("--vsini")
    ap.add_argument("--mass")
    ap.add_argument("--corrections")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the configs of an existing run folder")
    args = ap.parse_args(argv)

    data = os.path.abspath(os.path.expanduser(args.data_dir))
    p = {
        "fits_dir": args.fits_dir or os.path.join(data, "BLOeM_DR5.0_Combined"),
        "templates": args.templates or os.path.join(data, "Crowther_templates",
                                                    "per_star_templates"),
        "vsini": args.vsini or os.path.join(data, "master_table_vsini_v1.csv"),
        "mass": args.mass or os.path.join(data, "mass_bloem.csv"),
        "corrections": args.corrections or os.path.join(data, "RVcorrectionsDR5.csv"),
    }
    p = {k: os.path.abspath(os.path.expanduser(v)) for k, v in p.items()}
    run_dir = os.path.join(os.path.abspath(os.path.expanduser(args.out_dir)), "CCF", args.run)
    cfg_dir = os.path.join(run_dir, "configs")
    if os.path.exists(cfg_dir) and not args.force:
        raise SystemExit(f"{cfg_dir} exists; use --force to rewrite its configs")

    stars = read_stars(os.path.expanduser(args.stars))
    t = check_inputs(stars, p)
    bad = t[(t.n_epochs == 0) | ~t.template | ~t.vsini | ~t.mass | ~t.rv_corr_field]
    pd.set_option("display.width", 200)
    print(f"{len(t)} stars | epochs min/median {t.n_epochs.min()}/{int(t.n_epochs.median())} "
          f"| missing: FITS {int((t.n_epochs == 0).sum())}, template {int((~t.template).sum())}, "
          f"vsini {int((~t.vsini).sum())}, mass {int((~t.mass).sum())}, "
          f"RV-corr field {int((~t.rv_corr_field).sum())}")
    if len(bad):
        print(bad.to_string(index=False))
        raise SystemExit("\nremove these stars from the list (or add the missing inputs) "
                         "and run again; nothing was written")
    few = t[t.n_epochs < 20]
    if len(few):
        print(f"note: {len(few)} star(s) with < 20 epochs: {few.star.tolist()}")

    first, second, params = (os.path.join(REPO, f) for f in SETUPS[args.setup])
    os.makedirs(cfg_dir, exist_ok=True)
    stars_txt = os.path.join(run_dir, "stars.txt")
    with open(stars_txt, "w") as f:
        f.write("\n".join(stars) + "\n")
    pass1 = os.path.join(run_dir, "pass1") + "/"
    pass2 = os.path.join(run_dir, "pass2") + "/"
    header = (f"# Generated by pipeline/make_run.py (setup '{args.setup}') from\n"
              f"#   {{src}}\n# Only path keys were changed.\n")

    for src, dst, out, tmpl, meta in ((first, "ccf_pass1.yaml", pass1, p["templates"], p["vsini"]),
                                      (second, "ccf_pass2.yaml", pass2, pass1, "")):
        s = open(src).read()
        for key, val in (("path_to_observations", _q(p["fits_dir"] + "/")),
                         ("path_to_list_of_objects", _q(stars_txt)),
                         ("list_of_objects", "[]"),
                         ("path_to_output", _q(out)),
                         ("template_path", _q(tmpl)),
                         ("path_to_meta_data_csv", _q(meta))):
            s = _set_key(s, key, val)
        with open(os.path.join(cfg_dir, dst), "w") as f:
            f.write(header.format(src=os.path.relpath(src, REPO)) + s)

    s = open(params).read()
    s = _set_key(s, "base_dir", _q(pass2.rstrip("/")))
    s = _set_key(s, "mass_file", _q(p["mass"]), indent="  ")
    s = _set_key(s, "object_list", "[]", indent="  ")
    with open(os.path.join(cfg_dir, "params.yaml"), "w") as f:
        f.write(header.format(src=os.path.relpath(params, REPO)) + s)

    with open(os.path.join(run_dir, "run.env"), "w") as f:
        f.write(f'RV_CORRECTIONS="{p["corrections"]}"\n')
    t.to_csv(os.path.join(run_dir, "input_check.csv"), index=False)

    # sanity: the generated configs load with the pipeline's own readers
    from pipeline.config import load_args
    from spectroscopy.ccf_main import load_input_from_yaml
    c2 = load_input_from_yaml(os.path.join(cfg_dir, "ccf_pass2.yaml"))
    cp = load_args(os.path.join(cfg_dir, "params.yaml"))
    assert os.path.realpath(cp["pipeline_io"]["rv_input_dir"]) == os.path.realpath(c2.path_to_output)

    print(f"\nrun folder: {run_dir}\nnext:  caffeinate -i ./run_sample.sh {run_dir} --to ccf")


if __name__ == "__main__":
    main()
