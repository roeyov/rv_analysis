"""Resolve M1/R1/field/rv_err/gamma per star from BLOeM tables + catalogs.

Two public loaders:

- ``load_observed_star_properties`` — builds one row per row of the
  SB1/SB2 LaTeX tables; legacy entry point used by the closure-test
  generator that wants the detected-binary sample.
- ``load_star_properties`` — preferred. Builds one row per O-star
  catalog entry (full 134-star population). SB2/Higher-order systems
  pull primary-component RVs from ``sb2_analysis/``; everyone else uses
  the single-line CCF output.
"""

import os
import re
import numpy as np
import pandas as pd

from simulations.common import BLOEM_MJD_ARRAYS
from simulations.bias_grid_lib.logging_utils import logger


def _bloem_id_root(star_id):
    """Strip multi-component SB labels (e.g. ' Aa,Ab', ' B') so the star_id
    matches the underlying BLOeM star in mass_bloem.csv. Returns the
    cleaned root id, e.g. '4-080 Aa,Ab' -> '4-080'."""
    s = re.sub(r"\$\^\{[^}]*\}\$", "", str(star_id))  # drop $^{(a)}$
    s = re.sub(r"\s+(Aa,Ab|Aa|Ab|A|B|C).*$", "", s)
    return s.strip()


def _load_sb2_rv_from_analysis_dir(sb2_base_dir, star_key):
    """Load primary-component RVs from BLOeM_DR5 sb2_analysis output.

    Looks under ``<sb2_base_dir>/BLOeM_<star_key>/sb2_analysis/`` and
    selects the latest ``YYYYMMDD_HHMMSS`` subdirectory (lexicographic
    sort == chronological for that format). Priority chain:
    ``rv_final_for_mcmc.csv`` → ``rv_corrected.csv`` (post
    gamma-crossing correction) → ``rv_extracted.csv`` (pre-correction,
    last resort).

    Returns
    -------
    dict or None
        ``{"mjds": np.ndarray, "rv_err": float, "gamma": float,
        "source": "sb2_final" | "sb2_corrected" | "sb2_extracted"}``
        on success, ``None`` on any failure.
    """
    star_dir = os.path.join(sb2_base_dir, "BLOeM_%s" % star_key)
    sb2_dir = os.path.join(star_dir, "sb2_analysis")
    if not os.path.isdir(sb2_dir):
        return None
    try:
        timestamps = [d for d in os.listdir(sb2_dir)
                      if os.path.isdir(os.path.join(sb2_dir, d))]
        if not timestamps:
            return None
        latest = max(timestamps)
        data_dir = os.path.join(sb2_dir, latest, "data")
        for fname, source in (("rv_final_for_mcmc.csv", "sb2_final"),
                              ("rv_corrected.csv", "sb2_corrected"),
                              ("rv_extracted.csv", "sb2_extracted")):
            csv_path = os.path.join(data_dir, fname)
            if not os.path.isfile(csv_path):
                continue
            df = pd.read_csv(csv_path, comment="#")
            if not {"rv1", "rv1_err", "mjd"}.issubset(df.columns):
                continue
            return {
                "mjds": df["mjd"].values,
                "rv_err": float(df["rv1_err"].median()),
                "gamma": float(df["rv1"].median()),
                "source": source,
            }
        return None
    except Exception:
        return None


_LUMCLASS_RE = re.compile(
    # Order: longest tokens first so 'III' wins over 'II' and 'IV' over 'I'.
    r"O\s*(\d+(?:\.\d+)?)\s*"
    r"(Iaf\+?|Iab|Iaf|Ia|Ib|III|IV|II|I|V)?"
)


def _parse_spectral_type(sp_str):
    """Parse a BLOeM spectral type into ``(subclass_num, lum_class)``.

    Picks the first ``O<N> <LUMCLASS>`` token, so composite/SB2 strings
    like ``'O7.5 V: + O9.5 neb'`` resolve to the primary ``(7.5, 'V')``.
    Returns ``(None, None)`` if no O-type subclass can be parsed.
    """
    if sp_str is None or (isinstance(sp_str, float) and pd.isna(sp_str)):
        return (None, None)
    m = _LUMCLASS_RE.search(str(sp_str))
    if not m:
        return (None, None)
    subclass = float(m.group(1))
    lum = m.group(2) or None
    # Collapse Ia/Iab/Iaf+ into a coarser 'I' bucket so the small number
    # of supergiants is matchable.
    if lum and lum.startswith("I") and lum not in ("II", "III", "IV"):
        lum = "I"
    return (subclass, lum)


def _nearest_spectral_type_mass(target_sp, sp_table):
    """Find the M, R of the nearest-spectral-type star.

    Parameters
    ----------
    target_sp : tuple (subclass, lum_class)
    sp_table : list of dicts with keys 'key', 'subclass', 'lum', 'M', 'R'.
        Only stars that have both a parseable spectral type and valid
        mass/radius should be in this table.

    Returns
    -------
    (M, R, match_key) or (None, None, None) on no match.
    Matching rule: same luminosity class first, then minimum
    |subclass difference|. If the target has no lum class, match across
    all entries.
    """
    sub, lum = target_sp
    if sub is None:
        return (None, None, None)
    if lum is not None:
        same_lum = [r for r in sp_table if r["lum"] == lum]
    else:
        same_lum = []
    pool = same_lum or sp_table
    if not pool:
        return (None, None, None)
    best = min(pool, key=lambda r: abs(r["subclass"] - sub))
    return (best["M"], best["R"], best["key"])


def _load_sb2_ids_from_catalog(ostar_catalog_path,
                               statuses=("SB2", "Higher-order")):
    """Return the set of BLOeM IDs (cleaned, no 'BLOeM_' prefix) whose
    ``Binary status`` matches any of ``statuses`` in ``ostar_catalog.csv``.

    SB2 and Higher-order systems are analysed with the SB2 pipeline and
    have their primary-component RVs in
    ``sb2_analysis/<latest>/data/rv_*.csv`` — they should be pulled from
    there rather than from the single-line CCF output.

    Returns an empty set on any failure (caller falls back to CCF-only).
    """
    if not ostar_catalog_path or not os.path.isfile(ostar_catalog_path):
        return set()
    try:
        cat = pd.read_csv(ostar_catalog_path)
        status_col = cat["Binary status"].astype(str).str.strip()
        sel = cat.loc[status_col.isin(statuses), "BLOeM ID"]
        return set(s.replace("BLOeM_", "").strip() for s in sel.astype(str))
    except Exception:
        return set()


def load_observed_star_properties(mass_file, rv_dir, sb1_tex, sb2_tex,
                                  sb2_analysis_dir=None,
                                  ostar_catalog=None):
    """Build the bias-grid star sample from the SB1+SB2 LaTeX tables.

    Returns one row per detected binary in the LaTeX tables (~71 entries),
    with M1/R1 from mass_bloem.csv and rv_err/gamma/MJDs from per-star
    CCF_RVs CSVs in rv_dir. Uses sample-median fallbacks when an entry is
    missing — guarantees the full LaTeX sample size, mirroring how
    horvitz_thompson.py builds its star list.

    Returns
    -------
    DataFrame with columns: ID, Mspec, R_star, field, rv_err, gamma.
    """
    from simulations.horvitz_thompson import load_observed_binaries

    obs_binaries = load_observed_binaries(sb1_tex, sb2_tex)

    # Pre-load mass file (full BLOeM catalog) for M1/R1 lookup, with
    # cleaned key column.
    massdf = pd.read_csv(mass_file)
    massdf["_key"] = massdf["ID"].astype(str).str.replace(
        "BLOeM_", "", regex=False)
    M1_med = float(massdf["Mspec"].median())
    R1_med = float(massdf["R_star"].median())

    # Index CSV files in rv_dir by cleaned star id.
    csv_by_id = {}
    if rv_dir and os.path.isdir(rv_dir):
        import glob
        for f in glob.glob(os.path.join(rv_dir, "*_CCF_RVs.csv")):
            base = os.path.basename(f).replace("_CCF_RVs.csv", "")
            csv_by_id[base.replace("BLOeM_", "")] = f

    # SB2 ids from the O-star catalog: these stars' RVs always come from
    # rv_final_for_mcmc.csv (rv1/rv1_err) under sb2_analysis_dir.
    sb2_ids = _load_sb2_ids_from_catalog(ostar_catalog)

    def _field_from_mjds(star_mjds):
        best_field = 0
        best_overlap = 0
        for fi, fld_mjds in enumerate(BLOEM_MJD_ARRAYS):
            overlap = sum(1 for m in star_mjds
                          if any(abs(m - fm) < 0.5 for fm in fld_mjds))
            if overlap > best_overlap:
                best_overlap = overlap
                best_field = fi
        return best_field

    rows = []
    n_csv_hit = 0
    n_sb2_hit = 0
    n_mass_hit = 0
    for _, b in obs_binaries.iterrows():
        sid_raw = str(b["star_id"])
        sid = _bloem_id_root(sid_raw)

        # M1, R1 from mass file (median fallback).
        mrow = massdf[massdf["_key"] == sid]
        if len(mrow) > 0:
            M1 = float(mrow.iloc[0]["Mspec"])
            R1 = float(mrow.iloc[0]["R_star"])
            n_mass_hit += 1
        else:
            M1 = M1_med
            R1 = R1_med

        # Resolve rv_err / gamma / field.
        rv_err = None
        gamma_csv = None
        best_field = 0
        source = "default"

        if sid in sb2_ids and sb2_analysis_dir:
            info = _load_sb2_rv_from_analysis_dir(sb2_analysis_dir, sid)
            if info is not None:
                rv_err = info["rv_err"]
                gamma_csv = info["gamma"]
                best_field = _field_from_mjds(info["mjds"])
                source = info["source"]
                n_sb2_hit += 1
            else:
                logger.warning("load_observed_star_properties: SB2 star "
                               "%s flagged in catalog but no usable "
                               "rv_final_for_mcmc.csv / rv_extracted.csv "
                               "under %s", sid, sb2_analysis_dir)

        if rv_err is None:
            csv_path = csv_by_id.get(sid)
            if csv_path is not None:
                try:
                    rv_df = pd.read_csv(csv_path)
                    rv_err = float(rv_df["Mean RVsig"].median())
                    gamma_csv = float(rv_df["Mean RV"].median())
                    best_field = _field_from_mjds(rv_df["MJD"].values)
                    source = "ccf"
                    n_csv_hit += 1
                except Exception:
                    rv_err = 2.0
                    gamma_csv = float(b.get("gamma", 168.0))
            else:
                rv_err = 2.0
                gamma_csv = float(b.get("gamma", 168.0))

        # Prefer the orbital-solution gamma from the LaTeX table when
        # available; fall back to the per-CSV median.
        gamma = float(b["gamma"]) if pd.notna(b.get("gamma")) else gamma_csv

        rows.append({
            "ID": sid_raw,        # keep the raw label so 4-080 Aa,Ab and
                                  # 4-080 B remain distinct rows
            "_root": sid,
            "Mspec": M1,
            "R_star": R1,
            "field": best_field,
            "rv_err": rv_err,
            "gamma": gamma,
            "source": source,
        })

    df = pd.DataFrame(rows)
    logger.info("load_observed_star_properties: %d stars built "
                "(mass-file hits: %d, ccf hits: %d, sb2-analysis hits: %d)",
                len(df), n_mass_hit, n_csv_hit, n_sb2_hit)
    return df


def load_star_properties(mass_file, rv_dir=None, sb2_analysis_dir=None,
                         ostar_catalog=None):
    """
    Load star properties for the bias-grid injection sample.

    When ``ostar_catalog`` is provided, the iteration universe is the
    catalog (134 stars) — every catalog star gets considered. M, R are
    pulled from ``mass_bloem.csv`` when available; otherwise from the
    nearest-spectral-type catalog star that does have a mass fit (same
    luminosity class + closest subclass number), then from the sample
    median as a last resort.

    RV resolution per star:
        - If ``Binary status == "SB2"``: pull rv1/rv1_err/mjd from the
          latest ``sb2_analysis/<timestamp>/data/rv_final_for_mcmc.csv``
          (fall back to ``rv_extracted.csv`` if absent).
        - Else: use ``<rv_dir>/BLOeM_<id>_CCF_RVs.csv``.

    Returns
    -------
    DataFrame with columns:
        ID, Mspec, R_star, field, rv_err, gamma, source, mass_source.
    ``source`` ∈ {"ccf", "sb2_final", "sb2_extracted"}.
    ``mass_source`` ∈ {"mass_bloem", "spectral_type", "median"}.
    """
    massdf = pd.read_csv(mass_file)
    massdf["_key"] = massdf["ID"].astype(str).str.replace(
        "BLOeM_", "", regex=False)

    if not (rv_dir and os.path.isdir(rv_dir)):
        raise RuntimeError(
            "rv_dir is required so per-star MJDs/rv_err/gamma are loaded "
            "from real CCF outputs.")

    import glob
    rv_files = glob.glob(os.path.join(rv_dir, "*_CCF_RVs.csv"))
    rv_info = {}
    rv_files_by_key = {}  # remember source path for each key
    for f in rv_files:
        base = os.path.basename(f).replace("_CCF_RVs.csv", "")
        star = base.replace("BLOeM_", "")
        try:
            df = pd.read_csv(f)
            rv_info[star] = {
                "rv_err": float(df["Mean RVsig"].median()),
                "gamma": float(df["Mean RV"].median()),
                "field": None,
            }
            rv_files_by_key[star] = f
        except Exception:
            continue

    # Match CCF stars to fields by MJD-array overlap.
    for star, info in rv_info.items():
        try:
            df = pd.read_csv(rv_files_by_key[star])
            star_mjds = df["MJD"].values
            best_field = 0
            best_overlap = 0
            for fi, fld_mjds in enumerate(BLOEM_MJD_ARRAYS):
                overlap = sum(1 for m in star_mjds
                             if any(abs(m - fm) < 0.5
                                    for fm in fld_mjds))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_field = fi
            info["field"] = best_field
        except Exception:
            info["field"] = 0

    def _field_from_mjds(star_mjds):
        best_field = 0
        best_overlap = 0
        for fi, fld_mjds in enumerate(BLOEM_MJD_ARRAYS):
            overlap = sum(1 for m in star_mjds
                          if any(abs(m - fm) < 0.5 for fm in fld_mjds))
            if overlap > best_overlap:
                best_overlap = overlap
                best_field = fi
        return best_field

    # Sample medians: robust prior for stars whose individual mass fit
    # is broken or absent.
    M1_med = float(massdf["Mspec"].median())
    R1_med = float(massdf["R_star"].median())

    # Per-mass-file lookup.
    massdf_indexed = massdf.set_index("_key")

    # Catalog-driven iteration when a catalog is supplied; otherwise
    # fall back to mass_bloem iteration (legacy behaviour).
    if ostar_catalog and os.path.isfile(ostar_catalog):
        cat = pd.read_csv(ostar_catalog)
        cat["_key"] = cat["BLOeM ID"].astype(str).str.replace(
            "BLOeM_", "", regex=False).str.strip()
        cat["_status"] = cat["Binary status"].astype(str).str.strip()
        cat["_sp_parsed"] = cat["Spectral type"].apply(_parse_spectral_type)
        # SB2 + Higher-order systems both use the SB2 pipeline output
        # (sb2_analysis/.../rv_*.csv) for their primary-component RVs.
        sb2_ids = set(cat.loc[cat["_status"].isin(("SB2", "Higher-order")),
                              "_key"])
        # Build the spectral-type lookup table: catalog stars that have
        # a valid mass_bloem entry and a parseable spectral type.
        sp_table = []
        for _, crow in cat.iterrows():
            ckey = crow["_key"]
            sub, lum = crow["_sp_parsed"]
            if sub is None or ckey not in massdf_indexed.index:
                continue
            M_b = float(massdf_indexed.loc[ckey, "Mspec"])
            R_b = float(massdf_indexed.loc[ckey, "R_star"])
            if M_b < 5 or M_b > 120 or R_b < 2 or R_b > 30:
                continue
            sp_table.append({"key": ckey, "subclass": sub, "lum": lum,
                             "M": M_b, "R": R_b})
        iter_keys = list(cat["_key"])
        sp_by_key = dict(zip(cat["_key"], cat["_sp_parsed"]))
    else:
        sb2_ids = set()
        sp_table = []
        sp_by_key = {}
        iter_keys = list(massdf["_key"])

    rows = []
    n_median = 0
    n_sp_match = 0
    n_mass_hit = 0
    n_ccf = 0
    n_sb2 = 0
    for star_key in iter_keys:
        rv_err = None
        gamma_val = None
        field_val = None
        source = None

        # SB2 stars: pull from sb2_analysis (primary component).
        if star_key in sb2_ids and sb2_analysis_dir:
            info = _load_sb2_rv_from_analysis_dir(sb2_analysis_dir, star_key)
            if info is not None:
                rv_err = info["rv_err"]
                gamma_val = info["gamma"]
                field_val = _field_from_mjds(info["mjds"])
                source = info["source"]
                n_sb2 += 1
            else:
                logger.warning("load_star_properties: SB2 star %s flagged "
                               "in catalog but no usable "
                               "rv_final_for_mcmc.csv / rv_corrected.csv / "
                               "rv_extracted.csv under %s", star_key, sb2_analysis_dir)

        # Non-SB2 (or SB2 with no analysis dir): CCF RVs.
        if rv_err is None and star_key in rv_info:
            ri = rv_info[star_key]
            rv_err = ri["rv_err"]
            gamma_val = ri["gamma"]
            field_val = ri["field"]
            source = "ccf"
            n_ccf += 1

        # Last resort: a non-SB2 star may still have an sb2_analysis dir
        # with rv_corrected.csv (e.g. Higher-order systems analysed with
        # the SB2 pipeline). Try that before giving up.
        if rv_err is None and sb2_analysis_dir:
            info = _load_sb2_rv_from_analysis_dir(sb2_analysis_dir, star_key)
            if info is not None:
                rv_err = info["rv_err"]
                gamma_val = info["gamma"]
                field_val = _field_from_mjds(info["mjds"])
                source = info["source"]
                n_sb2 += 1

        if rv_err is None:
            # No RV data anywhere — can't simulate this star.
            continue

        # Mass / radius: prefer mass_bloem, then nearest spectral type,
        # then sample median.
        mass_source = None
        if star_key in massdf_indexed.index:
            M_val = float(massdf_indexed.loc[star_key, "Mspec"])
            R_val = float(massdf_indexed.loc[star_key, "R_star"])
            if 5 <= M_val <= 120 and 2 <= R_val <= 30:
                mass_source = "mass_bloem"
                n_mass_hit += 1
            else:
                # Broken fit — try spectral-type match before median.
                target_sp = sp_by_key.get(star_key, (None, None))
                M_sp, R_sp, match_key = _nearest_spectral_type_mass(
                    target_sp, sp_table)
                if M_sp is not None:
                    M_val, R_val = M_sp, R_sp
                    mass_source = "spectral_type"
                    n_sp_match += 1
                    logger.debug("load_star_properties: %s mass-fit broken "
                                 "-> spectral-type match %s "
                                 "(M=%.1f R=%.1f)",
                                 star_key, match_key, M_val, R_val)
                else:
                    M_val, R_val = M1_med, R1_med
                    mass_source = "median"
                    n_median += 1
        else:
            # Not in mass_bloem at all — spectral-type match.
            target_sp = sp_by_key.get(star_key, (None, None))
            M_sp, R_sp, match_key = _nearest_spectral_type_mass(
                target_sp, sp_table)
            if M_sp is not None:
                M_val, R_val = M_sp, R_sp
                mass_source = "spectral_type"
                n_sp_match += 1
                logger.info("load_star_properties: %s not in mass_bloem "
                            "-> spectral-type match %s (M=%.1f R=%.1f)",
                            star_key, match_key, M_val, R_val)
            else:
                M_val, R_val = M1_med, R1_med
                mass_source = "median"
                n_median += 1

        rows.append({
            "ID": star_key,
            "Mspec": float(M_val),
            "R_star": float(R_val),
            "field": field_val,
            "rv_err": rv_err,
            "gamma": gamma_val,
            "source": source,
            "mass_source": mass_source,
        })

    if not rows:
        raise RuntimeError(
            "No stars resolved. Check rv_dir (%s), sb2_analysis_dir (%s), "
            "and ostar_catalog (%s)." %
            (rv_dir, sb2_analysis_dir, ostar_catalog))
    result = pd.DataFrame(rows)
    logger.info("load_star_properties: %d stars loaded "
                "(%d CCF, %d SB2-analysis); "
                "mass source: %d mass_bloem, %d spectral-type, %d median",
                len(result), n_ccf, n_sb2,
                n_mass_hit, n_sp_match, n_median)
    return result
