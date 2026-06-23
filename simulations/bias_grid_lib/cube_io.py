"""Variant-aware readers for ``grid_cubes.npz`` (schema v4) + v3 fallback.

The all-variants cube (schema v4) namespaces every goodness cube as
``v__<tag>__gmf_<test>_cube`` / ``v__<tag>__<test>_<par>_cube``. Downstream
tools (closure aggregation, summary PNG/PDF, subsampled re-scoring) used to
read the bare single-mode keys (``gmf_ks_cube`` …). These helpers give them
one place to resolve a variant and read its cubes, transparently falling
back to the bare keys when handed a legacy v3 single-mode cube.

The default variant when none is requested is ``eccentric_only__numerical``
(the eccentric-only treatment with the numerical period cutoff), matching
the closure-test config historically used on astro3.
"""

from simulations.bias_grid_lib.constants import variant_tag, split_variant_tag


DEFAULT_VARIANT_TAG = variant_tag("eccentric_only", "numerical", False)


def is_multi_variant(cubes):
    """True for a schema-v4 (all-variants) cube, False for legacy v3."""
    return "variants" in getattr(cubes, "files", [])


def list_cube_variants(cubes):
    """Variant tags present in a loaded ``grid_cubes.npz``.

    v4 cubes carry an explicit ``variants`` array. A v3 single-mode cube
    synthesizes one tag from its stored scalar metadata (or the presence
    of ``*_e_circ_cube`` for split).
    """
    if is_multi_variant(cubes):
        return [str(t) for t in cubes["variants"]]
    # v3 single-mode cube: synthesize one tag from the stored scalars.
    if "e_score_mode" in cubes.files:
        em = str(cubes["e_score_mode"])
    else:
        em = "split" if any(("%s_e_circ_cube" % t) in cubes.files
                            for t in ("ks", "ad", "cvm", "wass")) \
            else "combined"
    cm = (str(cubes["logP_cutoff_mode"])
          if "logP_cutoff_mode" in cubes.files else "none")
    lucy = (bool(cubes["apply_lucy_sweeny_e"])
            if "apply_lucy_sweeny_e" in cubes.files else False)
    return [variant_tag(em, cm, lucy)]


def resolve_variant_tag(cubes, requested=None):
    """Pick a variant tag present in ``cubes``.

    Order of preference: the explicit ``requested`` tag (if present), then
    ``DEFAULT_VARIANT_TAG``, then the first available tag. Raises if the
    cube carries no variants at all.
    """
    tags = list_cube_variants(cubes)
    if not tags:
        raise KeyError("no variants found in cube")
    if requested and requested in tags:
        return requested
    if DEFAULT_VARIANT_TAG in tags:
        return DEFAULT_VARIANT_TAG
    # Match on (e_score_mode, logP_cutoff_mode) ignoring the lucy element,
    # so the default resolves across schema versions (e.g. a v4 cube whose
    # tags are 2-part 'eccentric_only__numerical'), and a 2-part `requested`
    # still selects the right v5 variant family.
    d_em, d_cm, _ = split_variant_tag(DEFAULT_VARIANT_TAG)
    want = split_variant_tag(requested)[:2] if requested else (d_em, d_cm)
    for t in tags:
        em, cm, _ = split_variant_tag(t)
        if (em, cm) == want:
            return t
    if requested:  # requested family absent — fall back to the default
        for t in tags:
            em, cm, _ = split_variant_tag(t)
            if (em, cm) == (d_em, d_cm):
                return t
    return tags[0]


def gmf_cube_key(cubes, test="ks", tag=None):
    """npz key for the (variant, test) log-GMF cube; resolves the variant."""
    tag = resolve_variant_tag(cubes, tag)
    if is_multi_variant(cubes):
        return "v__%s__gmf_%s_cube" % (tag, test), tag
    return "gmf_%s_cube" % test, tag


def read_gmf_cube(cubes, test="ks", tag=None):
    """Return ``(log_gmf_cube, resolved_tag)`` for (variant, test).

    Falls back to a legacy bare ``gmf_<test>_cube`` (and the very old
    ``gmf_cube`` for KS) when handed a v3 cube. Raises ``KeyError`` if no
    matching cube exists.
    """
    key, tag = gmf_cube_key(cubes, test=test, tag=tag)
    if key in cubes.files:
        return cubes[key], tag
    if not is_multi_variant(cubes) and test == "ks" \
            and "gmf_cube" in cubes.files:
        return cubes["gmf_cube"], tag
    raise KeyError("gmf cube %r not found in %s" % (key, list(cubes.files)[:8]))


def pval_cube_key(cubes, test, par, tag=None):
    """npz key for a (variant, test, par) p-value/distance cube."""
    tag = resolve_variant_tag(cubes, tag)
    if is_multi_variant(cubes):
        return "v__%s__%s_%s_cube" % (tag, test, par)
    return "%s_%s_cube" % (test, par)
