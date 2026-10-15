"""Quantiles of a posterior marginal evaluated on a grid.

Every reader that turns a bias-grid posterior into quantiles or intervals
should call this module: the closure coverage harness, the explorer, the
static plots and posterior_slices. The paper-side corner script
(``Ostars_article/scripts/make_bias_corner_figure.py``) cannot import the
``simulations`` package, so it carries a copy of ``grid_quantiles``. Keep the
two in sync.

Convention. The 1-D marginal ``post`` is a density sampled at the grid
nodes. Each node owns its cell: the boundaries are the midpoints between
neighbouring nodes, clamped to the grid box ``[grid[0], grid[-1]]``, which
is the support of the flat prior. The mass of node k is
``post[k] * width[k]``, so the end nodes get half-width cells, as in the
trapezoid rule. Within a cell the density is uniform, so the CDF is
piecewise linear on the cell edges and inverts exactly. A marginal that is
symmetric about a node has its median on that node.

The estimator this replaces, ``np.interp(q, cumsum(post) / sum, grid)``,
placed the cumulative mass *including* node k at the node position x_k. In
effect it put each node's mass in ``[x_{k-1}, x_k]``, so on a uniform grid
every interior quantile came out h/2 low. Interval widths were unaffected.

Alternative (``method="linear"``). The density is interpolated linearly
between neighbouring nodes (the trapezoid rule's model), so the CDF is
piecewise quadratic and is inverted exactly on each segment. On a marginal
that spans many cells the two estimators agree to a small fraction of a
cell; they differ when the posterior is only one or two cells wide, where
the uniform-in-cell model puts the quantiles at cell-phase-dependent
positions. ``QUANTILE_METHOD`` is the package-wide default.
"""

import numpy as np

QUANTILE_METHOD = "cell"    # "cell" (uniform within cells) or "linear"


def cell_edges(grid):
    """Cell boundaries of the grid nodes, clamped to the grid box.

    Returns ``len(grid) + 1`` edges: ``grid[0]``, the midpoints between
    neighbouring nodes, and ``grid[-1]``. The grid must be strictly
    increasing. It may be non-uniform.
    """
    g = np.asarray(grid, dtype=float)
    if g.ndim != 1 or g.size == 0:
        raise ValueError("grid must be a non-empty 1-D array")
    if g.size > 1 and np.any(np.diff(g) <= 0):
        raise ValueError("grid must be strictly increasing")
    if g.size == 1:
        return np.array([g[0], g[0]])
    return np.concatenate(([g[0]], 0.5 * (g[1:] + g[:-1]), [g[-1]]))


def grid_quantiles(post, grid, qs, method=None):
    """Quantiles ``qs`` of a 1-D marginal sampled at the nodes of ``grid``.

    ``post`` may be unnormalized; NaN counts as zero mass. Returns a list of
    floats, one per quantile. All values are NaN if the marginal carries no
    mass. A single-node grid returns that node. ``method`` is "cell"
    (default, uniform density within each node's cell) or "linear"
    (density interpolated linearly between nodes); None uses
    ``QUANTILE_METHOD``.
    """
    method = QUANTILE_METHOD if method is None else method
    if method not in ("cell", "linear"):
        raise ValueError("method must be 'cell' or 'linear', got %r" % method)
    g = np.asarray(grid, dtype=float)
    p = np.nan_to_num(np.asarray(post, dtype=float), nan=0.0)
    q = np.atleast_1d(np.asarray(qs, dtype=float))
    if p.shape != g.shape:
        raise ValueError("post shape %s does not match grid shape %s"
                         % (p.shape, g.shape))
    if np.any(p < 0):
        raise ValueError("posterior marginal has negative entries")
    if np.any((q < 0) | (q > 1)):
        raise ValueError("quantiles must lie in [0, 1]")
    if not p.sum() > 0:
        return [float("nan")] * q.size
    if g.size == 1:
        return [float(g[0])] * q.size
    if method == "linear":
        return _linear_quantiles(p, g, q)
    edges = cell_edges(g)
    mass = p * np.diff(edges)
    total = mass.sum()
    if not total > 0:
        return [float("nan")] * q.size
    cdf = np.concatenate(([0.0], np.cumsum(mass) / total))
    return [float(np.interp(qi, cdf, edges)) for qi in q]


def _linear_quantiles(p, g, q):
    """Exact quantiles of the piecewise-linear density through (g, p)."""
    if np.any(np.diff(g) <= 0):
        raise ValueError("grid must be strictly increasing")
    h = np.diff(g)
    seg = 0.5 * (p[1:] + p[:-1]) * h              # mass of each segment
    cdf = np.concatenate(([0.0], np.cumsum(seg)))
    total = cdf[-1]
    if not total > 0:
        return [float("nan")] * q.size
    out = []
    for qi in q:
        target = qi * total
        if target <= 0.0:
            out.append(float(g[0]))
            continue
        if target >= total:
            out.append(float(g[-1]))
            continue
        # segment k holds cdf[k] <= target < cdf[k+1]; skip empty segments
        k = int(np.searchsorted(cdf, target, side="right") - 1)
        k = min(max(k, 0), h.size - 1)
        rem = target - cdf[k]
        p0, p1, hk = p[k], p[k + 1], h[k]
        # mass within the segment: p0 t + (p1 - p0) t^2 / (2 hk) = rem
        a = 0.5 * (p1 - p0) / hk
        if abs(a) * hk < 1e-12 * max(p0, p1, 1e-300):
            t = rem / p0 if p0 > 0 else 0.0
        else:
            disc = p0 * p0 + 4.0 * a * rem
            t = (-p0 + np.sqrt(max(disc, 0.0))) / (2.0 * a)
        t = min(max(float(t), 0.0), float(hk))
        out.append(float(g[k] + t))
    return out


def c_marginals(lnc, c1_grid, c2_grid):
    """Marginal densities of the circular fractions c1 and c2.

    ``lnc[i1, i2]`` is the log posterior density at the node pair
    ``(c1_grid[i1], c2_grid[i2])``. Each marginal integrates the other
    fraction over its cells (the flat prior of this module's convention), so
    the nodes of an uneven lattice are not weighted equally. Returns the two
    densities at their own nodes, ready for ``grid_quantiles``.
    """
    lnc = np.asarray(lnc, dtype=float)
    pc = np.exp(lnc - np.max(lnc))

    def widths(grid):
        g = np.asarray(grid, dtype=float)
        return np.ones(1) if g.size == 1 else np.diff(cell_edges(g))
    w1, w2 = widths(c1_grid), widths(c2_grid)
    if pc.shape != (w1.size, w2.size):
        raise ValueError("c table shape %s does not match the grids (%d, %d)"
                         % (pc.shape, w1.size, w2.size))
    return pc @ w2, w1 @ pc
