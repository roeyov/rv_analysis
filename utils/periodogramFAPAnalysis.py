import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, wasserstein_distance


def analyze_permutation_convergence(
    Z,                      # array-like of permutation max-powers
    Z_obs=None,             # observed max-power from real data
    out_dir=None,
    tag="LS",
    star_name="",
    Ns=None,                # optional explicit checkpoints
    quantiles=(0.90, 0.95, 0.99, 0.999),
    window_frac=0.5,        # compare early vs late chunks of size floor(window_frac*N)
    n_checkpoints=45,       # <-- MORE points => denser curves
    n_checkpoints_early=20, # <-- extra density at small N (where things move fast)
    min_N=50,
    hist_bins=50,
    snapshot_Ns=(200, 2000, 20000, None),  # last None => use max N
):
    """
    Saves 5 convergence diagnostics (+ ECDF helper included):
      1) Histogram snapshots vs N
      2) Quantiles vs N
      3) FAP(Z_obs) vs N + ~95% (±2σ) binomial band
      4) Distribution distance vs N (KS + Wasserstein) early vs late chunks
      5) Survival function overlay (1-ECDF) vs N (log-y)

    Returns a dict of summary stats (last values + last-step relative quantile changes).
    """

    # --------------------------
    # helpers (local)
    # --------------------------
    def _ensure_dir(d):
        if d:
            os.makedirs(d, exist_ok=True)

    def _savefig(path):
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()

    def ecdf(x):
        """Empirical CDF: returns sorted x and cumulative probabilities."""
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    def _rel_change_last(vals, k=6):
        """Max relative step change over last k points (k>=2)."""
        vals = np.asarray(vals, dtype=float)
        if len(vals) < k:
            return np.nan
        dv = np.diff(vals[-k:])
        denom = np.maximum(np.abs(vals[-k:-1]), 1e-12)
        return float(np.max(np.abs(dv) / denom))

    # --------------------------
    # prep
    # --------------------------
    Z = np.asarray(Z, dtype=float)
    Z = Z[np.isfinite(Z)]
    _ensure_dir(out_dir)
    # --------------------------
    # Save permutation samples
    # --------------------------
    base = f"{star_name}_{tag}".strip("_")
    if out_dir:
        z_path = os.path.join(out_dir, f"{base}_perm_max_power.npy")
        np.save(z_path, Z)

    Ntot = len(Z)
    if Ntot < min_N:
        return {"warning": f"Too few permutations ({Ntot}) to assess convergence robustly.", "n": int(Ntot)}

    # --------------------------
    # choose denser checkpoints
    # --------------------------
    if Ns is None:
        # Dense in log-space overall + extra density at small N (linear-ish)
        # 1) log-spaced checkpoints
        Ns_log = np.logspace(np.log10(max(10, min_N)), np.log10(Ntot), n_checkpoints)
        Ns_log = np.unique(Ns_log.astype(int))

        # 2) add many early checkpoints (where convergence changes quickly)
        early_max = min(Ntot, max(500, int(0.02 * Ntot)))  # up to 2% of Ntot or 500
        if early_max > min_N:
            Ns_early = np.linspace(min_N, early_max, n_checkpoints_early)
            Ns_early = np.unique(Ns_early.astype(int))
        else:
            Ns_early = np.array([min_N], dtype=int)

        Ns = np.unique(np.concatenate([Ns_early, Ns_log, [Ntot]]))
        Ns = Ns[Ns >= min_N]
    else:
        Ns = np.unique(np.asarray(Ns, dtype=int))
        Ns = Ns[(Ns >= min_N) & (Ns <= Ntot)]
        if len(Ns) == 0:
            Ns = np.array([min_N, Ntot], dtype=int)

    results = {}

    # --------------------------
    # (1) Histogram snapshots
    # --------------------------
    snap = []
    for s in snapshot_Ns:
        if s is None:
            snap.append(Ntot)
        else:
            snap.append(int(np.clip(s, min_N, Ntot)))
    snap = np.unique(snap)

    plt.figure(figsize=(7.5, 4.4))
    for N in snap:
        plt.hist(Z[:N], bins=hist_bins, alpha=0.30, density=True, label=f"N={N}")
    if Z_obs is not None:
        plt.axvline(Z_obs, linestyle="--", linewidth=2)
    plt.xlabel(f"{tag} permutation max power")
    plt.ylabel("Density")
    plt.title(f"{base}: Histogram snapshots")
    plt.legend()
    if out_dir:
        _savefig(os.path.join(out_dir, f"{base}_perm_hist_snapshots.png"))

    # --------------------------
    # (2) Quantiles vs N
    # --------------------------
    q_curves = {q: np.empty(len(Ns), dtype=float) for q in quantiles}
    for j, N in enumerate(Ns):
        x = Z[:N]
        for q in quantiles:
            q_curves[q][j] = np.quantile(x, q)

    plt.figure(figsize=(7.5, 4.4))
    for q in quantiles:
        plt.plot(Ns, q_curves[q], label=f"q={q} (FAP~{1-q:g})")
    plt.xscale("log")
    plt.xlabel("Number of permutations (N)")
    plt.ylabel(f"{tag} max power quantile")
    plt.title(f"{base}: Quantile convergence")
    plt.legend()
    if out_dir:
        _savefig(os.path.join(out_dir, f"{base}_perm_quantiles_vs_N.png"))

    results["quantile_rel_change_last"] = {str(q): _rel_change_last(q_curves[q], k=6) for q in quantiles}

    # --------------------------
    # (3) FAP(Z_obs) vs N + ~95% band (±2σ)
    # --------------------------
    if Z_obs is not None:
        fap = np.empty(len(Ns), dtype=float)
        fap_lo = np.empty(len(Ns), dtype=float)
        fap_hi = np.empty(len(Ns), dtype=float)

        for j, N in enumerate(Ns):
            x = Z[:N]
            k = np.sum(x >= Z_obs)
            p = k / N
            se = np.sqrt(max(p * (1 - p), 0.0) / N)
            fap[j] = p
            fap_lo[j] = max(p - 2 * se, 0.0)
            fap_hi[j] = min(p + 2 * se, 1.0)

        plt.figure(figsize=(7.5, 4.4))
        plt.plot(Ns, fap, label="FAP estimate")
        plt.fill_between(Ns, fap_lo, fap_hi, alpha=0.25, label="~95% band (±2σ)")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of permutations (N)")
        plt.ylabel("Estimated FAP(Z_obs)")
        plt.title(f"{base}: FAP convergence")
        plt.legend()
        if out_dir:
            _savefig(os.path.join(out_dir, f"{base}_perm_fap_vs_N.png"))

        results["fap_last"] = float(fap[-1])
        results["fap_band_last"] = (float(fap_lo[-1]), float(fap_hi[-1]))

    # --------------------------
    # (4) KS + Wasserstein distances vs N (early vs late chunks)
    # --------------------------
    ks_vals, w_vals, Ns_dist = [], [], []
    for N in Ns:
        m = int(np.floor(window_frac * N))
        if m < 40:
            continue
        a = Z[:m]
        b = Z[N - m:N]
        ks_vals.append(ks_2samp(a, b).statistic)
        w_vals.append(wasserstein_distance(a, b))
        Ns_dist.append(N)

    if len(Ns_dist) >= 3:
        plt.figure(figsize=(7.5, 4.4))
        plt.plot(Ns_dist, ks_vals, label="KS statistic")
        plt.xscale("log")
        plt.xlabel("Number of permutations (N)")
        plt.ylabel("KS distance (early vs late)")
        plt.title(f"{base}: KS convergence")
        plt.legend()
        if out_dir:
            _savefig(os.path.join(out_dir, f"{base}_perm_KS_vs_N.png"))

        plt.figure(figsize=(7.5, 4.4))
        plt.plot(Ns_dist, w_vals, label="Wasserstein distance")
        plt.xscale("log")
        plt.xlabel("Number of permutations (N)")
        plt.ylabel("Wasserstein distance (early vs late)")
        plt.title(f"{base}: Wasserstein convergence")
        plt.legend()
        if out_dir:
            _savefig(os.path.join(out_dir, f"{base}_perm_Wasserstein_vs_N.png"))

        results["ks_last"] = float(ks_vals[-1])
        results["wasserstein_last"] = float(w_vals[-1])

    # --------------------------
    # (5) Survival overlays (1-ECDF) for several N
    # --------------------------
    # choose 4 overlay sizes: small / mid / large / max (from Ns list)
    overlay_idxs = np.unique(np.clip(
        np.array([0, len(Ns)//3, 2*len(Ns)//3, len(Ns)-1]),
        0, len(Ns)-1
    ))
    overlay_Ns = Ns[overlay_idxs]

    plt.figure(figsize=(7.5, 4.4))
    for N in overlay_Ns:
        xs, ys = ecdf(Z[:N])
        plt.plot(xs, 1 - ys, label=f"N={N}")
    if Z_obs is not None:
        plt.axvline(Z_obs, linestyle="--", linewidth=2)
    plt.yscale("log")
    plt.xlabel(f"{tag} permutation max power")
    plt.ylabel("Survival: P(Z ≥ z) (log scale)")
    plt.title(f"{base}: Tail stability (survival)")
    plt.legend()
    if out_dir:
        _savefig(os.path.join(out_dir, f"{base}_perm_survival_overlay.png"))

    return results
