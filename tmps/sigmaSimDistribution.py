#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# ── 1) Your fitted log-normal parameters ───────────────────────────
# Replace these with your actual fit results:
# shape = σ of ln(X); loc = shift; scale = exp(μ of ln(X))

# ── 2) Sampling functions ────────────────────────────────────────────
def sample_method1(shape, loc, scale, n_objects=139, n_measurements=25, random_state=None):
    """
    Method 1:
      - for each object i, draw σ_i ~ LogNormal(shape,loc,scale)
      - then draw 25 errors ~ Normal(0, σ_i)
    Returns an (n_objects × n_measurements) array of RV shifts.
    """
    rng = np.random.default_rng(random_state)
    sigmas = lognorm.rvs(shape, loc=loc, scale=scale, size=n_objects, random_state=rng)
    return [[sigma] * n_measurements for sigma in sigmas], rng.normal(loc=0, scale=sigmas[:, None],
                      size=(n_objects, n_measurements))

def sample_method2(shape, loc, scale, n_objects=139, n_measurements=25, random_state=None):
    """
    Method 2:
      - draw one σ per each of the 139×25 measurements
      - then draw one error per measurement with its σ
    Returns an (n_objects × n_measurements) array of RV shifts.
    """
    rng = np.random.default_rng(random_state)
    total = n_objects * n_measurements
    sigmas = lognorm.rvs(shape, loc=loc, scale=scale, size=total, random_state=rng)
    noise  = rng.normal(loc=0, scale=sigmas)
    return sigmas.reshape(n_objects, n_measurements), noise.reshape(n_objects, n_measurements)

def main():
    # ── 3) Generate the samples ──────────────────────────────────────────
    N_OBJECTS = 1
    N_MEASUREMENTS = int(25)
    shape, loc, scale = 0.8702, 0.0, 3.4602

    _,samp1 = sample_method1(shape, loc, scale, random_state=42,n_objects=N_OBJECTS, n_measurements=N_MEASUREMENTS)
    samp2 = sample_method2(shape, loc, scale, random_state=42, n_objects=N_OBJECTS, n_measurements=N_MEASUREMENTS)

    # ── 4) Plot overlayed histograms ────────────────────────────────────
    data1 = np.log10(np.abs(samp1.ravel()))
    data2 = np.log10(np.abs(samp2.ravel()))

    bins = np.linspace(min(data1.min(), data2.min()),
                       max(data1.max(), data2.max()),
                       100)

    plt.figure(figsize=(8, 5))
    plt.hist(data1, bins=bins, alpha=0.5,
             label='Method 1 (σ per object)', density=True)
    plt.hist(data2, bins=bins, alpha=0.5,
             label='Method 2 (σ per meas.)', density=True)

    plt.xlabel('Simulated RV Shift')
    plt.ylabel('Density')
    plt.title('Overlayed Histograms of Simulated RV Shifts')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()