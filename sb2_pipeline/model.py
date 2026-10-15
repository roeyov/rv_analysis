"""
sb2_pipeline.model — joint two-component Keplerian RV model.

Convention (identical to the archived SB2 analyses and simulations.common.RV12):
    cos_term = cos(omega + nu) + e*cos(omega)
    rv1 = gamma + K1 * cos_term
    rv2 = gamma - K2 * cos_term
with a single shared gamma and omega referring to the primary.
"""

from orbital.kepler import true_anomaly
from simulations.common import RV12


def sb2_rv_model(t, P, T0, omega, e, K1, K2, gamma, solver="iterative"):
    """Return (rv1_model, rv2_model) at times t. solver: "iterative" | "newton"."""
    nu = true_anomaly(t, P, T0, e, solver=solver)
    return RV12(nu, gamma, K1, K2, omega, e)
