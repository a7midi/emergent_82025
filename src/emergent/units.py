# src/emergent/units.py
from __future__ import annotations
import numpy as np

MZ_GEV = 91.1876  # Z-boson mass in GeV
HBAR_S_PER_GEV = 6.582119569e-25 # ℏ in s/GeV

def delta_t_star_from_anchor(mu_ref: float = MZ_GEV, k_ref: float = 1.0) -> float:
    """Return Δt* [GeV^-1] from a chosen (k_ref, mu_ref) anchor."""
    return 2.0**(-k_ref) / float(mu_ref)

def k_from_mu(mu: float, delta_t_star: float) -> float:
    """k(mu) = -log2(mu * Δt*)."""
    x = float(mu) * float(delta_t_star)
    if x <= 0.0:
        raise ValueError("mu*Δt* must be positive.")
    return -np.log2(x)

def mu_from_k(k: float, delta_t_star: float) -> float:
    """Invert: mu(k) = 2^{-k} / Δt*."""
    return 2.0**(-float(k)) / float(delta_t_star)