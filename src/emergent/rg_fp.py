# src/emergent/rg_fp.py
from __future__ import annotations
from typing import Callable, Optional
import numpy as np
import emergent.rg as _rg

CouplingVector = _rg.CouplingVector

def _assert_well_posed(q: int, R: int) -> None:
    if q <= 1 or R <= 1:
        raise ValueError("RG beta: q and R must be > 1.")

def beta_exact(g: CouplingVector, q: int, R: int) -> CouplingVector:
    """
    Third‑order paper‑style beta (stable numerics, monotone in typical ranges).
    """
    _assert_well_posed(q, R)
    g_star = float(g.g_star)
    lam    = float(g.lambda_mix)
    th     = float(g.theta_cp)

    # Spectral constants
    delta = (q - 1.0) / (q + R - 1.0)
    chi   = (1.0 - delta) / (2.0 * R)
    # use physical (positive) branch for mixing damping
    denom = max(1e-12, (delta - chi))
    sigma_mix = np.log2(delta / denom)

    # β(1), β(2), β(3)
    inv_qm1 = 1.0 / (q - 1.0)
    inv_qm1_sq = inv_qm1 * inv_qm1

    beta1_g = 0.0
    beta1_l = -sigma_mix * lam
    beta1_t = 0.0

    beta2_g = 0.0
    beta2_l = inv_qm1 * g_star * lam
    beta2_t = ((R - 1.0) / ((q - 1.0) * (q + R - 1.0))) * g_star * lam

    beta3_g = 0.5 * inv_qm1_sq * (g_star ** 2) * lam
    beta3_l = 1.5 * (R - 1.0) * inv_qm1_sq * (lam ** 3)
    beta3_t = 0.0

    return CouplingVector(
        g_star=beta1_g + beta2_g + beta3_g,
        lambda_mix=beta1_l + beta2_l + beta3_l,
        theta_cp=beta1_t + beta2_t + beta3_t,
    )

# ---- Register with the integrator (convert to IVP RHS) ----------------------
_default_beta_rhs: Optional[Callable] = getattr(_rg, "beta_function", None)

def _wrap(beta_g: Callable[[CouplingVector, int, int], CouplingVector]
          ) -> Callable[[float, np.ndarray, int, int], np.ndarray]:
    def rhs(k: float, y: np.ndarray, q: int, R: int) -> np.ndarray:
        g = CouplingVector(g_star=y[0], lambda_mix=y[1], theta_cp=y[2])
        d = beta_g(g, q, R)
        return np.array([d.g_star, d.lambda_mix, d.theta_cp], dtype=float)
    return rhs

def set_beta_functions_fp(beta_g: Callable[[CouplingVector, int, int], CouplingVector]) -> None:
    _rg.beta_function = _wrap(beta_g)

def restore_default_beta() -> None:
    if _default_beta_rhs is not None:
        _rg.beta_function = _default_beta_rhs

# install at import
set_beta_functions_fp(beta_exact)

__all__ = ["beta_exact", "set_beta_functions_fp", "restore_default_beta"]
