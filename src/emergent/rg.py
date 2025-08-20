# src/emergent/rg.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import numpy as np
from scipy.integrate import solve_ivp

# -------------------------
# Coupling container
# -------------------------
@dataclass(frozen=True)
class CouplingVector:
    g_star: float
    lambda_mix: float
    theta_cp: float

    def to_array(self) -> np.ndarray:
        return np.array([float(self.g_star),
                         float(self.lambda_mix),
                         float(self.theta_cp)], dtype=float)

    @staticmethod
    def from_array(y: np.ndarray) -> "CouplingVector":
        return CouplingVector(g_star=float(y[0]),
                              lambda_mix=float(y[1]),
                              theta_cp=float(y[2]))

# ------------------------------------------------------------
# Default beta; can be replaced at import time by emergent.rg_fp
# ------------------------------------------------------------
def _default_beta_function(k: float, y: np.ndarray, q: int, R: int) -> np.ndarray:
    """
    A numerically tame RHS used until rg_fp installs the paper beta.
    It gently damps lambda_mix and lets g_star drift weakly.
    """
    g, lam, th = float(y[0]), float(y[1]), float(y[2])
    inv_qm1 = 1.0 / max(1.0, (q - 1.0))

    # soft dynamics (stable even with large steps)
    dg = 0.05 * inv_qm1 * g * lam * 0.1
    dlam = -0.10 * lam + 0.02 * inv_qm1 * g * lam
    dth = 0.0
    return np.array([dg, dlam, dth], dtype=float)

# The global RHS; rg_fp may monkey‑patch this
beta_function: Callable[[float, np.ndarray, int, int], np.ndarray] = _default_beta_function

# ------------------------------------------------------------
# Integration helpers
# ------------------------------------------------------------
def run_rg_flow(
    g_initial: CouplingVector, *,
    k_start: float, k_end: float, q: int, R: int
) -> CouplingVector:
    """Integrate g'(k)=beta(g) and return g(k_end)."""
    sol = solve_ivp(
        fun=beta_function,
        t_span=[float(k_start), float(k_end)],
        y0=g_initial.to_array(),
        args=(int(q), int(R)),
        method="RK45",
        dense_output=True,
        rtol=1e-8, atol=1e-10,
    )
    return CouplingVector.from_array(sol.sol(float(k_end)))

def solve_rg_flow(
    g_initial: CouplingVector, *,
    k_start: float, k_end: float, q: int, R: int,
    k_eval: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Callable[[float], np.ndarray]]:
    """
    Integrate once; return (solution_on_grid, interpolator).
    """
    if k_eval is None:
        k_eval = np.linspace(k_start, k_end, 121)

    sol = solve_ivp(
        fun=beta_function,
        t_span=[float(k_start), float(k_end)],
        y0=g_initial.to_array(),
        args=(int(q), int(R)),
        method="RK45",
        dense_output=True,
        rtol=1e-9, atol=1e-11,
        t_eval=k_eval,
    )
    return sol.y.T, sol.sol
