# src/emergent/calibration.py
"""
M13-calibration: Lattice↔physical map C_{k0} and Jacobian.

Provides:
  • LatticeParams, PhysicalConstants
  • calibrate_lattice_to_physical (forward map)
  • calibrate_physical_to_lattice (inverse map)
  • get_calibration_jacobian (Jacobian of forward map)

Matches round-trip and Jacobian invertibility tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatticeParams:
    delta_t: float
    phi_min: float
    g_star: float
    lambda_mix: float
    theta_cp: float


@dataclass(frozen=True)
class PhysicalConstants:
    c: float
    hbar: float
    G: float
    m_f0: float
    theta_cp_obs: float


def calibrate_lattice_to_physical(
    lat_params: LatticeParams, k0: int, sigma_mix: float = 0.24
) -> PhysicalConstants:
    """Forward map to (c, ħ, G, m_f0, θ_CP)."""
    c = 2.0 ** (-k0) / lat_params.delta_t
    hbar = lat_params.phi_min * c
    G = lat_params.g_star / (8.0 * np.pi)

    mu0 = c  # 2^{-k0}/dt
    m_f0 = lat_params.lambda_mix * mu0 * np.exp(-sigma_mix * k0)

    return PhysicalConstants(c=c, hbar=hbar, G=G, m_f0=m_f0, theta_cp_obs=lat_params.theta_cp)


def calibrate_physical_to_lattice(
    phys_consts: PhysicalConstants, k0: int, sigma_mix: float = 0.24
) -> LatticeParams:
    """Inverse map back to lattice parameters (used by tests)."""
    delta_t = 2.0 ** (-k0) / phys_consts.c
    phi_min = phys_consts.hbar / phys_consts.c
    g_star = 8.0 * np.pi * phys_consts.G

    mu0 = phys_consts.c
    lambda_mix = phys_consts.m_f0 / (mu0 * np.exp(-sigma_mix * k0))

    return LatticeParams(
        delta_t=delta_t,
        phi_min=phi_min,
        g_star=g_star,
        lambda_mix=lambda_mix,
        theta_cp=phys_consts.theta_cp_obs,
    )


def get_calibration_jacobian(
    lat_params: LatticeParams, k0: int, sigma_mix: float = 0.24
) -> np.ndarray:
    """
    Jacobian J = d(physical) / d(lattice). Rows: (c, ħ, G, m_f0, θ_obs),
    Cols: (Δt, φ_min, g_*, λ_mix, θ_CP).
    """
    J = np.zeros((5, 5), dtype=float)
    dt, p_min, g_s, l_mix = lat_params.delta_t, lat_params.phi_min, lat_params.g_star, lat_params.lambda_mix

    # c = 2^{-k0} / dt
    J[0, 0] = - (2.0 ** (-k0)) / (dt ** 2)

    # ħ = φ_min * c
    J[1, 0] = - p_min * (2.0 ** (-k0)) / (dt ** 2)  # via c(dt)
    J[1, 1] = (2.0 ** (-k0)) / dt                   # via φ_min

    # G = g_*/(8π)
    J[2, 2] = 1.0 / (8.0 * np.pi)

    # m_f0 = λ_mix * (2^{-k0}/dt) * e^{-σ k0}
    mu0_factor = np.exp(-sigma_mix * k0)
    J[3, 0] = - l_mix * (2.0 ** (-k0)) / (dt ** 2) * mu0_factor
    J[3, 3] = (2.0 ** (-k0) / dt) * mu0_factor

    # θ_obs = θ_CP
    J[4, 4] = 1.0
    return J
