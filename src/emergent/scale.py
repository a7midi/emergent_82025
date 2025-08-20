# src/emergent/scale.py
"""
Physical scale helpers: convert between the dimensionless depth k
and a physical energy scale mu [GeV] using
    mu = GeV0 * 2^{-k}

These helpers are centralized here and re-exported from paper_maps.v8
for backwards compatibility.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

# Global pivot (can be set from a Z- or GUT-anchor)
_GEV0: float = 1.0


def get_GeV0() -> float:
    """Return the current reference scale GeV0."""
    return float(_GEV0)


def set_GeV0(value: float) -> None:
    """Set the reference scale GeV0."""
    global _GEV0
    _GEV0 = float(value)


def set_GeV0_by_anchors(
    *,
    mu_Z: float,
    k_Z: float,
    mu_GUT: Optional[float] = None,
    k_GUT: Optional[float] = None,
    prefer: str = "Z",
    return_log10: bool = False,
) -> tuple[float, Optional[float]]:
    """
    Fix GeV0 from one or two physical anchors.

    GeV0 = mu * 2^{k}.

    If both anchors are provided, we also report their mismatch
    (ratio or log10(ratio)). If `prefer="GUT"`, GeV0 is taken
    from the GUT anchor; otherwise from the Z anchor.
    """
    GeV0_Z = float(mu_Z) * (2.0 ** float(k_Z))

    mismatch: Optional[float] = None
    GeV0 = GeV0_Z

    if (mu_GUT is not None) and (k_GUT is not None):
        GeV0_GUT = float(mu_GUT) * (2.0 ** float(k_GUT))
        if GeV0_Z > 0.0:
            ratio = GeV0_GUT / GeV0_Z
            mismatch = np.log10(ratio) if return_log10 else ratio
        else:
            mismatch = None
        if str(prefer).upper() == "GUT":
            GeV0 = GeV0_GUT

    set_GeV0(GeV0)
    return GeV0, mismatch


def k_to_GeV(k: float) -> float:
    """mu(k) = GeV0 * 2^{-k}."""
    return get_GeV0() * (2.0 ** (-float(k)))


def GeV_to_k(mu_GeV: float) -> float:
    """k(mu) = -log2(mu / GeV0)."""
    mu = max(float(mu_GeV), 1e-300)
    return -np.log2(mu / get_GeV0())


__all__ = [
    "get_GeV0",
    "set_GeV0",
    "set_GeV0_by_anchors",
    "k_to_GeV",
    "GeV_to_k",
]
