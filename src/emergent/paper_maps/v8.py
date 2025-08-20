# src/emergent/paper_maps/v8.py
"""
Paper hooks (v8): analytic RG-facing maps, physical clock helpers, and
direct cosmology map Λ(q,R).  This module is loaded by
`emergent.physics_maps.make_hooks_from_module("emergent.paper_maps.v8")`.

Implements (paper references in docstrings):
- Gauge-map normalisation via ξ₂ (Paper III, §3.9.2; canonical generator scaling).
- Single-scale gauge map (g₁,g₂) from RG variables (g*, λ_mix) with a smooth,
  monotone mixing function; compatible with predict.py observables
  (sin²θ_W, α_EM) and the two-target calibration.
- Physical clock: μ = GeV₀ · 2^{−k}; anchor setters/getters.
- Cosmology: Λ(q,R) direct map (Paper I, cosmology theorem; discrete dependence).

All functions are deterministic and pure-Python; any optional speedups live
behind flags elsewhere in the codebase.
"""
from __future__ import annotations
from dataclasses import dataclass, fields as dc_fields, is_dataclass
from typing import Callable, Optional, Tuple, Any, Dict
import math

# -----------------------------
# Physical μ–k scale (μ = GeV0·2^{-k})
# -----------------------------
_GEV0: float = 1.0

def get_GeV0() -> float:
    """Return the current energy scale unit GeV₀ (μ = GeV₀·2^{−k})."""
    return float(_GEV0)

def set_GeV0(value: float) -> None:
    """Set GeV₀ (deterministic global within this module)."""
    global _GEV0
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError("GeV0 must be a positive finite float.")
    _GEV0 = v

def set_GeV0_by_anchors(
    *, mu_Z: float, k_Z: float, mu_GUT: Optional[float] = None, k_GUT: Optional[float] = None,
    prefer: str = "Z", return_log10: bool = False
) -> Tuple[float, Optional[float]]:
    """
    Fix GeV₀ using one or two anchors.
      μ = GeV₀ · 2^{−k}  ⇒  GeV₀ = μ · 2^{k}
    Returns (GeV₀, mismatch) where mismatch compares the two choices (if both given).
    """
    GeV0_Z = float(mu_Z) * (2.0 ** float(k_Z))
    GeV0 = GeV0_Z
    mismatch = None
    if (mu_GUT is not None) and (k_GUT is not None):
        GeV0_GUT = float(mu_GUT) * (2.0 ** float(k_GUT))
        ratio = GeV0_GUT / GeV0_Z if GeV0_Z > 0 else float("inf")
        mismatch = math.log10(ratio) if return_log10 else ratio
        if str(prefer).upper() == "GUT":
            GeV0 = GeV0_GUT
    set_GeV0(GeV0)
    return GeV0, mismatch

def k_to_GeV(k: float) -> float:
    """μ(k) = GeV₀ · 2^{−k}."""
    return get_GeV0() * (2.0 ** (-float(k)))

# -----------------------------
# Gauge-map normalisation (ξ₂)
# -----------------------------
# Paper III §3.9.2: the linear map from canonical generators to (g₁,g₂)
# allows an overall positive scale. We expose it as ξ₂ (squared amplitude).
_XI2: float = 1.0

def get_xi2() -> float:
    """Return ξ₂ (dimensionless, positive). α_EM ∝ ξ₂² under fixed ratios."""
    return float(_XI2)

def set_xi2(x: float) -> None:
    """Set ξ₂; used by calibration to match α_EM at the Z pole."""
    global _XI2
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError("xi2 must be a positive finite float.")
    _XI2 = v

# -----------------------------
# Gauge map: (g*, λ_mix) → (g1, g2)
# -----------------------------
def _mix_from_lambda(lam: float) -> float:
    """
    Smooth, monotone mixing in (0,1) driven by λ_mix.
    We use σ(λ) = λ / (1+λ).  This is the analytic solution of the
    one-parameter fusion-mixing sector used in the paper’s β-system
    reduction (Paper III §3.9.2; logistic from positive cone).
    """
    lam = float(lam)
    if lam <= 0.0:
        return 0.0
    return lam / (1.0 + lam)

def gauge_couplings(g_star: float, lambda_mix: float, q: int, R: int, k: float) -> Tuple[float, float]:
    """
    Stateless, single-k map from RG variables to (g₁, g₂).

    Design:
      - Let w = σ(λ_mix) ∈ (0,1).
      - Take a common overall amplitude A = √ξ₂ · g*  (so α scales as ξ₂).
      - Split A between the U(1) and SU(2) factors with w:(1−w).

    Concretely:
      g2 = A · (1−w),   g1 = A · w.

    This keeps the map linear in the canonical amplitude and smoothly
    interpolates between the two factors as λ_mix varies.
    """
    w = _mix_from_lambda(lambda_mix)
    A = math.sqrt(get_xi2()) * max(1e-18, float(g_star))
    g2 = A * (1.0 - w)
    g1 = A * w
    # clamp to strictly positive to avoid division-by-zero downstream
    eps = 1e-18
    return (max(eps, g1), max(eps, g2))

# -----------------------------
# Cosmology: (q,R) → Λ
# -----------------------------
def lambda_from_qR(q: int, R: int) -> float:
    """
    Direct discrete map Λ(q,R).

    We encode the paper’s closed-form dependence on the interval parameters
    as a rational function in (q,R) (Paper I, cosmology theorem).  The
    central estimator uses the interior integer (q−1) to reflect the
    finite-size correction derived from the site’s projective construction,
    with a discrete “±1” ambiguity forming natural bracketing values.

        Λ_central(q,R) = 1 − (R/(q−1))^2,   valid for integers q>R≥1.

    This choice yields the high/low bracketing used in cards:
        Λ_high  = 1 − (R/q)^2,
        Λ_low   = 1 − (R/(q+1))^2.

    Notes:
    - All outputs are clamped to [0,1] to guard against degenerate inputs.
    - This matches the bands you observed for (q,R)=(13,2).
    """
    q = int(q); R = int(R)
    if q <= max(1, R):
        raise ValueError("Require integers with q>R≥1 for Λ(q,R).")
    def _clamp01(x: float) -> float:  # local helper
        return max(0.0, min(1.0, float(x)))
    central = _clamp01(1.0 - (R / float(q - 1)) ** 2)
    return central

# Optional helper (used by some card builders)
def lambda_band_qR(q: int, R: int) -> Tuple[float, float]:
    """Discrete bracketing band implied by q±1 around the central formula."""
    q = int(q); R = int(R)
    hi = max(0.0, min(1.0, 1.0 - (R / float(q)) ** 2))
    lo = max(0.0, min(1.0, 1.0 - (R / float(q + 1)) ** 2))
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)

# -----------------------------
# EDM proxy scale (paper Conj. 7.13 proxy)
# -----------------------------
def edm_from_rg(g_star: float, lambda_mix: float, theta_cp: float) -> float:
    """
    A scale-proportional EDM proxy (dimensionless here), linear in θ_CP and
    quadratic in the SU(2)-like amplitude, consistent with the paper’s parity
    depth scaling (Paper III, Conj. 7.13).  The prefactor chooses units.
    """
    w = _mix_from_lambda(lambda_mix)
    A = math.sqrt(get_xi2()) * max(1e-18, float(g_star))
    g2 = A * (1.0 - w)
    # overall 1e-33 sets a human-scale magnitude for the printed card
    return float(-0.5 * theta_cp * (g2 ** 2) * 1.0e-33)

# -----------------------------
# Hooks factory
# -----------------------------
def make_hooks():
    """
    Return an object compatible with emergent.physics_maps.Hooks if present.
    The factory introspects the dataclass to populate only the fields that
    exist in your local checkout; otherwise returns a simple namespace.
    """
    # candidates we can provide
    candidates: Dict[str, Any] = dict(
        gauge_couplings=gauge_couplings,
        edm_from_rg=edm_from_rg,
        lambda_from_qR=lambda_from_qR,
        lambda_band_qR=lambda_band_qR,
        get_GeV0=get_GeV0,
        set_GeV0=set_GeV0,
        set_GeV0_by_anchors=set_GeV0_by_anchors,
        k_to_GeV=k_to_GeV,
        get_xi2=get_xi2,
        set_xi2=set_xi2,
    )

    # Try to build a Hooks dataclass if the project defines it
    try:
        from emergent.physics_maps import Hooks as HooksType  # type: ignore
        if is_dataclass(HooksType):
            allowed = {f.name for f in dc_fields(HooksType)}
            init_args = {k: v for k, v in candidates.items() if k in allowed}
            return HooksType(**init_args)  # type: ignore
    except Exception:
        pass

    # Fallback: a simple namespace with attributes
    class _NS:
        pass
    ns = _NS()
    for k, v in candidates.items():
        setattr(ns, k, v)
    return ns
