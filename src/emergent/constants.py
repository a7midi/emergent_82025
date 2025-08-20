# src/emergent/constants.py
"""
Causal-site constants and derived analytic knobs (M00).

Implements the depth-one contraction δ(q,R), spectral gap γ(q,R) = 1 - δ,
and a log–Sobolev constant χ(q,R) = γ/(2R). These are the exact constants
used throughout the numerical suite (entropy law, spectral gap, curvature tails).

Equations (from Paper III):
    δ = (q - 1) / (q + R - 1)                         (Eq. (3), §2.4.1)
    γ = 1 - δ = R / (q + R - 1)                       (Cor. 2.26)
    χ = γ / (2R)                                      (Lemma 3.40 / Cor. 2.26)

Constraints:
    q ∈ ℕ, q ≥ 2  (hidden alphabet size, |A_h|)
    R ∈ ℕ, R ≥ 2  (max out-degree + 1)
    Standing hypothesis requires q > R.

References: Paper III §§2.4, 3.8 (explicit constants as above).
:contentReference[oaicite:1]{index=1}
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Defaults chosen to satisfy the standing constraint q > R (Paper I/III).
DEFAULT_Q: int = 5
DEFAULT_R: int = 3
# Deterministic seed used unless user overrides (no reliance on OS entropy).
DEFAULT_SEED: int = 20250809


def validate_qR(q: int, R: int) -> None:
    """
    Validate (q, R) against the standing hypotheses.

    Parameters
    ----------
    q : int
        Hidden alphabet size, must be >= 2 and strictly larger than R.
    R : int
        1 + (max out-degree), must be >= 2.

    Raises
    ------
    ValueError
        If constraints are violated.
    """
    if q < 2:
        raise ValueError(f"q must be >= 2, got {q}")
    if R < 2:
        raise ValueError(f"R must be >= 2, got {R}")
    if not (q > R):
        raise ValueError(f"Standing hypothesis requires q > R (got q={q}, R={R})")


def delta(q: int, R: int) -> float:
    """
    Depth-one total-variation contraction constant δ.

    δ(q, R) = (q - 1) / (q + R - 1)

    See Eq. (3) in §2.4.1 of Paper III.
    :contentReference[oaicite:2]{index=2}
    """
    validate_qR(q, R)
    return (q - 1.0) / (q + R - 1.0)


def gamma(q: int, R: int) -> float:
    """
    Spectral gap of the depth-one transfer (Koopman adjoint) on cylinders.

    γ(q, R) = 1 - δ(q, R) = R / (q + R - 1)

    See Corollary 2.26 in Paper III.
    :contentReference[oaicite:3]{index=3}
    """
    d = delta(q, R)
    return 1.0 - d


def chi(q: int, R: int) -> float:
    """
    Log–Sobolev constant for R-local graphs (Bakry–Émery bound).

    χ(q, R) = γ(q, R) / (2R)

    See Lemma 3.40 (with γ from Cor. 2.26) in Paper III.
    :contentReference[oaicite:4]{index=4}
    """
    g = gamma(q, R)
    return g / (2.0 * R)


@dataclass(frozen=True)
class FloatDTypes:
    """Central float dtypes to ensure shape- and type-stability."""
    prob: np.dtype = np.float64
    metric: np.dtype = np.float64
    scalar: np.dtype = np.float64


@dataclass(frozen=True)
class IntDTypes:
    """Central int dtypes."""
    index: np.dtype = np.int64
    depth: np.dtype = np.int64
    tag: np.dtype = np.int64


FLOAT_DTYPES = FloatDTypes()
INT_DTYPES = IntDTypes()
