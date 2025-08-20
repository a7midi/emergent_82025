# src/emergent/gravity_fp.py
# -*- coding: utf-8 -*-
r"""
Gravity-facing wrappers for the 4D Benincasa–Dowker curvature stencil and witnesses.

This module provides:
  • A 4D scalar-curvature estimator `bd_curvature_edge_4d` that wraps the exact
    combinatorial stencil from `geom_fp` (no placeholders, no interpolation).
  • A light-weight “block witness” helper that keeps notebooks’ “non-edge raises”
    behavior stable: non-edges are ignored, edges are mapped to interval-layer
    contributions fed to the BD stencil.

All functions are pure-Python and deterministic for fixed inputs.

References
----------
• Benincasa–Dowker 4D stencil (exact integer weights), implemented in `geom_fp`.
• Paper I v8 invariants (spectral/adjointness/measure) remain unaffected by these wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import math

from .geom_fp import (
    BD4_WEIGHTS_LAYERS,
    BD4_CURVATURE_NORMALISATION,
    bd_scalar_curvature_from_counts_4d,
)

__all__ = [
    "bd_curvature_edge_4d",
    "bd_block_witness_4d",
    "NON_EDGE_POLICY",
]

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

NON_EDGE_POLICY: str = "ignore"  # historic notebook behavior ("non-edge raises" are suppressed)


def bd_curvature_edge_4d(
    counts: Mapping[int, int],
    *,
    rho: float,
    weights: Sequence[int] | None = None,
    normalisation: float | None = None,
) -> float:
    r"""
    Paper-faithful 4D BD scalar curvature estimator at a site/edge context.

    Parameters
    ----------
    counts :
        Mapping k ↦ N_k for k=0..4 (missing keys treated as 0). See `geom_fp.bd_scalar_curvature_from_counts_4d`.
    rho :
        Effective density (or site-scale proxy) > 0, determines the normalisation scale in 4D.
    weights :
        Optional override of integer weights (defaults to exact BD4 weights).
    normalisation :
        Optional override of BD4 curvature normalisation.

    Returns
    -------
    float
        Scalar-curvature estimator value.

    Notes
    -----
    Deterministic; this is a thin wrapper to keep gravity-facing code decoupled from geometry internals.
    """
    return bd_scalar_curvature_from_counts_4d(
        counts,
        rho=rho,
        weights=weights if weights is not None else BD4_WEIGHTS_LAYERS,
        normalisation=BD4_CURVATURE_NORMALISATION if normalisation is None else normalisation,
    )


def bd_block_witness_4d(
    edge: Tuple[int, int],
    *,
    adjacency: Mapping[int, Iterable[int]],
    layer_counts_provider: Optional[callable] = None,
    rho: float = 1.0,
    non_edge_policy: str = NON_EDGE_POLICY,
) -> Dict[str, float]:
    r"""
    Compute a minimal “witness” for the BD stencil along an edge; ignore non-edges by policy.

    Parameters
    ----------
    edge :
        A directed pair (u, v).
    adjacency :
        A DAG adjacency mapping node -> iterable of successors (duplicates ignored).
    layer_counts_provider :
        Optional callable (u, v, adjacency) -> Mapping[int,int] that returns {k: N_k} for k=0..4.
        If None, a trivial zero-counts provider is used (all N_k=0), suitable for structural smoke tests.
    rho :
        Effective density/scale used by the curvature estimator (must be > 0).
    non_edge_policy :
        One of {"ignore", "raise"}. When "ignore", a non-edge (u,v) returns a witness with
        {"is_edge": 0.0, "curvature": 0.0}. When "raise", a ValueError is raised.

    Returns
    -------
    dict
        {
          "is_edge": 1.0 or 0.0,
          "curvature": float,
          "rho": float
        }

    Notes
    -----
    • This helper preserves the legacy “non-edge raises” behavior: default = "ignore".
    • Use it in notebooks to probe blocks/witnesses without disturbing determinism.
    """
    u, v = edge
    succ = set(adjacency.get(u, ()))
    if v not in succ:
        if str(non_edge_policy).lower() == "raise":
            raise ValueError(f"pair {edge} is not an edge (non-edge policy='raise')")
        # Default legacy behavior: ignore non-edges, deterministic zero contribution.
        return {"is_edge": 0.0, "curvature": 0.0, "rho": float(rho)}

    # Obtain layer counts; default provider yields all zeros (smoke test)
    if layer_counts_provider is None:
        counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    else:
        counts = dict(layer_counts_provider(u, v, adjacency))
        # Ensure well-formedness
        for k in range(5):
            counts[k] = int(counts.get(k, 0))

    curv = bd_curvature_edge_4d(counts, rho=rho)
    return {"is_edge": 1.0, "curvature": float(curv), "rho": float(rho)}
