# src/emergent/entropy_max.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .poset import CausalSite
from .spectral import get_spectral_constants

# --- Finite-size correction (FSC) hook and constants ---

# Use placeholders for a1, a2 until they are derived from Appendix H.
# These values are physically motivated: a1 from 1D chain models, a2 is a small mixing term.
_A1: float = 0.5
_A2: float = 0.05

def set_fsc_coefficients(*, a1: float, a2: float) -> None:
    """
    Load the analytically-derived finite-size-correction constants from Paper III, App. H.
    """
    global _A1, _A2
    _A1, _A2 = float(a1), float(a2)

def _finite_size_correction(R: int, site: CausalSite) -> float:
    """
    Calculates the leading-order finite-size correction f(N) ≈ a₁/N + a₂(R-1)/N.
    """
    N = max(1, len(site.nodes))
    f = (_A1 / N) + (_A2 * (R - 1) / N)
    return float(max(0.0, min(0.95, f))) # Clamp for numerical stability

# --- Scoring / search utilities ---

@dataclass
class AnnealConfig:
    # ... (rest of the class is unchanged)

def _score_single_graph(q: int, R: int, site: CausalSite) -> float:
    """
    Scores (q, R) using the principled objective from the papers:
      score = (entropy-rate proxy) * (spectral gap) * (1 - FSC)
    """
    consts = get_spectral_constants(q, R)
    l_gap = consts["l_gap"]  # = 1 - delta
    rate = np.log2(max(2.0, q))

    fsc = _finite_size_correction(R, site)
    
    return float(rate * l_gap * (1.0 - fsc))

def measure_entropy_score(
    q: int, R: int, *,
    n_layers: int = 14, nodes_per_layer: int = 32,
    edge_prob: float = 0.5, seed: int = 0,
) -> float:
    """Constructs a single random site and returns the corrected entropy score."""
    rng = np.random.default_rng(seed)
    site = CausalSite.generate(
        n_layers=n_layers, nodes_per_layer=nodes_per_layer,
        R=R, edge_prob=edge_prob, rng=rng,
    )
    return _score_single_graph(q, R, site)

# The run_entropy_annealing function remains unchanged but now uses the new _score_single_graph
# ... (rest of the file is unchanged) ...