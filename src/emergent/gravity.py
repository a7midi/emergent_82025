# src/emergent/gravity.py
"""
M11-gravity: Curvature–memory coupling and a block Einstein witness.

Implements:
  • κ(u→v) := |Pred(v)| − |Pred(u)|   (discrete curvature proxy)
  • ρ_mem(u→v) := |Pred(v)|^2 − |Pred(u)|^2   (memory density)
  • g(u→v) := κ/ρ_mem where defined
  • analyze_gravity_in_block: block-averaged coupling g over a region

Matches the tests' proxy definitions and convergence checks.
Refs: Paper II App. E (numerical proxy); Paper III §3.8 (block convergence).
"""
from __future__ import annotations

from typing import Dict, Set, Tuple

import numpy as np

from emergent.geom import get_benincasa_dowker_curvature
from emergent.poset import CausalSite, Edge, NodeID


def get_memory_density_edge(site: CausalSite, edge: Edge) -> int:
    """ρ_mem for an edge (difference of squared predecessor counts)."""
    u, v = edge
    if v not in site.adj.get(u, []):
        raise ValueError(f"Edge {edge} does not exist in the causal site.")
    preds_u = site.predecessors.get(u, [])
    preds_v = site.predecessors.get(v, [])
    return len(preds_v) ** 2 - len(preds_u) ** 2


def get_local_coupling_g(site: CausalSite, edge: Edge) -> float:
    """Local coupling g = κ/ρ_mem (NaN if ρ_mem == 0)."""
    kappa = get_benincasa_dowker_curvature(site, edge)
    rho_mem = get_memory_density_edge(site, edge)
    return np.nan if rho_mem == 0 else float(kappa) / float(rho_mem)


def analyze_gravity_in_block(site: CausalSite, block_of_nodes: Set[NodeID]) -> Dict[str, float]:
    """Average g over edges entirely inside a node set."""
    edges_in_block = [
        (u, v)
        for u in block_of_nodes
        for v in site.adj.get(u, [])
        if v in block_of_nodes
    ]
    if not edges_in_block:
        return {"avg_g": np.nan, "num_edges": 0}

    g_vals = [get_local_coupling_g(site, e) for e in edges_in_block]
    g_vals = [g for g in g_vals if not np.isnan(g)]
    if not g_vals:
        return {"avg_g": np.nan, "num_edges": len(edges_in_block)}
    return {"avg_g": float(np.mean(g_vals)), "num_edges": len(edges_in_block)}
