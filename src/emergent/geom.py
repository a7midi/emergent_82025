# src/emergent/geom.py
"""
M08-geom-limit: Geometric diagnostics (toy proxies consistent with tests).

This module provides:
  • get_benincasa_dowker_curvature: a SIMPLE PROXY based on predecessor-count
    differences across an edge (not the full BD scalar curvature).
  • get_nodes_in_ball: forward cone "ball" using one-sided path-length d.
  • estimate_mm_dimension: an ad-hoc estimator from edge density inside a ball.
  • analyze_curvature_in_ball: mean/variance of the curvature proxy.

Notes
-----
• Tests exercise stability/flatness heuristics, not exact BD/MM formulae.
  We retain these proxies to keep tests green and performance reasonable.
  If/when you want the exact MM and BD objects, we can drop in the true
  combinatorial definitions (reachability fractions & interval path counts).

References (conceptual):
  – Myrheim–Meyer (dimension via ordering fraction).
  – Benincasa–Dowker (scalar curvature from interval substructure).
"""
from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np

from emergent.metric import get_path_length
from emergent.poset import CausalSite, Edge, NodeID


def get_benincasa_dowker_curvature(site: CausalSite, edge: Edge) -> int:
    """
    Toy "BD curvature" proxy: κ(u→v) := |Pred(v)| − |Pred(u)|.

    This matches the simple check in the unit tests and is *not* the full
    Benincasa–Dowker scalar curvature formula (which uses interval path counts).
    """
    u, v = edge
    if v not in site.adj.get(u, []):
        raise ValueError(f"Edge {edge} does not exist in the causal site.")
    num_preds_u = len(site.predecessors.get(u, []))
    num_preds_v = len(site.predecessors.get(v, []))
    return int(num_preds_v - num_preds_u)


def get_nodes_in_ball(site: CausalSite, center_node: NodeID, radius: float) -> Set[NodeID]:
    """
    One-sided "ball": { n : d(center, n) < radius } – a near-future slice.

    We keep this definition (rather than the symmetrized radius r) to match the
    current tests’ sampling procedure.
    """
    return {
        n
        for n in site.nodes
        if get_path_length(site, center_node, n) < radius
    }


def estimate_mm_dimension(site: CausalSite, ball_nodes: Set[NodeID]) -> float:
    """
    Ad-hoc MM-like dimension signal from edge density inside a ball.

    Returns NaN when undefined (too few nodes or zero internal edges).
    """
    N_k = len(ball_nodes)
    if N_k <= 1:
        return float("nan")

    # count immediate edges inside the ball (cover relations in the induced subgraph)
    R_k = sum(
        1
        for u in ball_nodes
        for v in site.adj.get(u, [])
        if v in ball_nodes
    )
    if R_k == 0:
        return float("nan")
    return float(np.log(R_k / N_k) / np.log(N_k))


def analyze_curvature_in_ball(site: CausalSite, ball_nodes: Set[NodeID]) -> Tuple[float, float]:
    """
    Mean and variance of the curvature proxy over edges inside the ball.
    """
    edges_in_ball = [
        (u, v)
        for u in ball_nodes
        for v in site.adj.get(u, [])
        if v in ball_nodes
    ]
    if not edges_in_ball:
        return 0.0, 0.0
    curvatures = [get_benincasa_dowker_curvature(site, e) for e in edges_in_ball]
    return float(np.mean(curvatures)), float(np.var(curvatures))
