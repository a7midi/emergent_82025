# src/emergent/metric.py
"""
M02-metric: Geometric structures on the causal site.

Implements:
  • Shortest-path distance d(u,v) on the graded DAG (Paper II, Def. 4.1).
  • Symmetrized radius r(u,v) = max{ d(u,v), d(v,u) } (Paper III, Remark 2.13),
    which is a true metric used for numerics and diagnostics.
  • Alexandrov future/past cones and intervals I^+(·), I^-(·), I(p,q).

NOTES
-----
• This module does NOT implement the depth-scaled quasi-metric d_∞; we use r
  for numerics as recommended in the papers and your plan.
• We rely on the site’s precomputed predecessor map to avoid O(|V||E|) scans.
• We add a small in-module cache for forward BFS distances keyed by
  (id(site), source). Sites are immutable post-construction, so this is safe.

References:
  – Paper II, Definition 4.1 (Path distance).
  – Paper III, Remark 2.13 (Symmetrised radius r and metric properties).
  – Paper III, Lemma 2.11 / Theorem 2.12 (Alexandrov topology & balls).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from emergent.poset import CausalSite, NodeID

__all__ = [
    "get_path_length",
    "get_symmetrized_radius",
    "get_future_cone",
    "get_past_cone",
    "get_alexandrov_interval",
    "get_alexandrov_ball_r",
    "clear_metric_caches",
]

# -----------------------------------------------------------------------------
# Internal caches (site-identity keyed). These are intentionally simple to keep
# determinism obvious and avoid surprises from functools.lru_cache on objects.
# -----------------------------------------------------------------------------
# Forward shortest paths: key = (id(site), source)
_FORWARD_DIST_CACHE: Dict[Tuple[int, NodeID], Dict[NodeID, int]] = {}


def clear_metric_caches(site: Optional[CausalSite] = None) -> None:
    """
    Clear cached forward distance maps.

    Args:
        site: if provided, only clear entries for this site; otherwise clear all.
    """
    if site is None:
        _FORWARD_DIST_CACHE.clear()
        return
    sid = id(site)
    keys_to_del = [k for k in _FORWARD_DIST_CACHE.keys() if k[0] == sid]
    for k in keys_to_del:
        _FORWARD_DIST_CACHE.pop(k, None)


# -----------------------------------------------------------------------------
# Core BFS helpers
# -----------------------------------------------------------------------------
def _forward_distances(site: CausalSite, source: NodeID) -> Dict[NodeID, int]:
    """
    Compute (and cache) forward shortest-path distances from `source` using BFS.
    Returns a dict mapping reachable node -> distance (in edges).
    """
    key = (id(site), source)
    cached = _FORWARD_DIST_CACHE.get(key)
    if cached is not None:
        return cached

    if source not in site.nodes:
        _FORWARD_DIST_CACHE[key] = {}
        return _FORWARD_DIST_CACHE[key]

    distances: Dict[NodeID, int] = {source: 0}
    q: deque[Tuple[NodeID, int]] = deque([(source, 0)])

    while q:
        u, d = q.popleft()
        for v in site.adj.get(u, []):
            if v not in distances:
                distances[v] = d + 1
                q.append((v, d + 1))

    _FORWARD_DIST_CACHE[key] = distances
    return distances


def _backward_distances(site: CausalSite, target: NodeID) -> Dict[NodeID, int]:
    """
    Compute backward (reverse-edge) shortest-path distances to `target` using BFS
    over the site.predecessors map. Returns pred node -> distance to `target`.
    """
    if target not in site.nodes:
        return {}

    distances: Dict[NodeID, int] = {target: 0}
    q: deque[Tuple[NodeID, int]] = deque([(target, 0)])

    # Use precomputed predecessor lists: O(|E|) total over the traversal.
    preds = site.predecessors
    while q:
        v, d = q.popleft()
        for p in preds.get(v, []):
            if p not in distances:
                distances[p] = d + 1
                q.append((p, d + 1))
    return distances


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def get_path_length(site: CausalSite, u: NodeID, v: NodeID) -> float:
    """
    Shortest-path distance d(u,v) on the DAG (Paper II, Def. 4.1).

    Returns:
        0.0 if u==v;
        a nonnegative integer (as float) if v is reachable from u;
        math.inf if no directed path u→…→v exists.
    """
    if u == v:
        return 0.0
    dist_map = _forward_distances(site, u)
    d = dist_map.get(v)
    return float(d) if d is not None else math.inf


def get_symmetrized_radius(site: CausalSite, u: NodeID, v: NodeID) -> float:
    """
    Symmetrized radius r(u,v) = max{ d(u,v), d(v,u) } (Paper III, Remark 2.13).

    Properties:
      • r is a true metric on nodes (nonnegativity, identity, symmetry, triangle).
      • r may be math.inf if u and v are causally unrelated in both directions.
    """
    d_uv = get_path_length(site, u, v)
    d_vu = get_path_length(site, v, u)
    return max(d_uv, d_vu)


def get_future_cone(site: CausalSite, u: NodeID) -> Set[NodeID]:
    """
    Alexandrov future cone I^+(u): all nodes reachable from u (including u).
    """
    return set(_forward_distances(site, u).keys())


def get_past_cone(site: CausalSite, v: NodeID) -> Set[NodeID]:
    """
    Alexandrov past cone I^-(v): all nodes that can reach v (including v).

    Uses the site's precomputed predecessor map for O(|E|) traversal rather than
    scanning adjacency.
    """
    return set(_backward_distances(site, v).keys())


def get_alexandrov_interval(site: CausalSite, p: NodeID, q: NodeID) -> Set[NodeID]:
    """
    Alexandrov interval I(p,q) = I^+(p) ∩ I^-(q).
    Includes endpoints if p→…→q or p==q.
    """
    return get_future_cone(site, p).intersection(get_past_cone(site, q))


def get_alexandrov_ball_r(site: CausalSite, center: NodeID, radius: float) -> Set[NodeID]:
    """
    r-ball B_r(center, radius) = { v : r(center, v) <= radius }.

    Notes:
      • radius is interpreted as an integer edge radius; non-integers are allowed
        but the useful regime is integer radii.
      • Returns an empty set if center is not in site.
    """
    if center not in site.nodes:
        return set()
    out: Set[NodeID] = set()
    for v in site.nodes:
        if get_symmetrized_radius(site, center, v) <= radius:
            out.add(v)
    return out
