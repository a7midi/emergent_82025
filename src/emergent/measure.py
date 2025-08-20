# src/emergent/measure.py
"""
M03-measure: Site and configuration probability measures.

Implements:
  • Depth-weighted site measure μ_n on C_n:
      μ_n({c}) = q^(-depth(c)) / Z_n
    with Z_n = sum_{v in V_n} q^(-depth(v)).
  • Configuration measure on Ω = A_h^{V_n} (tag configurations), including
    finite-cylinder probabilities:
      P[{x : x|_S = a}] = ∏_{v in S} p_v(a_v),
    which is uniform when p_v(·) ≡ 1/q.

Notes
-----
• This module stays agnostic about dynamics (T, K, L). Invariance checks live
  in M04 and beyond. Here we provide the projective / truncation structure and
  the measure-building blocks the rest of the suite relies on.
• Cylinders are essential for entropy, CHSH, and Born tests in later modules.

References:
  – Paper III, Def. 2.15 (Depth-weighted site measure) and Thm. 2.20 (Kolmogorov).
  – Paper III, §2.5 (Configuration measure & cylinders).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set

import numpy as np

from emergent.poset import CausalSite, NodeID


# -----------------------------------------------------------------------------
# Site measure μ_n on C_n
# -----------------------------------------------------------------------------
class SiteMeasure:
    """
    Depth-weighted probability measure μ_n on a finite causal site C_n.

    μ_n({c}) = q^(-depth(c)) / Z_n, with normalization Z_n computed over V_n.

    Parameters
    ----------
    site : CausalSite
        Finite graded site C_n = (V_n, ->).
    q : int
        Alphabet size (>1) used as the base for depth weighting.

    Attributes
    ----------
    site : CausalSite
    q : int
    z_n : float
        Normalization constant Z_n.
    """

    def __init__(self, site: CausalSite, q: int):
        if q <= 1:
            raise ValueError("Alphabet size q must be > 1.")
        self.site = site
        self.q = q

        depths = np.array([site.depths[n] for n in site.nodes], dtype=np.int64)
        weights = np.power(float(q), -depths, dtype=float)
        z = float(np.sum(weights))
        if not np.isfinite(z) or z <= 0.0:
            raise ValueError("Normalization constant Z_n must be positive and finite.")
        self.z_n: float = z

        self._pmf_vec = (weights / self.z_n).astype(float)
        # map node -> probability (node order per site.nodes)
        self._probs: Dict[NodeID, float] = {
            node: float(p) for node, p in zip(site.nodes, self._pmf_vec)
        }

    # -- basic API -------------------------------------------------------------
    def prob(self, node: NodeID) -> float:
        """Return μ_n({node})."""
        return self._probs.get(node, 0.0)

    def prob_set(self, nodes: Set[NodeID]) -> float:
        """Return μ_n(nodes) for a subset of V_n."""
        return float(sum(self.prob(n) for n in nodes))

    def pmf_vector(self) -> np.ndarray:
        """Return the probabilities aligned with `site.nodes` order."""
        return self._pmf_vec.copy()

    # -- truncation / projective checks ---------------------------------------
    def prob_of_truncation(self, max_depth: int) -> float:
        """
        Return μ_n(V_{≤max_depth}) where V_{≤k} is all nodes with depth ≤ k.
        Useful for conditioning identities in Kolmogorov checks.
        """
        allowed = {n for n in self.site.nodes if self.site.depths[n] <= max_depth}
        return self.prob_set(allowed)

    def conditional_prob_in_full_on_subset(
        self, nodes_subset: Set[NodeID], max_depth: int
    ) -> float:
        """
        For a SiteMeasure on a full site C_n, compute
            μ_{n+1}(nodes_subset) / μ_{n+1}(V_{≤max_depth}),
        which should equal μ_n(nodes_subset) when nodes_subset ⊆ V_{≤max_depth}.
        """
        mass_subset = self.prob_set(nodes_subset)
        mass_trunc = self.prob_of_truncation(max_depth)
        if mass_trunc <= 0:
            raise ValueError("Truncation has zero mass under full μ_n.")
        return float(mass_subset / mass_trunc)

    def __repr__(self) -> str:
        return f"SiteMeasure(nodes={len(self.site.nodes)}, q={self.q}, Z_n={self.z_n:.6f})"


# -----------------------------------------------------------------------------
# Configuration measure on Ω = A_h^{V_n} with cylinder sets
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Cylinder:
    """
    A cylinder event on tag space Ω: a finite assignment of tags on a subset S.
    Interpreted as { x in Ω : x|_S = a }.

    Attributes
    ----------
    assignments : Dict[NodeID, int]
        Mapping node -> tag value (0..q-1).
    """
    assignments: Mapping[NodeID, int]

    def support(self) -> Set[NodeID]:
        return set(self.assignments.keys())

    def restricted_to(self, allowed_nodes: Iterable[NodeID]) -> "Cylinder":
        """
        Restrict the cylinder to allowed nodes (e.g., a truncation).
        """
        allowed = set(allowed_nodes)
        sub = {v: t for v, t in self.assignments.items() if v in allowed}
        return Cylinder(sub)


class ConfigurationMeasure:
    """
    Product configuration measure on Ω = A_h^{V_n}, with per-node categorical
    distributions p_v over tags {0, ..., q-1}. If `tag_probs` is omitted, the
    measure is the uniform product: p_v(a) = 1/q.

    Parameters
    ----------
    site : CausalSite
    q : int
        Alphabet size (>1).
    tag_probs : Optional[Mapping[NodeID, Sequence[float]]]
        Optional node-wise categorical distributions of length q summing to 1.
    """

    def __init__(
        self,
        site: CausalSite,
        q: int,
        tag_probs: Optional[Mapping[NodeID, Sequence[float]]] = None,
    ):
        if q <= 1:
            raise ValueError("Alphabet size q must be > 1.")
        self.site = site
        self.q = q
        self.n_sites = len(site.nodes)

        if tag_probs is None:
            # uniform p_v(·) = 1/q for all v
            self._p = {v: np.full(q, 1.0 / q, dtype=float) for v in site.nodes}
        else:
            # validate and copy to arrays
            _p: Dict[NodeID, np.ndarray] = {}
            for v in site.nodes:
                pv = np.asarray(tag_probs.get(v, None), dtype=float)
                if pv.size != q:
                    raise ValueError(f"tag_probs[{v}] must have length q.")
                if np.any(pv < 0):
                    raise ValueError("tag probabilities must be nonnegative.")
                s = float(np.sum(pv))
                if not np.isfinite(s) or s <= 0.0:
                    raise ValueError("tag probabilities must sum to a positive value.")
                _p[v] = pv / s
            self._p = _p

    # -- uniform configuration stats ------------------------------------------
    def log_prob_uniform_configuration(self) -> float:
        """
        Return log P[{specific configuration}] under uniform product measure.
        Equals -n_sites * log(q).
        """
        return -self.n_sites * float(np.log(self.q))

    # -- cylinders -------------------------------------------------------------
    def prob_cylinder(self, cyl: Cylinder) -> float:
        """
        Probability of a cylinder {x : x|_S = a} under the product measure.

        For uniform case, this reduces to (1/q)^{|S|}.
        """
        prob = 1.0
        for v, tag in cyl.assignments.items():
            if v not in self._p:
                # node outside the site: event has probability 0
                return 0.0
            pv = self._p[v]
            if tag < 0 or tag >= self.q:
                return 0.0
            prob *= float(pv[tag])
            if prob == 0.0:
                break
        return prob

    def prob_cylinder_on_truncation(
        self, cyl: Cylinder, truncated_site: CausalSite
    ) -> float:
        """
        Probability of the cylinder restricted to a truncation. Because the
        measure is product (and independent across sites), the probability is
        simply the product over the specified nodes that survive the truncation.
        """
        allowed = set(truncated_site.nodes)
        restricted = cyl.restricted_to(allowed)
        return self.prob_cylinder(restricted)

    # -- helpers ---------------------------------------------------------------
    def node_probs(self, node: NodeID) -> np.ndarray:
        """Return the node-wise probability vector p_v(·)."""
        return self._p[node].copy()
