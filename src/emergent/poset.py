# src/emergent/poset.py
"""
M01-poset: Core implementation of the finite, acyclic, graded causal site.

This module provides the CausalSite class, which generates and represents the
fundamental structure C_n = (V_n, ->) as described in the source papers.
The generation algorithm ensures that all standing hypotheses regarding the
structure of the site are met.

- Paper I, Axiom 2.1 (Causal Universe): The universe is a finite acyclic poset.
- Standing Hypotheses:
  H1: Finite height; H2: unit-grade arrows (depth(v)=depth(u)+1);
  H3: Finite local branching (bounded out-degree).
- Paper III, Definition 2.1 (Depth truncations): C_n as a finite prefix by depth.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Iterable

import networkx as nx
import numpy as np

NodeID = int
Edge = Tuple[NodeID, NodeID]
AdjacencyList = Dict[NodeID, List[NodeID]]


class CausalSite:
    """
    Represents a finite, acyclic, and graded causal site (poset).

    The site is a Directed Acyclic Graph (DAG) organized into layers (depths),
    where arrows only connect nodes from depth d to depth d+1.
    """

    def __init__(
        self,
        nodes_by_depth: Dict[int, List[NodeID]],
        adj: AdjacencyList,
        R: int,
    ):
        """Initialize the site with precomputed structure."""
        self._nodes_by_depth: Dict[int, List[NodeID]] = nodes_by_depth
        self._adj: AdjacencyList = adj
        self.R: int = R  # Max out-degree + 1

        # Flattened node list and per-node depths
        self._nodes: List[NodeID] = sorted(
            node for depth_nodes in nodes_by_depth.values() for node in depth_nodes
        )
        self._depths: Dict[NodeID, int] = {
            node: depth for depth, nodes in nodes_by_depth.items() for node in nodes
        }
        # Edge list
        self._edges: List[Edge] = sorted(
            (u, v) for u, succs in adj.items() for v in succs
        )
        # Predecessors
        self._preds: Dict[NodeID, List[NodeID]] = {n: [] for n in self._nodes}
        for u, succs in self._adj.items():
            for v in succs:
                self._preds[v].append(u)

        # Invariants (lightweight checks; can be disabled if needed)
        self._validate_graded()
        self._validate_outdegree()

    # -------------------------
    # Public accessors
    # -------------------------
    @property
    def nodes(self) -> List[NodeID]:
        return self._nodes

    @property
    def edges(self) -> List[Edge]:
        return self._edges

    @property
    def depths(self) -> Dict[NodeID, int]:
        return self._depths

    @property
    def adj(self) -> AdjacencyList:
        return self._adj

    @property
    def predecessors(self) -> Dict[NodeID, List[NodeID]]:
        """Mapping from node to its list of predecessors."""
        return self._preds

    @property
    def nodes_by_depth(self) -> Dict[int, List[NodeID]]:
        """Read-only view of nodes organized by depth."""
        return self._nodes_by_depth

    @property
    def depth_levels(self) -> List[int]:
        """Sorted depth indices (topological order)."""
        return sorted(self._nodes_by_depth.keys())

    # -------------------------
    # Construction
    # -------------------------
    @classmethod
    def generate(
        cls,
        *,
        n_layers: int,
        nodes_per_layer: int,
        R: int,
        edge_prob: float,
        rng: np.random.Generator,
        ensure_successor: bool = False,
    ) -> CausalSite:
        """
        Generate a random graded DAG satisfying H1–H3 by construction.

        Args:
            n_layers: number of depth layers (>=1). Depths are 0..n_layers-1.
            nodes_per_layer: number of nodes in each layer (>=1).
            R: maximal out-degree + 1 (>1). Each node has ≤ R-1 successors.
            edge_prob: independent edge probability in [0, 1].
            rng: seeded NumPy Generator.
            ensure_successor: if True, non-top-layer nodes with zero selected
                successors get 1 successor deterministically (promotes mixing).

        Returns:
            CausalSite instance with grading and out-degree bound enforced.
        """
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1.")
        if nodes_per_layer < 1:
            raise ValueError("nodes_per_layer must be >= 1.")
        if R <= 1:
            raise ValueError("R (max out-degree + 1) must be > 1.")
        if not (0.0 <= edge_prob <= 1.0):
            raise ValueError("edge_prob must be in [0, 1].")

        nodes_by_depth: Dict[int, List[NodeID]] = {}
        node_counter = 0
        for depth in range(n_layers):
            nodes_by_depth[depth] = list(
                range(node_counter, node_counter + nodes_per_layer)
            )
            node_counter += nodes_per_layer

        adj: AdjacencyList = {i: [] for i in range(node_counter)}
        max_out_degree = R - 1

        for depth in range(n_layers - 1):
            current_layer_nodes = nodes_by_depth[depth]
            next_layer_nodes = nodes_by_depth[depth + 1]

            # Cap the number of Bernoulli trials by the population size
            n_trials = min(max_out_degree, len(next_layer_nodes))

            for u in current_layer_nodes:
                if n_trials == 0:
                    continue
                # Draw #successors; cap selection size by population
                n_successors = int(rng.binomial(n_trials, edge_prob))
                if ensure_successor and n_successors == 0:
                    n_successors = 1

                if n_successors > 0:
                    chosen = rng.choice(
                        next_layer_nodes,
                        size=min(n_successors, n_trials),
                        replace=False,
                    )
                    # Ensure no duplicates; store as ints
                    adj[u].extend(int(v) for v in chosen)

        return cls(nodes_by_depth, adj, R)

    # -------------------------
    # Utilities
    # -------------------------
    def to_networkx(self) -> nx.DiGraph:
        """Convert to a networkx.DiGraph for analysis."""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        return G

    def truncate_to_depth(self, max_depth: int) -> CausalSite:
        """
        Return a new CausalSite containing only nodes up to max_depth.
        Implements the depth truncation C_n (finite prefix by depth).
        """
        new_nodes_by_depth = {
            d: nodes for d, nodes in self._nodes_by_depth.items() if d <= max_depth
        }
        truncated_nodes = {n for nodes in new_nodes_by_depth.values() for n in nodes}
        new_adj: AdjacencyList = {
            u: [v for v in succs if v in truncated_nodes]
            for u, succs in self._adj.items()
            if u in truncated_nodes
        }
        return CausalSite(new_nodes_by_depth, new_adj, self.R)

    def __repr__(self) -> str:
        return (
            f"CausalSite(nodes={len(self.nodes)}, edges={len(self.edges)}, "
            f"layers={len(self._nodes_by_depth)}, R={self.R})"
        )

    # -------------------------
    # Internal validation
    # -------------------------
    def _validate_graded(self) -> None:
        """Ensure every edge increases depth by exactly 1."""
        for u, v in self._edges:
            du = self._depths[u]
            dv = self._depths[v]
            if dv != du + 1:
                raise ValueError(
                    f"Ungraded edge {u}->{v}: depth {du} -> {dv} (expected {du+1})"
                )

    def _validate_outdegree(self) -> None:
        """Ensure out-degree ≤ R-1 for all nodes."""
        max_out = self.R - 1
        for u, succs in self._adj.items():
            if len(succs) > max_out:
                raise ValueError(
                    f"Node {u} has out-degree {len(succs)} > {max_out}"
                )
