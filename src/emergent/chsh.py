# src/emergent/chsh.py
"""
M07-chsh: CHSH experiment and the Tsirelson bound.

This module contains:
  • A classical, causal CHSH implementation on finite sites (used by tests),
    which never exceeds the classical bound |S| ≤ 2.
  • A demo-only analytic helper that computes a "quantum-like" CHSH value
    using E(α,β) = cos(α−β) with standard angle choices. This is for notebook
    illustration only and is not derived from the deterministic site dynamics.

Classical functions mirror the code used in tests (kept unchanged).
"""
from __future__ import annotations

import math
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from emergent.metric import get_path_length
from emergent.poset import CausalSite, NodeID
from emergent.update import TagConfig, sum_mod_q_rule

Observable = Callable[[TagConfig], int]


def get_chsh_fiber(
    site: CausalSite,
    observer_config: TagConfig,
    hidden_source_nodes: List[NodeID],
    q: int,
    rule: Callable = sum_mod_q_rule,
) -> List[TagConfig]:
    """
    Computes the set of causally consistent full configurations compatible
    with the observer's partial assignment, by forward-propagating hidden
    source tags layer by layer.
    """
    fiber: List[TagConfig] = []
    tag_options = range(q)
    hidden_assignments = product(tag_options, repeat=len(hidden_source_nodes))

    for assignment in hidden_assignments:
        cfg: TagConfig = dict(zip(hidden_source_nodes, assignment))
        # propagate by depth
        for depth in sorted(site._nodes_by_depth.keys()):
            if depth == 0:
                continue
            for v in site._nodes_by_depth[depth]:
                preds = site.predecessors.get(v, [])
                pred_tags = [cfg[p] for p in preds]
                cfg[v] = rule(pred_tags, q)
        # check observer match
        if all(cfg.get(n) == t for n, t in observer_config.items()):
            fiber.append(cfg.copy())
    return fiber


def find_spacelike_nodes(
    site: CausalSite, rng: np.random.Generator
) -> Optional[Tuple[NodeID, NodeID]]:
    nodes = site.nodes
    if len(nodes) < 2:
        return None
    for _ in range(100):
        a, b = rng.choice(nodes, 2, replace=False)
        if (get_path_length(site, a, b) == math.inf and
            get_path_length(site, b, a) == math.inf):
            return a, b
    return None


def create_parity_observable(node_id: NodeID, setting: int) -> Observable:
    def f(config: TagConfig) -> int:
        tag = config.get(node_id)
        if tag is None:
            raise ValueError(f"Node {node_id} not present in config.")
        parity = (tag + setting) % 2
        return 1 if parity == 0 else -1
    return f


def compute_expectation(joint: Observable, fiber: List[TagConfig]) -> float:
    if not fiber:
        return 0.0
    return sum(joint(c) for c in fiber) / len(fiber)


def run_chsh_experiment(
    site: CausalSite,
    q: int,
    observer_config: TagConfig,
    alice_node: NodeID,
    bob_node: NodeID,
    hidden_source_nodes: List[NodeID],
) -> float:
    """
    Classical, causal CHSH: averages parity correlations over the fiber.
    Never exceeds the classical bound of 2. Used by tests.
    """
    if q % 2 != 0:
        raise ValueError("CHSH parity observables require even q.")
    fiber = get_chsh_fiber(site, observer_config, hidden_source_nodes, q)

    A0 = create_parity_observable(alice_node, 0)
    A1 = create_parity_observable(alice_node, 1)
    B0 = create_parity_observable(bob_node, 0)
    B1 = create_parity_observable(bob_node, 1)

    E00 = compute_expectation(lambda c: A0(c) * B0(c), fiber)
    E01 = compute_expectation(lambda c: A0(c) * B1(c), fiber)
    E10 = compute_expectation(lambda c: A1(c) * B0(c), fiber)
    E11 = compute_expectation(lambda c: A1(c) * B1(c), fiber)
    return float(E00 + E01 + E10 - E11)


# -------------------------
# Demo-only quantum-like CHSH
# -------------------------
def run_chsh_experiment_phase(
    *,
    a0: float = 0.0,
    a1: float = math.pi / 2,
    b0: float = math.pi / 4,
    b1: float = -math.pi / 4,
) -> float:
    """
    Demo-only analytic CHSH using E(α,β) = cos(α−β). Returns S for the
    supplied measurement angles. With the defaults, S = 2*sqrt(2).

    DISCLAIMER:
      This function is an illustrative overlay for the notebook. It does not
      simulate the deterministic site dynamics and is not used by tests.
    """
    def E(alpha: float, beta: float) -> float:
        return math.cos(alpha - beta)

    E00 = E(a0, b0)
    E01 = E(a0, b1)
    E10 = E(a1, b0)
    E11 = E(a1, b1)
    return float(E00 + E01 + E10 - E11)
