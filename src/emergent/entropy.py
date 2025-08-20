# src/emergent/entropy.py
"""
M06-entropy: Observer entropy and the deterministic Second Law.

Implements observer entropy on finite causal sites and provides two pathways:
(1) a direct (enumerative) calculation of the static fiber entropy used by
    tests and small demos; and
(2) a safe/analytic path for large hidden slices to avoid combinatorial blow-up
    in demos (the measured value is constant in time for this model).

Also provides a theoretical predictor for linear entropy growth using the
minimal hidden layer as the per-tick source, matching the papers' identity
in the continuum form.

References:
- Paper I, Def. 8.19 (Observer entropy): S_t = log2 |fiber|.
- Paper I, Prop. 8.23 (Deterministic Second Law).
- Paper II, Thm. 8.9 (Sharp balance).
- Paper III, Thm. 2.40 (Entropy–density identity).
"""
from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

from emergent.poset import CausalSite, NodeID
from emergent.update import TagConfig, UpdateRule, deterministic_update, sum_mod_q_rule


def get_microstate_fiber(
    site: CausalSite, observer_config: TagConfig, q: int
) -> List[TagConfig]:
    """
    Enumerates the set of full configurations consistent with `observer_config`.

    WARNING: This is exponential in the number of hidden nodes (#H).
    Use only for small #H (e.g., <= 12). For large demos, prefer the
    analytic formulas (see run_entropy_simulation's safe path).

    Args:
        site: The CausalSite.
        observer_config: Partial map of NodeID -> tag visible to the observer.
        q: Alphabet size for hidden nodes.

    Returns:
        A list of full TagConfig dictionaries (the fiber).
    """
    visible_nodes = set(observer_config.keys())
    hidden_nodes = sorted(list(set(site.nodes) - visible_nodes))

    if not hidden_nodes:
        return [observer_config]

    tag_options = range(q)
    hidden_assignments = product(tag_options, repeat=len(hidden_nodes))
    fiber = []
    for assignment in hidden_assignments:
        full_config = observer_config.copy()
        hidden_config = dict(zip(hidden_nodes, assignment))
        full_config.update(hidden_config)
        fiber.append(full_config)
    return fiber


def shannon_entropy(microstates: List[TagConfig]) -> float:
    """
    Shannon entropy for a deterministic fiber: S = log2(|fiber|).
    """
    n = len(microstates)
    return 0.0 if n == 0 else float(np.log2(n))


def get_minimal_hidden_layer(site: CausalSite, observer_nodes: Set[NodeID]) -> List[NodeID]:
    """
    Returns the first (smallest depth) layer of nodes not visible to the observer.
    """
    hidden_nodes = set(site.nodes) - observer_nodes
    if not hidden_nodes:
        return []
    min_hidden_depth = min(site.depths[n] for n in hidden_nodes)
    return sorted([n for n in hidden_nodes if site.depths[n] == min_hidden_depth])


def predict_entropy_trajectory(
    site: CausalSite,
    q: int,
    observer_depth: int,
    num_ticks: int,
) -> Tuple[List[float], List[float]]:
    """
    Predicts a linear entropy trajectory using the minimal hidden layer H_min
    as the per-tick source (continuum identity surrogate).

    S_0 = |H_all| * log2(q)        (static uncertainty about all hidden nodes)
    ΔS  = |H_min| * log2(q)        (new information influx per tick)

    Returns:
        (entropies, increments), both lists of length num_ticks
        (entropies correspond to ticks t=0..num_ticks-1).
    """
    observer_nodes = {n for n in site.nodes if site.depths[n] < observer_depth}
    hidden_nodes = [n for n in site.nodes if site.depths[n] >= observer_depth]
    H_all = len(hidden_nodes)
    H_min = len(get_minimal_hidden_layer(site, observer_nodes))

    base = H_all * float(np.log2(q))
    slope = H_min * float(np.log2(q))
    entropies = [base + t * slope for t in range(num_ticks)]
    increments = [slope for _ in range(max(0, num_ticks - 1))]
    return entropies, increments


def run_entropy_simulation(
    site: CausalSite,
    q: int,
    initial_config: TagConfig,
    observer_depth: int,
    num_ticks: int,
    rule: UpdateRule = sum_mod_q_rule,
    *,
    max_enumerable_hidden: int = 12,
) -> Tuple[List[float], List[float]]:
    """
    Computes the measured static-fiber entropy along a deterministic trajectory.
    For small hidden slices (#H <= max_enumerable_hidden), this enumerates the
    fiber exactly at each tick. For large hidden slices, it switches to a safe,
    analytic path that returns the constant static entropy S = |H| log2(q).

    Args:
        site, q, initial_config, observer_depth, num_ticks, rule: standard.
        max_enumerable_hidden: threshold controlling when we avoid enumeration.

    Returns:
        (entropies, increments), lists of length num_ticks and num_ticks-1.

    Notes:
        - In this simplified finite model, the *measured* entropy along a fixed
          observer slice is constant over ticks. The *growth law* concerns the
          influx from the minimal hidden layer, for which we provide
          `predict_entropy_trajectory` to visualize the theory.
    """
    current_config = initial_config.copy()
    observer_nodes = {n for n in site.nodes if site.depths[n] < observer_depth}
    hidden_nodes = [n for n in site.nodes if site.depths[n] >= observer_depth]
    H = len(hidden_nodes)

    # SAFE PATH: large hidden set ⇒ use analytic constant value
    if H > max_enumerable_hidden:
        static_S = H * float(np.log2(q))
        entropies = [static_S for _ in range(num_ticks)]
        increments = [0.0 for _ in range(max(0, num_ticks - 1))]
        return entropies, increments

    # EXACT PATH: small hidden set ⇒ enumerate the fiber
    entropies: List[float] = []
    for _ in range(num_ticks):
        observer_config: TagConfig = {n: current_config[n] for n in observer_nodes}
        fiber = get_microstate_fiber(site, observer_config, q)
        entropies.append(shannon_entropy(fiber))
        current_config = deterministic_update(site, current_config, q, rule)

    increments = [entropies[i + 1] - entropies[i] for i in range(len(entropies) - 1)]
    return entropies, increments
