"""
Entropy-MC (Phase A, corrected): Monte-Carlo estimator for dynamic observer entropy.

This MC now estimates the size of the consistent hidden microstate fiber by:
  • Maintaining an ensemble of hidden assignments at t=0,
  • Evolving each sample deterministically,
  • Conditioning on the observer slice at each tick,
  • Estimating |Fiber_t| ≈ q^{|H|} * (consistent / samples).

This replaces the prior behavior that (incorrectly) returned log2(samples).

References:
  - Papers I–III (observer entropy via cylinder/fiber).
  - Suite: M01 (poset), M04 (update), M06 (entropy).
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple
import numpy as np

from emergent.poset import CausalSite, NodeID
from emergent.update import TagConfig, deterministic_update, sum_mod_q_rule


def _split_visible_hidden(site: CausalSite, observer_depth: int) -> Tuple[Set[NodeID], Set[NodeID]]:
    visible = {n for n, d in site.depths.items() if d < observer_depth}
    hidden = set(site.nodes) - visible
    return visible, hidden


def _draw_hidden_samples(hidden_nodes: List[NodeID], q: int, samples: int, rng: np.random.Generator) -> List[TagConfig]:
    draws = rng.integers(low=0, high=q, size=(samples, len(hidden_nodes)))
    return [dict(zip(hidden_nodes, row.tolist())) for row in draws]


def run_entropy_monte_carlo(
    site: CausalSite,
    q: int,
    observer_depth: int,
    ticks: int,
    samples: int = 512,
    seed: int = 0,
) -> Tuple[List[float], List[float]]:
    """
    Monte-Carlo dynamic observer entropy.

    Args:
      site: CausalSite
      q: alphabet size
      observer_depth: nodes with depth < observer_depth are visible
      ticks: number of updates to simulate
      samples: Monte-Carlo ensemble size
      seed: RNG seed

    Returns:
      (S_t list, ΔS_t list), where S_t estimates log2(|Fiber_t|).
    """
    rng = np.random.default_rng(seed)
    visible, hidden = _split_visible_hidden(site, observer_depth)
    hidden_nodes = sorted(list(hidden))

    # Ground-truth full history (to define the observed slices each tick)
    gt = [{n: int(rng.integers(q)) for n in site.nodes}]
    for _ in range(ticks):
        gt.append(deterministic_update(site, gt[-1], q, sum_mod_q_rule))

    # Hidden ensemble at t=0 (unconstrained)
    H = _draw_hidden_samples(hidden_nodes, q, samples, rng)

    # Helper: compute S_t from acceptance rate at tick t
    def estimate_S(accept_rate: float) -> float:
        # |H_total| = q^{|hidden|}
        if accept_rate <= 0.0:
            return 0.0  # empty fiber ⇒ S=0 (finite-ensemble floor)
        return float(len(hidden_nodes) * np.log2(q) + np.log2(accept_rate))

    S_series: List[float] = []

    # Process ticks; at each step we condition on observed slice and evolve
    for t in range(ticks):
        obs_slice_t = {n: gt[t][n] for n in visible}
        obs_slice_t1 = {n: gt[t + 1][n] for n in visible}

        # Acceptance step: keep only those hidden samples that produce the correct obs at t+1
        kept_hidden_next: List[TagConfig] = []
        consistent = 0
        for h in H:
            full_t = obs_slice_t | h
            full_t1 = deterministic_update(site, full_t, q, sum_mod_q_rule)
            # Check consistency at t+1 on *visible* nodes
            ok = all(full_t1[n] == obs_slice_t1[n] for n in visible)
            if ok:
                consistent += 1
                # Store the *hidden* part at t+1
                kept_hidden_next.append({n: full_t1[n] for n in hidden_nodes})

        accept_rate = consistent / max(1, len(H))
        S_series.append(estimate_S(accept_rate))

        # Resample to keep ensemble size stable (multinomial with replacement)
        if kept_hidden_next:
            idx = rng.integers(0, len(kept_hidden_next), size=samples)
            H = [kept_hidden_next[i] for i in idx]
        else:
            # Degenerate fiber: restart from fresh hidden draws (rare on small sites)
            H = _draw_hidden_samples(hidden_nodes, q, samples, rng)

    dS = [S_series[i+1] - S_series[i] for i in range(len(S_series)-1)]
    return S_series, dS
