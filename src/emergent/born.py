# src/emergent/born.py
"""
M09-quantum-born: Finite-Ω Hilbert model and the Born rule on cylinders.

Implements:
  • get_configuration_space: enumerate Ω = A_h^{V_n} (for small sites).
  • get_cyclic_born_vector: uniform superposition |Ψ⟩ (|Ψ(a)|^2 = 1/N).
  • get_cylinder_event_mask: indicator of a cylinder E ⊂ Ω.
  • verify_born_rule: checks ⟨Ψ|P_E|Ψ⟩ = μ(E) (uniform μ on Ω).

This is the minimal, finite setting used by the current tests. It matches the
cylinder/Born statements in the plan, with uniform μ on configurations.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Tuple

import numpy as np

from emergent.poset import CausalSite, NodeID
from emergent.update import TagConfig


def get_configuration_space(
    site: CausalSite, q: int
) -> Tuple[List[TagConfig], Dict[Tuple[Tuple[NodeID, int], ...], int]]:
    """Enumerate Ω and return (configs_list, hashable->index map)."""
    nodes = site.nodes
    configs: List[TagConfig] = [
        dict(zip(nodes, tags)) for tags in product(range(q), repeat=len(nodes))
    ]
    cfg2idx = {tuple(sorted(c.items())): i for i, c in enumerate(configs)}
    return configs, cfg2idx


def get_cyclic_born_vector(n_configs: int) -> np.ndarray:
    """Uniform |Ψ⟩ with amplitudes 1/√N (returns empty vector for N=0)."""
    if n_configs <= 0:
        return np.array([], dtype=np.complex128)
    val = 1.0 / np.sqrt(n_configs)
    return np.full(n_configs, val, dtype=np.complex128)


def get_cylinder_event_mask(
    cylinder_def: Dict[NodeID, int],
    configs_list: List[TagConfig],
) -> np.ndarray:
    """Boolean mask for the cylinder E = {x : x|_S = a}."""
    mask = np.ones(len(configs_list), dtype=bool)
    for i, cfg in enumerate(configs_list):
        for n, tag in cylinder_def.items():
            if cfg[n] != tag:
                mask[i] = False
                break
    return mask


def verify_born_rule(psi: np.ndarray, event_mask: np.ndarray, uniform_prob: float) -> bool:
    """Check ⟨Ψ|P_E|Ψ⟩ == μ(E) under the uniform configuration measure."""
    quantum_prob = float(np.sum(np.abs(psi[event_mask]) ** 2))
    return bool(np.isclose(quantum_prob, float(uniform_prob)))
