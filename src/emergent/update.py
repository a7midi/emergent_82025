# src/emergent/update.py
"""
M04-update: Deterministic evolution operators T, K, and L.

Implements the core one-tick update T on tag configurations, plus small
linear-algebra helpers for the Koopman K and Transfer L operators.

Also exposes a deterministic hidden-drive for depth-0 nodes, and an optional
phase-based evolution used by the CHSH demo’s interference variant.

References:
- Paper I, Def. 1.2 (Tag-fusion update T)
- Paper II, Def. 7.3 (Koopman K f = f ∘ T)
- Paper III, Def. 2.31 (Transfer / Perron–Frobenius L = K*)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Iterable
import numpy as np

from emergent.poset import CausalSite, NodeID

# --------------------------
# Types
# --------------------------
Tag = int
TagConfig = Dict[NodeID, Tag]
UpdateRule = Callable[[List[Tag], int], Tag]

Phase = complex
PhaseConfig = Dict[NodeID, Phase]
PhaseUpdateRule = Callable[[List[Phase], int, int], Phase]

__all__ = [
    # tags
    "Tag", "TagConfig", "UpdateRule", "sum_mod_q_rule", "deterministic_update",
    "hidden_drive",
    # operators on the finite configuration space
    "build_update_permutation", "koopman_apply_vector", "transfer_apply_vector",
    # optional phase evolution (used by notebook demo)
    "Phase", "PhaseConfig", "PhaseUpdateRule",
    "phase_fusion_rule", "deterministic_update_phase",
]


# --------------------------
# Utilities
# --------------------------
def _stable_hash_64(*ints: int) -> int:
    """
    Small 64-bit LCG-style mixer in pure Python (no NumPy) to avoid overflow warnings.
    Deterministic across platforms (we mask to 64 bits).
    """
    a = 6364136223846793005
    c = 1442695040888963407
    x = 0xCBF29CE484222325
    mask = (1 << 64) - 1
    for k in ints:
        x = (a * (x ^ (int(k) & mask)) + c) & mask
    return x


# --------------------------
# Tag evolution (default)
# --------------------------
def sum_mod_q_rule(predecessor_tags: List[Tag], q: int) -> Tag:
    """
    Commutative tag-fusion rule: sum of predecessor tags modulo q.
    Depth-0 nodes (no predecessors): default tag 0.
    """
    if not predecessor_tags:
        return 0
    return int(sum(predecessor_tags) % q)


def deterministic_update(
    site: CausalSite,
    config: TagConfig,
    q: int,
    rule: UpdateRule = sum_mod_q_rule,
) -> TagConfig:
    """
    Applies one tick of the deterministic tag-fusion update T.

    Nodes are processed in increasing depth to respect causality.
    """
    new_config: TagConfig = {}
    for depth in sorted(site._nodes_by_depth.keys()):
        for v in site._nodes_by_depth[depth]:
            preds = site.predecessors.get(v, [])
            pred_tags = [config[p] for p in preds]
            new_config[v] = rule(pred_tags, q)
    return new_config


def hidden_drive(site: CausalSite, t: int, q: int, seed: int) -> TagConfig:
    """
    Deterministically assigns tags to the depth-0 (source) layer for tick t.

    Signature kept to match earlier tests:
        hidden_drive(site, t, q, seed)

    Each depth-0 node v gets: tag[v] = H(seed, t, v) mod q
    """
    if q < 1:
        raise ValueError("Alphabet size q must be >= 1.")
    cfg: TagConfig = {}
    for v in site._nodes_by_depth.get(0, []):
        cfg[v] = _stable_hash_64(seed, t, v) % max(q, 1)
    return cfg


# --------------------------
# Finite-state operator helpers (K and L on vectors)
# --------------------------
def _enumerate_configurations(nodes: List[NodeID], q: int) -> List[TagConfig]:
    """Enumerate all tag configurations over ordered `nodes` with alphabet size q."""
    n = len(nodes)
    n_configs = int(q ** n)
    configs: List[TagConfig] = []
    for idx in range(n_configs):
        rem = idx
        tags = [0] * n
        for pos in range(n - 1, -1, -1):
            tags[pos] = rem % q
            rem //= q
        configs.append(dict(zip(nodes, tags)))
    return configs


def build_update_permutation(site: CausalSite, q: int) -> np.ndarray:
    """
    Build the permutation-like map π on the finite configuration space Ω_q^|V|,
    where π[i] = j if configuration i maps to configuration j under T.

    NOTE: T is generally many-to-one; π maps indices forward, while the
    preimage sets {i : π[i] = j} are used for L via summation.
    """
    nodes = site.nodes
    configs = _enumerate_configurations(nodes, q)
    to_idx = {tuple(sorted(c.items())): i for i, c in enumerate(configs)}

    pi = np.zeros(len(configs), dtype=int)
    for i, cfg in enumerate(configs):
        nxt = deterministic_update(site, cfg, q)
        pi[i] = to_idx[tuple(sorted(nxt.items()))]
    return pi


def koopman_apply_vector(f: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """
    Apply the Koopman operator K to a function-vector f on configurations:
      (K f)[i] = f[π[i]]
    """
    return f[pi]


def transfer_apply_vector(g: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """
    Apply the Transfer/Perron–Frobenius L to a function-vector g on configurations:
      (L g)[i] = sum_{j : π[j] = i} g[j]
    """
    return np.bincount(pi, weights=g, minlength=len(g))


# --------------------------
# Optional: complex-phase evolution (interference demo)
# --------------------------
def phase_fusion_rule(predecessor_phases: List[Phase], q: int, seed: int) -> Phase:
    """
    Non-linear, deterministic fusion rule for complex phases.
    This is used only in the notebook to generate interference-like correlations.
    """
    if q < 1:
        return 1.0 + 0j
    if not predecessor_phases:
        idx = _stable_hash_64(seed) % q
        return np.exp(2j * np.pi * (idx / q))
    prod = complex(1.0 + 0j)
    for z in predecessor_phases:
        prod *= z
    twist_idx = _stable_hash_64(seed, int(abs(prod) * 1e6)) % q
    kick = np.exp(2j * np.pi * (twist_idx / q))
    return prod * kick


def deterministic_update_phase(
    site: CausalSite,
    config: PhaseConfig,
    q: int,
    tick_seed: int,
    rule: PhaseUpdateRule = phase_fusion_rule,
) -> PhaseConfig:
    """
    Phase-valued analogue of `deterministic_update`.
    Each node’s update receives a unique (tick_seed + node_id) seed, so the
    map remains deterministic.
    """
    new_cfg: PhaseConfig = {}
    for depth in sorted(site._nodes_by_depth.keys()):
        for v in site._nodes_by_depth[depth]:
            preds = site.predecessors.get(v, [])
            pred_vals = [config.get(p, 1.0 + 0j) for p in preds]
            node_seed = int(tick_seed + v)
            new_cfg[v] = rule(pred_vals, q, node_seed)
    return new_cfg
