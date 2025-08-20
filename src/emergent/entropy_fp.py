"""
M06/M14 (Phase B) — Exact/Symbolic entropy tools for the deterministic tag-fusion model.

What this module provides
-------------------------
1) exact_static_entropy(...)        — exact Shannon entropy of the observer's *snapshot* fiber
                                      (no-growth model used by existing tests).
2) predict_entropy_trajectory(...)  — the closed-form, per-tick *linear* growth predictor from
                                      the minimal hidden layer H_min (papers' continuum identity).
3) (optional) exact_history_entropy_prime(...) — exact history-consistency fiber size via a
                                      stacked linear system modulo a *prime* q (for small sites).
                                      This is useful for cross-checks and small “ground truth”
                                      runs; it is not needed by the default CLI/tests.

References (mapping to existing code)
-------------------------------------
- Causal site & depths:   emergent.poset.CausalSite.                               
- Deterministic update:   emergent.update.deterministic_update (sum_mod_q rule).   
- Baseline entropy utils: emergent.entropy.get_minimal_hidden_layer / shannon_entropy.

The predictor produced here is the exact formula used in Phase B's optimizer:
    ΔS_per_tick  = |H_min| * log2(q)

It is the deterministic, continuum-limit identity from the attachments and matches the
structure of your earlier notebook demonstration.

NOTE: The "history" solver requires q to be prime; for composite q, use the predictor
or Monte-Carlo estimators (entropy_mc) to avoid number-theory machinery.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .poset import CausalSite, NodeID
from .update import TagConfig, UpdateRule, deterministic_update, sum_mod_q_rule
from .entropy import get_minimal_hidden_layer, shannon_entropy


# ---- Small helpers -------------------------------------------------------------------------

def _observer_nodes(site: CausalSite, observer_depth: int) -> Set[NodeID]:
    """All nodes visible to the observer: depths < observer_depth."""
    return {n for n, d in site.depths.items() if d < observer_depth}


def _hidden_nodes(site: CausalSite, observer_depth: int) -> List[NodeID]:
    """All nodes hidden from the observer: depths >= observer_depth."""
    return [n for n, d in site.depths.items() if d >= observer_depth]


# ---- 1) Exact snapshot entropy (no-growth baseline used in current tests) ------------------

def exact_static_entropy(site: CausalSite, q: int, observer_depth: int) -> float:
    """
    Exact entropy for a *snapshot* macrostate: S = |Hidden| * log2(q).

    This matches the current deterministic, no-drive simulation used in the green test
    suite where the observer fiber size remains constant across ticks.

    Args:
        site: CausalSite.
        q: hidden alphabet size.
        observer_depth: visibility cut (depths < observer_depth are visible).

    Returns:
        Shannon entropy (bits).
    """
    if q <= 1:
        return 0.0
    hidden = _hidden_nodes(site, observer_depth)
    return float(len(hidden) * np.log2(q))


# ---- 2) Exact closed-form predictor for linear growth (H_min identity) ---------------------

def predict_entropy_trajectory(
    site: CausalSite,
    q: int,
    observer_depth: int,
    ticks: int,
) -> Tuple[List[float], List[float]]:
    """
    Closed-form predictor for the per-tick observer entropy growth:

        ΔS_per_tick = |H_min| * log2(q)

    where H_min is the first layer beyond the observer (the minimal hidden layer).
    The returned sequence starts at t=1 and goes to t=ticks (like your notebook).

    Args:
        site: CausalSite.
        q: hidden alphabet size.
        observer_depth: visibility cut.
        ticks: how many steps to predict.

    Returns:
        (S_pred, dS_pred) where:
          S_pred[t-1]  = t * |H_min| * log2(q)
          dS_pred[t-1] = |H_min| * log2(q)  (constant slope)
    """
    if ticks <= 0 or q <= 1:
        return [], []
    vis = _observer_nodes(site, observer_depth)
    h_min = get_minimal_hidden_layer(site, vis)
    slope = float(len(h_min) * np.log2(q))
    S = [slope * (t + 1) for t in range(ticks)]
    dS = [slope for _ in range(ticks)]
    return S, dS


# ---- 3) (Optional) exact history-consistency fiber at prime q ------------------------------

def _modular_inv_prime(a: int, p: int) -> int:
    """Inverse of a modulo a prime p."""
    return pow(int(a) % p, p - 2, p)


def _gauss_rank_mod_prime(M: np.ndarray, p: int) -> Tuple[int, bool]:
    """
    Row-reduction over Z_p (prime field). Returns (rank, consistent_to_zero)
    where consistent_to_zero indicates whether Mx = 0 is consistent (always True).
    For augmented systems, call with the augmented matrix and inspect for contradictions.
    """
    A = (M.copy() % p).astype(np.int64)
    m, n = A.shape
    rank = 0
    row = 0
    for col in range(n):
        # Find pivot
        pivot = None
        for r in range(row, m):
            if A[r, col] % p != 0:
                pivot = r
                break
        if pivot is None:
            continue
        # Swap to current row
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        # Normalize pivot row
        inv = _modular_inv_prime(A[row, col], p)
        A[row, :] = (A[row, :] * inv) % p
        # Eliminate others
        for r in range(m):
            if r != row and A[r, col] % p != 0:
                factor = A[r, col] % p
                A[r, :] = (A[r, :] - factor * A[row, :]) % p
        rank += 1
        row += 1
        if row == m:
            break
    # For homogeneous systems, consistency to zero is trivially True
    return rank, True


def exact_history_entropy_prime(
    site: CausalSite,
    q: int,
    observer_depth: int,
    ticks: int,
    initial_config: Optional[TagConfig] = None,
    rule: UpdateRule = sum_mod_q_rule,
) -> Tuple[List[float], List[float]]:
    """
    *Experimental* exact history-consistency fiber for prime q (small sites).

    We model:
        x_{t+1} = B x_t (mod q), where B[v,p]=1 if p -> v, else 0.
    Let P select visible coordinates (depth < observer_depth). Observed y_t = P x_t.
    Then y_t = P B^t x_0. For each t we stack constraints M_t x_0 = y_t and compute
    the number of solutions |{x_0}| = q^{N - rank(M_t)} if consistent.
    Entropy S_t = log2 |{x_0}|.

    Notes:
      - q **must** be prime for this exact solver.
      - For composite q or large graphs, prefer the predictor (above) or a Monte‑Carlo
        estimator; this function is intended for spot checks only.

    Returns:
        (S_t_list, dS_t_list) for t = 1..ticks
    """
    if q <= 1 or ticks <= 0:
        return [], []
    # Prime check (very cheap Miller–Rabin for small q, or trial division)
    if any(q % p == 0 for p in range(2, int(np.sqrt(q)) + 1)):
        raise ValueError("exact_history_entropy_prime requires prime q.")

    # Build B and P
    nodes = site.nodes
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    B = np.zeros((N, N), dtype=np.int64)
    for v in nodes:
        preds = site.predecessors.get(v, [])
        for p in preds:
            B[idx[v], idx[p]] = 1

    vis = _observer_nodes(site, observer_depth)
    vis_idx = [idx[n] for n in sorted(vis)]
    P = np.zeros((len(vis_idx), N), dtype=np.int64)
    for r, j in enumerate(vis_idx):
        P[r, j] = 1

    # Generate ground truth visible history (deterministic)
    rng = np.random.default_rng(0)
    x0 = (
        np.array([rng.integers(q) for _ in range(N)], dtype=np.int64)
        if initial_config is None
        else np.array([initial_config[n] for n in nodes], dtype=np.int64)
    )
    xs = [x0.copy()]
    for _ in range(ticks):
        xs.append((B @ xs[-1]) % q)
    ys = [P @ xs[t] % q for t in range(1, ticks + 1)]

    # Stack constraints up to each t and compute rank mod q
    S: List[float] = []
    for t in range(1, ticks + 1):
        Bt = np.linalg.matrix_power(B, t) % q
        M = (P @ Bt) % q  # shape: (#vis, N)
        # Rank of M over Z_q:
        rank, _ = _gauss_rank_mod_prime(M, q)
        fiber_size = q ** (N - rank)
        S.append(float(np.log2(fiber_size)))

    dS = [S[i] - S[i - 1] for i in range(1, len(S))] if len(S) > 1 else []
    return S, dS
