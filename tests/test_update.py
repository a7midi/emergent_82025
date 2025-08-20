# tests/test_update.py
"""
Tests for M04-update: Deterministic update T and its associated operators.

Verifies:
- One-tick deterministic update using the local fusion rule.
- Adjoint relation between Koopman and Transfer operators (finite space).
- Deterministic hidden-layer drive reproducibility.

Notes:
- T is deterministic and generally MANY-TO-ONE; it does not preserve the
  counting (uniform) measure (see module docstring for context).
"""
import math
from typing import Dict

import numpy as np
import pytest

from emergent.poset import CausalSite
from emergent.update import (
    TagConfig,
    build_update_permutation,
    deterministic_update,
    hidden_drive,
    koopman_apply_vector,
    transfer_apply_vector,
)


@pytest.fixture(scope="module")
def update_test_site():
    """Small graded site for update tests.
    Structure:
      d=0: 0, 1
      d=1: 2 (preds: 0,1), 3 (preds: 1)
      d=2: 4 (preds: 2,3)
    """
    nodes_by_depth = {0: [0, 1], 1: [2, 3], 2: [4]}
    adj = {0: [2], 1: [2, 3], 2: [4], 3: [4]}
    return CausalSite(nodes_by_depth, adj, R=3)


def test_deterministic_update(update_test_site):
    """A single step of T with the default rule (no hidden input)."""
    q = 5
    # Initial state x_0 with arbitrary tags
    config_t0: TagConfig = {0: 1, 1: 3, 2: 0, 3: 0, 4: 0}

    # Update rule uses ONLY predecessor tags from x_0;
    # roots (0,1) have no predecessors -> 0 at t+1.
    expected_config_t1: TagConfig = {0: 0, 1: 0, 2: 4, 3: 3, 4: 0}

    config_t1 = deterministic_update(update_test_site, config_t0, q)
    assert config_t1 == expected_config_t1


def test_koopman_transfer_are_adjoints(update_test_site):
    """Finite-space adjoint property: <Kf, g> == <f, Lg>."""
    q = 2
    nodes = update_test_site.nodes
    n_configs = q ** len(nodes)

    # Build index map and π via helper
    pi = build_update_permutation(update_test_site, q)

    rng = np.random.default_rng(seed=252)
    f = rng.random(n_configs)
    g = rng.random(n_configs)

    Kf = koopman_apply_vector(f, pi)
    Lg = transfer_apply_vector(g, pi)

    inner_product_1 = float(np.dot(Kf, g))
    inner_product_2 = float(np.dot(f, Lg))
    assert math.isclose(inner_product_1, inner_product_2)


def test_hidden_drive_is_deterministic(update_test_site):
    """Hidden inputs for roots are reproducible and in-range."""
    q = 11
    seed = 12345
    t = 7
    h1 = hidden_drive(update_test_site, t, q, seed)
    h2 = hidden_drive(update_test_site, t, q, seed)

    assert h1 == h2  # deterministic repeat
    # All keys must be depth-0 nodes; values are 0..q-1
    roots = [v for v, d in update_test_site.depths.items() if d == 0]
    assert set(h1.keys()) == set(roots)
    assert all((0 <= tag < q) for tag in h1.values())
