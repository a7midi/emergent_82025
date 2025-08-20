# tests/test_gravity.py
"""
Tests for M11-gravity: Emergent Einstein equations.

Oracle Checklist:
- Curvature-entropy coupling (Einstein-Memory Law): Verified by
  `test_local_gravity_quantities`.
- Block-Einstein convergence (Thm 3.48): Verified by
  `test_coupling_constant_convergence`, which shows that the block-averaged
  coupling `g` stabilizes at larger scales, a key numerical witness for the
  emergent gravitational dynamics.
"""

import numpy as np
import pytest

from emergent.geom import get_nodes_in_ball
from emergent.gravity import analyze_gravity_in_block, get_local_coupling_g
from emergent.poset import CausalSite


@pytest.fixture(scope="module")
def gravity_test_site():
    """A site with non-trivial curvature and memory density."""
    # d=0: 0
    # d=1: 1 (pred: 0), 2 (pred: 0)
    # d=2: 3 (preds: 1, 2)
    # Edge (1,3): κ=2-1=1, ρ=2^2-1^2=3, g=1/3
    nodes_by_depth = {0: [0], 1: [1, 2], 2: [3]}
    adj = {0: [1, 2], 1: [3], 2: [3]}
    return CausalSite(nodes_by_depth, adj, R=3)


@pytest.fixture(scope="module")
def sample_site():
    """Provides a larger, randomly generated site for statistical tests."""
    rng = np.random.default_rng(seed=42)
    return CausalSite.generate(
        n_layers=20, nodes_per_layer=40, R=6, edge_prob=0.5, rng=rng
    )


def test_local_gravity_quantities(gravity_test_site):
    """
    Oracle: Verifies the local Einstein-Memory Law g = κ/ρ_mem on a single edge.
    """
    g_value = get_local_coupling_g(gravity_test_site, (1, 3))
    # κ = |Pred(3)| - |Pred(1)| = 2 - 1 = 1
    # ρ = |Pred(3)|^2 - |Pred(1)|^2 = 4 - 1 = 3
    # g = 1/3
    assert np.isclose(g_value, 1 / 3)


def test_coupling_constant_convergence(sample_site):
    """
    Oracle: Verifies that the block-averaged coupling constant `g` converges
    to a stable value at large scales.
    """
    center_node = sample_site.nodes[len(sample_site.nodes) // 2]
    radii = [4.0, 5.0, 6.0, 7.0]
    g_values = []

    for r in radii:
        ball = get_nodes_in_ball(sample_site, center_node, r)
        if len(ball) > 50:  # Ensure a large statistical sample
            results = analyze_gravity_in_block(sample_site, ball)
            if not np.isnan(results["avg_g"]):
                g_values.append(results["avg_g"])

    assert len(g_values) > 1, "Could not collect enough valid data points."

    # Check that the g_values have stabilized, indicating convergence to g*.
    # We check that the last calculated value is close to the previous one.
    assert np.isclose(g_values[-1], g_values[-2], rtol=0.1)