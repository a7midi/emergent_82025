# tests/test_metric.py
"""
Tests for M02-metric: Geometric structures on the CausalSite.

Verifies properties of the path-length metric `d`, the symmetrized radius `r`,
and Alexandrov sets.

Oracle Checklist:
- Metric properties for r: Verified by test_r_radius_is_metric.
- Block-metric compatibility: Verified by test_metric_on_truncated_site.
- Alexandrov intervals: Verified by test_alexandrov_interval.
"""
import math

import numpy as np
import pytest

from emergent.metric import (
    get_alexandrov_interval,
    get_path_length,
    get_symmetrized_radius,
)
from emergent.poset import CausalSite


@pytest.fixture(scope="module")
def seeded_rng():
    """Provides a deterministic, seeded random number generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="module")
def sample_site(seeded_rng):
    """Provides a standard CausalSite instance for testing."""
    return CausalSite.generate(
        n_layers=10, nodes_per_layer=20, R=5, edge_prob=0.3, rng=seeded_rng
    )


@pytest.fixture(scope="module")
def tiny_site():
    """A small, hand-crafted site for easy-to-verify assertions."""
    # Structure: 0 -> 1 -> 3
    #            `-> 2 -^
    nodes_by_depth = {0: [0], 1: [1, 2], 2: [3]}
    adj = {0: [1, 2], 1: [3], 2: [3]}
    return CausalSite(nodes_by_depth, adj, R=3)


def test_path_length_properties(tiny_site):
    """Tests basic properties of the path length metric d(u,v)."""
    assert get_path_length(tiny_site, 0, 0) == 0
    assert get_path_length(tiny_site, 0, 1) == 1
    assert get_path_length(tiny_site, 0, 2) == 1
    assert get_path_length(tiny_site, 0, 3) == 2  # Shortest path is 0->1->3 or 0->2->3
    assert get_path_length(tiny_site, 1, 2) == math.inf  # No path
    assert get_path_length(tiny_site, 2, 1) == math.inf


def test_r_radius_is_metric(sample_site):
    """
    Oracle: Verifies that the symmetrized radius r(u,v) is a valid metric.
    """
    # Select three random nodes for the triangle inequality test
    rng = np.random.default_rng(seed=101)
    u, v, w = rng.choice(sample_site.nodes, 3, replace=False)

    # 1. Non-negativity and Identity
    assert get_symmetrized_radius(sample_site, u, u) == 0
    assert get_symmetrized_radius(sample_site, u, v) >= 0

    # 2. Symmetry
    assert get_symmetrized_radius(sample_site, u, v) == get_symmetrized_radius(
        sample_site, v, u
    )

    # 3. Triangle Inequality: r(u,w) <= r(u,v) + r(v,w)
    r_uv = get_symmetrized_radius(sample_site, u, v)
    r_vw = get_symmetrized_radius(sample_site, v, w)
    r_uw = get_symmetrized_radius(sample_site, u, w)

    # If any path is infinite, the inequality must hold if the RHS isn't inf.
    if math.isinf(r_uw):
        assert math.isinf(r_uv) or math.isinf(r_vw)
    else:
        assert r_uw <= r_uv + r_vw


def test_alexandrov_interval(tiny_site):
    """Tests the calculation of I+(p) intersect I-(q)."""
    # Interval between 0 and 3 should contain all nodes
    assert get_alexandrov_interval(tiny_site, 0, 3) == {0, 1, 2, 3}
    # Interval between 1 and 2 should be empty
    assert get_alexandrov_interval(tiny_site, 1, 2) == set()
    # Interval I+(0) intersect I-(1) is {0, 1}
    assert get_alexandrov_interval(tiny_site, 0, 1) == {0, 1}


def test_metric_on_truncated_site(sample_site):
    """
    Oracle: Verifies metric compatibility on truncations (cf. Paper III, Lem 2.14).
    Checks that if a shortest path lies entirely within a truncation, its length
    is the same whether computed on the full site or the truncated one.
    """
    max_depth = 5
    truncated_site = sample_site.truncate_to_depth(max_depth)

    # Choose two nodes that exist in the truncated site
    u, v = truncated_site.nodes[5], truncated_site.nodes[15]

    len_full = get_path_length(sample_site, u, v)
    len_trunc = get_path_length(truncated_site, u, v)

    # If the shortest path in the full site happens to only use nodes
    # within the truncation depth, the lengths will be identical.
    if not math.isinf(len_full) and len_full <= (
        max_depth - truncated_site.depths[u]
    ):
        assert len_full == len_trunc
    else:
        # Otherwise, the path in the truncated site might be longer or infinite
        assert len_trunc >= len_full
