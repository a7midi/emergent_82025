# tests/test_measure.py
"""
Tests for M03-measure: Site and configuration measures.

Verifies:
- SiteMeasure normalization and tiny closed-form values.
- Kolmogorov projective consistency across truncations (site measure).
- Cylinder probabilities for ConfigurationMeasure (uniform product).
- Projective consistency for cylinders under truncation (configuration measure).

Oracle Checklist:
- Projective Limit of μ_n: test_kolmogorov_projective_limit
- Cylinder probabilities: test_configuration_cylinder_uniform
- Cylinder consistency under truncation: test_configuration_projective_consistency
"""
import math
from dataclasses import dataclass

import numpy as np
import pytest

from emergent.measure import Cylinder, ConfigurationMeasure, SiteMeasure
from emergent.poset import CausalSite


@pytest.fixture(scope="module")
def sample_site():
    """Provides a standard CausalSite instance for testing."""
    rng = np.random.default_rng(seed=42)
    return CausalSite.generate(
        n_layers=10, nodes_per_layer=20, R=5, edge_prob=0.3, rng=rng
    )


@pytest.fixture(scope="module")
def tiny_site():
    """A small, hand-crafted site for easy-to-verify assertions.
    Structure: 0 -> 1 -> 3
               `-> 2 -^
    """
    nodes_by_depth = {0: [0], 1: [1, 2], 2: [3]}
    adj = {0: [1, 2], 1: [3], 2: [3]}
    return CausalSite(nodes_by_depth, adj, R=3)


# -----------------------------------------------------------------------------
# SiteMeasure tests
# -----------------------------------------------------------------------------
def test_site_measure_properties(tiny_site):
    """Normalization and closed-form probabilities on a toy site."""
    q = 3
    measure = SiteMeasure(tiny_site, q)

    # 1) Normalization
    total_prob = sum(measure.prob(n) for n in tiny_site.nodes)
    assert math.isclose(total_prob, 1.0)

    # 2) Individual probabilities (depths: 0,1,1,2)
    # Z = 3^-0 + 3^-1 + 3^-1 + 3^-2 = 1 + 1/3 + 1/3 + 1/9 = 16/9
    # μ(0) = 1 / (16/9) = 9/16
    # μ(1) = (1/3) / (16/9) = 3/16
    # μ(2) = (1/3) / (16/9) = 3/16
    # μ(3) = (1/9) / (16/9) = 1/16
    assert math.isclose(measure.prob(0), 9 / 16)
    assert math.isclose(measure.prob(1), 3 / 16)
    assert math.isclose(measure.prob(2), 3 / 16)
    assert math.isclose(measure.prob(3), 1 / 16)


def test_kolmogorov_projective_limit(sample_site):
    """
    Kolmogorov consistency: For nodes within a truncation V_{≤k},
    μ_{full}(A) / μ_{full}(V_{≤k}) == μ_{trunc}(A).
    """
    q = 6
    max_depth = 5
    full_measure = SiteMeasure(sample_site, q)

    truncated_site = sample_site.truncate_to_depth(max_depth)
    truncated_measure = SiteMeasure(truncated_site, q)

    # Random subset of nodes inside the truncation
    rng = np.random.default_rng(seed=123)
    subset_size = min(20, len(truncated_site.nodes))
    nodes_A = set(rng.choice(truncated_site.nodes, subset_size, replace=False))

    prob_A_in_trunc = truncated_measure.prob_set(nodes_A)
    prob_A_conditioned = full_measure.conditional_prob_in_full_on_subset(
        nodes_A, max_depth=max_depth
    )
    assert math.isclose(prob_A_in_trunc, prob_A_conditioned)


# -----------------------------------------------------------------------------
# ConfigurationMeasure + cylinders tests
# -----------------------------------------------------------------------------
def test_configuration_cylinder_uniform(tiny_site):
    """
    For uniform product measure, P[cylinder on k nodes] = (1/q)^k.
    """
    q = 5
    cm = ConfigurationMeasure(tiny_site, q)

    cyl = Cylinder({1: 2, 3: 4})  # two fixed tags
    p = cm.prob_cylinder(cyl)
    assert math.isclose(p, (1 / q) ** 2)

    # degenerate / invalid tags -> probability 0
    cyl_bad = Cylinder({1: q})  # out of range
    assert cm.prob_cylinder(cyl_bad) == 0.0


def test_configuration_projective_consistency(sample_site):
    """
    Cylinder probability is consistent under truncation: restricting the
    cylinder to nodes in V_{≤k} does not change its probability within that
    truncation (product measure).
    """
    q = 7
    cm_full = ConfigurationMeasure(sample_site, q)

    # choose a truncation and a random cylinder on that truncation
    k = 4
    trunc = sample_site.truncate_to_depth(k)
    rng = np.random.default_rng(seed=222)

    # choose 10 distinct nodes in trunc and random tags for them
    m = min(10, len(trunc.nodes))
    nodes = list(rng.choice(trunc.nodes, size=m, replace=False))
    tags = [int(t) for t in rng.integers(low=0, high=q, size=m)]
    cyl = Cylinder(dict(zip(nodes, tags)))

    # Both computations should coincide
    p_trunc = cm_full.prob_cylinder_on_truncation(cyl, trunc)
    cm_trunc = ConfigurationMeasure(trunc, q)  # same (uniform) product on trunc
    p_direct = cm_trunc.prob_cylinder(cyl)
    assert math.isclose(p_trunc, p_direct)
