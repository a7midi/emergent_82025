# tests/test_poset.py
"""
Tests for M01-poset: CausalSite generation and properties.

Verifies that the generated sites adhere to the standing hypotheses:
- Acyclic (DAG)
- Graded arrows (depth increases by 1 along edges)
- Bounded out-degree (≤ R-1)
- Depth truncation behaves as expected
"""
import networkx as nx
import numpy as np
import pytest

from emergent.poset import CausalSite


@pytest.fixture(scope="module")
def seeded_rng():
    """Deterministic RNG for reproducibility."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="module")
def sample_site(seeded_rng):
    """A standard CausalSite instance for testing."""
    return CausalSite.generate(
        n_layers=10, nodes_per_layer=20, R=5, edge_prob=0.3, rng=seeded_rng
    )


def test_site_generation(sample_site):
    """Basic properties of a generated site (deterministic structure checks)."""
    assert len(sample_site.nodes) == 200
    assert len(sample_site.nodes_by_depth) == 10
    assert sample_site.R == 5
    assert max(sample_site.depths.values()) == 9


def test_site_is_acyclic(sample_site):
    """Oracle: generated site is a DAG."""
    G = sample_site.to_networkx()
    assert nx.is_directed_acyclic_graph(G)


def test_arrows_are_graded(sample_site):
    """Oracle: every arrow increases depth by exactly 1 (H2)."""
    for u, v in sample_site.edges:
        assert sample_site.depths[v] == sample_site.depths[u] + 1, (
            f"Ungraded arrow {u}->{v}: "
            f"{sample_site.depths[u]} -> {sample_site.depths[v]}"
        )


def test_out_degree_is_bounded(sample_site):
    """Oracle: out-degree is bounded by R-1 (H3)."""
    max_out_degree = sample_site.R - 1
    for u in sample_site.nodes:
        out_degree = len(sample_site.adj.get(u, []))
        assert out_degree <= max_out_degree, (
            f"Node {u} has out-degree {out_degree}, "
            f"exceeding max of {max_out_degree}"
        )


def test_truncation(sample_site):
    """Oracle: depth truncation C_n."""
    max_depth = 4
    truncated = sample_site.truncate_to_depth(max_depth)

    assert len(truncated.nodes_by_depth) == max_depth + 1
    assert all(d <= max_depth for d in truncated.depths.values())
    assert truncated.R == sample_site.R

    # No edge should point beyond max_depth
    for u, v in truncated.edges:
        assert truncated.depths[v] <= max_depth


def test_generate_caps_successors_when_R_exceeds_width():
    """
    Robustness: when R-1 exceeds the next layer width, generation should not fail
    and must cap out-degree by the population. With edge_prob=1.0 we select all
    available successors up to the cap.
    """
    rng = np.random.default_rng(seed=7)
    site = CausalSite.generate(
        n_layers=3, nodes_per_layer=2, R=10, edge_prob=1.0, rng=rng
    )
    # Next layer has width=2, so out-degree ≤ min(R-1, 2) = 2
    for u in site.nodes_by_depth[0]:
        assert len(site.adj.get(u, [])) <= 2
