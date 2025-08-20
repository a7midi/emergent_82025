# tests/test_properties.py
"""
M16—Property-based tests (lightweight, deterministic profile).

Properties covered:
  • Metric: Triangle inequality for the symmetrized radius r (Paper III, Remark 2.13).
  • Spectral constants: 0 < δ < 1, 0 < gaps < 1 when q > R-1 (Cor. 2.26).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from emergent.metric import get_symmetrized_radius
from emergent.poset import CausalSite
from emergent.spectral import get_spectral_constants


def _gen_site(n_layers: int, nodes_per_layer: int, R: int, edge_prob: float, seed: int) -> CausalSite:
    rng = np.random.default_rng(seed)
    return CausalSite.generate(
        n_layers=n_layers, nodes_per_layer=nodes_per_layer, R=R, edge_prob=edge_prob, rng=rng
    )


@st.composite
def site_and_triple(draw):
    n_layers = draw(st.integers(min_value=4, max_value=6))
    nodes_per_layer = draw(st.integers(min_value=4, max_value=6))
    R = draw(st.integers(min_value=2, max_value=4))
    edge_prob = draw(st.sampled_from([0.3, 0.5, 0.7]))
    seed = draw(st.integers(min_value=1, max_value=10_000))

    site = _gen_site(n_layers, nodes_per_layer, R, edge_prob, seed)
    # Pick 3 distinct nodes (Hypothesis ensures enough nodes given bounds)
    u, v, w = np.random.default_rng(seed + 1).choice(site.nodes, 3, replace=False)
    return site, int(u), int(v), int(w)


@given(site_and_triple())
def test_triangle_inequality_for_r(data):
    """r(u,w) ≤ r(u,v) + r(v,w) whenever RHS is finite."""
    site, u, v, w = data
    r_uv = get_symmetrized_radius(site, u, v)
    r_vw = get_symmetrized_radius(site, v, w)
    r_uw = get_symmetrized_radius(site, u, w)

    if math.isinf(r_uv) or math.isinf(r_vw):
        # Triangle inequality is vacuous if RHS is ∞
        assert True
    else:
        assert r_uw <= r_uv + r_vw + 1e-9  # small float slop


@given(
    st.integers(min_value=3, max_value=12),  # q
    st.integers(min_value=2, max_value=8),   # R
)
def test_spectral_constants_well_posed(q, R):
    """For q > R-1: 0 < δ < 1 and gaps in (0,1)."""
    if q <= R - 1:
        with pytest.raises(ValueError):
            get_spectral_constants(q, R)
        return

    c = get_spectral_constants(q, R)
    delta, l_gap, k_gap = c["delta"], c["l_gap"], c["k_gap"]
    assert 0.0 < delta < 1.0
    assert 0.0 < l_gap < 1.0
    assert 0.0 < k_gap < 1.0
