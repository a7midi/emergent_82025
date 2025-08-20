import numpy as np
import pytest

from emergent.poset import CausalSite
from emergent.entropy_fp import exact_static_entropy, predict_entropy_trajectory

def test_exact_snapshot_entropy_matches_count():
    rng = np.random.default_rng(0)
    site = CausalSite.generate(n_layers=5, nodes_per_layer=4, R=4, edge_prob=0.6, rng=rng)
    q = 3
    observer_depth = 2
    # Hidden nodes are layers 2,3,4 -> 3 * 4 = 12
    S0 = exact_static_entropy(site, q, observer_depth)
    assert np.isclose(S0, 12 * np.log2(q))

def test_predictor_has_positive_linear_slope():
    rng = np.random.default_rng(1)
    site = CausalSite.generate(n_layers=6, nodes_per_layer=6, R=4, edge_prob=0.6, rng=rng)
    q = 5
    observer_depth = 3
    ticks = 5

    S_pred, dS_pred = predict_entropy_trajectory(site, q, observer_depth, ticks)
    assert len(S_pred) == ticks and len(dS_pred) == ticks
    # Constant positive slope if minimal hidden layer non-empty
    assert dS_pred[0] > 0
    assert all(np.isclose(dS_pred[i], dS_pred[0]) for i in range(1, ticks))
    # Linear growth
    for t in range(1, ticks):
        assert np.isclose(S_pred[t] - S_pred[t-1], dS_pred[0])
