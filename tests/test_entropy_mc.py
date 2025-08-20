# tests/test_entropy_mc.py
import os
import numpy as np
import pytest

# Corrected import
from emergent.entropy_mc import run_entropy_monte_carlo
from emergent.entropy_fp import predict_entropy_trajectory
from emergent.poset import CausalSite

# Skip this whole module unless explicitly enabled
if not bool(int(os.environ.get("EMERGENT_RUN_FP_TESTS", "0"))):
    pytest.skip("Entropy-MC tests disabled by default.", allow_module_level=True)


def test_entropy_mc_growth_is_monotone():
    rng = np.random.default_rng(73)
    site = CausalSite.generate(n_layers=6, nodes_per_layer=6, R=4, edge_prob=0.6, rng=rng)
    q = 4
    obs_depth = 3
    ticks = 6

    ent, inc = run_entropy_monte_carlo(site, q, obs_depth, ticks, samples=128, seed=11)
    assert len(ent) == ticks
    # Monotone non-decreasing (MC may plateau briefly but not drop)
    assert all(ent[i+1] >= ent[i] - 1e-9 for i in range(len(ent)-1))

    pred_ent, _ = predict_entropy_trajectory(site, q, obs_depth, ticks)
    # Predicted slope is positive when H_min non-empty
    assert pred_ent[-1] >= pred_ent[0]
