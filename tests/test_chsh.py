# tests/test_chsh.py
"""
Tests for M07-chsh: CHSH experiment and the Tsirelson bound.
...
"""
import math

import numpy as np
import pytest

from emergent.chsh import find_spacelike_nodes, run_chsh_experiment
from emergent.metric import get_path_length
from emergent.poset import CausalSite
from emergent.update import TagConfig


@pytest.fixture(scope="module")
def chsh_test_site():
    """
    A site designed for a CHSH test.
    ...
    """
    nodes_by_depth = {0: [0, 1], 1: [2, 3], 2: [4]}
    adj = {0: [2], 1: [3], 2: [4], 3: [4]}
    return CausalSite(nodes_by_depth, adj, R=3)


def test_find_spacelike_nodes(chsh_test_site):
    """Tests that the function can identify causally disconnected nodes."""
    rng = np.random.default_rng(seed=303)
    found_pair = find_spacelike_nodes(chsh_test_site, rng)

    assert found_pair is not None
    u, v = found_pair

    assert get_path_length(chsh_test_site, u, v) == math.inf
    assert get_path_length(chsh_test_site, v, u) == math.inf


def test_tsirelson_bound(chsh_test_site):
    """
    Oracle: Verifies that the CHSH S-value from the deterministic model
    does not exceed the Tsirelson bound of 2*sqrt(2).
    """
    site = chsh_test_site
    q = 2  # CHSH requires binary outcomes
    
    alice_node, bob_node = 2, 3
    tsirelson_limit = 2 * math.sqrt(2)

    max_s_value = 0
    
    hidden_source_nodes = [0, 1]
    
    # Iterate through all possible final outcomes that the observer could see.
    for final_v2_tag in range(q):
        observer_config: TagConfig = {4: final_v2_tag}
        
        # The experiment averages over the fiber of possibilities consistent
        # with this observation.
        s_value = run_chsh_experiment(
            site=site,
            q=q,
            observer_config=observer_config,
            alice_node=alice_node,
            bob_node=bob_node,
            hidden_source_nodes=hidden_source_nodes,
        )
        
        if abs(s_value) > max_s_value:
            max_s_value = abs(s_value)

    assert max_s_value <= tsirelson_limit + 1e-9
    
    # For this specific local model, the maximum correlation is classical.
    assert np.isclose(max_s_value, 2.0)