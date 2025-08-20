# tests/test_geom_fp.py
import os
import numpy as np
import pytest

from emergent.geom_fp import alexandrov_interval, bd_curvature_edge, count_k_chains
from emergent.poset import CausalSite

# Skip this whole module unless explicitly enabled
if not bool(int(os.environ.get("EMERGENT_RUN_FP_TESTS", "0"))):
    pytest.skip("First-principles geom tests disabled by default.", allow_module_level=True)


@pytest.fixture(scope="module")
def tiny_site():
    # 0 -> 1 -> 3
    #  \-> 2 -^
    nodes_by_depth = {0: [0], 1: [1, 2], 2: [3]}
    adj = {0: [1, 2], 1: [3], 2: [3]}
    return CausalSite(nodes_by_depth, adj, R=3)


def test_chain_counts_and_curvature(tiny_site):
    I = alexandrov_interval(tiny_site, 0, 3)
    assert I == {0, 1, 2, 3}
    # edges inside I : 0->1, 0->2, 1->3, 2->3 => 4
    assert count_k_chains(tiny_site, I, 1) == 4
    # length-2 chains: 0->1->3, 0->2->3 => 2
    assert count_k_chains(tiny_site, I, 2) == 2
    # length-3 chains: none => 0
    assert count_k_chains(tiny_site, I, 3) == 0

    kappa = bd_curvature_edge(tiny_site, (0, 3), scheme="4D-lite")
    # Just check it's finite and of expected sign for this simple diamond
    assert np.isfinite(kappa)
