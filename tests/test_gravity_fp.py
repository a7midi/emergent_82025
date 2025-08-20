# tests/test_gravity_fp.py
import os
import numpy as np
import pytest

from emergent.geom_fp import alexandrov_interval
from emergent.gravity_fp import analyze_gravity_in_block_fp, get_local_coupling_g_fp
from emergent.poset import CausalSite

# Skip this whole module unless explicitly enabled
if not bool(int(os.environ.get("EMERGENT_RUN_FP_TESTS", "0"))):
    pytest.skip("First-principles gravity tests disabled by default.", allow_module_level=True)


@pytest.fixture(scope="module")
def small_site():
    nodes_by_depth = {0: [0], 1: [1, 2], 2: [3]}
    adj = {0: [1, 2], 1: [3], 2: [3]}
    return CausalSite(nodes_by_depth, adj, R=3)


def test_local_and_block_coupling(small_site):
    g = get_local_coupling_g_fp(small_site, (1, 3))
    assert isinstance(g, float)

    block = set(small_site.nodes)
    res = analyze_gravity_in_block_fp(small_site, block)
    assert "avg_g_fp" in res and "num_edges" in res
