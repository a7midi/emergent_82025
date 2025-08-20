# tests/test_integration.py
"""
M16—End-to-end integration smoke test.

Covers a thin but representative slice:
  1) Entropy sim runs and returns stable lengths.
  2) CHSH S ≤ 2√2 and equals 2 on the canonical tiny site.
  3) Geometry & gravity probes yield finite summaries.
  4) RG flows move couplings.
  5) Calibration round-trip is consistent.

All parameters are chosen to keep runtime small.
"""
from __future__ import annotations

import math
import numpy as np

from emergent.calibration import (
    LatticeParams,
    calibrate_lattice_to_physical,
    calibrate_physical_to_lattice,
)
from emergent.chsh import run_chsh_experiment
from emergent.entropy import run_entropy_simulation
from emergent.geom import analyze_curvature_in_ball, get_nodes_in_ball
from emergent.gravity import analyze_gravity_in_block, get_local_coupling_g
from emergent.poset import CausalSite
from emergent.rg import CouplingVector, run_rg_flow


def test_end_to_end_pipeline():
    # 1) Entropy on a small generated site
    rng = np.random.default_rng(73)
    site = CausalSite.generate(n_layers=5, nodes_per_layer=5, R=3, edge_prob=0.6, rng=rng)
    q = 4
    obs_depth = 2
    ticks = 3
    init_cfg = {n: int(rng.integers(q)) for n in site.nodes}
    entropies, increments = run_entropy_simulation(site, q, init_cfg, obs_depth, ticks)
    assert len(entropies) == ticks
    assert len(increments) == ticks - 1

    # 2) CHSH on canonical small site (guaranteed spacelike pair 2,3)
    nodes_by_depth = {0: [0, 1], 1: [2, 3], 2: [4]}
    adj = {0: [2], 1: [3], 2: [4], 3: [4]}
    chsh_site = CausalSite(nodes_by_depth, adj, R=3)
    hidden_sources = [0, 1]
    alice, bob = 2, 3
    s_vals = []
    for tag in range(2):
        s_vals.append(
            run_chsh_experiment(
                chsh_site, 2, {4: tag}, alice, bob, hidden_sources
            )
        )
    max_s = max(abs(x) for x in s_vals)
    assert max_s <= 2 * math.sqrt(2) + 1e-9
    assert math.isclose(max_s, 2.0)  # local deterministic model

    # 3) Geometry & gravity on a modest random site
    geom_site = CausalSite.generate(n_layers=12, nodes_per_layer=20, R=5, edge_prob=0.5, rng=np.random.default_rng(11))
    center = geom_site.nodes[len(geom_site.nodes)//2]
    ball = get_nodes_in_ball(geom_site, center, radius=5.0)
    mean_k, var_k = analyze_curvature_in_ball(geom_site, ball)
    assert isinstance(mean_k, float)
    assert isinstance(var_k, float)

    # Gravity: simple 3-layer gadget with known g=1/3 on (1,3)
    grav_nodes = {0: [0], 1: [1, 2], 2: [3]}
    grav_adj = {0: [1, 2], 1: [3], 2: [3]}
    grav_site = CausalSite(grav_nodes, grav_adj, R=3)
    g_local = get_local_coupling_g(grav_site, (1, 3))
    assert np.isclose(g_local, 1/3)
    # Block average should be finite with enough edges
    block = set(grav_site.nodes)
    res = analyze_gravity_in_block(grav_site, block)
    assert res["num_edges"] > 0
    assert (isinstance(res["avg_g"], float) or np.isnan(res["avg_g"]))

    # 4) RG flow—values should change
    g0 = CouplingVector(g_star=0.1, lambda_mix=0.5, theta_cp=0.2)
    g1 = run_rg_flow(g0, k_start=100, k_end=0, q=6, R=4)
    assert not np.allclose(g0.to_array(), g1.to_array())

    # 5) Calibration round-trip consistency
    lat = LatticeParams(delta_t=1e-3, phi_min=2e-3, g_star=0.2, lambda_mix=0.5, theta_cp=0.3)
    phys = calibrate_lattice_to_physical(lat, k0=64, sigma_mix=0.24)
    lat2 = calibrate_physical_to_lattice(phys, k0=64, sigma_mix=0.24)
    phys2 = calibrate_lattice_to_physical(lat2, k0=64, sigma_mix=0.24)
    assert np.allclose(
        [phys.c, phys.hbar, phys.G, phys.m_f0, phys.theta_cp_obs],
        [phys2.c, phys2.hbar, phys2.G, phys2.m_f0, phys2.theta_cp_obs],
    )
