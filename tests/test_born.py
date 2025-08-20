# tests/test_born.py
"""
Tests for M09-quantum-born: Hilbert space and the Born rule.

Oracle Checklist:
- Props. 3.21-3.27: The core result, <Ψ|P_E|Ψ> = μ(E), is directly
  verified for cylinder events in `test_born_rule_on_cylinders`. The test
  confirms that the quantum probability derived from the cyclic state's
  amplitudes correctly reproduces the underlying classical measure.
"""
import numpy as np
import pytest

from emergent.born import (
    get_configuration_space,
    get_cyclic_born_vector,
    get_cylinder_event_mask,
    verify_born_rule,
)
from emergent.poset import CausalSite


@pytest.fixture(scope="module")
def born_test_site():
    """A small site for manageable configuration space size."""
    nodes_by_depth = {0: [0], 1: [1, 2], 2: [3]}
    adj = {0: [1, 2], 1: [3]}
    return CausalSite(nodes_by_depth, adj, R=3)


def test_get_cyclic_born_vector():
    """Tests the construction and normalization of the Born vector."""
    n_configs = 16  # e.g., 4 nodes, q=2
    psi = get_cyclic_born_vector(n_configs)

    assert psi.shape == (n_configs,)
    # Test normalization: ||Ψ||^2 should be 1.
    assert np.isclose(np.linalg.norm(psi), 1.0)
    assert np.allclose(psi, np.full(n_configs, 1.0 / np.sqrt(n_configs)))


def test_born_rule_on_cylinders(born_test_site):
    """
    Oracle: Verifies the Born rule for cylinder events.
    """
    q = 2
    site = born_test_site
    configs_list, _ = get_configuration_space(site, q)
    n_configs = len(configs_list)

    # 1. Construct the cyclic Born vector Ψ for this space.
    psi = get_cyclic_born_vector(n_configs)

    # 2. Define a non-trivial cylinder event E.
    # E = "all configurations where node 0 has tag 1 and node 3 has tag 0"
    cylinder_def = {0: 1, 3: 0}
    event_mask = get_cylinder_event_mask(cylinder_def, configs_list)

    # 3. Calculate the classical probability μ(E) for the uniform measure.
    # μ(E) = (number of configs in E) / (total number of configs)
    num_events = np.sum(event_mask)
    assert num_events > 0 and num_events < n_configs
    classical_prob = num_events / n_configs

    # 4. Verify the Born rule: <Ψ|P_E|Ψ> should equal μ(E).
    # The projector P_E is represented by the boolean mask.
    assert verify_born_rule(psi, event_mask, classical_prob)

    # 5. Test another, simpler cylinder event
    cylinder_def_2 = {1: 0}
    event_mask_2 = get_cylinder_event_mask(cylinder_def_2, configs_list)
    classical_prob_2 = np.sum(event_mask_2) / n_configs
    assert verify_born_rule(psi, event_mask_2, classical_prob_2)