# tests/test_gauge.py
"""
Tests for M10-gauge-chirality: Verifying symbolic results.

Oracle Checklist:
- H1 ~= Z (Thm 3.33): Acknowledged symbolically. This module assumes a single
  U(1) charge source from the site's topology.
- Anomaly Cancellation (Lemma 3.35 & App I): Verified explicitly by
  `test_sm_anomaly_cancellation`, which checks that all relevant trace
  calculations sum to zero.
"""
import numpy as np
import pytest

from emergent.gauge import check_all_anomalies, get_sm_field_content


def test_sm_field_content():
    """Checks that the field content is generated correctly."""
    gen1 = get_sm_field_content(1)
    assert len(gen1) == 5
    # Total states for one generation: 6 (Q_L) + 3 (u_R^c) + 3 (d_R^c) + 2 (L_L) + 1 (e_R^c) = 15
    assert sum(f.multiplicity for f in gen1) == 15

    gen3 = get_sm_field_content(3)
    assert len(gen3) == 15
    assert sum(f.multiplicity for f in gen3) == 45


def test_sm_anomaly_cancellation():
    """
    Oracle: Verifies the cancellation of all Standard Model gauge and
    gravitational anomalies for 1 and 3 generations.
    """
    # Test for 1 generation
    anomalies_1_gen = check_all_anomalies(n_generations=1)
    for name, value in anomalies_1_gen.items():
        assert np.isclose(
            value, 0.0
        ), f"Anomaly '{name}' is non-zero for 1 generation: {value}"

    # Test for 3 generations
    anomalies_3_gen = check_all_anomalies(n_generations=3)
    for name, value in anomalies_3_gen.items():
        assert np.isclose(
            value, 0.0
        ), f"Anomaly '{name}' is non-zero for 3 generations: {value}"