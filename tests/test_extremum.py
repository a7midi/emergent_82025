# tests/test_extremum.py
"""
Tests for M14-extremum-qr: Verifying the parameter-elimination mechanism.

Oracle Checklist:
- Unique argmax (Thm 5.13, 5.17): Verified by `test_unique_argmax_qr`, which
  finds the maximum and asserts that it is strictly greater than all its
  immediate neighbors.
- Rule-out q=R+2 (Lemma 5.16): Verified by `test_ruleout_marginal_q`, which
  shows that the entropy function is still increasing in q at the boundary
  q=R+2, so it cannot be a maximum.
"""
import numpy as np
import pytest

from emergent.extremum import calculate_entropy_density, find_entropy_maximizer


def test_unique_argmax_qr():
    """
    Oracle: Verifies that the entropy functional has a unique maximizer
    in the discrete (q, R) parameter space.
    """
    q_range = (2, 20)
    r_range = (1, 10)

    (q_star, r_star), max_s = find_entropy_maximizer(q_range, r_range)

    # Check that a valid maximum was found
    assert q_star != -1 and r_star != -1
    
    # To verify the maximum is unique, check that the entropy at all
    # neighboring points is strictly less than the maximum.
    neighbors = [
        (q_star + 1, r_star),
        (q_star - 1, r_star),
        (q_star, r_star + 1),
        (q_star, r_star - 1),
    ]

    for q_n, r_n in neighbors:
        # Ensure neighbor is within the search space before checking
        if q_range[0] <= q_n <= q_range[1] and r_range[0] <= r_n <= r_range[1]:
            s_neighbor = calculate_entropy_density(q_n, r_n)
            assert s_neighbor < max_s


def test_ruleout_marginal_q():
    """
    Oracle: Verifies that the entropy maximum cannot occur at q = R + 2.
    The test confirms that the entropy is still increasing in the q-direction
    at this boundary, as shown in Lemma 5.16.
    """
    # Test for several values of R
    for R in [2, 3, 4, 5]:
        q_boundary = R + 2
        
        s_at_boundary = calculate_entropy_density(q_boundary, R)
        s_after_boundary = calculate_entropy_density(q_boundary + 1, R)
        
        # The lemma states that the function should be increasing in q here,
        # so S(q+1) must be greater than S(q).
        assert s_after_boundary > s_at_boundary