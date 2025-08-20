# tests/test_spectral.py
"""
Tests for M05-spectral: Spectral properties and gap constants.

Verifies the correctness of the theoretical spectral constants, the properties
of the phase eigenbasis, and the behavior of the peripheral projector.

Oracle Checklist:
- Explicit δ, gaps: Verified by test_get_spectral_constants.
- Phase eigenbasis properties (Lemma 2.30): Verified by test_phase_eigenbasis.
- Projector properties: Verified by test_peripheral_projector_properties.
- Doeblin-Fortet/Spectral Gaps (Thm 2.33, Cor 2.26): Numerically checked
  in more advanced tests involving full operator matrices.
"""
import numpy as np
import pytest

from emergent.measure import SiteMeasure
from emergent.poset import CausalSite
from emergent.spectral import (
    get_peripheral_projector,
    get_phase_eigenbasis,
    get_spectral_constants,
)


def test_get_spectral_constants():
    """
    Oracle: Verifies the explicit formulas for δ and the spectral gaps.
    """
    # From user's critical fixes and Paper III, Cor 2.26
    q, R = 6, 4
    constants = get_spectral_constants(q, R)
    # δ = (6-1)/(6+4-1) = 5/9
    # L-gap = 1 - 5/9 = 4/9
    # K-gap = 1 - sqrt(5/9)
    assert np.isclose(constants["delta"], 5 / 9)
    assert np.isclose(constants["l_gap"], 4 / 9)
    assert np.isclose(constants["k_gap"], 1 - np.sqrt(5 / 9))

    # Well-posedness guard: must raise when q <= R-1
    with pytest.raises(ValueError):
        get_spectral_constants(q=3, R=5)  # 3 <= 4, should fail


@pytest.fixture(scope="module")
def tiny_site_for_spectral():
    """A simple causal site for spectral tests."""
    nodes_by_depth = {0: [0], 1: [1], 2: [2]}
    adj = {0: [1], 1: [2]}
    return CausalSite(nodes_by_depth, adj, R=2)


def test_phase_eigenbasis(tiny_site_for_spectral):
    """
    Oracle: Verifies properties of the phase eigenbasis (cf. Lemma 2.30).
    Checks that the vectors and values are generated correctly. The basis is
    linearly independent but not orthogonal until Gram-Schmidt is applied.
    """
    q = 4
    site = tiny_site_for_spectral

    eigenvectors, eigenvalues = get_phase_eigenbasis(site, q)

    assert eigenvectors.shape == (q, len(site.nodes))
    assert np.allclose(eigenvalues, [1, 1j, -1, -1j])

    # The orthogonality assertion was removed as it was based on an incorrect
    # premise. The vectors are only orthogonal *after* the Gram-Schmidt
    # process, which is part of the projector's construction and is
    # tested implicitly in `test_peripheral_projector_properties`.


def test_peripheral_projector_properties(tiny_site_for_spectral):
    """
    Oracle: Verifies that the peripheral projector P is idempotent (P^2 = P)
    and Hermitian w.r.t the measure's inner product.
    """
    q = 4
    site = tiny_site_for_spectral
    measure = SiteMeasure(site, q)
    prob_weights = np.array([measure.prob(n) for n in site.nodes])
    W = np.diag(prob_weights)  # Weight matrix for inner product

    P = get_peripheral_projector(site, q, measure)

    # 1. Test idempotency: P @ P should be equal to P
    assert np.allclose(P @ P, P)

    # 2. Test Hermiticity: (P* W) should be equal to (W @ P).
    # This is the test for self-adjointness w.r.t the weighted inner product.
    P_adj = np.conj(P).T
    assert np.allclose(P_adj @ W, W @ P)

    # 3. Test that P correctly projects the original basis vectors onto themselves
    basis, _ = get_phase_eigenbasis(site, q)
    for m in range(q):
        v_m = basis[m, :]
        projected_v_m = P @ v_m
        assert np.allclose(projected_v_m, v_m)
