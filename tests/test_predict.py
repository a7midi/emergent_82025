# tests/test_predict.py
"""
Phase D — prediction cards: fast sanity tests.

We check shapes/ranges for the weak-mixing curve, and that EDM/cosmology
cards produce finite, positive proxies (with conservative ranges).
"""
import numpy as np
import pytest

from emergent.predict import (
    predict_weak_mixing_curve,
    make_card_weakmix,
    make_card_cosmology,
    make_card_edm,
)
from emergent.rg import CouplingVector


def test_weakmix_curve_shapes_and_bounds():
    g0 = CouplingVector(g_star=0.1, lambda_mix=0.5, theta_cp=0.2)
    curve, summary = predict_weak_mixing_curve(
        g0, q=6, R=4, k_start=50, k_end=1, n_grid=51, bootstrap=8, seed=1
    )
    assert len(curve.k) == len(curve.mean) == len(curve.lo) == len(curve.hi) == 51
    assert np.all((np.array(curve.mean) >= 0.0) & (np.array(curve.mean) <= 1.0))
    assert 0.0 <= summary["alpha_EM_EW"] <= 1.0


def test_cards_construct():
    g0 = CouplingVector(g_star=0.1, lambda_mix=0.5, theta_cp=0.2)

    card_w = make_card_weakmix(g0, q=6, R=4, k_start=30, k_end=1, n_grid=31, bootstrap=8)
    assert "sin2_thetaW_EW" in card_w.central

    card_c = make_card_cosmology()
    assert "Lambda_proxy" in card_c.central

    card_e = make_card_edm(g0, q=6, R=4, k_start=30, k_end=1)
    assert "d_n_EDM_proxy" in card_e.central
    assert card_e.central["d_n_EDM_proxy"] >= 0.0
