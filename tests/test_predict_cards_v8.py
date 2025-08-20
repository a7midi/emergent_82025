# tests/test_predict_cards_v8.py
import numpy as np
import pytest

from emergent.physics_maps import make_hooks_from_module
from emergent.rg import CouplingVector
from emergent.predict import (
    predict_weak_mixing_curve,
    make_card_weakmix, make_card_cosmology, make_card_edm
)


def test_weakmix_curve_band_and_summary():
    hooks = make_hooks_from_module("emergent.paper_maps.v8")
    g0 = CouplingVector(g_star=0.40, lambda_mix=0.50, theta_cp=0.20)
    curve, summ = predict_weak_mixing_curve(
        g0, q=13, R=2, k_start=120.0, k_end=1.0, n_grid=41,
        bootstrap=16, seed=42, hooks=hooks
    )
    assert curve.mean.shape == curve.k.shape == curve.lo.shape == curve.hi.shape
    # physical range checks
    assert np.all(curve.mean > 0.0) and np.all(curve.mean < 1.0)
    # EW band should widen with bootstrap
    assert float(curve.hi[-1] - curve.lo[-1]) > 0.0
    # summary keys
    assert 0.0 < float(summ["sin2_thetaW_EW"]) < 1.0
    assert 0.0 < float(summ["alpha_EM_EW"]) < 1.0


def test_cards_dataclass_and_json():
    hooks = make_hooks_from_module("emergent.paper_maps.v8")
    g0 = CouplingVector(g_star=0.30, lambda_mix=0.50, theta_cp=0.10)

    cw = make_card_weakmix(g0, q=13, R=2, k_start=120.0, k_end=1.0, n_grid=41, bootstrap=8, seed=0, hooks=hooks)
    d = cw.to_dict()
    assert "sin2_thetaW_EW" in d["central"]
    lo, hi = d["interval"]["sin2_thetaW_band@EW"]
    assert hi > lo

    cc = make_card_cosmology(q=13, R=2, hooks=hooks)
    d2 = cc.to_dict()
    assert "Lambda" in d2["central"]
    lo2, hi2 = d2["interval"]["Lambda_discrete_band"]
    assert hi2 >= lo2

    ce = make_card_edm(g0, q=13, R=2, k_start=120.0, k_end=1.0, hooks=hooks, bootstrap=8, seed=1)
    d3 = ce.to_dict()
    assert "d_n_EDM" in d3["central"]
    lo3, hi3 = d3["interval"]["d_n_EDM_band"]
    assert hi3 >= lo3
    # EDM is a magnitude-like proxy; keep non-negative
    assert d3["central"]["d_n_EDM"] >= 0.0
