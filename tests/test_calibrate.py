# tests/test_calibrate.py
import math
from emergent.rg import CouplingVector
from emergent.physics_maps import make_hooks_from_module
from emergent.calibrate import calibrate_weakmix_gstar

def test_calibrates_weakmix_within_tol():
    hooks = make_hooks_from_module("emergent.paper_maps.v8")
    g_template = CouplingVector(g_star=0.35, lambda_mix=0.5, theta_cp=0.2)
    cal = calibrate_weakmix_gstar(
        g_template, q=13, R=2, k_start=120.0, k_end=1.0,
        target_sin2=0.23122, hooks=hooks
    )
    assert cal["success"], cal
    assert cal["residual"] <= 5e-4
