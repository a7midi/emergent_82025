# src/emergent/calibrate.py
"""
Calibration helpers used by notebooks and tests.

This version guarantees that, when `target_alpha_EW` is supplied, the
overall gauge-map scale ξ₂ is *always* calibrated after matching
sin²θ_W at the electroweak (EW) scale.  This enforces the paper’s
normalisation freedom (Paper III §3.9.2) without touching the shape
of the weak-mixing curve.
"""
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from .rg import CouplingVector
from .predict import predict_weak_mixing_curve

# --------------------------
# utilities
# --------------------------
def _find_bracket(f, lo: float, hi: float, n_scan: int = 33, expand_steps: int = 2, expand: float = 1.7):
    a, b = float(lo), float(hi)
    for _ in range(expand_steps + 1):
        xs = np.linspace(a, b, n_scan)
        ys = [float(f(x)) for x in xs]
        for i in range(len(xs) - 1):
            if ys[i] == 0.0:
                return xs[i], xs[i]
            if ys[i] * ys[i + 1] < 0.0:
                return xs[i], xs[i + 1]
        a, b = a / expand, b * expand
    return None

# --------------------------
# one-target (sin²) helper
# --------------------------
def calibrate_weakmix_gstar(
    g_template: CouplingVector, *, q: int, R: int, k_start: float, k_end: float,
    target_sin2: float, hooks, scan_lo: float = 0.05, scan_hi: float = 1.5
) -> Dict[str, Any]:
    def f(g_star: float) -> float:
        g0 = CouplingVector(g_star=float(g_star), lambda_mix=g_template.lambda_mix, theta_cp=g_template.theta_cp)
        _, s = predict_weak_mixing_curve(g0, q=q, R=R, k_start=k_start, k_end=k_end, n_grid=91, hooks=hooks)
        return float(s["sin2_thetaW_EW"] - target_sin2)

    br = _find_bracket(f, scan_lo, scan_hi)
    if br is not None:
        a, b = br
        root = a if a == b else brentq(f, a, b)
        res = abs(f(root))
        return {"g_star_cal": float(root), "residual": float(res), "success": True, "message": "brentq", "bracket": (float(a), float(b))}
    # bounded fallback
    sol = minimize_scalar(lambda x: abs(f(x)), bounds=(scan_lo, scan_hi), method="bounded")
    return {"g_star_cal": float(sol.x), "residual": float(abs(f(sol.x))), "success": True, "message": "bounded minimization", "bracket": None}

# --------------------------
# two-target (sin² & α) helper
# --------------------------
def calibrate_two_anchors(
    g_template: CouplingVector, *, q: int, R: int, k_start: float, k_end: float,
    target_sin2_EW: float, mu_EW_GeV: float, hooks,
    mode: str = "g_and_lambda", target_alpha_EW: Optional[float] = None,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.05, 1.50), (0.05, 1.50)),
    n_grid: int = 17
) -> Dict[str, Any]:
    """
    First match sin²θ_W at EW by scanning (g*, λ_mix) in a bounded box.
    Then, if target_alpha_EW is provided, set ξ₂ to match α_EM at EW.
    """
    # 0) anchor GeV0 immediately
    from .paper_maps import v8 as paper
    GeV0 = float(mu_EW_GeV) * (2.0 ** float(k_end))
    paper.set_GeV0(GeV0)

    # 1) grid search to match sin²θ_W
    (g_lo, g_hi), (l_lo, l_hi) = bounds
    Gs = np.linspace(g_lo, g_hi, n_grid)
    Ls = np.linspace(l_lo, l_hi, n_grid)
    t_s2 = float(target_sin2_EW)

    def eval_obs(g_star: float, lam: float) -> Tuple[float, float]:
        g0 = CouplingVector(g_star=float(g_star), lambda_mix=float(lam), theta_cp=g_template.theta_cp)
        _, s = predict_weak_mixing_curve(g0, q=q, R=R, k_start=k_start, k_end=k_end, n_grid=81, bootstrap=0, seed=0, hooks=hooks)
        return float(s["sin2_thetaW_EW"]), float(s["alpha_EM_EW"])

    best_x, best_val = None, float("inf")
    for gv in Gs:
        for lv in Ls:
            s2, _ = eval_obs(gv, lv)
            val = abs(s2 - t_s2)
            if val < best_val:
                best_val, best_x = val, (gv, lv)
    g_star_cal, lambda_cal = best_x
    cal: Dict[str, Any] = {
        "g_star_cal": float(g_star_cal),
        "lambda_mix_cal": float(lambda_cal),
        "success": True,
        "message": "bounded minimization (grid refine)",
        "residual": float(best_val),
        "bracket": None,
        "GeV0": GeV0,
    }

    # 2) set ξ₂ if α target provided
    if target_alpha_EW is not None:
        # evaluate α at EW once with ξ₂=1
        old = paper.get_xi2()
        paper.set_xi2(1.0)
        g0 = CouplingVector(g_star=float(g_star_cal), lambda_mix=float(lambda_cal), theta_cp=g_template.theta_cp)
        _, s_base = predict_weak_mixing_curve(g0, q=q, R=R, k_start=k_start, k_end=k_end, n_grid=21, bootstrap=0, seed=0, hooks=hooks)
        alpha_base = float(s_base["alpha_EM_EW"])
        # α ∝ ξ₂^2  ⇒  ξ₂ = sqrt(alpha_target / alpha_base)
        xi2 = math.sqrt(max(1e-30, float(target_alpha_EW)) / max(1e-30, alpha_base))
        paper.set_xi2(xi2)  # persist for the session
        cal["xi2_cal"] = float(xi2)
        # restore old if you want isolation (but we keep calibrated value on purpose)
        # paper.set_xi2(old)

    return cal
