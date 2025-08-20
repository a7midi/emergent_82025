# tests/test_entropy_max.py
import numpy as np
from emergent.entropy_max import s_infty_from_series, argmax_by_grid

def test_geometric_tail_exact():
    # ρ_k = S*(1 - r^{k+1}) → Aitken Δ² is exact for any N ≥ 3
    S_star = 0.375
    r = 0.6
    N = 32
    k = np.arange(N, dtype=float)
    rho = S_star * (1.0 - r ** (k + 1.0))
    est = s_infty_from_series(rho, q=7, R=3, window=6)
    assert abs(est.S_infty - S_star) < 1e-12
    assert abs(est.tail) < 1e-12
    assert 0.0 <= est.gamma < 1.0
    assert est.band[0] <= est.S_infty <= est.band[1]

def test_argmax_stability_and_ci():
    Q = list(range(8, 15))
    R = [2, 3, 4]
    res = argmax_by_grid(Q, R, N=48, bootstrap=64, seed=0)
    assert res.q_star in Q and res.R_star in R
    assert res.stability >= 0.90
    assert res.ci_90[0] <= res.S_star <= res.ci_90[1]
