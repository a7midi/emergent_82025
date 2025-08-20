# tests/test_paper_v8_gauge.py
"""
Phase E.1 — analytic gauge map tests (paper v8)

We check:
- α_EM ∈ (0,1) and sin^2 θ_W ∈ (0,1) across a grid of (g_star, lambda_mix).
- Monotonic sensitivity of sin^2 θ_W to |lambda_mix| at fixed g_star.
"""

import numpy as np

from emergent.paper_maps import v8 as paper_v8


def _observables_from_gauge(g1: float, g2: float):
    """SM electroweak identities (GUT-normalised): alpha_EM, sin^2 theta_W."""
    # α1 = g1^2/(4π), α2 = g2^2/(4π)
    a1 = (g1 * g1) / (4.0 * np.pi)
    a2 = (g2 * g2) / (4.0 * np.pi)
    alpha_em = (a1 * a2) / max(1e-24, (a1 + a2))
    sin2 = a1 / max(1e-24, (a1 + a2))
    return float(alpha_em), float(sin2)


def test_alpha_and_weakmix_ranges():
    hooks = paper_v8.make_hooks()

    q, R = 6, 4
    # grid of small couplings within the analytic polydisc region (Paper III)
    vals = []
    for gs in [1e-3, 5e-3, 1e-2, 5e-2]:
        for lm in [0.0, 1e-3, 5e-3, 1e-2]:
            g1, g2 = hooks.gauge_couplings(gs, lm, q, R, k=10.0)
            a, s2 = _observables_from_gauge(g1, g2)
            vals.append((a, s2))

    # Check ranges
    for alpha_em, sin2 in vals:
        assert 0.0 < alpha_em < 1.0
        assert 0.0 <= sin2 <= 1.0


def test_monotone_sensitivity_in_lambda():
    hooks = paper_v8.make_hooks()
    q, R = 6, 4
    gs = 0.02  # fixed g_star

    lam_seq = np.linspace(0.0, 0.02, 9)
    s2_seq = []
    for lm in lam_seq:
        g1, g2 = hooks.gauge_couplings(gs, float(lm), q, R, k=8.0)
        _, s2 = _observables_from_gauge(g1, g2)
        s2_seq.append(s2)

    # As |lambda_mix| grows, the Abelian piece increases ⇒ sin^2 θ_W should not decrease.
    # (Not necessarily strictly monotone for tiny steps; we require non-decreasing.)
    assert all(s2_seq[i] <= s2_seq[i+1] + 1e-12 for i in range(len(s2_seq)-1))
