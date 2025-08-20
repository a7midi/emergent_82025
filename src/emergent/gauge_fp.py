# src/emergent/gauge_fp.py
"""
Stateless analytic gauge coupling map used by the paper hooks.

This is *not* the RG integrator. It simply maps the running state
(g_star, lambda_mix) at a particular depth k to (g1, g2), in the spirit
of the paper’s map where g* anchors SU(2)_L and lambda_mix tilts U(1)_Y.

The actual RG running of (g_star, lambda_mix, theta_cp) should be handled
by your ODE integrator; this file is just a pointwise map.
"""
from __future__ import annotations
import math


def g1_g2_analytic(
    g_star: float, lambda_mix: float, q: int, R: int, k: float
) -> tuple[float, float]:
    """
    Map (g_star, lambda_mix, q, R, k) -> (g1, g2) at scale k.

    - g2 is identified with the SU(2)_L coupling.
    - g1 is constructed from g2 and a saturating function of lambda_mix with
      a mild k-dependence to keep behavior well-posed across your scan.

    This is intentionally stateless and numerically stable.
    """
    g2 = max(1e-12, float(g_star))
    lm = max(0.0, float(lambda_mix))

    # Smooth, bounded “tilt”. 0 <= w0 < 0.75 ; decays slowly with k.
    w0 = 0.75 * (1.0 - math.exp(-0.25 * lm))
    wk = w0 / (1.0 + 0.01 * max(0.0, float(k)))

    g1 = g2 * (1.0 + wk)
    return float(g1), float(g2)


__all__ = ["g1_g2_analytic"]
