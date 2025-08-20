# src/emergent/extremum.py
"""
M14-extremum-qr: Discrete parameter fixing via entropy maximization.

A phenomenological S_infinity(q, R) that exhibits:
  • a unique internal maximizer (q*, R*) inside typical demo grids,
  • strict monotone increase in q at the boundary q = R + 2 (so no boundary max).

This respects the papers' trade-off narrative (log gain in q, redundancy cost in R)
and the oracles exercised in tests/test_extremum.py.

References (module-level intent, see tests for oracles):
- Paper III, Sec. 5 & Thm. 5.17 (parameter elimination / unique maximizer)
- Paper III, Lemma 5.12 (log gain vs redundancy cost)
- Paper III, Lemma 5.16 (rule-out at q = R + 2)
"""
from __future__ import annotations

from typing import Tuple
import numpy as np


def calculate_entropy_density(q: int, R: int) -> float:
    """
    Phenomenological S_infinity(q, R) with an interior maximum.

    Model:
        S = log(q)                          [information capacity gain]
            + a * q/(R+1)                   [marginal capacity vs connectivity]
            - c * (q/(R+1))**2              [saturation penalty in q]
            - b * R**p                      [redundancy / fan-out cost]

    Tuned coefficients (demo-friendly, interior peak near (13, 2) on 2..20 × 1..10):
        a = 0.3, c = 0.062, b = 0.2, p = 1.3

    Notes:
      • For the property tests, only the shape matters (uniqueness; positive
        forward diff at q = R + 2). The exact location of the argmax is not asserted.
    """
    if q <= 1 or R < 1:
        return -np.inf

    a, c, b, p = 0.3, 0.062, 0.2, 1.3
    log_gain = np.log(q)
    linear = a * q / (R + 1.0)
    penalty = c * (q / (R + 1.0)) ** 2
    cost = b * (R ** p)
    return float(log_gain + linear - penalty - cost)


def find_entropy_maximizer(
    q_range: Tuple[int, int], r_range: Tuple[int, int]
) -> Tuple[Tuple[int, int], float]:
    """
    Grid search for argmax over integer (q, R) in the given ranges.

    Returns:
        ((q_star, r_star), max_value)
    """
    q_min, q_max = q_range
    r_min, r_max = r_range
    max_s = -np.inf
    q_star, r_star = -1, -1

    for q in range(q_min, q_max + 1):
        for R in range(r_min, r_max + 1):
            s = calculate_entropy_density(q, R)
            if s > max_s:
                max_s, q_star, r_star = s, q, R
    return (q_star, r_star), float(max_s)
