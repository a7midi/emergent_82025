# tests/conftest.py
"""
M16—Global test configuration:

- Registers a deterministic Hypothesis profile to avoid flakiness in CI.
- Sets global NumPy print/float options for stable snapshots.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import settings

# Deterministic property-based tests: fixed number of examples, no timeouts.
settings.register_profile(
    "ci",
    max_examples=25,
    derandomize=True,   # stable across runs
    deadline=None,      # avoid timeouts on slower CI machines
)
settings.load_profile("ci")

@pytest.fixture(scope="session", autouse=True)
def _numpy_print_options():
    old = np.get_printoptions()
    np.set_printoptions(precision=6, suppress=True)
    yield
    np.set_printoptions(**old)
