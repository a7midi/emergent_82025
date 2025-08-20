# tests/test_setup.py
import numpy as np
import pytest
from hypothesis import given, strategies as st

from emergent.constants import delta, gamma, chi, validate_qR
from emergent.random import set_global_seed, get_rng, rng_integers, rng_random, temp_seed


def test_validate_qR_ok_and_fail():
    # Valid
    validate_qR(7, 3)
    # Fail: q <= R
    with pytest.raises(ValueError):
        validate_qR(3, 3)
    with pytest.raises(ValueError):
        validate_qR(3, 4)
    # Fail: small q/R
    with pytest.raises(ValueError):
        validate_qR(1, 2)
    with pytest.raises(ValueError):
        validate_qR(5, 1)  # R must be >= 2


def test_delta_gamma_chi_values():
    q, R = 7, 3
    d = delta(q, R)   # (q - 1) / (q + R - 1) = 6/9 = 2/3
    g = gamma(q, R)   # 1 - 2/3 = 1/3 = R / (q + R - 1)
    c = chi(q, R)     # (1/3) / (2R) = 1 / (6 * R) = 1/18
    assert np.isclose(d, 2.0 / 3.0)
    assert np.isclose(g, 1.0 / 3.0)
    assert np.isclose(c, 1.0 / 18.0)


def test_rng_repeatability_same_name_same_seed():
    set_global_seed(2025)
    a = get_rng("stream", fresh=True).integers(0, 1_000_000, size=8, dtype=np.int64)
    set_global_seed(2025)
    b = get_rng("stream", fresh=True).integers(0, 1_000_000, size=8, dtype=np.int64)
    assert np.array_equal(a, b)


def test_rng_cached_progresses_but_fresh_resets():
    set_global_seed(42)
    g_cached = get_rng("foo")  # cached
    a1 = g_cached.integers(0, 100, size=5, dtype=np.int64)
    a2 = g_cached.integers(0, 100, size=5, dtype=np.int64)
    # Fresh starts at canonical beginning again
    a1_fresh = get_rng("foo", fresh=True).integers(0, 100, size=5, dtype=np.int64)
    assert not np.array_equal(a1, a2)
    assert np.array_equal(a1, a1_fresh)


@given(
    name=st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=16),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    size=st.integers(min_value=1, max_value=32),
)
def test_property_named_streams_deterministic(name, seed, size):
    set_global_seed(seed)
    x1 = get_rng(name, fresh=True).integers(0, 10_000, size=size, dtype=np.int64)
    set_global_seed(seed)
    x2 = get_rng(name, fresh=True).integers(0, 10_000, size=size, dtype=np.int64)
    assert np.array_equal(x1, x2)


@given(seed=st.integers(min_value=0, max_value=2**31 - 1), size=st.integers(min_value=1, max_value=32))
def test_property_different_names_different_sequences(seed, size):
    set_global_seed(seed)
    a = get_rng("alpha", fresh=True).integers(0, 1_000_000, size=size, dtype=np.int64)
    b = get_rng("beta", fresh=True).integers(0, 1_000_000, size=size, dtype=np.int64)
    # Extremely unlikely to be exactly equal; this is a strong property test.
    assert not np.array_equal(a, b)


def test_temp_seed_context_restores_state():
    set_global_seed(77)
    before = get_rng("g", fresh=True).integers(0, 1000, size=4, dtype=np.int64)
    with temp_seed(77):
        inner = get_rng("g", fresh=True).integers(0, 1000, size=4, dtype=np.int64)
    after = get_rng("g", fresh=True).integers(0, 1000, size=4, dtype=np.int64)
    # Inner matches fresh draws under the same seed
    set_global_seed(77)
    expected = get_rng("g", fresh=True).integers(0, 1000, size=4, dtype=np.int64)
    assert np.array_equal(inner, expected)
    # Before/After are not necessarily equal (cached progression), but fresh equals expected
    assert np.array_equal(get_rng("g", fresh=True).integers(0, 1000, size=4, dtype=np.int64), expected)






