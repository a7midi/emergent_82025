# src/emergent/random.py
"""
Deterministic RNG pool (M00).

We use NumPy's Generator(PCG64) with a SeedSequence derived from a base seed
and a (stable) hash of the requested stream name. This gives:

- Order-independent named substreams (same name => same sequence),
- Reproducibility across processes/platforms (PCG64 is stable across NumPy 1.26),
- Pure functions available via helpers (`rng_integers`, `rng_random`).

Notes
-----
- Avoid Python's built-in hash (salted). We use BLAKE2b to derive stable 128-bit keys.
- All helpers return arrays with explicit dtypes: int64 for integers, float64 for floats.
- Absolutely no access to OS entropy; default seed is fixed (constants.DEFAULT_SEED).

This RNG supports the mixing/spectral-gap numerics that depend on δ, γ, χ
(see constants module for equations). :contentReference[oaicite:5]{index=5}
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from numpy.random import Generator, PCG64, SeedSequence

from .constants import DEFAULT_SEED, INT_DTYPES, FLOAT_DTYPES


def _blake2b_u32_words(seed: int, name: str) -> list[int]:
    """Return 4 uint32 words from blake2b(seed||'|'||name)."""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    h.update(str(int(seed)).encode("utf-8"))
    h.update(b"|")
    h.update(name.encode("utf-8"))
    raw = h.digest()  # 16 bytes
    # interpret as four uint32 words in little-endian
    return list(np.frombuffer(raw, dtype=np.uint32))


@dataclass
class RNGPool:
    """
    A pool of named, order-independent deterministic RNG streams.

    Parameters
    ----------
    base_seed : int
        Global base seed. Changing it reinitialises all streams.

    Notes
    -----
    - Substream "names" are mapped to 128-bit keys via BLAKE2b(seed,name).
    - `generator(name, fresh=True)` creates a fresh Generator whose first draw
      is independent of previous requests; `fresh=False` caches and continues.
    """
    base_seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        self._cache: Dict[str, Generator] = {}
        self._base_ss = SeedSequence([np.uint32(self.base_seed)])

    def _make_generator(self, name: str) -> Generator:
        words = _blake2b_u32_words(self.base_seed, name)
        ss = SeedSequence(words)
        return Generator(PCG64(ss))

    def generator(self, name: Optional[str] = None, *, fresh: bool = False) -> Generator:
        """
        Get a Generator for a named stream.

        Parameters
        ----------
        name : Optional[str]
            Stream name. None means the "global" stream "global".
        fresh : bool
            If True, return a new generator at the canonical start of the named stream.
            If False, return a cached generator (create and cache if missing).

        Returns
        -------
        numpy.random.Generator
        """
        key = "global" if name is None else str(name)
        if fresh:
            return self._make_generator(key)
        if key not in self._cache:
            self._cache[key] = self._make_generator(key)
        return self._cache[key]

    def integers(
        self, low: int, high: Optional[int] = None, size: int | tuple[int, ...] = 1, name: Optional[str] = None
    ) -> np.ndarray:
        """Draw deterministic int64 integers from the named stream."""
        g = self.generator(name)
        return g.integers(low=low, high=high, size=size, dtype=INT_DTYPES.index, endpoint=False)

    def random(
        self, size: int | tuple[int, ...] = 1, name: Optional[str] = None
    ) -> np.ndarray:
        """Draw deterministic float64 uniform(0,1) from the named stream."""
        g = self.generator(name)
        return g.random(size=size, dtype=FLOAT_DTYPES.scalar)


# module-level singleton pool (mutable via set_global_seed)
_pool = RNGPool(DEFAULT_SEED)


def set_global_seed(seed: int) -> None:
    """
    Set the global deterministic seed for all named RNG streams.

    This resets the internal pool; all `fresh=True` generators and new cached
    generators will reproduce sequences implied by the new seed.
    """
    global _pool
    _pool = RNGPool(int(seed))


def get_rng(name: Optional[str] = None, *, fresh: bool = False) -> Generator:
    """Convenience accessor for the module-level pool."""
    return _pool.generator(name, fresh=fresh)


def rng_integers(
    low: int, high: Optional[int] = None, size: int | tuple[int, ...] = 1, name: Optional[str] = None
) -> np.ndarray:
    """Draw deterministic int64 integers from the module-level pool."""
    return _pool.integers(low=low, high=high, size=size, name=name)


def rng_random(size: int | tuple[int, ...] = 1, name: Optional[str] = None) -> np.ndarray:
    """Draw deterministic float64 U(0,1) from the module-level pool."""
    return _pool.random(size=size, name=name)


@contextmanager
def temp_seed(seed: int):
    """
    Context manager that temporarily sets the global seed.

    Examples
    --------
    >>> import numpy as np
    >>> from emergent.random import rng_random, temp_seed
    >>> with temp_seed(123):
    ...     a = rng_random(3, name="demo")
    >>> with temp_seed(123):
    ...     b = rng_random(3, name="demo")
    >>> np.allclose(a, b)
    True
    """
    global _pool
    old_pool = _pool
    try:
        _pool = RNGPool(int(seed))
        yield
    finally:
        _pool = old_pool