# src/emergent/types.py
"""
Global type aliases used across modules (M00).

We keep numeric dtypes explicit to avoid accidental dtype upcasts.

Notes
-----
- All integer quantities that count vertices, depths, or arrows are int64.
- Probabilities are float64 in [0, 1].
"""
from __future__ import annotations

from typing import NewType, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt

# Scalar type aliases
Depth = NewType("Depth", np.int64)
VertexId = NewType("VertexId", np.int64)
Tag = NewType("Tag", np.int64)
Prob = NewType("Prob", np.float64)

# Compound structures
Arrow: TypeAlias = Tuple[VertexId, VertexId]

# Array type aliases
FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
