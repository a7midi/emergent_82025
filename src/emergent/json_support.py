# src/emergent/json_support.py
"""
Utilities to make prediction outputs JSON-serializable.

This keeps the public API stable: you can still return Card objects
from make_card_*(), and the default= argument in json.dumps will
convert them (and numpy scalars/arrays) into plain Python types.

Usage:
    import json
    from emergent.json_support import to_jsonable
    json.dumps(obj, default=to_jsonable)

Author: Emergent Suite v8 hardening
"""

from __future__ import annotations
from dataclasses import is_dataclass, asdict
from typing import Any
import numpy as np

def to_jsonable(obj: Any) -> Any:
    """
    Convert common scientific/Python objects to JSON-serializable forms:
    - objects with .to_dict() -> dict
    - dataclasses -> dict
    - numpy scalars -> native Python scalars
    - numpy arrays -> lists
    - sets/tuples -> lists
    - anything else -> TypeError (let json fall back to its own error)
    """
    # 1) User-defined types with explicit converter
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return obj.to_dict()
        except Exception:
            # Fall through to dataclass/other handling
            pass

    # 2) Dataclasses
    if is_dataclass(obj):
        return asdict(obj)

    # 3) Numpy types
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # 4) Simple containers that JSON can take as lists
    if isinstance(obj, (set, tuple)):
        return list(obj)

    # 5) Leave dict/list/str/int/float/bool/None unchanged
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj

    # 6) Let json raise on unknowns (helps surface hidden problems)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable; "
                    f"provide .to_dict() or convert it before dumping.")
