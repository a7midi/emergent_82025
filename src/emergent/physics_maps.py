# src/emergent/physics_maps.py
# Phase E: uniform hooks loader (used by the notebook and predict.py)
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple
import importlib

# Signature the rest of the suite expects
GaugeMap = Callable[[float, float, int, int, float], Tuple[float, float]]
LambdaMap = Callable[[int, int], float]
EDMMap    = Callable[[float, float, float], float]

@dataclass(frozen=True)
class Hooks:
    gauge_couplings: GaugeMap
    lambda_from_qR: LambdaMap
    edm_from_rg:    EDMMap

def make_hooks_from_module(module_path: str) -> Hooks:
    """
    Import `module_path` and call its make_hooks(). Duck-type check to ensure
    gauge_couplings/lambda_from_qR/edm_from_rg exist.
    """
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "make_hooks"):
        raise ImportError(f"{module_path} does not define make_hooks().")
    h = mod.make_hooks()
    for name in ("gauge_couplings", "lambda_from_qR", "edm_from_rg"):
        if not hasattr(h, name):
            raise TypeError(f"{module_path}.make_hooks() returned an object without '{name}'.")
    return Hooks(
        gauge_couplings=getattr(h, "gauge_couplings"),
        lambda_from_qR=getattr(h, "lambda_from_qR"),
        edm_from_rg=getattr(h, "edm_from_rg"),
    )

__all__ = ["Hooks", "make_hooks_from_module"]
