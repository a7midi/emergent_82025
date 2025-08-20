# src/emergent/gauge_fp.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Iterable
import numpy as np
from .poset import CausalSite
from .update import deterministic_update, TagConfig
from .measure import SiteMeasure
from .rg import CouplingVector

@dataclass(frozen=True)
class GaugeMapConfig:
    n_samples: int = 256
    n_burn: int = 1
    seed: int = 0

def _draw_configs(site: CausalSite, q: int, n: int, seed: int) -> Iterable[TagConfig]:
    """Draws sample tag configurations from the site measure."""
    rng = np.random.default_rng(seed)
    nodes = site.nodes
    tags = range(q)
    for _ in range(n):
        cfg: TagConfig = {v: int(rng.choice(tags)) for v in nodes}
        yield cfg

def _currents_for_config(site: CausalSite, cfg: TagConfig, q: int) -> Tuple[float, float]:
    """
    Defines local (additive) currents J_Y, J_2 from tags on edges.
    """
    JY = 0.0
    J2 = 0.0
    for u in site.nodes:
        out = site.adj.get(u, [])
        t_u = cfg[u]
        for v in out:
            JY += float(t_u)
            # FIX: Access depth via the `depths` dictionary attribute, not a method.
            J2 += float(((-1) ** site.depths[u]) * t_u)
    
    Z = max(1, len(site.edges))
    return JY / Z, J2 / Z

def estimate_kappa_matrix(
    site: CausalSite, q: int, gk: GaugeMapConfig
) -> np.ndarray:
    """
    Estimates the 2x2 connected correlator matrix K = <JJ>_c on one tick.
    """
    J = []
    for cfg in _draw_configs(site, q, gk.n_samples, seed=gk.seed):
        cfg_b = cfg
        for _ in range(gk.n_burn):
            cfg_b = deterministic_update(site, cfg_b, q)
        jy, j2 = _currents_for_config(site, cfg_b, q)
        J.append((jy, j2))
    
    J_arr = np.asarray(J)
    mean = J_arr.mean(axis=0, keepdims=True)
    X = J_arr - mean
    K = (X.T @ X) / max(1, len(J_arr))
    return K

def gauge_couplings_analytic(
    g: CouplingVector, q: int, R: int, k: float, *, site_builder: Callable[[], CausalSite],
    gk: GaugeMapConfig = GaugeMapConfig()
) -> Tuple[float, float]:
    """
    Paper-faithful gauge map: computes (g1,g2) by diagonalizing the connected
    current-current matrix K at depth k.
    """
    site = site_builder()
    K = estimate_kappa_matrix(site, q, gk)
    
    evals = np.linalg.eigvalsh(K)
    gY_inv2, g2_inv2 = float(evals[0]), float(evals[1])

    if g2_inv2 <= 1e-16:
        scale = 1.0
    else:
        scale = (1.0 / (g.g_star ** 2)) / g2_inv2
        
    g2 = float(np.sqrt(1.0 / (scale * g2_inv2 + 1e-24)))
    gY = float(np.sqrt(1.0 / (scale * gY_inv2 + 1e-24)))
    
    g1 = float(np.sqrt(5.0/3.0) * gY)
    return g1, g2