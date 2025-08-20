# src/emergent/predict.py
"""
Prediction utilities and publication-ready cards (Phase E—P).

This module provides:
- A reproducible weak-mixing curve prediction using the *paper hooks* gauge map.
- Paper-style publication cards for weak mixing, cosmology (Λ from (q,R)), and the neutron EDM.
- Deterministic uncertainty bands via physics-driven priors or bootstrap perturbations.

Key paper references (as implemented in hooks):
- Gauge map (g1,g2) from (g*, λ_mix): Paper III §3.9.2 (analytic β-system solution via fusion algebra).
- Cosmology Λ(q,R): direct map Λ = Λ(q,R) (Paper III, Theorem “(q,R)→Λ” implemented in hooks.lambda_from_qR).
- Entropy/complexity facts used for discrete (q,R) selection bands: Paper I Prop. 8.23.

API stability: existing call sites that import `predict_weak_mixing_curve` and the
`make_card_*` helpers continue to work. We add a light Card dataclass with `.to_dict()`
but still return the same (Curve, summary) from `predict_weak_mixing_curve`.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional, Callable, Any
import math
import numpy as np
from scipy.integrate import solve_ivp

from .rg import CouplingVector, beta_function

try:
    # Optional: paper hooks loader
    from .physics_maps import make_hooks_from_module, Hooks
except Exception:  # pragma: no cover – tight import guard
    make_hooks_from_module = None
    Hooks = object  # type: ignore


# ------------------------------
# Data containers
# ------------------------------

@dataclass(frozen=True)
class Curve:
    k: np.ndarray
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


@dataclass(frozen=True)
class Card:
    """Publication-ready card container.

    Attributes
    ----------
    title : str
        Human-readable name of the prediction.
    central : Dict[str, float]
        Central (point-estimate) values.
    interval : Dict[str, Tuple[float, float]]
        Uncertainty bands, always (lo, hi).
    meta : Dict[str, Any]
        Optional metadata: inputs, seeds, hooks module, notes.
    """
    title: str
    central: Dict[str, float]
    interval: Dict[str, Tuple[float, float]]
    meta: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # numpy scalars are not JSON serializable by default
        def _clean(v):
            if isinstance(v, np.generic):
                return v.item()
            if isinstance(v, (np.ndarray,)):
                return v.tolist()
            return v
        d["central"] = {k: _clean(v) for k, v in d["central"].items()}
        d["interval"] = {k: tuple(map(float, v)) for k, v in d["interval"].items()}
        if d.get("meta") is None:
            d["meta"] = {}
        return d


# ------------------------------
# Helpers
# ------------------------------

def _observables_from_gauge(g1: float, g2: float) -> tuple[float, float]:
    """Compute s2 := sin^2(theta_W) and α_EM from SU(2)_L and U(1)_Y couplings.

    Uses the canonical GUT normalisation g_Y^2 = (3/5) g1^2.
    α_EM = g_Y^2 g2^2 / [4π (g_Y^2 + g2^2)].
    """
    gY2 = (3.0 / 5.0) * (g1 * g1)
    denom = gY2 + g2 * g2
    s2 = gY2 / denom if denom > 0 else 0.0
    alpha = (gY2 * g2 * g2) / (4.0 * math.pi * denom) if denom > 0 else 0.0
    return float(s2), float(alpha)


# ------------------------------
# Prediction core
# ------------------------------

def predict_weak_mixing_curve(
    g0: CouplingVector, *, q: int, R: int,
    k_start: float, k_end: float, n_grid: int = 121,
    bootstrap: int = 0, seed: Optional[int] = None,
    hooks=None,
    param_prior: Optional[Callable[[np.random.Generator], CouplingVector]] = None,
) -> Tuple[Curve, Dict[str, float]]:
    """
    Predict sin^2(theta_W)(k) with uncertainties.

    Parameters
    ----------
    g0 : CouplingVector
        Initial RG couplings at k_start.
    q, R : int
        Paper parameters (alphabet and branching proxy).
    k_start, k_end : float
        RG depth range (μ = GeV0·2^{-k}).
    n_grid : int
        Number of k-samples (inclusive). Default 121.
    bootstrap : int
        If >0, draw this many bootstrap samples either from a `param_prior`
        callback (preferred) or from a small deterministic Gaussian jitter.
    seed : Optional[int]
        Seed for deterministic sampling.
    hooks : module or Hooks
        Paper hooks providing `gauge_couplings(g*, λ_mix, q, R, k)`.
    param_prior : Optional[Callable[[np.random.Generator], CouplingVector]]
        If provided, used to draw bootstrap initial conditions; otherwise we
        look for `hooks.draw_params(rng)`; otherwise we fall back to jitters.

    Returns
    -------
    (Curve, summary) where summary contains EW-endpoint observables.
    """
    if hooks is None:
        # Conservative default: load paper v8 hooks if available
        if make_hooks_from_module is None:
            raise ValueError("hooks must be provided (paper hooks module).")
        hooks = make_hooks_from_module("emergent.paper_maps.v8")

    k_grid = np.linspace(float(k_start), float(k_end), int(n_grid))

    # integrate once with dense output
    sol = solve_ivp(
        fun=beta_function,
        t_span=(float(k_start), float(k_end)),
        y0=g0.to_array(), args=(q, R), method="RK45",
        dense_output=True,
    )

    def g_at(k: float) -> CouplingVector:
        y = sol.sol(float(k))
        return CouplingVector.from_array(y)

    # central curve
    s2_list: List[float] = []
    a_list:  List[float] = []
    for kk in k_grid:
        gv = g_at(kk)
        g1, g2 = hooks.gauge_couplings(gv.g_star, gv.lambda_mix, q, R, kk)
        s2, a = _observables_from_gauge(g1, g2)
        s2_list.append(s2); a_list.append(a)

    # (optional) bootstrap on paper-driven priors
    lo = np.array(s2_list); hi = np.array(s2_list)
    if bootstrap and int(bootstrap) > 0:
        rng = np.random.default_rng(seed)
        all_curves = []
        for _ in range(int(bootstrap)):
            if param_prior is not None:
                g0b = param_prior(rng)
            elif hasattr(hooks, "draw_params") and callable(hooks.draw_params):  # type: ignore[attr-defined]
                gp = hooks.draw_params(rng)  # expected to return dict or CouplingVector
                if isinstance(gp, CouplingVector):
                    g0b = gp
                else:
                    # tolerate dict-like
                    g0b = CouplingVector(
                        g_star=float(gp.get("g_star", g0.g_star)),
                        lambda_mix=float(gp.get("lambda_mix", g0.lambda_mix)),
                        theta_cp=float(gp.get("theta_cp", g0.theta_cp)),
                    )
            else:
                # deterministic small jitter around g0 (keeps tests fast)
                g0b = CouplingVector(
                    g_star=g0.g_star * (1.0 + 0.01 * rng.normal()),
                    lambda_mix=g0.lambda_mix * (1.0 + 0.03 * rng.normal()),
                    theta_cp=g0.theta_cp,
                )

            solb = solve_ivp(beta_function, (k_start, k_end), g0b.to_array(), args=(q, R), dense_output=True)
            cb = []
            for kk in k_grid:
                yb = solb.sol(float(kk))
                gb = CouplingVector.from_array(yb)
                g1b, g2b = hooks.gauge_couplings(gb.g_star, gb.lambda_mix, q, R, kk)
                s2b, _ = _observables_from_gauge(g1b, g2b)
                cb.append(s2b)
            all_curves.append(cb)
        all_curves = np.array(all_curves, dtype=float)
        lo = np.percentile(all_curves, 16, axis=0)
        hi = np.percentile(all_curves, 84, axis=0)

    curve = Curve(k=k_grid, mean=np.array(s2_list, dtype=float), lo=lo, hi=hi)

    # summary at the EW end
    gv_end = g_at(float(k_end))
    g1_EW, g2_EW = hooks.gauge_couplings(gv_end.g_star, gv_end.lambda_mix, q, R, float(k_end))
    s2_EW, a_EW = _observables_from_gauge(g1_EW, g2_EW)
    summary: Dict[str, float] = {
        "sin2_thetaW_EW": float(s2_EW),
        "alpha_EM_EW": float(a_EW),
        "g_star_EW": float(gv_end.g_star),
        "lambda_mix_EW": float(gv_end.lambda_mix),
        "theta_cp_EW": float(gv_end.theta_cp),
    }
    return curve, summary


# ------------------------------
# Cards
# ------------------------------

def _infer_hooks_module_name(hooks) -> str:
    try:
        mod = hooks.__module__  # type: ignore[attr-defined]
    except Exception:
        mod = ""
    return str(mod)


def make_card_weakmix(
    g0: CouplingVector, *, q: int, R: int, k_start: float, k_end: float,
    n_grid: int = 121, bootstrap: int = 0, seed: Optional[int] = None, hooks=None,
    param_prior: Optional[Callable[[np.random.Generator], CouplingVector]] = None,
) -> Card:
    """Paper-ready weak mixing card.

    Uses the analytic paper gauge map exposed by `hooks` and returns
    a `Card` with both the EW-endpoint estimates and an uncertainty band
    computed via the same priors used in the weak-mixing curve.
    """
    curve, summary = predict_weak_mixing_curve(
        g0, q=q, R=R, k_start=k_start, k_end=k_end, n_grid=n_grid,
        bootstrap=bootstrap, seed=seed, hooks=hooks, param_prior=param_prior
    )
    ew_band = (float(curve.lo[-1]), float(curve.hi[-1]))
    card = Card(
        title="Weak mixing prediction",
        central=summary,
        interval={"sin2_thetaW_band@EW": ew_band},
        meta={
            "k_start": float(k_start), "k_end": float(k_end), "n_grid": int(n_grid),
            "bootstrap": int(bootstrap), "seed": (None if seed is None else int(seed)),
            "hooks": _infer_hooks_module_name(hooks),
        },
    )
    return card


def make_card_cosmology(
    *, q: int = 6, R: int = 4, hooks=None,
    include_discrete_uncertainty: bool = True
) -> Card:
    """Cosmology card using the paper's direct map Λ = Λ(q,R).

    The central value uses `hooks.lambda_from_qR`. We attach a discrete-selection
    band by evaluating neighbouring integer choices (q±1, R±1) where valid.
    This provides a falsifiable, parameter-selection uncertainty without any
    phenomenological fudge factors.

    Paper reference: see Paper III Theorem mapping (q,R) to Λ (implemented
    in hooks.lambda_from_qR). Entropy slope variants are deprecated.

    Also referenced: deterministic entropy increment that motivates the discrete
    site-count priors on q and branching R (Paper I Prop. 8.23).
    """
    if hooks is None:
        if make_hooks_from_module is None:
            raise ValueError("hooks must be provided (paper hooks module).")
        hooks = make_hooks_from_module("emergent.paper_maps.v8")

    lam0 = float(hooks.lambda_from_qR(int(q), int(R)))
    lo = hi = lam0
    if include_discrete_uncertainty:
        candidates = set()
        for dq in (-1, 0, 1):
            for dR in (-1, 0, 1):
                qq = int(q) + dq; RR = int(R) + dR
                if qq >= 2 and RR >= 2:
                    candidates.add((qq, RR))
        vals = [float(hooks.lambda_from_qR(qq, RR)) for (qq, RR) in sorted(candidates)]
        lo = float(min(vals)); hi = float(max(vals))
    card = Card(
        title="Cosmology (Λ from (q,R))",
        central={"Lambda": lam0},
        interval={"Lambda_discrete_band": (lo, hi)},
        meta={"q": int(q), "R": int(R), "hooks": _infer_hooks_module_name(hooks)},
    )
    return card


def make_card_edm(
    g0: CouplingVector, *, q: int, R: int, k_start: float, k_end: float,
    hooks=None, bootstrap: int = 0, seed: Optional[int] = None,
    param_prior: Optional[Callable[[np.random.Generator], CouplingVector]] = None,
) -> Card:
    """Neutron EDM card from depth-parity CPV with proper RG-evolved θ_CP^EW.

    Central value is computed at the EW end using `hooks.edm_from_rg(g*, λ_mix, θ_cp)`
    with the RG-evolved couplings at k_end. An optional bootstrap (with the same
    priors as weak-mixing) produces a ±1σ interval.

    Paper reference: Paper III, Conj. 7.13 (depth-parity CPV scaling of EDM proxy).
    """
    if hooks is None:
        if make_hooks_from_module is None:
            raise ValueError("hooks must be provided (paper hooks module).")
        hooks = make_hooks_from_module("emergent.paper_maps.v8")

    # Evolve once to EW
    sol = solve_ivp(beta_function, (k_start, k_end), g0.to_array(), args=(q, R), dense_output=True)
    y = sol.sol(float(k_end))
    gv = CouplingVector.from_array(y)

    d0 = float(hooks.edm_from_rg(gv.g_star, gv.lambda_mix, gv.theta_cp))

    lo = hi = d0
    if bootstrap and int(bootstrap) > 0:
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(int(bootstrap)):
            if param_prior is not None:
                g0b = param_prior(rng)
            elif hasattr(hooks, "draw_params") and callable(hooks.draw_params):  # type: ignore[attr-defined]
                gp = hooks.draw_params(rng)
                if isinstance(gp, CouplingVector):
                    g0b = gp
                else:
                    g0b = CouplingVector(
                        g_star=float(gp.get("g_star", g0.g_star)),
                        lambda_mix=float(gp.get("lambda_mix", g0.lambda_mix)),
                        theta_cp=float(gp.get("theta_cp", g0.theta_cp)),
                    )
            else:
                g0b = CouplingVector(
                    g_star=g0.g_star * (1.0 + 0.01 * rng.normal()),
                    lambda_mix=g0.lambda_mix * (1.0 + 0.03 * rng.normal()),
                    theta_cp=g0.theta_cp,
                )
            solb = solve_ivp(beta_function, (k_start, k_end), g0b.to_array(), args=(q, R), dense_output=True)
            yb = solb.sol(float(k_end))
            gb = CouplingVector.from_array(yb)
            samples.append(float(hooks.edm_from_rg(gb.g_star, gb.lambda_mix, gb.theta_cp)))
        samples = np.array(samples, dtype=float)
        lo = float(np.percentile(samples, 16))
        hi = float(np.percentile(samples, 84))

    card = Card(
        title="Neutron EDM prediction (proxy scale)",
        central={"d_n_EDM": d0, "theta_cp_EW": float(gv.theta_cp)},
        interval={"d_n_EDM_band": (lo, hi)},
        meta={
            "k_start": float(k_start), "k_end": float(k_end),
            "bootstrap": int(bootstrap), "seed": (None if seed is None else int(seed)),
            "hooks": _infer_hooks_module_name(hooks),
        },
    )
    return card


# --- Paper reference note (explicit comment for the entropy/complexity lemma) ---
# The deterministic entropy increment used to motivate discrete q,R priors is
# Proposition 8.23 in Paper I (uploaded with the suite).


# Convenience export (kept minimal to avoid extra deps)
def export_card_json(card: Card, path: str) -> None:
    """Write a Card to JSON on disk (UTF-8)."""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card.to_dict(), f, indent=2, sort_keys=False)
