# src/emergent/cli.py
from __future__ import annotations

import json
from typing import Optional, Sequence, List, Tuple

import click

# Public CLI depends only on the entropy module for the "extremum" path.
from .entropy_max import argmax_by_grid


@click.group(help="Emergent — research utilities CLI")
def app() -> None:
    """Root click Group. Tests import this symbol."""


@app.command(help="Scan a (q,R) grid and report the entropy-extremum argmax.")
@click.option("--qmin", type=int, required=True, help="Minimum q (inclusive).")
@click.option("--qmax", type=int, required=True, help="Maximum q (inclusive).")
@click.option("--rmin", type=int, required=True, help="Minimum R (inclusive).")
@click.option("--rmax", type=int, required=True, help="Maximum R (inclusive).")
@click.option("--N", "N_series", type=int, default=64, show_default=True,
              help="Length of the synthetic depth series ρ_k used by the proxy estimator.")
@click.option("--bootstrap", type=int, default=0, show_default=True,
              help="Number of bootstrap replicates for CI/stability (0 = off).")
@click.option("--seed", type=int, default=0, show_default=True,
              help="Seed for deterministic bootstrap.")
def extremum(qmin: int, qmax: int, rmin: int, rmax: int,
             N_series: int, bootstrap: int, seed: int) -> None:
    """
    CLI wrapper around `emergent.entropy_max.argmax_by_grid`. Returns JSON on stdout.
    """
    if qmin > qmax or rmin > rmax:
        raise click.UsageError("Require qmin <= qmax and rmin <= rmax.")

    Q = list(range(int(qmin), int(qmax) + 1))
    R = list(range(int(rmin), int(rmax) + 1))

    res = argmax_by_grid(Q, R, N=N_series, bootstrap=bootstrap, seed=seed)

    payload = {
        "q_star": int(res.q_star),
        "R_star": int(res.R_star),
        "S_star": float(res.S_star),
        "ci_90": (float(res.ci_90[0]), float(res.ci_90[1])),
        "stability": float(res.stability),
    }
    click.echo(json.dumps(payload, indent=2))


# Maintain a runnable module for manual use:
if __name__ == "__main__":
    app()
