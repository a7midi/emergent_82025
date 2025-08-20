# scripts/emergent_fp
#!/usr/bin/env python3
"""
CLI add-on exposing first-principles & MC demos without touching your main CLI.

Usage:
  python scripts/emergent_fp --help
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import typer

from emergent.entropy_mc import predict_entropy_trajectory, run_entropy_monte_carlo
from emergent.geom import get_nodes_in_ball  # reuse existing radius tool
from emergent.gravity_fp import analyze_gravity_in_block_fp
from emergent.poset import CausalSite

app = typer.Typer(no_args_is_help=True, help="Emergent (add-on) FP utilities.")


@app.command("entropy-mc")
def entropy_mc(
    layers: int = typer.Option(6, help="Number of layers"),
    nodes: int = typer.Option(6, help="Nodes per layer"),
    R: int = typer.Option(4, help="Max out-degree + 1"),
    q: int = typer.Option(4, help="Alphabet size"),
    edge_prob: float = typer.Option(0.6, help="Edge probability"),
    obs_depth: int = typer.Option(3, help="Observer depth (< obs_depth is visible)"),
    ticks: int = typer.Option(8, help="Number of ticks"),
    samples: int = typer.Option(256, help="Monte Carlo histories"),
    seed: int = typer.Option(0, help="RNG seed"),
):
    rng = np.random.default_rng(seed)
    site = CausalSite.generate(n_layers=layers, nodes_per_layer=nodes, R=R, edge_prob=edge_prob, rng=rng)

    ent, inc = run_entropy_monte_carlo(site, q, obs_depth, ticks, samples=samples, seed=seed)
    pred_ent, pred_inc = predict_entropy_trajectory(site, q, obs_depth, ticks)

    typer.echo(f"Observed S_t  : {[round(x,4) for x in ent]}")
    typer.echo(f"Observed dS_t : {[round(x,4) for x in inc]}")
    typer.echo(f"Predicted S_t : {[round(x,4) for x in pred_ent]}")
    typer.echo(f"Predicted dS_t: {[round(x,4) for x in pred_inc]}")


@app.command("gravity-fp")
def gravity_fp(
    layers: int = typer.Option(20),
    nodes: int = typer.Option(40),
    R: int = typer.Option(6),
    edge_prob: float = typer.Option(0.5),
    radius: float = typer.Option(6.0, help="Ball radius (one-sided metric)"),
    scheme: str = typer.Option("4D-lite", help="Curvature scheme"),
    seed: int = typer.Option(7),
):
    rng = np.random.default_rng(seed)
    site = CausalSite.generate(n_layers=layers, nodes_per_layer=nodes, R=R, edge_prob=edge_prob, rng=rng)
    center = site.nodes[len(site.nodes) // 2]
    ball = get_nodes_in_ball(site, center, radius)
    res = analyze_gravity_in_block_fp(site, ball, scheme=scheme)
    avg = res["avg_g_fp"]
    val = "NaN" if (avg is None or (isinstance(avg, float) and math.isnan(avg))) else f"{avg:.6f}"
    typer.echo(f"radius={radius:.1f} | edges={res['num_edges']} | avg g_fp={val}")


if __name__ == "__main__":
    app()
