# src/emergent/cli_predict.py
"""
Optional CLI addon for Phase D prediction cards.
Mount into your main Typer app with:

    from emergent.cli_predict import app as predict_app
    app.add_typer(predict_app, name="predict")

This keeps your existing CLI intact.
"""
from __future__ import annotations

import json
from pathlib import Path
import typer

from emergent.rg import CouplingVector
from emergent.predict import (
    make_card_weakmix,
    make_card_cosmology,
    make_card_edm,
    export_card_json,
)

app = typer.Typer(no_args_is_help=True, help="Prediction cards (weakmix, cosmology, edm)")

@app.command("weakmix")
def cli_weakmix(
    q: int = typer.Option(6, help="Alphabet size"),
    R: int = typer.Option(4, help="Max out-degree+1"),
    k_start: float = typer.Option(100.0, help="RG start scale"),
    k_end: float = typer.Option(1.0, help="RG end scale (EW)"),
    g_star: float = typer.Option(0.1, help="Initial g_star"),
    lambda_mix: float = typer.Option(0.5, help="Initial lambda_mix"),
    theta_cp: float = typer.Option(0.2, help="Initial theta_cp"),
    save_json: Path = typer.Option(None, help="Optional JSON output path"),
    save_png: Path = typer.Option(None, help="Optional PNG plot path"),
):
    g0 = CouplingVector(g_star=g_star, lambda_mix=lambda_mix, theta_cp=theta_cp)
    card = make_card_weakmix(g0, q=q, R=R, k_start=k_start, k_end=k_end,
                             save_plot=str(save_png) if save_png else None)
    if save_json:
        export_card_json(card, str(save_json))
    typer.echo(json.dumps(card.to_dict(), indent=2))

@app.command("cosmo")
def cli_cosmo(
    q: int = typer.Option(6),
    observer_depth: int = typer.Option(3),
    ticks: int = typer.Option(8),
):
    card = make_card_cosmology(q=q, observer_depth=observer_depth, ticks=ticks)
    typer.echo(json.dumps(card.to_dict(), indent=2))

@app.command("edm")
def cli_edm(
    q: int = typer.Option(6),
    R: int = typer.Option(4),
    k_start: float = typer.Option(100.0),
    k_end: float = typer.Option(1.0),
    g_star: float = typer.Option(0.1),
    lambda_mix: float = typer.Option(0.5),
    theta_cp: float = typer.Option(0.2),
):
    g0 = CouplingVector(g_star=g_star, lambda_mix=lambda_mix, theta_cp=theta_cp)
    card = make_card_edm(g0, q=q, R=R, k_start=k_start, k_end=k_end)
    typer.echo(json.dumps(card.to_dict(), indent=2))
