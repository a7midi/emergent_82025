# tests/test_cli.py
"""
M16—CLI smoke tests (Typer CliRunner).

Covers commands present in the current repo version:
  • emergent extremum
  • emergent entropy

We intentionally avoid `chsh` here to steer clear of private-field access
differences across versions of cli.py in the branch history.
"""
from __future__ import annotations

from typer.testing import CliRunner

from emergent.cli import app


def test_cli_extremum_smoke():
    runner = CliRunner()
    result = runner.invoke(app, ["extremum", "--qmin", "2", "--qmax", "8", "--rmin", "1", "--rmax", "5"])
    assert result.exit_code == 0
    assert "Argmax" in result.stdout or "maximum" in result.stdout


def test_cli_entropy_smoke():
    runner = CliRunner()
    # Keep the instance tiny to run fast
    result = runner.invoke(app, ["entropy", "--layers", "4", "--nodes", "3", "--q", "2", "--r", "3", "--ticks", "3"])
    assert result.exit_code == 0
    assert "Entropy Simulation" in result.stdout
