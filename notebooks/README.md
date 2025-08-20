# notebooks/README.md
# M17 — Demos & Reproducible Notebooks

## How to run
1) From the repo root:
   python -m venv .venv && source .venv/bin/activate   # (or .venv\Scripts\activate on Windows)
   pip install -e .
   pip install jupyter matplotlib

2) Launch:
   jupyter notebook notebooks/

Each notebook echoes the seed and parameters. All runs are deterministic.

## Notebook map
01_entropy.ipynb          — Observer entropy on finite sites (M06)
02_chsh_tsirelson.ipynb   — CHSH experiment and Tsirelson bound (M07)
03_lorentz_flatness.ipynb — Dimension/curvature concentration (M08)
04_einstein_limit.ipynb   — Block Einstein witness via g=κ/ρ_mem (M11)
05_rg_window.ipynb        — Depth-ordered RG flow window (M12)
06_extremum_qR.ipynb      — Discrete search for (q*, R*) (M14)
