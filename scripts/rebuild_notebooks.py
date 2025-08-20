# scripts/rebuild_notebooks.py
from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf
from nbformat.validator import validate

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)

def nb(*cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = list(cells)
    nb["metadata"].update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    })
    return nb

def md(s: str): return nbf.v4.new_markdown_cell(s.strip("\n"))
def code(s: str): return nbf.v4.new_code_cell(s.strip("\n"))

# 01 — Entropy
nb01 = nb(
    md("""
# 01 · Entropy demo (M06)

Observer entropy on a finite causal site using `run_entropy_simulation`.

**Refs:** `emergent.entropy`, `emergent.poset`.
    """),
    code("""
import numpy as np
from emergent.poset import CausalSite
from emergent.entropy import run_entropy_simulation

seed = 73
rng = np.random.default_rng(seed)
layers, nodes, R = 6, 6, 4
q = 4
observer_depth = 3
ticks = 6

site = CausalSite.generate(n_layers=layers, nodes_per_layer=nodes, R=R, edge_prob=0.6, rng=rng)
initial_config = {n: int(rng.integers(q)) for n in site.nodes}

entropies, increments = run_entropy_simulation(site, q, initial_config, observer_depth, ticks)
print(f"seed={seed} layers={layers} nodes={nodes} R={R} q={q} obs_depth={observer_depth} ticks={ticks}")
print("Entropy S_t:", [round(x,4) for x in entropies])
print("Increments ΔS_t:", [round(x,4) for x in increments])
    """),
)

# 02 — CHSH / Tsirelson
nb02 = nb(
    md("""
# 02 · CHSH / Tsirelson demo (M07)

Runs the deterministic CHSH experiment on:
1) A canonical 3-layer site (guaranteed spacelike pair).
2) A random generated site (if a spacelike pair is found).

**Refs:** `emergent.chsh`, `emergent.poset`, `emergent.metric`.
    """),
    code("""
import math, numpy as np
from emergent.poset import CausalSite
from emergent.chsh import run_chsh_experiment, find_spacelike_nodes

q = 2

# Canonical site
nodes_by_depth = {0: [0,1], 1: [2,3], 2: [4]}
adj = {0:[2], 1:[3], 2:[4], 3:[4]}
site = CausalSite(nodes_by_depth, adj, R=3)

alice, bob = 2, 3
hidden_sources = [0,1]
s_vals = []
for tag in range(q):
    s_vals.append(run_chsh_experiment(site, q, {4: tag}, alice, bob, hidden_sources))

max_s = max(abs(x) for x in s_vals)
print("Canonical site S-values:", [round(x,4) for x in s_vals], "| max |S|=", round(max_s,4))
print("Classical bound=2.0, Tsirelson bound=", round(2*math.sqrt(2),4))

# Generated site
rng = np.random.default_rng(101)
gsite = CausalSite.generate(n_layers=5, nodes_per_layer=5, R=4, edge_prob=0.7, rng=rng)
pair = find_spacelike_nodes(gsite, rng)
if pair:
    alice2, bob2 = pair
    finals = [n for n,d in gsite.depths.items() if d == max(gsite.depths.values())]
    hidden2 = [n for n,d in gsite.depths.items() if d == 0]
    if finals:
        obs = finals[0]
        s2 = [run_chsh_experiment(gsite, q, {obs: tag}, alice2, bob2, hidden2) for tag in range(q)]
        print("Generated site S-values:", [round(x,4) for x in s2], "max=", round(max(abs(x) for x in s2),4))
else:
    print("No spacelike pair found (try another seed)")
    """),
)

# 03 — Lorentz/flatness
nb03 = nb(
    md("""
# 03 · Lorentz/flatness diagnostics (M08)

Myrheim–Meyer dimension and curvature concentration as radius grows.

**Refs:** `emergent.geom`, `emergent.metric`, `emergent.poset`.
    """),
    code("""
import numpy as np
from emergent.poset import CausalSite
from emergent.geom import get_nodes_in_ball, estimate_mm_dimension, analyze_curvature_in_ball

rng = np.random.default_rng(42)
site = CausalSite.generate(n_layers=20, nodes_per_layer=40, R=6, edge_prob=0.5, rng=rng)
center = site.nodes[len(site.nodes)//2]
radii = [3.0,4.0,5.0,6.0]
rows = []
for r in radii:
    ball = get_nodes_in_ball(site, center, r)
    if len(ball) < 15: 
        continue
    dim = estimate_mm_dimension(site, ball)
    m,v = analyze_curvature_in_ball(site, ball)
    rows.append((r, len(ball), dim, m, v))
print("radius | |ball| | MM-dim | mean κ | var κ")
for r, n, d, m, v in rows:
    print(f"{r:5.1f} | {n:5d} | {d:7.4f} | {m:7.4f} | {v:7.4f}")
    """),
)

# 04 — Einstein witness
nb04 = nb(
    md("""
# 04 · Einstein/Gravity witness (M11)

Block-averaged `g = κ/ρ_mem` vs radius as a convergence witness.

**Refs:** `emergent.gravity`, `emergent.geom`, `emergent.poset`.
    """),
    code("""
import numpy as np
from emergent.poset import CausalSite
from emergent.geom import get_nodes_in_ball
from emergent.gravity import analyze_gravity_in_block

rng = np.random.default_rng(7)
site = CausalSite.generate(n_layers=20, nodes_per_layer=40, R=6, edge_prob=0.5, rng=rng)
center = site.nodes[len(site.nodes)//2]
radii = [4.0, 5.0, 6.0, 7.0]
print("radius | edges | avg g")
for r in radii:
    ball = get_nodes_in_ball(site, center, r)
    res = analyze_gravity_in_block(site, ball)
    val = res['avg_g']
    s = 'NaN' if (val is None or (isinstance(val,float) and np.isnan(val))) else f"{val:.6f}"
    print(f"{r:5.1f} | {res['num_edges']:5d} | {s}")
    """),
)

# 05 — RG window
nb05 = nb(
    md("""
# 05 · RG window (M12)

Integrate the truncated 3-loop β and visualize trajectories.

**Refs:** `emergent.rg`.
    """),
    code("""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from emergent.rg import beta_function, CouplingVector, run_rg_flow

q, R = 6, 4
k_start, k_end = 100, 0
g0 = CouplingVector(g_star=0.1, lambda_mix=0.5, theta_cp=0.2)

sol = solve_ivp(fun=beta_function, t_span=[k_start, k_end], y0=g0.to_array(), args=(q,R), dense_output=True)
ks = np.linspace(k_start, k_end, 101)
vals = sol.sol(ks)

plt.figure(); plt.plot(ks, vals[0]); plt.title("g_* vs k"); plt.xlabel("k"); plt.ylabel("g_*"); plt.show()
plt.figure(); plt.plot(ks, vals[1]); plt.title("lambda_mix vs k"); plt.xlabel("k"); plt.ylabel("lambda_mix"); plt.show()
plt.figure(); plt.plot(ks, vals[2]); plt.title("theta_cp vs k"); plt.xlabel("k"); plt.ylabel("theta_cp"); plt.show()

g_final = run_rg_flow(g0, k_start, k_end, q, R)
print("Initial:", g0)
print("Final:", g_final)
    """),
)

# 06 — Extremum in (q,R)
nb06 = nb(
    md("""
# 06 · Discrete extremum in (q, R) (M14)

Scan `S_infinity(q,R)` over a grid, visualize, and report the unique max.

**Refs:** `emergent.extremum`.
    """),
    code("""
import numpy as np
import matplotlib.pyplot as plt
from emergent.extremum import calculate_entropy_density, find_entropy_maximizer

qmin, qmax = 2, 15
rmin, rmax = 1, 10
(q_star, r_star), max_s = find_entropy_maximizer((qmin,qmax),(rmin,rmax))
print(f"Argmax (q*,R*)=({q_star},{r_star}), S_inf={max_s:.6f}")

Q = np.arange(qmin, qmax+1)
R = np.arange(rmin, rmax+1)
Z = np.zeros((len(R), len(Q)))
for i,Rv in enumerate(R):
    for j,Qv in enumerate(Q):
        Z[i,j] = calculate_entropy_density(Qv, Rv)

plt.figure()
plt.imshow(Z, origin='lower', extent=[Q.min(), Q.max(), R.min(), R.max()], aspect='auto')
plt.colorbar(label='S_inf(q,R)')
plt.scatter([q_star],[r_star], marker='x')
plt.xlabel('q'); plt.ylabel('R'); plt.title('Entropy density landscape')
plt.show()
    """),
)

def write_and_validate(path: Path, notebook):
    with path.open("w", encoding="utf-8") as f:
        nbf.write(notebook, f)
    # Validate structure to catch any mistakes
    validate(notebook)

def main():
    mapping = {
        "01_entropy.ipynb": nb01,
        "02_chsh_tsirelson.ipynb": nb02,
        "03_lorentz_flatness.ipynb": nb03,
        "04_einstein_limit.ipynb": nb04,
        "05_rg_window.ipynb": nb05,
        "06_extremum_qR.ipynb": nb06,
    }
    for name, notebook in mapping.items():
        path = NB_DIR / name
        write_and_validate(path, notebook)
        print(f"Wrote and validated: {path}")

    # Optional: write a tiny README
    readme = (NB_DIR / "README.md")
    readme.write_text("""# M17 — Demos & Reproducible Notebooks

How to run:
1) Activate your venv and install the package: `pip install -e .`
2) Install notebook deps: `pip install jupyter matplotlib`
3) Launch: `jupyter notebook notebooks/`

Notebooks:
01_entropy.ipynb          — Observer entropy (M06)
02_chsh_tsirelson.ipynb   — CHSH & Tsirelson (M07)
03_lorentz_flatness.ipynb — Dimension/curvature (M08)
04_einstein_limit.ipynb   — Block Einstein witness (M11)
05_rg_window.ipynb        — RG flow window (M12)
06_extremum_qR.ipynb      — (q,R) extremum (M14)
""", encoding="utf-8")
    print(f"Wrote: {readme}")

if __name__ == "__main__":
    main()
