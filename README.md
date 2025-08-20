Of course. Here is a comprehensive README file for your project. You can copy and paste the content directly into your README.md file on GitHub.

Markdown

# Emergent: A Causal Site Numerical Suite

This repository contains the official numerical suite for the paper series "A Parameter-Free Causal Theory of Spacetime and Matter." It provides a full, end-to-end Python pipeline for generating finite causal sites, simulating their deterministic evolution, and deriving falsifiable, first-principles predictions for low-energy physics.

The suite is deterministic, reproducible, and provides command-line and notebook interfaces for all major experiments described in the papers.

## Features

* **Causal Site Engine**: Generate finite, acyclic, graded posets (`C_n`) satisfying all standing hypotheses from the papers.
* **Deterministic Dynamics**: Simulate the one-tick tag-fusion update (`T`), the Koopman operator (`K`), and the transfer operator (`L`) with provable spectral properties.
* **First-Principles Physics**:
    * **Emergent Geometry**: Calculate geometric properties like Alexandrov intervals, path-length metrics, and Benincasa-Dowker-style curvature.
    * **Observer Entropy**: Measure observer entropy via Monte Carlo simulation and predict its linear growth using the paper-derived `ΔS = |H_min| * log₂(q)` identity.
    * **Quantum Correlations**: Run CHSH experiments and verify consistency with the Tsirelson bound.
    * **Emergent Gravity**: Witness the convergence of the block-averaged Einstein-Memory coupling `g* = κ / ρ_mem`.
* **Parameter-Free Pipeline**:
    * **Entropy Maximizer**: Find the unique optimal discrete parameters `(q*, R*)` that maximize the theory's entropy functional.
    * **Forward Predictions**: Use the optimal `(q*, R*)` to run a deterministic Renormalization Group (RG) flow and produce testable predictions for `sin²(θ_W)`, the cosmological constant `Λ`, and the neutron EDM.

## Quickstart: Installation and Running

### 1. Installation

This project uses Python 3.11+. All commands should be run from the repository root.

```cmd
:: 1. Create a virtual environment
python -m venv .venv

:: 2. Activate the environment
.venv\Scripts\activate

:: 3. Install the suite in editable mode and all dependencies
pip install -e .
2. Verify with Tests
Run the test suite to confirm the installation is working correctly.

DOS

:: Run all fast tests
pytest -q

:: (Optional) Run slower, first-principles tests
pytest -q -m slow
3. Run Predictions from the CLI
The emergent command is the main entry point. The most important command is predict, which runs the full pipeline.

DOS

:: Generate the three main prediction cards using the optimal (q*, R*) = (13, 2)
emergent predict weakmix --q 13 --R 2
emergent predict cosmo --q 13 --R 2
emergent predict edm --q 13 --R 2
4. Explore with Jupyter Notebooks
For visual and interactive demos, use the Jupyter notebooks.

DOS

:: Install Jupyter if you haven't already
pip install jupyter matplotlib

:: Launch the notebook server from the project root
jupyter notebook
Navigate to the notebooks/ directory in your browser and open 07_prediction_cards.ipynb to see the final, first-principles prediction pipeline in action.

Module Overview
The suite is organized into a modular src/emergent/ package.

Module	M#	Responsibility
poset.py	M01	Core CausalSite class for generating graded DAGs.
metric.py	M02	Path-length distance d(u,v) and symmetrized metric r(u,v).
measure.py	M03	Depth-weighted site measure μ_n and configuration measures on tag space.
update.py	M04	Deterministic one-tick update T and operators K, L.
spectral.py	M05	Spectral constants (δ, gaps) and the peripheral phase projector P.
entropy.py	M06	Static observer entropy calculations (older, enumerative).
chsh.py	M07	Classical CHSH experiments on causal sites.
geom.py	M08	Geometric diagnostics (curvature/dimension proxies).
born.py	M09	Verification of the Born rule for cylinder events on a finite state space.
gauge.py	M10	Symbolic verification of Standard Model anomaly cancellation.
gravity.py	M11	Einstein-Memory coupling proxy g = κ / ρ_mem.
rg.py	M12	Renormalization Group (RG) integrator using the 3rd-order β-function.
calibration.py	M13	Mapping between lattice parameters and physical constants.
extremum.py	M14	Phenomenological entropy maximizer (legacy).
cli.py	M15	Main Typer CLI application.
First-Principles		
geom_fp.py	-	Benincasa-Dowker-style curvature from k-chain counting.
gravity_fp.py	-	First-principles gravity witness using geom_fp curvature.
entropy_fp.py	-	First-principles linear entropy predictor (`ΔS =
entropy_mc.py	-	Monte Carlo estimator for dynamic observer entropy.
entropy_max.py	-	First-principles entropy maximizer using a measured objective.
physics_maps.py	-	Pluggable "hooks" interface for physical formulas.
paper_maps/v8.py	-	The final, paper-derived physics maps used by the prediction pipeline.
predict.py	-	The final prediction engine that generates falsifiable cards.

Export to Sheets
Command-Line Interface (CLI)
The CLI provides access to all major experiments.

Command	Description
emergent predict <card>	(Recommended) Generate prediction cards (weakmix, cosmo, edm).
emergent entropy-max run	Find the optimal (q*, R*) via simulated annealing.
emergent chsh	Run a classical CHSH experiment.
emergent lorentz	Run geometric diagnostics for flatness/curvature concentration.
emergent extremum	Run the legacy (phenomenological) entropy maximizer.
