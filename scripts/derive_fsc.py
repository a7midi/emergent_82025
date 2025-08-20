# scripts/derive_fsc.py
"""
Derives the Finite-Size Correction (FSC) constants a₁ and a₂ numerically.

This script runs the entropy score measurement for a range of system sizes (N)
and connectivities (R). It then performs a linear regression on the results
to fit the theoretical model:

  (S_inf - S_N) / S_inf ≈ a₁ * (1/N) + a₂ * ((R-1)/N)

and solves for the best-fit (a₁, a₂).
"""
import numpy as np
from scipy.optimize import curve_fit

# Ensure the local emergent package is in the path
# This might require running from the project root or setting PYTHONPATH
try:
    from emergent.entropy_max import measure_entropy_score
except ImportError:
    print("Error: Could not import from 'emergent' package.")
    print("Please run this script from the project's root directory.")
    exit(1)

def fsc_scaling_law(X, a1, a2):
    """The theoretical model we are fitting to: f(N,R) = a₁/N + a₂(R-1)/N"""
    N_inv, R_minus_1_div_N = X.T
    return a1 * N_inv + a2 * R_minus_1_div_N

def run_fsc_scan(q: int, R_values: list[int], N_values: list[int]) -> tuple[float, float]:
    """
    Performs a grid scan over R and N to collect data for the FSC fit.
    """
    print(f"--- Starting FSC Scan for q={q} ---")
    print(f"R values: {R_values}")
    print(f"N values (approx): {N_values}")
    
    # Estimate S_inf by measuring the score at a very large N
    R_mid = R_values[len(R_values) // 2]
    N_large = N_values[-1] * 2
    
    print(f"\nEstimating S_inf(q={q}, R={R_mid}) at large N={N_large}...")
    s_inf_proxy = measure_entropy_score(
        q, R_mid, n_layers=40, nodes_per_layer=N_large//40, seed=999
    )
    if not np.isfinite(s_inf_proxy) or s_inf_proxy <= 0:
        print(f"Error: Invalid S_inf proxy ({s_inf_proxy}). Aborting.")
        return 0.0, 0.0
    print(f"Estimated S_inf ≈ {s_inf_proxy:.4f}")

    X_data = []
    Y_data = []

    for i, R in enumerate(R_values):
        for N in N_values:
            width = N // 20
            if width < 2: continue
            
            current_N = width * 20
            print(f"Measuring for (R={R}, N≈{current_N})...", end='', flush=True)
            
            s_N = measure_entropy_score(q, R, n_layers=20, nodes_per_layer=width, seed=i*len(N_values) + N)
            
            y = (s_inf_proxy - s_N) / s_inf_proxy
            
            x1 = 1.0 / current_N
            x2 = (R - 1.0) / current_N
            
            X_data.append([x1, x2])
            Y_data.append(y)
            print(f" -> S_N = {s_N:.4f}, y = {y:.4f}")

    # Perform the fit
    X_data_arr = np.array(X_data)
    Y_data_arr = np.array(Y_data)
    
    try:
        popt, _ = curve_fit(fsc_scaling_law, X_data_arr, Y_data_arr)
        a1, a2 = popt
        print("\n--- Fit completed ---")
        return float(a1), float(a2)
    except Exception as e:
        print(f"\n--- Fit failed: {e} ---")
        return 0.0, 0.0

if __name__ == "__main__":
    Q_STAR = 13
    
    R_SCAN_VALUES = [2, 3, 4, 5]
    N_SCAN_VALUES = [200, 400, 800, 1600]
    
    a1_derived, a2_derived = run_fsc_scan(Q_STAR, R_SCAN_VALUES, N_SCAN_VALUES)
    
    print("\n" + "="*40)
    print("DERIVED FINITE-SIZE CORRECTION CONSTANTS")
    print("="*40)
    print(f"  a₁ = {a1_derived:.6f}")
    print(f"  a₂ = {a2_derived:.6f}")
    print("\nNext step: Use these values in `set_fsc_coefficients`")
    print("in your `paper_maps/v8.py` file to finalize your theory.")
    print("="*40)