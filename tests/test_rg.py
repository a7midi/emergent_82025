# tests/test_rg.py
"""
Tests for M12-rg: Renormalisation Group flow.
...
"""
import numpy as np
import pytest

from emergent.rg import CouplingVector, beta_function, run_rg_flow


@pytest.fixture
def rg_params():
    """Provides standard parameters for RG tests."""
    return {"q": 6, "R": 4}


def test_beta_function_coefficients(rg_params):
    """
    Oracle: Verifies the implementation of the β-function from Paper III, App C.
    """
    g = CouplingVector(g_star=0.1, lambda_mix=0.5, theta_cp=0.2).to_array()
    dg_dk = beta_function(0, g, **rg_params)

    # Precise manual calculation of all terms for dg_dk[1] (d_lambda_mix / dk)
    # Using sigma_mix = 0.24 (fallback value from paper)
    # b1 = -0.24 * 0.5 = -0.12
    # b2 = (1/5) * 0.1 * 0.5 = 0.01
    # b3 = (3*3 / (2*25)) * 0.5**3 = (9/50)*0.125 = 0.0225
    # total = -0.12 + 0.01 + 0.0225 = -0.0875
    expected_d_lambda_dk = -0.0875

    assert dg_dk.shape == (3,)
    # FIX: Update the assertion to use the precise expected value.
    assert np.isclose(dg_dk[1], expected_d_lambda_dk)
    assert dg_dk[0] != 0  # 3-loop term for g_star should be non-zero
    assert dg_dk[2] != 0  # 2-loop term for theta_cp should be non-zero


def test_rg_flow_integration(rg_params):
    """
    Oracle: Verifies the RG flow integration (Thm 4.20).
    Checks that the flow is non-trivial and stable.
    """
    g_gut = CouplingVector(g_star=0.1, lambda_mix=0.5, theta_cp=0.2)
    k_gut, k_z = 100, 0

    g_z = run_rg_flow(
        g_initial=g_gut, k_start=k_gut, k_end=k_z, **rg_params
    )

    # 1. Check that the flow is non-trivial (the values have changed)
    assert not np.allclose(g_z.to_array(), g_gut.to_array())

    # 2. Check for stability by running with a different solver and comparing
    from scipy.integrate import solve_ivp
    sol_lsoda = solve_ivp(
        fun=beta_function,
        t_span=[k_gut, k_z],
        y0=g_gut.to_array(),
        method="LSODA",
        args=(rg_params["q"], rg_params["R"]),
        dense_output=True,
    )
    g_z_lsoda = CouplingVector.from_array(sol_lsoda.sol(k_z))
    
    assert np.allclose(g_z.to_array(), g_z_lsoda.to_array(), rtol=1e-3)