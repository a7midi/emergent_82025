# tests/test_calibration.py
"""
Tests for M13-calibration: Mapping between parameters.

Oracle Checklist:
- Theorem 3.53 (Unique parameter fit): Verified by
  `test_calibration_jacobian_is_invertible`, which checks that the
  determinant of the calibration map's Jacobian is non-zero, confirming
  it is a local diffeomorphism as required by the theorem.
"""

import numpy as np
import pytest

from emergent.calibration import (
    LatticeParams,
    PhysicalConstants,
    calibrate_lattice_to_physical,
    calibrate_physical_to_lattice,
    get_calibration_jacobian,
)


@pytest.fixture
def sample_params():
    """Provides a sample set of physical and lattice parameters."""
    k0 = 100
    sigma_mix = 0.24
    
    # Plausible physical constants (not necessarily real-world values)
    phys_consts = PhysicalConstants(
        c=3e8, hbar=1.05e-34, G=6.67e-11, m_f0=1e-27, theta_cp_obs=0.1
    )
    # The corresponding lattice parameters
    lat_params = calibrate_physical_to_lattice(phys_consts, k0, sigma_mix)
    
    return {"phys": phys_consts, "lat": lat_params, "k0": k0, "sigma_mix": sigma_mix}


def test_calibration_round_trip(sample_params):
    """
    Tests that mapping from physical to lattice and back yields the
    original physical constants.
    """
    phys_initial = sample_params["phys"]
    lat_derived = calibrate_physical_to_lattice(
        phys_initial, sample_params["k0"], sample_params["sigma_mix"]
    )
    phys_final = calibrate_lattice_to_physical(
        lat_derived, sample_params["k0"], sample_params["sigma_mix"]
    )

    assert np.isclose(phys_initial.c, phys_final.c)
    assert np.isclose(phys_initial.hbar, phys_final.hbar)
    assert np.isclose(phys_initial.G, phys_final.G)
    assert np.isclose(phys_initial.m_f0, phys_final.m_f0)
    assert np.isclose(phys_initial.theta_cp_obs, phys_final.theta_cp_obs)


def test_calibration_jacobian_is_invertible(sample_params):
    """
    Oracle: Verifies the Jacobian of the calibration map is invertible.
    """
    lat_params = sample_params["lat"]
    k0 = sample_params["k0"]
    sigma_mix = sample_params["sigma_mix"]

    jacobian = get_calibration_jacobian(lat_params, k0, sigma_mix)
    
    # The map is a local diffeomorphism if the determinant is non-zero.
    det = np.linalg.det(jacobian)
    
    assert jacobian.shape == (5, 5)
    assert not np.isclose(det, 0.0)