from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from crazyflow.control.transform import force2pwm, motor_force2rotor_vel, pwm2force
from crazyflow.drones import load_params


@pytest.fixture(scope="module")
def core_params() -> dict[str, Any]:
    return {k: np.asarray(v) for k, v in load_params("cf2x_L250").items()}


@pytest.mark.unit
def test_force2pwm_pwm2force_roundtrip(core_params: dict[str, Any]) -> None:
    thrust_max = float(core_params["thrust_max"])
    pwm_max = float(core_params["pwm_max"])
    forces = np.array([0.0, thrust_max * 0.25, thrust_max * 0.5, thrust_max])
    assert np.allclose(
        pwm2force(force2pwm(forces, thrust_max, pwm_max), thrust_max, pwm_max), forces
    )


@pytest.mark.unit
def test_force2pwm_boundary(core_params: dict[str, Any]) -> None:
    thrust_max = float(core_params["thrust_max"])
    pwm_max = float(core_params["pwm_max"])
    assert force2pwm(0.0, thrust_max, pwm_max) == pytest.approx(0.0)
    assert force2pwm(thrust_max, thrust_max, pwm_max) == pytest.approx(pwm_max)


@pytest.mark.unit
def test_motor_force2rotor_vel_shape(core_params: dict[str, Any]) -> None:
    rpm2thrust = core_params["rpm2thrust"]
    assert motor_force2rotor_vel(np.full(4, 0.05), rpm2thrust).shape == (4,)
    assert motor_force2rotor_vel(np.full((3, 2, 4), 0.05), rpm2thrust).shape == (3, 2, 4)


@pytest.mark.unit
def test_motor_force2rotor_vel_positive(core_params: dict[str, Any]) -> None:
    rpm2thrust = core_params["rpm2thrust"]
    forces = np.linspace(0.02, 0.12, 10)
    assert np.all(motor_force2rotor_vel(forces, rpm2thrust) > 0)


@pytest.mark.unit
def test_motor_force2rotor_vel_batched_coefficients(core_params: dict[str, Any]) -> None:
    """Shared (1, 3) and per-motor (4, 3) coefficients broadcast, with and without batch axes."""
    N, M = 3, 2  # Worlds, drones
    rpm2thrust = core_params["rpm2thrust"]  # (3,)
    forces = np.random.rand(N, M, 4) * 0.1 + 0.01  # (N, M, 4), one force per motor
    ref = motor_force2rotor_vel(forces, rpm2thrust)  # (N, M, 4)
    # Copies of the shared curve must be identical to the (3,) call
    shared = rpm2thrust[None]  # (1, 3)
    assert np.allclose(motor_force2rotor_vel(forces, shared), ref)
    batched = np.tile(shared, (N, M, 1, 1))  # (N, M, 1, 3)
    assert np.allclose(motor_force2rotor_vel(forces, batched), ref)
    # Per-motor and per-batch coefficients must match a per-motor reference
    per_motor = np.tile(shared, (N, M, 4, 1))  # (N, M, 4, 3), one curve per motor
    per_motor = per_motor * (1 + 0.1 * np.random.rand(N, M, 4, 3))
    out = motor_force2rotor_vel(forces, per_motor)
    assert out.shape == (N, M, 4)
    for i in np.ndindex(N, M, 4):  # Single motor: force () with its own curve (3,)
        assert np.allclose(out[i], motor_force2rotor_vel(forces[i], per_motor[i]))
    # Lists of coefficients also work
    assert np.allclose(motor_force2rotor_vel(forces, rpm2thrust.tolist()), ref)
