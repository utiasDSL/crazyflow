"""Tests of the numeric dynamics."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Callable

import array_api_strict as xp
import casadi as cs
import jax
import jax.numpy as jp
import numpy as np
import pytest
from array_api_compat import device as xp_device

from crazyflow.drones import available_drones
from crazyflow.dynamics import available_dynamics, dynamics_features
from crazyflow.dynamics.core import parametrize

if TYPE_CHECKING:
    from types import ModuleType

    from crazyflow._typing import Array  # To be changed to array_api_typing later


@pytest.fixture(autouse=True)
def _enable_x64():
    """Run only this module in float64 so jax matches numpy precision, then restore the default.

    jax_enable_x64 is a global flag, so we scope it to this file's tests to keep every other test
    running in float32.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# Parameters that optionally accept one value per motor. Defaults to single values.
_PER_MOTOR = ("L", "prop_inertia", "rpm2thrust", "rpm2torque", "rotor_dyn_coef")


def create_rnd_states(
    shape: tuple[int, ...] = (),
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    x = np.random.randn(*shape, 3 + 4 + 3 + 3 + 4 + 3 + 3)
    pos = xp.asarray(x[..., :3])
    quat = xp.asarray(x[..., 3:7])
    vel = xp.asarray(x[..., 7:10])
    ang_vel = xp.asarray(x[..., 10:13])
    rotor_vel = xp.abs(xp.asarray(x[..., 13:17]))  # Rotor velocities must be positive
    dist_f = xp.asarray(x[..., 17:20])
    dist_t = xp.asarray(x[..., 20:23])
    return pos, quat, vel, ang_vel, rotor_vel, dist_f, dist_t


def create_rnd_commands(shape: tuple[int, ...] = (), dim: int = 4) -> Array:
    """Creates N random inputs with size dim."""
    return xp.abs(xp.asarray(np.random.randn(*shape, dim)))  # Motor forces must be positive


def make_inputs(
    dynamics: Callable,
    *,
    batch: tuple[int, ...] = (),
    rotor_vel: bool = True,
    ext_wrench: bool = False,
) -> dict[str, Array]:
    """Build random inputs for a parametrized dynamics function, to splat as ``dynamics(**inp)``.

    Only the requested inputs are included, so the dict doubles as the call signature: dynamics
    without a rotor_vel parameter (so_rpy) simply never get one, and omitted optional inputs fall
    back to the function defaults.

    Args:
        dynamics: The (parametrized) dynamics function.
        batch: Batch shape of the inputs.
        rotor_vel: Whether to provide rotor_vel (ignored for dynamics without rotor dynamics).
            Set to False to exercise the commanded-rotor-velocity fallback.
        ext_wrench: Whether to provide external force/torque disturbances.
    """
    pos, quat, vel, ang_vel, rv, dist_f, dist_t = create_rnd_states(batch)
    cmd = create_rnd_commands(batch)
    inp = {"pos": pos, "quat": quat, "vel": vel, "ang_vel": ang_vel, "cmd": cmd}
    if rotor_vel and dynamics_features(dynamics)["rotor_dynamics"]:
        inp["rotor_vel"] = rv
    if ext_wrench:
        inp["dist_f"], inp["dist_t"] = dist_f, dist_t
    return inp


def make_params(
    dynamics: Callable, batch: tuple[int, ...] = (), scale: float = 0.0, xp: ModuleType = np
) -> dict[str, Array]:
    """Build batched parameters for a parametrized dynamics function."""
    params = {}
    for name, value in dynamics.keywords.items():
        core = np.shape(value) or (1,)
        if name in _PER_MOTOR:
            core = (4,) + np.shape(value)  # Quadrotors only
        value = np.broadcast_to(value, batch + core) * (1 + scale * np.random.randn(*batch, *core))
        params[name] = xp.asarray(value)
    return params


def state_vector(inp: dict) -> Array:
    """Stacked state vector matching the symbolic state X (present inputs in canonical order)."""
    order = ("pos", "quat", "vel", "ang_vel", "rotor_vel", "dist_f", "dist_t")
    return xp.concat([inp[k] for k in order if k in inp], axis=-1)


def symbolic_flags(dynamics: Callable, dist: bool = False) -> dict[str, bool]:
    """Build the symbolic_dynamics flags, gating model_rotor_vel on the rotor dynamics feature."""
    flags = {}
    if dynamics_features(dynamics)["rotor_dynamics"]:
        flags["model_rotor_vel"] = True
    if dist:
        flags["model_dist_f"] = flags["model_dist_t"] = True
    return flags


def assert_array_meta(x: Array | None, y: Array | None, name: str | None = None):
    """Assert the output is on the correct device, has the correct type and shape."""
    if x is None and y is None:
        return
    prefix = "" if name is None else f"{name}: "
    assert isinstance(x, type(y)), (
        f"{prefix}Output type {type(x)} does not match expected {type(y)}"
    )
    assert xp_device(x) == xp_device(y), (
        f"{prefix}Output device {xp_device(x)} does not match expected {xp_device(y)}"
    )
    assert x.shape == y.shape, f"{prefix}Output shape {x.shape} does not match expected {y.shape}"
    assert np.all(np.isnan(x) == np.isnan(y)), f"{prefix}Derivative of non-nan values are NaN"


def assert_shapes(dynamics: Callable, inp: dict):
    """Assert the dynamics output has the correct type, device and shape for each derivative."""
    out = dynamics(**inp)
    names = ["dpos", "dquat", "dvel", "dang_vel"]
    expected = [inp["pos"], inp["quat"], inp["vel"], inp["ang_vel"]]
    if dynamics_features(dynamics)["rotor_dynamics"]:
        names.append("drotor_vel")
        expected.append(inp.get("rotor_vel"))
    for name, dx, x in zip(names, out, expected, strict=True):
        assert_array_meta(dx, x, name=name)


def check_shapes(dynamics: Callable, batch: tuple[int, ...] = ()):
    """Check output shapes with/without external wrench, and the rotor_vel fallback warning."""
    assert_shapes(dynamics, make_inputs(dynamics, batch=batch))
    assert_shapes(dynamics, make_inputs(dynamics, batch=batch, ext_wrench=True))
    if not dynamics_features(dynamics)["rotor_dynamics"]:
        return
    for ext_wrench in (False, True):
        inp = make_inputs(dynamics, batch=batch, rotor_vel=False, ext_wrench=ext_wrench)
        with pytest.warns(UserWarning, match="Rotor velocity not provided"):
            assert_shapes(dynamics, inp)


@pytest.mark.unit
@pytest.mark.parametrize("dynamics_name, dynamics", available_dynamics.items())
def test_dynamics_features(dynamics_name: str, dynamics: Callable):
    """Tests if the dynamics features are correctly set."""
    assert hasattr(dynamics, "__dynamics_features__"), (
        f"Dynamics function {dynamics_name} does not have __dynamics_features__ attribute"
    )
    features = dynamics_features(dynamics)
    assert isinstance(features, dict), (
        f"dynamics features should be a dict, got {type(features)} for {dynamics_name}"
    )
    assert "rotor_dynamics" in features, (
        f"dynamics features should contain 'rotor_dynamics' key for {dynamics_name}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("dynamics_name, dynamics", available_dynamics.items())
@pytest.mark.parametrize("drone", available_drones)
def test_dynamics_shapes(dynamics_name: str, dynamics: Callable, drone: str):
    check_shapes(parametrize(dynamics, drone))


@pytest.mark.unit
@pytest.mark.parametrize("dynamics_name, dynamics", available_dynamics.items())
@pytest.mark.parametrize("drone", available_drones)
def test_dynamics_shapes_batched(dynamics_name: str, dynamics: Callable, drone: str):
    dynamics = parametrize(dynamics, drone, xp=xp)
    batch = (10, 5)
    check_shapes(dynamics, batch=batch)
    # Batched parameters
    dynamics.keywords.update(make_params(dynamics, batch, xp=xp))
    check_shapes(dynamics, batch=batch)


@pytest.mark.unit
@pytest.mark.parametrize("dynamics_name, dynamics", available_dynamics.items())
@pytest.mark.parametrize("drone", available_drones)
@pytest.mark.parametrize("ext_wrench", [False, True])
@pytest.mark.parametrize("per_motor_params", [False, True])
def test_symbolic_dynamics(
    dynamics_name: str, dynamics: Callable, drone: str, ext_wrench: bool, per_motor_params: bool
):
    """Tests if the symbolic and numeric dynamics produce the same output."""
    symbolic_dynamics = getattr(sys.modules[dynamics.__module__], "symbolic_dynamics")
    symbolic_dynamics = parametrize(symbolic_dynamics, drone)
    dynamics = parametrize(dynamics, drone)
    if per_motor_params:
        params = make_params(dynamics, scale=0.1)
        dynamics.keywords.update(params)
        symbolic_dynamics.keywords.update(params)
    inp = make_inputs(dynamics, batch=(10, 5), ext_wrench=ext_wrench)

    X_dot, X, U, _ = symbolic_dynamics(**symbolic_flags(dynamics, dist=ext_wrench))
    symbolic2numeric = cs.Function(dynamics_name, [X, U], [X_dot])

    order = ("pos", "quat", "vel", "ang_vel", "rotor_vel", "dist_f", "dist_t")
    for i in np.ndindex(np.shape(inp["pos"])[:-1]):  # casadi only supports non batched calls
        inp_i = {k: v[i + (...,)] for k, v in inp.items()}
        x_dot = xp.concat([x for x in dynamics(**inp_i) if x is not None], axis=-1)
        X = np.asarray(xp.concat([inp_i[k] for k in order if k in inp_i], axis=-1))
        U = np.asarray(inp_i["cmd"])
        x_dot_symbolic2numeric = xp.squeeze(xp.asarray(symbolic2numeric(X, U)), axis=-1)
        assert np.allclose(x_dot, x_dot_symbolic2numeric), (
            "Symbolic and numeric dynamics have different output"
        )


@pytest.mark.unit
@pytest.mark.parametrize("dynamics_name, dynamics", available_dynamics.items())
@pytest.mark.parametrize("drone", available_drones)
def test_compare_batched_non_batched(dynamics_name: str, dynamics: Callable, drone: str):
    """Tests if batching works and if the results are identical to the non-batched version."""
    dynamics = parametrize(dynamics, drone)
    inp = make_inputs(dynamics, batch=(10, 5))

    x_dot_batched = xp.concat([x for x in dynamics(**inp) if x is not None], axis=-1)
    for i in np.ndindex(np.shape(inp["pos"])[:-1]):
        out = dynamics(**{k: v[i + (...,)] for k, v in inp.items()})
        x_dot = xp.concat([x for x in out if x is not None], axis=-1)
        assert np.allclose(x_dot_batched[i + (...,)], x_dot, atol=1e-5), (
            "Non-batched and batched results are not the same"
        )


@pytest.mark.unit
@pytest.mark.parametrize("dynamics_name, dynamics", available_dynamics.items())
@pytest.mark.parametrize("drone", available_drones)
def test_batched_params(dynamics_name: str, dynamics: Callable, drone: str):
    """Tests if batched parameters give the same results as the shared parameters."""
    dynamics = parametrize(dynamics, drone, xp=xp)
    batch = (4, 3)
    inp = make_inputs(dynamics, batch=batch, ext_wrench=True)

    x_dot = dynamics(**inp)
    x_dot_batched = dynamics(**inp, **make_params(dynamics, batch, xp=xp))
    for dx, dx_batched in zip(x_dot, x_dot_batched, strict=True):
        assert_array_meta(dx_batched, dx)

    x_dot_batched = xp.concat([x for x in x_dot_batched if x is not None], axis=-1)
    x_dot = xp.concat([x for x in x_dot if x is not None], axis=-1)
    assert np.allclose(x_dot_batched, x_dot)

    params = make_params(dynamics, batch, scale=0.1, xp=xp)
    x_dot_batched = xp.concat([x for x in dynamics(**inp, **params) if x is not None], axis=-1)
    assert not np.allclose(x_dot_batched, x_dot), "Perturbation has no effect"
    for i in np.ndindex(batch):
        inp_i = {k: v[i + (...,)] for k, v in inp.items()}
        params_i = {k: v[i + (...,)] for k, v in params.items()}
        x_dot = xp.concat([x for x in dynamics(**inp_i, **params_i) if x is not None], axis=-1)
        assert np.allclose(x_dot_batched[i + (...,)], x_dot), f"Drone {i} differs"


@pytest.mark.unit
@pytest.mark.parametrize("dynamics_name, dynamics", available_dynamics.items())
@pytest.mark.parametrize("drone", available_drones)
def test_numeric_jit(dynamics_name: str, dynamics: Callable, drone: str):
    """Tests if the dynamics are jitable and if the results are identical to the array API ones."""
    dynamics = parametrize(dynamics, drone)
    inp = make_inputs(dynamics, batch=(10, 5))
    xp_dot = dynamics(**inp)
    jp_dot = jax.jit(dynamics)(**{k: jp.asarray(np.asarray(v)) for k, v in inp.items()})

    assert isinstance(jp_dot[0], jp.ndarray), "Results are not jax arrays"
    xp_dot = xp.concat([x for x in xp_dot if x is not None], axis=-1)
    jp_dot = jp.concat([x for x in jp_dot if x is not None], axis=-1)
    assert np.allclose(xp_dot, jp_dot), "numpy and jax results differ"


# TODO test if external wrench gets applied properly
# Maybe apply and predict how much higher the (angular) acceleration should be?
@pytest.mark.unit
def test_external_wrench(): ...
