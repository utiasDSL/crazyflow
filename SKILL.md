---
name: crazyflow
description: Non-obvious traps and conventions in the crazyflow drone simulator. Use when working
  in the crazyflow codebase, adding a dynamics model or drone platform, reusing the dynamics or
  controllers outside Sim for estimation or MPC, or writing code under jax.jit, grad or vmap.
---

# Crazyflow

Settings are in `pyproject.toml`, layout in the tree, API reference in `docs/`.

## Conventions

- Ruff settings live in `pyproject.toml` and must pass before committing.
- Dynamics and controllers follow the array API standard. Take the namespace from the inputs with
  `array_namespace(...)`, do all math through it, coerce bound parameters with `to_xp`, and never
  import `numpy` into a computation path.
- `crazyflow/control/mellinger/params.toml` deliberately holds values that differ from the true
  physical constants in `crazyflow/dynamics/*/params.toml`, reproducing the real firmware. Do not
  unify them.

## Testing

```bash
pixi run -e tests tests        # what CI runs
pixi run -e tests test-docs    # every example in docstrings and docs
```

Use pixi, not uv. `pixi run <cmd>` resolves task names only, so arbitrary commands need `-e tests`.

- `addopts = "-m 'not render'"` deselects every render test, and `tests/unit/test_visualizations.py`
  is `@skip_if_headless` without the marker, so it is skipped silently without `DISPLAY`. Rendering
  regressions pass CI green. Run `-m render` locally with a display.
- Docs and docstrings are executable. The markdown runner catches exceptions but never compares
  output, so a `print` with the result in a trailing comment proves nothing. Output that must be
  checked goes in a `pycon` fence with `>>>` prompts, which doctest picks up.
- `tests/integration/test_examples.py` runs every script under `examples/`, so a new example is a
  new test.
- `tests/conftest.py` forces JAX's persistent cache on at `/tmp/jax_cache-<uid>`, shared across
  branches. Delete it when failures make no sense.
- Request the `device` fixture rather than a GPU marker. It falls back to CPU silently, so
  `gpu-tests` asserts nothing about placement on a machine without CUDA.

## Adding a dynamics model or drone

Grepping the name of an existing model or drone finds every registration site, except when matching
against all models in the simulation's `build_control_fns`.

Define the function in `dynamics.py` and never in the package `__init__.py`, because
`load_fn_params` derives the model name from `fn.__module__.split(".")[-2]`. `parametrize`
binds exactly the keyword-only parameters after the bare `*`, so anything before it is never bound.
`Drone` in `crazyflow/drones/__init__.py` is a `StrEnum` with one member per MJCF file in
`crazyflow/drones`, asserted on import. A model supports a drone when its
`crazyflow/dynamics/<model>/params.toml` has a section for it. The commented example at the
top of each file lists the keys. `supported_drones` and `supported_dynamics` report the pairs, and the
tests only run those. Only `gravity_vec` is global, in `crazyflow/dynamics/params.toml`.

Registration alone produces roughly 40 parametrized tests. These do not include derivatives tests.

## Using dynamics and controllers outside Sim

Pure, batched, array-API functions with no dependency on `Sim`.

- Import crazyflow before scipy. `crazyflow/__init__.py` sets `SCIPY_ARRAY_API=1` and imports scipy
  immediately, and scipy cannot be reconfigured once loaded. Transitive imports through acados or
  sklearn trigger this too.
- `dynamics` and `control` each have `load_params(name, drone)`, returning everything defined for
  the drone, and `load_fn_params(fn, drone)`, which filters to the signature of `fn` and
  drops the rest. Additional platform data is stored as a comment in the drone MJCF.
- `parametrize` returns a `functools.partial` whose `keywords` dict is shared by every reference to
  it. Call `parametrize` again for an independent copy.
- Leading batch dimensions, trailing feature axis. `quat` is scalar-last xyzw, `force` is `(..., 1)`
  rather than a scalar, `ang_vel` is body frame.
- `ctrl_freq` scales only the integral and derivative terms, so it must match your real loop rate.
  Under jit, initialize integral states to zeros rather than `None`, since the pytree structure
  change forces a recompile.
- `symbolic_dynamics` monkey-patches module-global symbols while building, so it is not reentrant.
  Do not build two concurrently.

## The functional API and jit

`Sim` rebinds `sim.data` as a Python side effect that JAX cannot trace, so anything inside `jit`,
`grad`, `vmap` or `scan` must use `crazyflow.sim.functional` with `Sim` only as the builder. See
[`functional-api.md`](docs/user-guide/functional-api.md).

- **After editing a pipeline, call `build_step_fn()` or `build_reset_fn()` again.** The builders
  snapshot the stages, so a later edit is silently ignored and the stage never runs.
- `n_steps` is static, so each distinct value compiles separately. Build once and reuse.
- `states.force` and `states.torque` are inputs, not outputs, and nothing clears them, so writing
  them applies a persistent disturbance.
- Mutating state does not invalidate `core.mjx_synced`, so `contacts()` can query stale geometry.
  Set the flag false yourself.
- Gradients vanish at `clip_floor_pos` (`jnp.where` on floor contact) and at any saturation bound.

## Splats

`attach_splats` performs no calibration. Both `.ply` files must already sit in the simulator's
frames, the scene in the MuJoCo world frame at metric scale and the drone in its body frame. An
uncalibrated splat looks correct alone but does not match the simulated geometry. See
[`splat-rendering.md`](docs/user-guide/splat-rendering.md).
