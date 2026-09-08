# Dynamics

Crazyflow supports four dynamics, selectable via the `Dynamics` enum. All dynamics share the same state representation and control interface, so you can swap them at construction time without changing any other code.

```python
from crazyflow.sim import Sim, Dynamics

sim = Sim(dynamics=Dynamics.first_principles)
```

## Available dynamics

| Dynamics | Enum value | Command input | Description |
|---|---|---|---|
| First principles | `Dynamics.first_principles` | Rotor RPM | Full analytical dynamics with identified parameters |
| SO(3) + RPY | `Dynamics.so_rpy` | Roll/pitch/yaw + thrust | Simplified fitted dynamics |
| SO(3) + RPY + rotor | `Dynamics.so_rpy_rotor` | Roll/pitch/yaw + thrust | Adds first-order rotor dynamics |
| SO(3) + RPY + rotor + drag | `Dynamics.so_rpy_rotor_drag` | Roll/pitch/yaw + thrust | Adds translational and rotational drag |

`Dynamics.default` resolves to `Dynamics.first_principles`.

## First-principles dynamics

The first-principles dynamics derives forces and torques analytically from motor speeds using identified physical parameters: mass, arm length, propeller constants, and the full inertia tensor. It operates at the rotor-velocity level and is the most accurate dynamics for sim-to-real transfer.

```python
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

# Force-torque and rotor_vel control modes require first_principles
sim = Sim(dynamics=Dynamics.first_principles, control=Control.rotor_vel)
sim.reset()
```

Parameters accessible through `sim.data.params`:

| Parameter | Description |
|---|---|
| `mass` | Drone mass, kg |
| `J` | Inertia matrix, kg·m² |
| `L` | Motor arm length, m, shared `(1,)` or per motor `(4,)` |
| `rpm2thrust` | Thrust curve coefficients `[a, b, c]` of `f = a + b·rpm + c·rpm²`, shared `(1, 3)` or per motor `(4, 3)` |
| `rpm2torque` | Torque curve coefficients, shared `(1, 3)` or per motor `(4, 3)` |
| `mixing_matrix` | Maps rotor RPMs² to [thrust, tx, ty, tz] |
| `rotor_dyn_coef` | Rotor spin-up/down coefficients, shared `(1, 4)` or per motor `(4, 4)` |

## Fitted dynamics (so_rpy family)

The `so_rpy` dynamics are identified from flight data using a small number of flight minutes. They take higher-level commands (roll/pitch/yaw setpoints + collective thrust in Newtons) and are faster to simulate because they skip the rotor-velocity level.

These dynamics are a good choice when:

- You are training RL agents and want speed over fidelity
- Your controller outputs attitude targets (as most Crazyflie firmware does)
- You do not need rotor-level detail

```python
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

sim = Sim(
    dynamics=Dynamics.so_rpy_rotor_drag,  # most accurate of the fitted family
    control=Control.attitude,
)
sim.reset()
```

The `so_rpy_rotor_drag` variant includes translational drag, which captures the velocity-dependent deceleration effect visible in aggressive flights. It is the recommended fitted dynamics for sim-to-real experiments.

## Control mode compatibility

| Dynamics | `Control.state` | `Control.attitude` | `Control.force_torque` | `Control.rotor_vel` |
|---|---|---|---|---|
| `first_principles` | ✓ | ✓ | ✓ | ✓ |
| `so_rpy` | ✓ | ✓ | ✗ | ✗ |
| `so_rpy_rotor` | ✓ | ✓ | ✗ | ✗ |
| `so_rpy_rotor_drag` | ✓ | ✓ | ✗ | ✗ |

!!! warning
    Using `Control.force_torque` or `Control.rotor_vel` with a fitted dynamics raises `ConfigError` at construction time.

## Using the dynamics standalone

The `Dynamics` enum above is how you select a dynamics *inside the simulator*. The dynamics also live in `crazyflow.dynamics` as a self-contained library of pure functions, usable on their own for state estimation, control design, or as MPC models independent of `Sim`. The following guides cover that standalone API:

- [Dynamics functions](dynamics-functions.md) — the state representation, the four dynamics functions, and external disturbances
- [Parametrization](parametrize.md) — binding a dynamics function to a drone configuration
- [Batching & domain randomization](batching.md) — evaluating many drones at once and randomizing parameters
- [Symbolic dynamics](symbolic.md) — CasADi expressions for MPC and estimation
- [System identification](system-identification.md) — fitting dynamics coefficients from flight data

## Next steps

- [Control Modes](../control/index.md) — command shapes and the control hierarchy
- [Object-Oriented API](../oo-api.md) — full constructor arguments
