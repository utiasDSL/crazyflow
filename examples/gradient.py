import time

import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.data import SimData


def main():
    sim = Sim(control=Control.attitude, physics=Physics.first_principles, attitude_freq=50)
    sim_step = sim._step

    def step(cmd: NDArray, data: SimData) -> jax.Array:
        data = data.replace(
            controls=data.controls.replace(attitude=data.controls.attitude.replace(staged_cmd=cmd))
        )
        data = sim_step(data, 10)
        return (data.states.pos[0, 0, 2] - 1.0) ** 2  # Quadratic cost to reach 1m height

    step_grad = jax.jit(jax.grad(step))

    cmd = jnp.zeros((1, 1, 4), dtype=jnp.float32)
    cmd = cmd.at[..., 3].set(0.3)

    # Trigger jax's jit to compile the gradient function. This is not necessary, but it ensures that
    # the timings are not affected by the compilation time.
    step_grad(cmd, sim.data).block_until_ready()
    # JAX compiles again if static properties change. Not sure why this is happening here, but this
    # is a simple way to enforce all recompilations before measuring performance.
    step_grad(cmd - 0.1 * step_grad(cmd, sim.data), sim.data).block_until_ready()

    print(f"Initial command: {cmd}")
    t0 = time.perf_counter()
    for _ in range(10):
        grad = step_grad(cmd, sim.data)
        cmd = cmd - 0.1 * grad
    t1 = time.perf_counter()
    print(f"Loss: {step(cmd, sim.data)}\nGradient: {grad}")

    print(f"Time taken: {t1 - t0:.2e}s ({(t1 - t0) / 10:.2e}s per step)")
    # The final command should increase the z position (3rd array element) as well as the z velocity
    # (6th array element) to minimize the cost function.
    print(f"Final command: {cmd}")


if __name__ == "__main__":
    main()
