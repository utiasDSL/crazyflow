"""Shard the simulation across devices.

The host backend is asked for four logical devices, and the sharded results are compared against
running on a single device.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import numpy as np

from crazyflow.sim import Sim
from crazyflow.sim.sharding import world_mesh


def main():
    devices = jax.devices("cpu")
    sim = Sim(n_worlds=4 * len(devices), device="cpu")  # Worlds must be divisible by n_devices
    sim.step(10)
    single_device_pos = np.asarray(sim.data.states.pos)

    sim.reset()
    sim.shard(world_mesh(devices))
    sim.step(10)

    pos, gravity = sim.data.states.pos, sim.data.params.gravity_vec
    print(f"Distributing {sim.n_worlds} worlds over {len(devices)} devices")
    # Arrays that carry a world axis are partitioned. Shared params are replicated on every device
    print("  Placement")
    print(f"    {'position':<{10}}  {str(pos.shape):<{10}}  {pos.sharding.spec}")
    print(f"    {'gravity':<{10}}  {str(gravity.shape):<{10}}  {gravity.sharding.spec}")
    print("  Shards of states.pos")
    for shard in pos.addressable_shards:
        print(f"    {shard.device}  {shard.data.shape[0]} worlds")

    assert np.allclose(np.asarray(pos), single_device_pos, atol=1e-6)
    sim.close()


if __name__ == "__main__":
    main()
