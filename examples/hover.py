import numpy as np

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim


def main():
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        physics=Physics.analytical,
        control=Control.state,
        freq=500,
        attitude_freq=500,
        state_freq=100,
        device="cpu",
    )

    sim.reset()
    duration = 5.0
    fps = 60

    # State cmd is [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate]
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    cmd[..., :3] = 0.1

    for i in range(int(duration * sim.control_freq)):
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
