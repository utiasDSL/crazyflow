import numpy as np

from crazyflow.control import Control
from crazyflow.sim import Sim


def main():
    sim = Sim(control=Control.force_torque)
    sim.reset()
    duration = 5.0
    fps = 60

    cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))  # [fz, tx, ty, tz]
    mass, gravity = sim.data.params.mass, -sim.data.params.gravity_vec[-1]
    cmd[..., 0] = (mass + 1e-4) * gravity  # Plus a small margin to accelerate slightly
    for i in range(int(duration * sim.control_freq)):
        sim.force_torque_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
