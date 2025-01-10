import numpy as np

from crazyflow.control import Control
from crazyflow.sim import Sim


def control(start_xy: np.ndarray, t: float) -> np.ndarray:
    circle = np.array([np.cos(t) - 1, np.sin(t)])
    cmd = np.zeros((*start_xy.shape[:-1], 13))
    cmd[..., :2] = start_xy + circle
    cmd[..., 2] = 0.2 * t
    return cmd


def main():
    sim = Sim(n_drones=4, control=Control.state, integrator="rk4", physics="analytical")
    sim.reset()
    duration = 5.0
    fps = 60

    start_xy = sim.data.states.pos[..., :2]
    for i in range(int(duration * sim.control_freq)):
        sim.state_control(control(start_xy, i / sim.control_freq))
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
