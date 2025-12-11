import numpy as np

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.sim import use_box_collision


def main():
    """Example showing a drone crash."""
    sim = Sim(control=Control.state, freq=500, attitude_freq=500, state_freq=100)
    sim.reset()
    fps = 60
    use_box_collision(sim, enable=False)

    print("Phase 1: Hovering at [0, 0, 0.5] for 3 seconds")
    hover_duration = 3.0
    hover_cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    hover_cmd[..., :3] = [0.0, 0.0, 0.5]  # x, y, z position
    sim.state_control(hover_cmd)

    for i in range(int(hover_duration * sim.control_freq)):
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
            print(f"Crash detected: {sim.contacts().any()}")

    print("Phase 2: Dropping to [-5, 0, -0.5] for 3 seconds")
    drop_duration = 3.0
    drop_cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    drop_cmd[..., :3] = [-5.0, 0.0, -0.5]  # x, y, z position
    sim.state_control(drop_cmd)

    for i in range(int(drop_duration * sim.control_freq)):
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
            print(f"Crash detected: {sim.contacts().any()}")
    sim.close()


if __name__ == "__main__":
    main()
