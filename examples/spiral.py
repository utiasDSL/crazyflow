import numpy as np

from crazyflow.control.controller import Control, Controller
from crazyflow.sim.core import Sim
from crazyflow.sim.physics import Physics


def attitude_control(pos, quat, target_pos, target_quat, dt):
    # TODO: Implement attitude control
    return np.array([[[10, 0, 0, 0]]])


def main():
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        physics=Physics.analytical,
        control=Control.attitude,
        controller=Controller.emulatefirmware,
        freq=500,
        control_freq=500,
        device="cpu",
    )
    sim.reset()
    duration = 5.0
    fps = 60

    for i in range(int(duration * sim.freq)):
        if sim.controllable:
            des_pos = np.array([np.cos(i * 0.1), np.sin(i * 0.1), i * 0.1]) * i / sim.freq
            des_quat = np.array([0, 0, 0, 1])
            pos = sim.states["pos"][0, 0]
            quat = sim.states["quat"][0, 0]
            cmd = attitude_control(pos, quat, des_pos, des_quat, dt=sim.dt)
            sim.attitude_control(cmd)
        sim.step()
        if i * fps % sim.freq < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
