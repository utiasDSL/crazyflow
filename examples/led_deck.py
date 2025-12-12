import tempfile
from pathlib import Path

import numpy as np

from crazyflow.control.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.visualize import change_material

scene_dark_xml = """
<mujoco model="Drone scene">
    <option integrator="RK4" density="1.225" viscosity="1.8e-5" timestep="0.001"/>
    <compiler inertiafromgeom="false" meshdir="assets" autolimits="true"/>
    <statistic center="0 0 2" extent="2.5"/>

    <visual>
        <rgba haze="0.15 0.25 0.35 0" fog="1 1 1 0"/>
        <map fogstart="0" fogend="0"/>
        <global azimuth="-20" elevation="-20" ellipsoidinertia="true"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
        markrgb="0.8 0.8 0.8" width="512" height="512"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="2 2" reflectance="0.2"/>
    </asset>

    <worldbody>
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    </worldbody>
</mujoco>
"""  # noqa: E501


def main():
    """Spawn 25 drones in one world and activate led decks."""
    try:
        # Use a named temporary xml file
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
            tmp.write(scene_dark_xml.encode())
            tmp.flush()
            tmp_path = Path(tmp.name)

        sim = Sim(n_drones=25, drone_model="cf21B_500", control=Control.state, xml_path=tmp_path)
        fps = 60
        cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
        cmd[..., 3] = sim.data.params.mass[0, 0, 0] * 9.81
        rgbas = np.random.default_rng(0).uniform(0, 1, (sim.n_drones, 4))
        rgbas[..., 3] = 1.0

        init_pos = np.array(sim.data.states.pos[0, :, :])
        cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
        cmd[:, :, :3] = init_pos
        cmd[:, :, 2] += 1.5

        for i in range(int(10 * sim.control_freq)):
            sim.state_control(cmd)
            sim.step(sim.freq // sim.control_freq)
            if ((i * fps) % sim.control_freq) < fps:
                even_ids = np.arange(0, sim.n_drones, 2)
                odd_ids = np.arange(1, sim.n_drones, 2)
                emission = np.sin(i / sim.control_freq * np.pi)
                change_material(
                    sim,
                    mat_name="led_top",
                    drone_ids=even_ids,
                    rgba=rgbas[even_ids, :],
                    emission=emission,
                )
                change_material(
                    sim,
                    mat_name="led_bot",
                    drone_ids=odd_ids,
                    rgba=rgbas[odd_ids, :],
                    emission=emission,
                )
                sim.render()
        sim.close()
    finally:
        # clean up the temporary file
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
