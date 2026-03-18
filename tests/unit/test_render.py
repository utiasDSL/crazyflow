import mujoco
import pytest
from conftest import skip_if_headless

from crazyflow import Sim


@pytest.mark.unit
@pytest.mark.parametrize("cam_name", ["fpv_cam:0", "track_cam:0", "fpv_cam:1", "track_cam:1"])
@pytest.mark.render
@skip_if_headless
def test_render_camera_selection_from_name(cam_name: str):
    sim = Sim(drone_model="cf21B_500", n_drones=2)
    cam_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    sim.render(mode="human", camera=cam_name)
    viewer_cam = sim.viewer.viewer.cam
    assert viewer_cam.type == mujoco.mjtCamera.mjCAMERA_FIXED, "Camera type was not set to FIXED"
    assert viewer_cam.fixedcamid == cam_id, f"Expected cam ID {cam_id}, got {viewer_cam.fixedcamid}"
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("cam_id", [0, 1, 2, 3])
@pytest.mark.render
@skip_if_headless
def test_render_camera_selection_from_id(cam_id: int):
    sim = Sim(drone_model="cf21B_500", n_drones=2)
    sim.render(mode="human", camera=cam_id)
    viewer_cam = sim.viewer.viewer.cam
    assert viewer_cam.type == mujoco.mjtCamera.mjCAMERA_FIXED, "Camera type was not set to FIXED"
    assert viewer_cam.fixedcamid == cam_id, f"Expected cam ID {cam_id}, got {viewer_cam.fixedcamid}"
    sim.close()


@pytest.mark.unit
@pytest.mark.render
@skip_if_headless
def test_render_free_camera():
    sim = Sim(drone_model="cf21B_500", n_drones=2)
    sim.render(mode="human")
    viewer_cam = sim.viewer.viewer.cam
    assert viewer_cam.type == mujoco.mjtCamera.mjCAMERA_FREE, "Camera type was not set to FREE"
    sim.close()
