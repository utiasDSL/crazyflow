import os

import numpy as np

os.environ["SCIPY_ARRAY_API"] = "1"

from scipy.spatial.transform import Rotation as R

from crazyflow.control import Control
from crazyflow.sim import Sim

kp = np.array([0.4, 0.4, 1.25])
ki = np.array([0.05, 0.05, 0.05])
kd = np.array([0.2, 0.2, 0.4])
g = 9.81


def control(
    t: float, obs: dict[str, np.ndarray], pos_start: np.ndarray, drone_mass: float
) -> np.ndarray:
    des_pos = np.zeros(3)
    des_pos[..., :2] = pos_start[:2] + np.array([np.cos(t) - 1, np.sin(t)])
    des_pos[..., 2] = 0.2 * t
    des_vel = np.zeros_like(des_pos)
    des_yaw = t

    # Calculate the deviations from the desired trajectory
    pos_error = des_pos - np.array(obs["pos"])
    vel_error = des_vel - np.array(obs["vel"])

    # Compute target thrust
    target_thrust = np.zeros(3)
    target_thrust += kp * pos_error
    target_thrust += kd * vel_error
    target_thrust[2] += drone_mass * g

    # Update z_axis to the current orientation of the drone
    z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]

    # update current thrust
    thrust_desired = target_thrust.dot(z_axis)

    # update z_axis_desired
    z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
    x_c_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
    y_axis_desired = np.cross(z_axis_desired, x_c_des)
    y_axis_desired /= np.linalg.norm(y_axis_desired)
    x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

    R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
    euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

    action = np.concatenate([euler_desired, [thrust_desired]], dtype=np.float32)

    return action


def main():
    sim = Sim(control=Control.attitude)
    sim.reset()
    duration = 6.5
    fps = 60

    cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))  # [roll, pitch, yaw, thrust]
    pos_start = sim.data.states.pos
    for i in range(int(duration * sim.control_freq)):
        obs = {
            "pos": sim.data.states.pos[0, 0],
            "vel": sim.data.states.vel[0, 0],
            "quat": sim.data.states.quat[0, 0],
        }
        cmd[0, 0, :] = control(
            i / sim.control_freq, obs, pos_start[0, 0], sim.data.params.mass[0, 0, 0]
        )
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
