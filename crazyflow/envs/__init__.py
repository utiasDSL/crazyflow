from gymnasium.envs.registration import register

from crazyflow.envs.figure_8_env import FigureEightEnv
from crazyflow.envs.landing_env import LandingEnv
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from crazyflow.envs.reach_pos_env import ReachPosEnv
from crazyflow.envs.reach_vel_env import ReachVelEnv

__all__ = ["ReachPosEnv", "ReachVelEnv", "LandingEnv", "NormalizeActions", "FigureEightEnv"]

register(id="DroneReachPos-v0", vector_entry_point=ReachPosEnv)

register(id="DroneReachVel-v0", vector_entry_point=ReachVelEnv)

register(id="DroneLanding-v0", vector_entry_point=LandingEnv)

register(id="DroneFigureEightTrajectory-v0", vector_entry_point=FigureEightEnv)
