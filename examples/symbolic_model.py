from ml_collections import config_dict

from crazyflow.sim.core import Sim
from crazyflow.sim.symbolic import symbolic

device = "cpu"
sim_config = config_dict.ConfigDict()
sim_config.n_worlds = 1
sim_config.n_drones = 1
sim_config.physics = "sys_id"
sim_config.control = "attitude"
sim_config.controller = "emulatefirmware"
sim_config.device = device
sim = Sim(**sim_config)

symbolic_model = symbolic(sim, sim.dt)
