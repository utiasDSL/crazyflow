import jax

from crazyflow.sim import Physics, Sim
from crazyflow.sim.domain_randomizer import DomainRandomizer


def main():
    """Spawn multiple envs with domain randomization."""
    n_worlds, n_drones = 1, 1
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.sys_id, device="cpu")

    example_1(sim)
    example_2(sim)
    example_3(sim)
    example_4(sim)
    example_5(sim)
    example_6(sim)

    sim.close()

def example_1(sim):
    """Randomize friciton from a uniform distribution. """
    env_num = 10
    friction_range = (0.6, 1.4)
    domain_randomizer = DomainRandomizer(env_num=env_num, seed=0)
    batched_sys = domain_randomizer.randomize_friction_from_distribution(sim._mjx_model, "uniform", friction_range)

    print('Single env friction shape: ', sim._mjx_model.geom_friction.shape)
    print('Batched env friction shape: ', batched_sys.geom_friction.shape)

    print('Friction on geom 0: ', sim._mjx_model.geom_friction[0, 0])
    print('Random frictions on geom 0: ', batched_sys.geom_friction[:, 0, 0])

def example_2(sim):
    """Randomize friction from a list of values."""
    env_num = 10
    friction_range = (0.6, 1.4)
    domain_randomizer = DomainRandomizer(env_num=env_num, seed=0)

    def helper(key):
        return jax.random.uniform(key, shape=(1,), minval=friction_range[0], maxval=friction_range[1])
    
    frictions = jax.vmap(helper)(domain_randomizer.rng)
    batched_sys = domain_randomizer.randomize_friction_from_values(sim._mjx_model, frictions)

    print('Single env friction shape: ', sim._mjx_model.geom_friction.shape)
    print('Batched env friction shape: ', batched_sys.geom_friction.shape)

    print('Friction on geom 0: ', sim._mjx_model.geom_friction[0, 0])
    print('Random frictions on geom 0: ', batched_sys.geom_friction[:, 0, 0])

def example_3(sim):
    """Randomize mass from a uniform distribution."""
    env_num = 10
    mass_range = (0.022, 0.032)
    domain_randomizer = DomainRandomizer(env_num=env_num, seed=0)
    batched_sys = domain_randomizer.randomize_mass_from_distribution(sim._mjx_model, "uniform", mass_range)

    print('Single env mass shape: ', sim._mjx_model.body_mass.shape)
    print('Batched env mass shape: ', batched_sys.body_mass.shape)

    print('Mass on part 1: ', sim._mjx_model.body_mass[1])
    print('Random mass on part 1: ', batched_sys.body_mass[:, 1])

def example_4(sim):
    """Randomize mass from a list."""
    env_num = 10
    mass_range = (0.022, 0.032)
    domain_randomizer = DomainRandomizer(env_num=env_num, seed=0)

    def helper(key):
        return jax.random.uniform(key, shape=(), minval=mass_range[0], maxval=mass_range[1])
    
    masses = jax.vmap(helper)(domain_randomizer.rng)
    batched_sys = domain_randomizer.randomize_mass_from_values(sim._mjx_model, masses)

    print('Single env mass shape: ', sim._mjx_model.body_mass.shape)
    print('Batched env mass shape: ', batched_sys.body_mass.shape)

    print('Mass on part 1: ', sim._mjx_model.body_mass[1])
    print('Random mass on part 1: ', batched_sys.body_mass[:, 1])

def example_5(sim):
    """Randomize the position of the motors from a uniform distribution."""
    env_num = 10
    position_range = (-0.02, 0.02)
    domain_randomizer = DomainRandomizer(env_num=env_num, seed=0)
    batched_sys = domain_randomizer.randomize_body_positions_from_distribution(sim._mjx_model, "uniform", position_range)

    print('Single env motor position shape: ', sim._mjx_model.body_pos.shape)
    print('Batched env motor position shape: ', batched_sys.body_pos.shape)

    print('Motor positions: ', sim._mjx_model.body_pos)
    print('Random motor positions: ', batched_sys.body_pos)

def example_6(sim):
    """Randomize the position of the motors from a list of values."""
    env_num = 10
    position_range = (-0.02, 0.02)
    domain_randomizer = DomainRandomizer(env_num=env_num, seed=0)

    def helper(key):
        return jax.random.uniform(key, shape=(3,), minval=position_range[0], maxval=position_range[1])
    
    positions = jax.vmap(helper)(domain_randomizer.rng)
    batched_sys = domain_randomizer.randomize_body_positions_from_values(sim._mjx_model, positions)

    print('Single env motor position shape: ', sim._mjx_model.body_pos.shape)
    print('Batched env motor position shape: ', batched_sys.body_pos.shape)

    print('Motor positions: ', sim._mjx_model.body_pos)
    print('Random motor positions: ', batched_sys.body_pos)

if __name__ == "__main__":
    main()
