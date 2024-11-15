from crazyflow.sim import Physics, Sim
from crazyflow.sim.domain_randomizer import DomainRandomizer


def main():
    """Spawn multiple drones with domain randomization."""
    n_worlds, n_drones = 1, 1
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.sys_id, device="cpu")

    # Randomize friciton
    env_num = 10
    domain_randomizer = DomainRandomizer(env_num=env_num, seed=0)
    batched_sys = domain_randomizer.randomize_friction_from_distribution(
        sim._mjx_model, "uniform", (0.6, 1.4)
    )

    print("Single env friction shape: ", sim._mjx_model.geom_friction.shape)
    print("Batched env friction shape: ", batched_sys.geom_friction.shape)

    print("Friction on geom 0: ", sim._mjx_model.geom_friction[0, 0])
    print("Random frictions on geom 0: ", batched_sys.geom_friction[:, 0, 0])

    sim.close()


if __name__ == "__main__":
    main()
