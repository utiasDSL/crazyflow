import crazyflow  # noqa: F401, ensure gymnasium envs are registered
from crazyflow.sim import Sim


def main():
    """Main entry point for profiling."""
    sim = Sim(n_worlds=1, n_drones=1, physics="analytical", control="attitude")

    compiled_reset = sim._reset.lower(sim.data, sim.default_data, None).compile()
    compiled_step = sim._step.lower(sim.data, 1).compile()
    op_count_reset = compiled_reset.cost_analysis()["flops"]
    op_count_step = compiled_step.cost_analysis()["flops"]
    print(f"Op counts:\n Reset: {op_count_reset}\n Step: {op_count_step}")


if __name__ == "__main__":
    main()
