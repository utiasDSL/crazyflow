# Environment examples

Running the Gymnasium environments that wrap `Sim` for reinforcement learning. See [Gymnasium Environments](../user-guide/gymnasium-envs.md) for the available environments.

---

## Gymnasium environment

Evaluating a random policy in the figure-8 environment. The env wraps `Sim` behind the standard Gymnasium `VectorEnv` interface.

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/environments/figure8.py"
```
