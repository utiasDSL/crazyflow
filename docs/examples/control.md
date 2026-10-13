# Control examples

Flying a drone with the different control modes, from state commands down to body rates, and a sampling-based MPC built on batched rollouts. See [Control Modes](../user-guide/control/index.md) for the controller behind each mode.

---

## Hover

A single drone commanded to hold a fixed height using state control. This is the minimal end-to-end loop: create a `Sim`, reset it, apply a state command, and step forward.

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/control/hover.py"
```

```bash
python examples/control/hover.py
```

---

## Attitude control

Commanding roll, pitch, yaw, and collective thrust directly. This level bypasses the Mellinger position loop and is typical for RL agents that output attitude targets.

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/control/attitude.py"
```

---

## Body rate control

Commanding body-frame angular rates and collective thrust. The firmware controller has no dedicated body rate mode and levels the drone with its attitude terms, so the example sets the `kR` and `ki_m` gains of the body rate controller to zero. 

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/control/body_rate.py"
```

---

## Sampling-based MPC

A sampling-based model predictive controller tracks a Lissajous curve while avoiding a grid of obstacles. It rolls out thousands of candidate control sequences in parallel using identified dynamics, then applies the first action from a cost-weighted update of the best samples. The controller automatically uses a GPU when one is available and lowers the sample count on CPU.

```bash
python examples/control/sampling.py
```
