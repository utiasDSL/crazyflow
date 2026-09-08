# System identification

If your drone is not among the [supported configurations](parametrize.md#available-drone-configurations), or if you want to refine the existing parameters with your own hardware, the system identification pipeline fits the dynamics coefficients from recorded flight data. It handles data preprocessing, derivative estimation, and least-squares parameter fitting for both translational and rotational dynamics.

The pipeline is part of `crazyflow.dynamics.utils`. Plotting the fit (`plot=True`) additionally requires `matplotlib`.

## Required data format

The pipeline expects a Python dict of NumPy arrays assembled from your flight log. The keys below are required by [`preprocessing`][crazyflow.dynamics.utils.data_utils.preprocessing]:

| Key | Shape | Units | Description |
|---|---|---|---|
| `"time"` | `(N,)` | s | Timestamps (need not be evenly spaced) |
| `"pos"` | `(N, 3)` | m | Position in world frame |
| `"quat"` | `(N, 4)` | — | Orientation quaternion (xyzw) |
| `"cmd_rpy"` | `(N, 3)` | rad | Commanded roll/pitch/yaw |
| `"cmd_f"` | `(N,)` | N | Commanded collective thrust |

After `preprocessing` + [`derivatives_svf`][crazyflow.dynamics.utils.data_utils.derivatives_svf], the dict is augmented with filtered signals and numerical derivatives. The identification functions read `SVF_vel`, `SVF_acc`, `SVF_quat`, `SVF_cmd_f` (translation) and `SVF_rpy`, `SVF_cmd_rpy` (rotation).

## Full pipeline

<!-- notest: requires flight-log data -->
```{ .python notest }
from crazyflow.dynamics.utils.data_utils import preprocessing, derivatives_svf
from crazyflow.dynamics.utils.identification import sys_id_translation, sys_id_rotation

# Step 1 — assemble raw data dict from your flight log
data = {
    "time": time_array,  # (N,) seconds
    "pos": pos_array,  # (N, 3) meters
    "quat": quat_array,  # (N, 4) xyzw
    "cmd_rpy": cmd_rpy_array,  # (N, 3) radians
    "cmd_f": cmd_f_array,  # (N,)  Newtons
}

# Step 2 — outlier removal, quaternion normalisation, RPY calculation
data = preprocessing(data)

# Step 3 — low-pass filter and compute time derivatives via State Variable Filter
data = derivatives_svf(data)

# Step 4 — fit translational parameters
trans_params = sys_id_translation(
    dynamics="so_rpy_rotor_drag",
    mass=0.0319,  # drone mass in kg — measure this directly
    data=data,
    verbose=0,  # 0 = silent, 1 = progress, 2 = full optimizer output
    plot=True,  # show fit vs. measured plots
)
# Returns: {'cmd_f_coef': ..., 'thrust_time_coef': ...,
#           'drag_xy_coef': ..., 'drag_z_coef': ...}

# Step 5 — fit rotational parameters
rot_params = sys_id_rotation(data=data, verbose=0, plot=True)
# Returns: {'rpy_coef': (3,), 'rpy_rates_coef': (3,), 'cmd_rpy_coef': (3,)}
```

See [`sys_id_translation`][crazyflow.dynamics.utils.identification.sys_id_translation] and [`sys_id_rotation`][crazyflow.dynamics.utils.identification.sys_id_rotation] in the API reference for the full argument list.

## Validation

To check that the identified parameters generalise to unseen flight regimes, collect a second dataset of different trajectories and pass it as `data_validation`. RMSE and R² are then reported on both the training data and the validation data.

<!-- notest: requires flight-log data -->
```{ .python notest }
# Preprocess the validation dataset independently — it must come from
# different trajectories, not a split of the same recording.
data_valid = preprocessing(validation_raw_data)
data_valid = derivatives_svf(data_valid)

trans_params = sys_id_translation(
    dynamics="so_rpy_rotor_drag",
    mass=0.0319,
    data=data,
    data_validation=data_valid,
    plot=True,
)
```

## Using identified parameters

Once you have the identified coefficients, add them to the relevant `params.toml` file under a new drone name. Each dynamics sub-package ships its own `params.toml` — for example `crazyflow/dynamics/so_rpy_rotor_drag/params.toml` — and `load_params` reads from it when you call `parametrize`. Add a new section using the TOML table syntax:

```toml
[my_drone]
cmd_f_coef       = 0.983        # from trans_params["cmd_f_coef"]
thrust_time_coef = 0.121        # from trans_params["thrust_time_coef"]
drag_matrix      = [[-0.0147, 0.0, 0.0],
                    [0.0, -0.0147, 0.0],
                    [0.0, 0.0, -0.0128]]  # diag([drag_xy, drag_xy, drag_z])
rpy_coef         = [-245.67, -245.67, -227.78]  # from rot_params["rpy_coef"]
rpy_rates_coef   = [-17.32, -17.32, -25.63]     # from rot_params["rpy_rates_coef"]
cmd_rpy_coef     = [196.18, 196.18, 390.27]     # from rot_params["cmd_rpy_coef"]
```

!!! note
    `sys_id_translation` returns `drag_xy_coef` and `drag_z_coef` as scalars. Assemble the diagonal `drag_matrix` manually: `[drag_xy, drag_xy, drag_z]` on the diagonal.

Once the entry is in the TOML file, load the dynamics as usual:

<!-- notest: example drone not in params.toml -->
```{ .python notest }
from crazyflow.dynamics import parametrize
from crazyflow.dynamics.so_rpy_rotor_drag import dynamics

dynamics = parametrize(dynamics, drone="my_drone")
```

Support for new drones can be added to the shared parameter files via a pull request on [GitHub](https://github.com/learnsyslab/crazyflow).

## Which dynamics to identify

Choose based on which physical effects you need to capture:

- **`so_rpy`** — identifies only `cmd_f_coef`; no motor dynamics, no drag. Fastest to calibrate, good for slow flight.
- **`so_rpy_rotor`** — adds `thrust_time_coef` to model motor spin-up delay. Better for agile maneuvers.
- **`so_rpy_rotor_drag`** — adds `drag_xy_coef` and `drag_z_coef`. Best accuracy at higher speeds where aerodynamic drag is significant.

---
