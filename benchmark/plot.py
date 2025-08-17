from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_fps_data(data_folder: Path):
    """Read all CSVs from the data folder and plot the latest gym and sim fps by device.

    Args:
        data_folder: pathlib.Path object pointing to the folder containing CSV files
    """
    # Colors for different devices. Deep blue for CPU, NVIDIA green for GPU
    colors = {"cpu": "#0000AA", "gpu": "#76B900"}
    device_names = {"cpu": "Intel Core i9-13900KF", "gpu": "NVIDIA RTX 4090"}
    dfs = {}

    # Read all CSV files in the data folder
    for csv_file in sorted(data_folder.glob("*.csv")):
        df = pd.read_csv(csv_file)
        device = df["device"].iloc[-1].lower()
        # Group data by test_type and n_worlds
        dfs["sim_" + device] = df[df["test_type"] == "simulator"]
        dfs["gym_" + device] = df[df["test_type"] == "gym_env"]

    if not dfs:
        print("No valid data found for plotting")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Add a super title (suptitle) for the entire figure
    fig.suptitle("Crazyflow Performance", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the suptitle

    # Plot gym FPS
    for key, df in dfs.items():
        if key.startswith("gym_"):
            device = key.split("_")[1]
            color = colors[device]

            # Plot FPS
            ax1.plot(
                df["n_worlds"],
                df["fps"],
                marker="o",
                linestyle="-",
                color=color,
                label=f"{device_names[device]}",
            )

    ax1.set_title("Steps per second: Gym envs")
    ax1.set_xlabel("Number of Worlds")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    format_log_axes(ax1, dfs, "gym_")

    # Plot sim FPS
    for key, df in dfs.items():
        if key.startswith("sim_"):
            device = key.split("_")[1]
            color = colors[device]

            # Plot FPS
            ax2.plot(
                df["n_worlds"],
                df["fps"],
                marker="o",
                linestyle="-",
                color=color,
                label=f"{device_names[device]}",
            )

    ax2.set_title("Steps per second: Crazyflow")
    ax2.set_xlabel("Number of Worlds")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True)
    format_log_axes(ax2, dfs, "sim_")
    # Add legend for the axis
    ax2.legend(loc="upper left")

    plt.tight_layout()

    # Save the plot
    output_path = data_folder / "performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def format_log_axes(ax: plt.Axes, dfs: dict[str, pd.DataFrame], prefix: str):
    """Format logarithmic axes with nice labels.

    Args:
        ax: matplotlib axis to format
        dfs: dictionary of dataframes
        prefix: prefix for filtering dataframes (e.g., "gym_" or "sim_")
    """
    # Rename the axes labels
    xticks = np.array([1, 10, 100, 1000, 10000, 100000, 1000000])
    min_x = min([df["n_worlds"].min() for key, df in dfs.items() if key.startswith(prefix)])
    max_x = max([df["n_worlds"].max() for key, df in dfs.items() if key.startswith(prefix)])
    mask = (xticks >= min_x) & (xticks <= max_x)
    valid_indices = np.nonzero(mask)[0]
    ax.set_xticks(xticks[valid_indices])
    xticklabels = ["1", "10", "100", "1K", "10K", "100K", "1M"]
    ax.set_xticklabels([xticklabels[i] for i in valid_indices])

    # Get min and max y values for plots
    min_y = min([df["fps"].min() for key, df in dfs.items()])
    max_y = max([df["fps"].max() for key, df in dfs.items()])

    # Create logarithmic y-ticks
    # Generate yticks based on data range
    min_power = int(np.floor(np.log10(min_y)))
    max_power = int(np.ceil(np.log10(max_y)))
    yticks = np.array([10**i for i in range(min_power, max_power + 1)])
    ax.set_yticks(yticks)
    yticklabels = []
    abbrev = {1e9: "B", 1e6: "M", 1e3: "K"}
    for i in yticks:
        for divisor, suffix in sorted(abbrev.items(), reverse=True):
            if i >= divisor:
                yticklabels.append(f"{int(i // divisor)}{suffix}")
                break
        else:
            yticklabels.append(f"{int(i)}")
    ax.set_yticklabels(yticklabels)

    # Remove minor ticks for cleaner appearance
    ax.minorticks_off()


if __name__ == "__main__":
    plot_fps_data(Path(__file__).parent / "data")
