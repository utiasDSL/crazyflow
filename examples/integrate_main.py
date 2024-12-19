import matplotlib.pyplot as plt
import numpy as np

from crazyflow.sim.core import Sim


def control(t: float):
    pos = np.array([np.cos(t) - 1, np.sin(t), 0.2 * t])
    cmd = np.zeros((1, 1, 13))
    cmd[:, :, :3] = pos
    return cmd


def run_spiral(freq):
    sim = Sim(control="state", freq=freq, physics="analytical")
    # sim = Sim(control="state", freq=freq, physics="sys_id")
    duration = 5.0
    positions = []

    for i in range(int(duration * sim.freq)):
        if sim.controllable.any():
            cmd = control(i / sim.freq)
            sim.state_control(cmd)
        sim.step()
        if i % (sim.freq // 100) == 0:  # Sample 100 times per second
            # sim.render()
            positions.append(sim.states.pos[0, 0].copy())
    sim.close()
    return np.array(positions)


def main():
    # Run simulations at different frequencies
    if False:
        freqs = np.array([1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000])
        all_positions = {}

        for freq in freqs:
            print(f"Running simulation at {freq}Hz")
            all_positions[freq] = run_spiral(freq)

        # Get positions at highest frequency as reference
        ref_positions = all_positions[max(freqs)]
    else:
        save_file = "simulation_results_analytical.npy"
        data = np.load(save_file, allow_pickle=True).item()
        freqs = data["frequencies"]
        all_positions = data["positions"]
        ref_positions = data["reference"]
        freqs = freqs[:-1]

    # Compute and plot errors
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for freq in freqs[:-1]:  # Exclude highest frequency
        # Compute absolute error
        error = all_positions[freq] - ref_positions
        t = np.linspace(0, 5, len(error))

        ax1.plot(t, error[:, 0], label=f"{freq/1000:.0f}kHz")
        ax2.plot(t, error[:, 1], label=f"{freq/1000:.0f}kHz")
        ax3.plot(t, error[:, 2], label=f"{freq/1000:.0f}kHz")

    # Calculate and print total distance error for each frequency
    print("\nAverage distance error:")
    for freq in freqs[:-1]:
        total_dist_error = np.mean(np.linalg.norm(all_positions[freq] - ref_positions, axis=1))
        print(f"{freq/1000:.0f}kHz: {total_dist_error:.6f} meters")

    plt.suptitle("Position Error vs Reference (100kHz)")

    ax1.set_ylabel("X Error (m)")
    ax1.grid(True)
    ax1.legend()

    ax2.set_ylabel("Y Error (m)")
    ax2.grid(True)
    ax2.legend()

    ax3.set_ylabel("Z Error (m)")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True)
    ax3.legend()

    # Save results to disk
    save_data = {"frequencies": freqs, "positions": all_positions, "reference": ref_positions}
    np.save("simulation_results.npy", save_data)

    plt.xlabel("Time (s)")
    plt.ylabel("Absolute Position Error (m)")
    plt.title("Position Error vs Reference (100kHz)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
