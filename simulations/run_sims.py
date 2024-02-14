import os
import numpy as np
import argparse
from simulate import SimulationDataset


def main(sim, n, dim, nt, ns, seed, output_dir):
    sim_sets = {
        "r1": {"dt": 5e-3},
        "r2": {"dt": 1e-3},
        "spring": {"dt": 1e-2},
        "string": {"dt": 1e-2},
        "charge": {"dt": 1e-3},
        "superposition": {"dt": 1e-3},
        "damped": {"dt": 2e-2},
        "discontinuous": {"dt": 1e-2},
    }

    # Fetch hand picked dt for the sim.
    dt = sim_sets[sim]["dt"]

    # Create a title from the sim params
    title = (
        f"sim={sim}_ns{ns}_seed{seed}_n_body={n}_dim={dim}_nt={nt}_dt={dt:.0e}"
    )

    # Create a simulation dataset object.
    sim_obj = SimulationDataset(sim, n=n, dim=dim, nt=nt, dt=dt)

    # Run the simulation.
    print("Running simulation: ", title)
    sim_obj.simulate(ns=ns, key=seed)
    data = sim_obj.data
    accel_data = sim_obj.get_acceleration()

    # Save the data
    np.save(os.path.join(output_dir, title + "_data.npy"), data)
    np.save(os.path.join(output_dir, title + "_accel_data.npy"), accel_data)
    print("Simulation data saved to: ", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation sets.")
    parser.add_argument("sim", type=str, help="Simulation type")
    parser.add_argument(
        "output_dir",
        type=str,
        default=".",
        help="Output path for saving simulation data",
    )
    parser.add_argument(
        "--n", type=int, help="Number of bodies (default=4)", default=4
    )
    parser.add_argument(
        "--dim",
        type=int,
        help="Number of spatial dimensions - 2 (default) or 3",
        choices=[2, 3],
        default=2,
    )
    parser.add_argument(
        "--nt",
        type=int,
        help="Number of time steps to simulate (default=1000)",
        default=1000,
    )
    parser.add_argument(
        "--ns",
        type=int,
        help="Number of simulations to run (default = 5)",
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for random number generation, default=42",
        default=42,
    )

    args = parser.parse_args()

    # Check if output dir exists else create it.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Run simulations.
    main(
        args.sim,
        args.n,
        args.dim,
        args.nt,
        args.ns,
        args.seed,
        args.output_dir,
    )
