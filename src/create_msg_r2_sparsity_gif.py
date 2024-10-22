import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.optimize import minimize
import os
import imageio
import pandas as pd
import argparse
from utils import force_factory

from plotting_utils import (
    linear_transformation_2d,
    linear_transformation_3d,
    out_linear_transformation_2d,
    out_linear_transformation_3d,
    make_sparsity_plot,
    make_force_edge_msg_scatter,
)


def main(
    messages_over_time, output_dir, sim, plot_sparsity, delete_frames, eps=1e-2
):
    # Initalise list to store filenames for frames to be used in GIF.
    frame_filenames = []

    # Loop over each node message dataframe in messages_over_time.
    for idx, df in enumerate(messages_over_time):
        print(f"\r[INFO] Reading frame {idx}/{len(messages_over_time)}", end="")
        # Set dim by checking if z is present in columns.
        dim = 3 if "z1" in df.columns else 2

        pos_cols = ["dx", "dy"]

        if dim == 3:
            pos_cols.append("dz")

        # Fetch the message columns.
        msg_columns = [col for col in df.columns if "e" in col]

        # Fetch the message array.
        msgs_array = np.array(df[msg_columns])

        if plot_sparsity:
            # Visualise the sparsity via the std of the edge message components.
            fig, axes = make_sparsity_plot(
                msgs_array=msgs_array, dim=dim, top_n=15
            )

        else:
            force_fnc = force_factory(sim)
            # Calculate the expected forces, i.e. the 'labels'.
            expected_forces = force_fnc(df, eps)
            msgs_std = msgs_array.std(axis=0)
            most_important_msgs_idxs = np.argsort(msgs_std)[-dim:]
            most_important_msgs = msgs_array[:, most_important_msgs_idxs]

            # TODO: Generalise these transformations functions.
            # Find the best linear transformation force -> message.
            if dim == 2:
                min_result = minimize(
                    linear_transformation_2d,
                    x0=np.ones(dim**2 + dim),
                    args=(expected_forces, most_important_msgs),
                    method="Powell",
                )
            if dim == 3:
                min_result = minimize(
                    linear_transformation_3d,
                    x0=np.ones(dim**2 + dim),
                    args=(expected_forces, most_important_msgs),
                    method="Powell",
                )

            # Extract the linear transformation coefficients.
            alpha = min_result.x

            # Plot the edge messages vs the transformed forces.
            fig, ax = plt.subplots(1, dim, figsize=(4 * dim, 4))
            if dim == 3:
                transformed_forces = out_linear_transformation_3d(
                    alpha, expected_forces
                )
            else:
                transformed_forces = out_linear_transformation_2d(
                    alpha, expected_forces
                )

            # Plot transformed force components against the edge components.
            fig, ax, R2_stats = make_force_edge_msg_scatter(
                transformed_forces, most_important_msgs, dim
            )

        fig.suptitle("Frame %d" % idx)

        # Determine the file name prefix based on plot_sparsity flag
        filename_prefix = "sparsity" if plot_sparsity else "force"

        # Save the frame to a file.
        filename = os.path.join(
            output_dir, f"{filename_prefix}_frame_{idx}.png"
        )
        filename = os.path.join(output_dir, f"frame_{idx}.png")
        fig.savefig(filename)  # Save the plot to a file
        plt.close("all")
        frame_filenames.append(filename)

    # Create the GIF.
    output_gif_path = os.path.join(
        output_dir, "sparsity.gif" if plot_sparsity else "force.gif"
    )
    with imageio.get_writer(output_gif_path, mode="I", duration=0.5) as writer:
        for filename in frame_filenames:
            frame = imageio.imread(filename)
            writer.append_data(frame)

    if delete_frames:
        print("[INFO] Deleting frames...")
        for filename in set(frame_filenames):
            os.remove(filename)  # Remove the files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "edge_message_dir", type=str, help="Directory of edge message csvs"
    )
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("sim", type=str, help="Simulation type")
    parser.add_argument(
        "--plot_sparsity", action="store_true", help="Plot sparsity"
    )
    parser.add_argument(
        "--delete_frames", action="store_true", help="Delete frames"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-2,
        help="Epsilon for numerical stability",
    )
    args = parser.parse_args()

    input_path = args.edge_message_dir
    output_dir = args.output_dir
    sim = args.sim
    plot_sparsity = args.plot_sparsity
    delete_frames = args.delete_frames
    eps = args.eps

    try:
        os.makedirs(output_dir)
    except Exception:
        pass

    messages_over_time = []
    csv_files = sorted(
        os.listdir(input_path),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    for csv in csv_files:
        if csv.endswith(".csv"):
            messages_over_time.append(
                pd.read_csv(os.path.join(input_path, csv))
            )

    print(f"\n[INFO] Identified {len(messages_over_time)} frames.")

    main(messages_over_time, output_dir, sim, plot_sparsity, delete_frames, eps)
    print(f"[SUCCESS] Gif created at {output_dir}")
