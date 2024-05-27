import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.optimize import minimize
import os
import imageio
import pandas as pd
import argparse
from plotting_utils import (
    linear_transformation_2d,
    linear_transformation_3d,
    out_linear_transformation_2d,
    out_linear_transformation_3d,
)


def make_sparsity_plot(msg_importance):
    # Visualise the sparsity via the std of the edge message components.
    fig, ax = plt.subplots(1, 1)
    ax.pcolormesh(
        msg_importance[np.argsort(msg_importance)[::-1][None, :15]],
        cmap="gray_r",
        edgecolors="k",
    )

    # Write std under the plot.
    for i, std in enumerate(
        msg_importance[np.argsort(msg_importance)[::-1][:15]]
    ):
        ax.text(i + 0.5, -0.5, "%.2f" % std, ha="center", va="center")

    plt.axis("off")
    plt.grid(True)
    ax.set_aspect("equal")
    plt.text(15.5, 0.5, "...", fontsize=30)
    plt.tight_layout()
    return fig, ax


def main(messages_over_time, output_dir, plot_sparsity, delete_frames):
    # Initalise list to store filenames for frames to be used in GIF.
    frame_filenames = []

    # Loop over each node message dataframe in messages_over_time.
    for idx, df in enumerate(messages_over_time):
        # Set dim by checking if z is present in columns.
        dim = 3 if "z1" in df.columns else 2

        # TODO: is this even needed as simulation.py already adds eps...
        eps = 1e-2

        pos_cols = ["dx", "dy"]

        if dim == 3:
            pos_cols.append("dz")

        # Fetch the message columns.
        msg_columns = [col for col in df.columns if "e" in col]

        # Fetch the message array.
        msg_array = np.array(df[msg_columns])

        # Select only top dim features by standard deviation.
        msg_importance = msg_array.std(axis=0)

        if plot_sparsity:
            # Visualise the sparsity via the std of the edge message components.
            fig, ax = make_sparsity_plot(msg_importance)

        else:
            # TODO: Generalise force fnc.
            force_fnc = (
                lambda msg: -(msg.r + eps - 1).to_numpy()[:, None]
                * np.array(msg[pos_cols])
                / (msg.r + eps).to_numpy()[:, None]
            )

            # Plot force components.
            # Select the dim most important messages.
            most_important_idxs = np.argsort(msg_importance)[-dim:]
            msgs_to_compare = msg_array[:, most_important_idxs]

            # Standardise the messages.
            msgs_to_compare = (
                msgs_to_compare - np.average(msgs_to_compare, axis=0)
            ) / np.std(msgs_to_compare, axis=0)

            # Calculate the expected forces, i.e. the 'labels'.
            expected_forces = force_fnc(df)

            # Find the best linear transformation force -> message.
            if dim == 2:
                min_result = minimize(
                    linear_transformation_2d,
                    x0=np.ones(dim**2 + dim),
                    args=(expected_forces, msgs_to_compare),
                    method="Powell",
                )
            if dim == 3:
                min_result = minimize(
                    linear_transformation_3d,
                    x0=np.ones(dim**2 + dim),
                    args=(expected_forces, msgs_to_compare),
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

            for i in range(dim):
                ax[i].scatter(
                    transformed_forces[:, i],
                    msgs_to_compare[:, i],
                    alpha=0.1,
                    s=0.1,
                    color="k",
                )
                ax[i].set_xlabel("Linear Transformation of True Forces")
                ax[i].set_ylabel("Message Element %d" % (i + 1))

                xlim = np.array(
                    [
                        np.percentile(transformed_forces[:, i], q)
                        for q in [10, 90]
                    ]
                )
                ylim = np.array(
                    [np.percentile(msgs_to_compare[:, i], q) for q in [10, 90]]
                )

                xlim[0], xlim[1] = (
                    xlim[0] - (xlim[1] - xlim[0]) * 0.05,
                    xlim[1] + (xlim[1] - xlim[0]) * 0.05,
                )
                ylim[0], ylim[1] = (
                    ylim[0] - (ylim[1] - ylim[0]) * 0.05,
                    ylim[1] + (ylim[1] - ylim[0]) * 0.05,
                )

                ax[i].set_xlim(xlim)
                ax[i].set_ylim(ylim)

                plt.tight_layout()

        # Save the plot to a file.
        filename = os.path.join(output_dir, f"frame_{idx}.png")
        fig.savefig(filename)  # Save the plot to a file
        plt.close()
        frame_filenames.append(filename)

    # Create the GIF.
    with imageio.get_writer("force.gif", mode="I", duration=0.5) as writer:
        for filename in frame_filenames:
            frame = imageio.imread(filename)
            writer.append_data(frame)
        print("[INFO] GIF created.")

    if delete_frames:
        print("[INFO] Deleting frames...")
        for filename in set(frame_filenames):
            os.remove(filename)  # Remove the files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", type=str, help="Input pkl file or directory of csvs"
    )
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--plot_sparsity", action="store_true", help="Plot sparsity"
    )
    parser.add_argument(
        "--delete_frames", action="store_true", help="Delete frames"
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_dir = args.output_dir
    plot_sparsity = args.plot_sparsity
    delete_frames = args.delete_frames

    if input_path.endswith(".pkl"):
        with open(input_path, "rb") as f:
            messages_over_time = pkl.load(f)
    else:
        messages_over_time = []
        csv_files = sorted(
            os.listdir(input_path),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        for csv in csv_files:
            print("[INFO] Reading", csv)
            if csv.endswith(".csv"):
                messages_over_time.append(
                    pd.read_csv(os.path.join(input_path, csv))
                )

    print(f"[INFO] Identified {len(messages_over_time)} frames.")

    main(messages_over_time, output_dir, plot_sparsity, delete_frames)
