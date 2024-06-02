import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from scipy.optimize import minimize
from scipy.stats import linregress
import json
from plotting_utils import (
    linear_transformation_2d,
    linear_transformation_3d,
    out_linear_transformation_2d,
    out_linear_transformation_3d,
    # force_dict,
)
from pysr import PySRRegressor
import shutil


def main(input_csv, output_dir, eps=1e-2, standarise=True):
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataframe from a CSV file.
    df = pd.read_csv(input_csv)

    # Determine dimension based on columns present.
    dim = 3 if "z1" in df.columns else 2
    pos_cols = ["dx", "dy"] + (["dz"] if dim == 3 else [])

    # Get the message columns.
    msg_columns = [col for col in df.columns if "e" in col]
    msgs_array = np.array(df[msg_columns])

    # Fetch the dim most significant features.
    msgs_std = msgs_array.std(axis=0)
    most_important_msgs_idxs = np.argsort(msgs_std)[-dim:]
    most_important_msgs = msgs_array[:, most_important_msgs_idxs]

    if standarise:
        # Standardise the messages. (Get rid off... doesnt seem needed...)
        most_important_msgs = (
            most_important_msgs - np.average(most_important_msgs, axis=0)
        ) / np.std(most_important_msgs, axis=0)

    # Generate Sparsity plot.
    top_15_msgs_std = msgs_std[np.argsort(msgs_std)[::-1][None, :15]]
    fig, ax = plt.subplots(1, 1)

    ax.pcolormesh(
        top_15_msgs_std,
        cmap="gray_r",
        edgecolors="k",
    )

    # Write std under the plot.
    for i, std in enumerate(top_15_msgs_std.squeeze()):
        x_pos = i if top_15_msgs_std.shape[1] == 15 else i + 0.5
        y_pos = -0.95 if top_15_msgs_std.shape[1] == 15 else -0.45
        ax.text(
            x_pos,
            y_pos,
            f"{std: .2e}",
            ha="center",
            va="center",
            rotation=45,
        )

    plt.axis("off")
    plt.grid(True)
    ax.set_aspect("equal")
    plt.text(15.5, 0.5, "...", fontsize=30)

    # Save the sparsity plot.
    sparsity_plot_file = os.path.join(output_dir, "sparsity_plot.png")
    plt.savefig(sparsity_plot_file)
    print(f"[INFO] Sparsity plot saved to {sparsity_plot_file}")

    # Function to calculate forces. (Generalise)
    # A small epsilon to avoid division by zero
    force_fnc = (
        lambda msg: -(msg.r + eps - 1).to_numpy()[:, None]
        * np.array(msg[pos_cols])
        / (msg.r + eps).to_numpy()[:, None]
    )

    # Calculate forces.
    expected_forces = force_fnc(df)

    # Find the best linear transformation.
    x0 = np.ones(dim**2 + dim)  # Initial guess for minimisation
    if dim == 2:
        min_result = minimize(
            linear_transformation_2d,
            x0=x0,
            args=(expected_forces, most_important_msgs),
            method="Powell",
        )
    elif dim == 3:
        min_result = minimize(
            linear_transformation_3d,
            x0=x0,
            args=(expected_forces, most_important_msgs),
            method="Powell",
        )

    # Extract coefficients and transform forces.
    alpha = min_result.x

    if dim == 2:
        transformed_forces = out_linear_transformation_2d(
            alpha, expected_forces
        )
    elif dim == 3:
        transformed_forces = out_linear_transformation_3d(
            alpha, expected_forces
        )

    # Generate message vs transformed force plot.
    fig, axes = plt.subplots(1, dim, figsize=(4 * dim, 4))
    R2_stats = []
    for i in range(dim):
        ax = axes[i] if dim > 1 else axes
        ax.scatter(
            transformed_forces[:, i],
            most_important_msgs[:, i],
            alpha=0.1,
            s=0.1,
            c="black",
        )
        ax.set_xlabel("Transformed Forces")
        ax.set_ylabel(f"Message Component {i+1}")

        # Fit a linear regression model.
        slope, intercept, r_value, p_value, stderr = linregress(
            transformed_forces[:, i], most_important_msgs[:, i]
        )

        R2 = r_value**2
        R2_stats.append(R2)
        ax.title.set_text(f"Component {i+1} R^2: {R2: .2f}")

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "messages_vs_transformed_force.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"[INFO] Message vs Force plot saved to {plot_file}")

    # Save R2 statistics.
    R2_file = os.path.join(output_dir, "R2_stats.txt")
    with open(R2_file, "w") as f:
        f.write(json.dumps(R2_stats))
        print(f"[INFO] R2 statistics saved to {R2_file}")

    # Fit a symbolic regression model for each component
    X_cols = pos_cols + ["r", "m1", "m2"]
    for i in range(dim):
        X = df[X_cols].to_numpy()
        Y = most_important_msgs[:, i]

        # Random Sample 1000 points for faster fitting.
        idxs = np.random.choice(X.shape[0], 1000, replace=False)
        X_sample = X[idxs]
        Y_sample = Y[idxs]
        model = PySRRegressor(
            populations=50,
            model_selection="best",
            niterations=50,
            binary_operators=["+", "-", "*", "/"],
        )

        model.fit(X_sample, Y_sample)

        # Move model state pkl to output directory.
        model_state_src = os.path.join(
            os.getcwd(), model.equation_file_[:-3] + "pkl"
        )
        model_state_dst = os.path.join(output_dir, f"model_{i+1}.pkl")
        print(f"[INFO] Saved model state to {model_state_dst}")
        shutil.move(model_state_src, model_state_dst)

        # Clean up the saved csvs.
        os.remove(os.path.join(os.getcwd(), model.equation_file_))
        os.remove(os.path.join(os.getcwd(), model.equation_file_ + ".bkup"))
        break


if __name__ == "__main__":
    # force_key_list = list(force_dict.keys())
    parser = argparse.ArgumentParser(
        description="Process edge messages and plot transformations."
    )
    parser.add_argument(
        "input_csv", type=str, help="Path to the input node message CSV file"
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory to save outputs"
    )

    args = parser.parse_args()

    main(args.input_csv, args.output_dir)
