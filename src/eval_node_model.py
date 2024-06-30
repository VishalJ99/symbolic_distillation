import argparse
from pysr import PySRRegressor
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import calc_summary_stats
import pickle as pkl
import json


def main(input_csv_x, input_csv_y, output_dir, samples):
    # Load the edge features and node output CSV files.
    df_x = pd.read_csv(input_csv_x)
    df_y = pd.read_csv(input_csv_y)

    # Determine dimension based on columns present.
    dim = 3 if "z1" in df_x.columns else 2
    accel_cols = ["a1", "a2"] + (["a3"] if dim == 3 else [])

    # Labels for the pysr model are the instantaneous accelerations.
    Y = df_y[accel_cols].to_numpy()

    # Fetch the recieving node features.
    target_node_feat_cols = ["x1", "y1"] + (["z1"] if dim == 3 else [])
    target_node_feat_cols += ["vx1", "vy1"] + (["vz1"] if dim == 3 else [])
    target_node_feat_cols += ["q1", "m1"]

    # Fetch the sending node features.
    src_node_feat_cols = ["x2", "y2"] + (["z2"] if dim == 3 else [])
    src_node_feat_cols += ["vx2", "vy2"] + (["vz2"] if dim == 3 else [])
    src_node_feat_cols += ["q2", "m2"]

    # Fetch the edge message array.
    msg_columns = [col for col in df_x.columns if "e" in col]
    msgs_array = np.array(df_x[msg_columns])

    # Get indices of the most significant messages.
    msgs_std = msgs_array.std(axis=0)
    most_important_msgs_idxs = np.argsort(msgs_std)[-dim:]
    most_important_msg_columns = [
        msg_columns[i] for i in most_important_msgs_idxs
    ]

    # Aggregate the significant edge messages for all target nodes.
    grouped_df = (
        df_x.groupby(target_node_feat_cols)
        .agg({col: "sum" for col in most_important_msg_columns})
        .reset_index()
    )

    # Sorted rows based on order of the source nodes in df_x.
    grouped_df["target_key"] = grouped_df[target_node_feat_cols].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )

    df_x["src_key"] = df_x[src_node_feat_cols].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )

    order = pd.Index(df_x["src_key"].unique())
    grouped_df = grouped_df.set_index("target_key").reindex(order, fill_value=0)

    # Convert df to array w target node feats and summed important msgs.
    X = grouped_df.to_numpy()

    # Random sample points for faster fitting.
    train_idxs = np.random.choice(X.shape[0], samples, replace=False)

    # Use remaining points for testing.
    test_idxs = np.setdiff1d(np.arange(X.shape[0]), train_idxs)

    X_train, X_test = X[train_idxs], X[test_idxs]
    Y_train, Y_test = Y[train_idxs], Y[test_idxs]

    node_model = PySRRegressor(
        populations=100,
        model_selection="best",
        elementwise_loss="L1DistLoss()",
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
    )

    node_model.fit(X_train, Y_train)
    node_pred = node_model.predict(X_test)

    fig, ax = plt.subplots(ncols=dim)
    for i in range(dim):
        # Create a scatter plot of the true vs predicted accels.
        ax[i].scatter(
            Y_test[:, i], node_pred[:, i], alpha=0.1, s=0.1, c="black"
        )
        ax[i].set_xlabel("True Acceleration")
        ax[i].set_ylabel("Predicted Acceleration")

        os.remove(
            os.path.join(os.getcwd(), node_model.equation_file_ + f".out{i+1}")
        )

        os.remove(
            os.path.join(
                os.getcwd(), node_model.equation_file_ + f".out{i+1}" + ".bkup"
            )
        )

    # Save the plot to the output directory.
    a_plot_file = os.path.join(output_dir, "nn_a_vs_symbolic.png")
    plt.savefig(a_plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved NN accel vs symbolic accel plot to {a_plot_file}")

    # Calculate the diff statistics between the true and symbolic messages.
    accel_diff = node_pred - Y_test
    summary_stats = calc_summary_stats(accel_diff)
    a_diff_json_file = os.path.join(output_dir, "nn_a_symbolic_a_diff.json")

    with open(a_diff_json_file, "w") as f:
        f.write(json.dumps(summary_stats))
        print(
            "[INFO] NN accel symbolic accel difference saved to "
            f"{output_dir}/nn_a_symbolic_a_diff.json"
        )

    # Move model state pkl to output directory.
    node_model_state_src = os.path.join(
        os.getcwd(), node_model.equation_file_[:-3] + "pkl"
    )
    with open(node_model_state_src, "rb") as f:
        node_model_state = pkl.load(f)

    symbolic_node_dict = {
        "model": node_model_state,
        "var_names": target_node_feat_cols + most_important_msg_columns,
        "important_msg_idxs": most_important_msgs_idxs.tolist(),
    }

    with open(os.path.join(output_dir, "symbolic_node.pkl"), "wb") as f:
        pkl.dump(symbolic_node_dict, f)
        print(
            f"[INFO] Symbolic node model states saved to "
            f"{output_dir}/symbolic_node.pkl"
        )
    os.remove(os.path.join(os.getcwd(), node_model.equation_file_[:-3] + "pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit symbolic model to edge message."
    )

    parser.add_argument(
        "edge_message_csv", type=str, help="Path to the edge message CSV file"
    )

    parser.add_argument(
        "node_output_csv", type=str, help="Path to the node output CSV file"
    )

    parser.add_argument(
        "output_dir", type=str, help="Directory to save outputs"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to use for fitting, default=5000",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(
        args.edge_message_csv,
        args.node_output_csv,
        args.output_dir,
        args.samples,
    )
