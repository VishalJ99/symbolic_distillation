import argparse
from pysr import PySRRegressor
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import calc_summary_stats
import pickle as pkl
import json
from icecream import ic


def main(input_csv_x, input_csv_y, nbody, output_dir):

    # Load the edge features and node output CSV files.
    df_x = pd.read_csv(input_csv_x)
    df_y = pd.read_csv(input_csv_y)

    # Determine dimension based on columns present.
    dim = 3 if "z1" in df_x.columns else 2
    accel_cols = ["a1", "a2"] + (["a3"] if dim == 3 else [])

    # Fetch the recieving node features.
    target_node_feat_cols = ["x1", "y1"] + (["z1"] if dim == 3 else [])
    target_node_feat_cols += ["vx1", "vy1"] + (["vz1"] if dim == 3 else [])
    target_node_feat_cols += ["q1", "m1"]
    target_node_array = np.array(df_x[target_node_feat_cols])

    # Fetch the edge message array.
    msg_columns = [col for col in df_x.columns if "e" in col]
    msgs_array = np.array(df_x[msg_columns])

    # Get indices of the most significant messages.
    msgs_std = msgs_array.std(axis=0)
    most_important_msgs_idxs = np.argsort(msgs_std)[-dim:]
    most_important_msgs = msgs_array[:, most_important_msgs_idxs]

    # Aggregate the significant edge messages across all sending nodes.
    i = 0
    X = np.zeros((df_y.shape[0], 3*dim+2))
    for idx, row in enumerate(target_node_array[:10]):
        dst_node = row
        # Check if dst_node in the first column of x already
        if ic(np.any(ic(np.all(X[:, :2*dim+2] == dst_node, axis=1)))):
            print(f"Node {idx}: {dst_node} already in X, skipping...")
            continue
        # Find all row idxs with the same target node.
        target_node_idxs = np.where(np.all(target_node_array == dst_node, axis=1))

        # Get all edge messages for the target node.
        target_node_msgs = most_important_msgs[target_node_idxs]
        # Aggregate the edge messages via sum.
        agg_msg = np.sum(target_node_msgs, axis=0)
        x_el = np.concatenate([dst_node, agg_msg])
        X[idx] = x_el

    # Fit a symbolic regression model for each component.
    fig, ax = plt.subplots(ncols=dim)

    nn_a_symbolic_a_diff_dict = {}
    node_model_states = []

    for i in range(dim):
        # Get node features, acceleration and important messages.
        X = most_important_msgs
        # Reshape X to put all messages for each node in a single row.
        X = X.reshape(X.shape[0]//(nbody-1), X.shape[1]*(nbody-1))
        # Concatenate the node features with the important messages.
        feature_cols = ['q1', 'q2', 'm1', 'm2']
        node_feature_array = df_x[feature_cols].to_numpy()

        # Find all sending edges for a given node.
        
        
        # Note duplicate entries for the sending node features 
        # TODO remove duplicate entries.
        node_feature_array = node_feature_array.reshape(node_feature_array.shape[0]//(nbody-1), node_feature_array.shape[1]*(nbody-1))
        # Drop duplicate columns, need first 4 to get source node features, can drop it from the rest.
        first_four_cols = node_feature_array[:, :3]
        every_other_starting_fifth = node_feature_array[:, 3::2]
        # Combine the two selections
        final_node_feature_array = np.hstack((first_four_cols, every_other_starting_fifth))        

        X = np.concatenate([X, final_node_feature_array], axis=1)
        # Labels are the ith component of net acceleration for each node.
        Y = df_y[accel_cols].to_numpy()[:, i]

        # Random Sample 1000 points for faster fitting.
        train_idxs = np.random.choice(X.shape[0], 1000, replace=False)

        # Use remaining points for testing.
        test_idxs = np.setdiff1d(np.arange(X.shape[0]), train_idxs)

        X_train, X_test = X[train_idxs], X[test_idxs]
        Y_train, Y_test = Y[train_idxs], Y[test_idxs]
        
        node_model = PySRRegressor(
            populations=50,
            model_selection="best",
            elementwise_loss="L1DistLoss()",
            niterations=50,
            binary_operators=["+", "-", "*", "/"],
        )

        node_model.fit(X_train, Y_train)
        node_pred = node_model.predict(X_test)
        ic(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
        ic(node_pred.shape)
        # Calculate the diff statistics between the true and symbolic messages.
        accel_diff = node_pred - Y_test

        nn_a_symbolic_a_diff_dict[i + 1] = calc_summary_stats(accel_diff)

        # Create a scatter plot of the true vs predicted accels.
        ax[i].scatter(Y_test, node_pred, alpha=0.1, s=0.1, c="black")
        ax[i].set_xlabel("True Acceleration")
        ax[i].set_ylabel("Predicted Acceleration")

        # Move model state pkl to output directory.
        node_model_state_src = os.path.join(
            os.getcwd(), node_model.equation_file_[:-3] + "pkl"
        )

        with open(node_model_state_src, "rb") as f:
            node_model_state = pkl.load(f)
            node_model_states.append(node_model_state)

        os.remove(os.path.join(os.getcwd(), node_model.equation_file_))

        os.remove(
            os.path.join(os.getcwd(), node_model.equation_file_ + ".bkup")
        )
        os.remove(
            os.path.join(os.getcwd(), node_model.equation_file_[:-3] + "pkl")
        )
    a_plot_file = os.path.join(output_dir, "nn_a_vs_symbolic.png")

    plt.savefig(a_plot_file)
    plt.close()

    symbolic_node_dict = {
        "models": node_model_states,
        "important_msg_idxs": most_important_msgs_idxs.tolist(),
    }

    symbolic_node_dict = {
        "models": node_model_states,
        "important_msg_idxs": most_important_msgs_idxs.tolist(),
    }

    with open(os.path.join(output_dir, "symbolic_node.pkl"), "wb") as f:
        pkl.dump(symbolic_node_dict, f)
        print(
            f"[INFO] Symbolic node model states saved to "
            f"{output_dir}/symbolic_node.pkl"
        )

    a_diff_json_file = os.path.join(output_dir, "nn_a_symbolic_a_diff.json")

    with open(a_diff_json_file, "w") as f:
        f.write(json.dumps(nn_a_symbolic_a_diff_dict))
        print(
            "[INFO] NN accel symbolic accel difference saved to "
            f"{output_dir}/nn_a_symbolic_a_diff.json"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit symbolic model to edge message."
    )

    parser.add_argument(
        "input_csv_x", type=str, help="Path to the edge message CSV file"
    )
    
    parser.add_argument(
        "input_csv_y", type=str, help="Path to the node message CSV file"
    )
    
    parser.add_argument(
        "nbody", type=int, help="Number of bodies in the simulation"
    )

    parser.add_argument(
        "output_dir", type=str, help="Directory to save outputs"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args.input_csv_x, args.input_csv_y, args.nbody, args.output_dir)
