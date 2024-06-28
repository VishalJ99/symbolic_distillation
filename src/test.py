# Need to import juliacall before torch to avoid segfault issue.
# This is relevant when testing using symbolic pysr models.
import juliacall
import os
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
import argparse
import json
import yaml
from accelerate import Accelerator
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
import pandas as pd
from datasets import ParticleDynamicsDataset
from utils import (
    make_dir,
    model_factory,
    loss_factory,
    get_node_message_info_dfs,
    calc_summary_stats,
)


def main(config):
    output_dir = config["output_dir"]
    make_dir(output_dir)

    if config["save_messages"]:
        sr_csv_dirpath = os.path.join(output_dir, "symbolic_regression_csvs")
        make_dir(sr_csv_dirpath)

    # Save the config to the output directory for reproducibility.
    config_file = os.path.join(output_dir, "test_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    if os.path.exists(".git"):
        # Add the git hash to the config for reproducibility.
        config["git_hash"] = os.popen("git rev-parse HEAD").read().strip()

    print("-" * 50)
    print("[INFO] Config options set.")
    for key, val in config.items():
        print(f"[INFO] {key}: {val}")
    print("-" * 50)

    # Initialise Accelerator.
    accelerator = Accelerator()
    device = accelerator.device
    print(f"[INFO] Device set to: {device}")

    # Load the test dataset.
    test_dir = os.path.join(config["data_dir"], "test")
    test_dataset = ParticleDynamicsDataset(
        root=test_dir,
        pre_transform=None,
        prune_outliers=config["prune_outliers"],
    )
    if config["quick_test"]:
        # Create smaller subsets of the datasets for quick testing.
        test_indices = list(range(config["test_batch_size"]))

        test_dataset = Subset(test_dataset, test_indices)

    # Initialise DataLoader for the test dataset.
    test_loader = DataLoader(
        test_dataset, batch_size=config["test_batch_size"], shuffle=False
    )

    # Load model.
    model = model_factory(config["model"], config["model_params"]).to(device)

    # Load the saved model weights.
    model_path = os.path.join(config["model_weights_path"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("[INFO] Model loaded successfully.")

    # Move model and data loader to device.
    model, test_loader = accelerator.prepare(model, test_loader)

    # Set the model to evaluation mode.
    model.eval()

    # Initialise the loss function.
    loss_fn = loss_factory(config["loss"], config["loss_params"])

    if config["save_messages"]:
        df_x_list = []
        df_y_list = []
        msgs_recorded = 0

    # Test loop.
    # Create np array to store loss values
    loss_array = torch.zeros(len(test_loader))
    with torch.no_grad():
        test_loader_iter = (
            tqdm(test_loader, desc="Testing") if config["tqdm"] else test_loader
        )
        for idx, graph in enumerate(test_loader_iter):
            pred = model(graph)
            loss, _ = loss_fn(graph, pred, model)
            loss_array[idx] = loss

            if config["tqdm"]:
                test_loader_iter.set_postfix({"Loss": loss.item()})

            if (
                config["save_messages"]
                and msgs_recorded < config["message_save_limit"]
            ):
                # Record edge messages.
                df_x, df_y = get_node_message_info_dfs(
                    graph, model, dim=(graph.x.shape[1] - 2) // 2
                )
                msgs_recorded += len(df_x)
                df_x_list.append(df_x)
                df_y_list.append(df_y)

    # Save test results to a CSV.
    loss_np = loss_array.numpy()  # Convert to numpy for easier manipulation
    stats = calc_summary_stats(loss_np)
    stats.update(
        {
            "N": int(len(test_loader)),
            "Batch Size": int(config["test_batch_size"]),
        }
    )

    # Print summary statistics.
    print("-" * 50)
    print("[INFO] Test Results:")
    for key, val in stats.items():
        print(f"[INFO] {key}: {val}")
    print("-" * 50)

    # Save test results to a CSV.
    test_results = os.path.join(output_dir, "test_results.json")
    print(f"[INFO] Saving test results to {test_results}")
    with open(test_results, "w") as f:
        json.dump(stats, f)

    # Save test results and messages if required
    if config["save_messages"]:
        print("[INFO] Saving node messages...")
        df_x = pd.concat(df_x_list)
        node_message_save_path = os.path.join(
            sr_csv_dirpath, "edge_messages.csv"
        )
        df_x.to_csv(node_message_save_path, index=False)
        df_y = pd.concat(df_y_list)
        node_accel_save_path = os.path.join(sr_csv_dirpath, "node_accels.csv")
        df_y.to_csv(node_accel_save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a torch geometric message passing model."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the yaml test config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
    print("[SUCCESS] Test complete.")
