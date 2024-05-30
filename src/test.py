import argparse
import os
import json
import yaml
from accelerate import Accelerator
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import pandas as pd
from tqdm import tqdm

from datasets import ParticleDynamicsDataset
from utils import (
    make_dir,
    model_factory,
    loss_factory,
    get_node_message_info_df,
)


def main(config):
    output_dir = config["output_dir"]

    if config["save_messages"]:
        messages_dir_path = os.path.join(output_dir, "test_messages")
        make_dir(messages_dir_path)

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
    test_dataset = ParticleDynamicsDataset(root=test_dir, pre_transform=None)

    # Initialise DataLoader for the test dataset.
    test_loader = DataLoader(
        test_dataset, batch_size=config["test_batch_size"], shuffle=False
    )

    # Load model.
    model = model_factory(config["model"], config["model_params"]).to(device)

    # Load the saved model weights.
    model_path = os.path.join(config["model_weights_dir"], "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("[INFO] Model loaded successfully.")

    # Move model and data loader to device.
    model, test_loader = accelerator.prepare(model, test_loader)

    # Set the model to evaluation mode.
    model.eval()

    # Initialise the loss function.
    loss_fn = loss_factory(config["loss"], config["loss_params"])

    if config["save_messages"]:
        df_list = []
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
            loss = loss_fn(graph, pred, model)
            loss_array[idx] = loss

            if config["tqdm"]:
                test_loader_iter.set_postfix({"Loss": loss.item()})

            if (
                config["save_messages"]
                and msgs_recorded < config["message_save_limit"]
            ):
                # Record edge messages.
                df = get_node_message_info_df(
                    graph, model, dim=(graph.x.shape[1] - 2) // 2
                )
                msgs_recorded += len(df)
                df_list.append(df)

    # Save test results to a CSV.
    loss_np = loss_array.numpy()  # Convert to numpy for easier manipulation
    stats = {
        "Mean": float(np.mean(loss_np)),
        "Standard Deviation": float(np.std(loss_np)),
        "Median": float(np.median(loss_np)),
        "Lower Quartile": float(np.percentile(loss_np, 25)),
        "Upper Quartile": float(np.percentile(loss_np, 75)),
        "N": int(len(test_loader)),
        "Batch Size": int(config["test_batch_size"]),
    }

    # Print summary statistics.
    print("-" * 50)
    print("[INFO] Test Results:")
    for key, val in stats.items():
        print(f"[INFO] {key}: {val}")
    print("-" * 50)

    # Save test results to a CSV.
    test_results = os.path.join(output_dir, "test_results.json")
    with open(test_results, "w") as f:
        json.dump(stats, f)

    # Save test results and messages if required
    if config["save_messages"]:
        df = pd.concat(df_list)
        node_message_save_path = os.path.join(
            messages_dir_path, "node_messages.csv"
        )
        df.to_csv(node_message_save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a DDPM model.")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the yaml test config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
