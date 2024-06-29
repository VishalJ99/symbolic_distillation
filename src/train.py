import argparse
import os
import wandb
import yaml
from accelerate import Accelerator
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch.utils.data import Subset
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import pandas as pd
from datasets import ParticleDynamicsDataset
from utils import (
    make_dir,
    seed_everything,
    transforms_factory,
    model_factory,
    loss_factory,
    get_node_message_info_dfs,
    debug_logs,
)

import math


def main(config):
    # Set the random seed for reproducibility.
    seed_everything(config["seed"])

    # Create the output directory for the run if it does not exist.
    output_dir = config["output_dir"]
    make_dir(output_dir)

    # Create output directory to save model weights during training.
    weights_dir_path = os.path.join(output_dir, "model_weights")
    make_dir(weights_dir_path)

    if config["save_messages"]:
        # Create output directory to save messages during training.
        messages_dir_path = os.path.join(output_dir, "train_messages")
        make_dir(messages_dir_path)

    if os.path.exists(".git"):
        # Add the git hash to the config if the .git file exists.
        config["git_hash"] = os.popen("git rev-parse HEAD").read().strip()

    if config["wandb"]:
        # Use wandb to log the run.
        run = wandb.init(project=config["wandb_project"], config=config)
        run_url = run.get_url()
        config["run_url"] = run_url

    # Save the config to the output directory for reproducibility.
    config_file = os.path.join(output_dir, "train_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    print("-" * 50)
    print("[INFO] Config options set.")
    for key, val in config.items():
        print(f"[INFO] {key}: {val}")
    print("-" * 50)

    # Set the device to use for training. GPU -> MPS -> CPU.
    accelerator = Accelerator()
    device = accelerator.device
    print(f"[INFO] Device set to: {device}")

    # Initialise transforms if specified in the configuration.
    if "augmentations" in config:
        augmentations = Compose(
            transforms_factory(k, v) for k, v in config["augmentations"].items()
        )
    else:
        augmentations = None

    # Initialise pre-transforms if specified in the configuration.
    if "pre_transforms" in config:
        pre_transforms = Compose(
            transforms_factory(k, v)
            for k, v in config["pre_transforms"].items()
        )
    else:
        pre_transforms = None

    # Load the training and validation datasets.
    train_dir = os.path.join(config["data_dir"], "train")
    val_dir = os.path.join(config["data_dir"], "val")

    train_dataset = ParticleDynamicsDataset(
        root=train_dir,
        transform=augmentations,
        pre_transform=pre_transforms,
        prune_outliers=config["prune_graphs"],
    )

    val_dataset = ParticleDynamicsDataset(
        root=val_dir,
        pre_transform=pre_transforms,
        prune_outliers=config["prune_graphs"],
    )

    if config["quick_test"]:
        # Create smaller subsets of the datasets for quick testing.
        train_indices = list(range(config["train_batch_size"]))
        val_indices = list(range(config["val_batch_size"]))

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    # Initialise dataloaders.
    train_loader = DataLoader(
        train_dataset, batch_size=config["train_batch_size"], shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config["val_batch_size"], shuffle=False
    )

    # Load model.
    model = model_factory(config["model"], config["model_params"])

    # Load state dict if specified in the config.
    if config["model_state_path"]:
        model.load_state_dict(torch.load(config["model_state_path"]))
        print("[INFO] Loaded model state dict...")
    else:
        print(f"[INFO] Saving seed model weights to {weights_dir_path}...")
        torch.save(
            model.state_dict(),
            os.path.join(weights_dir_path, "seed_model.pt"),
        )

    # Initialise the optimiser.
    total_epochs = config["epochs"]
    lr = config["lr"]
    optim = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=config["weight_decay"]
    )

    # Initialise the learning rate scheduler.
    max_lr = config["scheduler_params"]["max_lr"]
    final_div_factor = config["scheduler_params"]["final_div_factor"]
    sched = OneCycleLR(
        optim,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=total_epochs,
        final_div_factor=final_div_factor,
    )

    # Initialise the loss function.
    loss_fn = loss_factory(config["loss"], config["loss_params"])

    # TODO: Check if optim and sched need to be moved?
    model, optim, sched, train_loader, val_loader = accelerator.prepare(
        model, optim, sched, train_loader, val_loader
    )

    # Set max validation loss to infinity.
    max_val_loss = float("inf")

    # Training loop.
    for epoch in range(1, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")

        # Training phase
        total_train_loss = 0
        num_train_items = 0
        model.train()
        train_loader_iter = (
            tqdm(train_loader, desc=f"Training Epoch {epoch}")
            if config["tqdm"]
            else train_loader
        )
        for idx, graph in enumerate(train_loader_iter):
            optim.zero_grad()
            pred = model(graph)
            # TODO: Handle saving of params dict...
            loss, _ = loss_fn(graph, pred, model)

            accelerator.backward(loss)
            optim.step()
            sched.step()

            total_train_loss += loss.item()
            num_train_items += 1
            avg_train_loss = total_train_loss / num_train_items

            if config["debug"]:
                debug_logs(graph, pred, loss, _)

                user_input = input("[DEBUG] Continue training? (y/n): ")
                if user_input == "n":
                    exit(1)

            if math.isnan(avg_train_loss):
                print("[INFO] Training loss is NaN. Exiting...")
                exit(1)

            if config["tqdm"]:
                train_loader_iter.set_postfix(avg_train_loss=avg_train_loss)

        print(f"[INFO] Average training loss for Epoch: {avg_train_loss}")

        if config["wandb"]:
            wandb.log({"avg_train_loss": avg_train_loss})

        # Validation phase
        total_val_loss = 0
        num_val_items = 0
        model.eval()
        with torch.no_grad():
            val_loader_iter = (
                tqdm(val_loader, desc=f"Validation Epoch {epoch}")
                if config["tqdm"]
                else val_loader
            )
            for graph in val_loader_iter:
                pred = model(graph)
                val_loss, _ = loss_fn(graph, pred, model)

                total_val_loss += val_loss.item()
                num_val_items += 1
                avg_val_loss = total_val_loss / num_val_items

                if config["tqdm"]:
                    val_loader_iter.set_postfix(avg_val_loss=avg_val_loss)

            print(f"[INFO] Average validation loss for Epoch: {avg_val_loss}")

            if math.isnan(avg_val_loss):
                print("[INFO] Validation loss is NaN. Exiting...")
                exit(1)

            if config["wandb"]:
                wandb.log({"avg_val_loss": avg_val_loss})

        if avg_val_loss < max_val_loss:
            torch.save(
                model.state_dict(),
                os.path.join(weights_dir_path, "best_model.pt"),
            )
            print(
                f"[INFO] Average validation loss improved from {max_val_loss}"
                f" to {avg_val_loss}, saving model weights..."
            )

            # Update the max validation loss.
            max_val_loss = avg_val_loss

        # Save the model weights every n epochs.
        if (epoch + 1) % config["save_every_n_epochs"] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(weights_dir_path, f"model_epoch_{epoch+1}.pt"),
            )
            print(f"Model saved at epoch {epoch+1}..")

            if config["save_messages"]:
                # Save node features and msgs for each edge in the val set as a df.
                pbar = (
                    tqdm(val_loader, desc="Saving node messages")
                    if config["tqdm"]
                    else val_loader
                )
                msgs_recorded = 0
                df_list = []
                for graph in pbar:
                    # Only record 10k messages per epoch to avoid large file sizes.
                    while msgs_recorded < config["message_save_limit"]:
                        df, _ = get_node_message_info_dfs(
                            graph, model, dim=(graph.x.shape[1] - 2) // 2
                        )
                        msgs_recorded += len(df)
                        df_list.append(df)

                df = pd.concat(df_list)
                node_message_save_path = os.path.join(
                    messages_dir_path, f"node_messages_epoch_{epoch}.csv"
                )

                df.to_csv(node_message_save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Train Script.")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the yaml train config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
    print("[SUCCESS] Training complete.")
