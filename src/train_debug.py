import argparse
import os
import wandb
import yaml
from accelerate import Accelerator
import torch
from torch_geometric.data import DataLoader
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
    get_node_message_info_df,
)


def main(config):
    # Set the random seed for reproducibility.
    seed_everything(config["seed"])

    # # Create the output directory for the run if it does not exist.
    # output_dir = config["output_dir"]
    # make_dir(output_dir)

    # # Create output directory to save model weights during training.
    # weights_dir_path = os.path.join(output_dir, "model_weights")
    # make_dir(weights_dir_path)

    # if config["save_messages"]:
    #     # Create output directory to save messages during training.
    #     messages_dir_path = os.path.join(output_dir, "training_messages")
    #     make_dir(messages_dir_path)

    # if os.path.exists(".git"):
    #     # Add the git hash to the config if the .git file exists.
    #     config["git_hash"] = os.popen("git rev-parse HEAD").read().strip()

    # if config["wandb"]:
    #     # Use wandb to log the run.
    #     run = wandb.init(project=config["wandb_project"], config=config)
    #     run_url = run.get_url()
    #     config["run_url"] = run_url

    # Save the config to the output directory for reproducibility.
    # config_file = os.path.join(output_dir, "train_config.yaml")
    # with open(config_file, "w") as f:
    #     yaml.dump(config, f)

    print("-" * 50)
    print("[INFO] Config options set.")
    for key, val in config.items():
        print(f"[INFO] {key}: {val}")
    print("-" * 50)

    # Set the device to use for training. GPU -> MPS -> CPU.
    accelerator = Accelerator()
    device = accelerator.device
    print(f"[INFO] Device set to: {device}")

    # # Initialise transforms if specified in the configuration.
    # if "augmentations" in config:
    #     augmentations = Compose(
    #         transforms_factory(k, v) for k, v in config["augmentations"].items()
    #     )
    # else:
    #     augmentations = None

    # if "pre_transforms" in config:
    #     pre_transforms = Compose(
    #         transforms_factory(k, v)
    #         for k, v in config["pre_transforms"].items()
    #     )
    # else:
    #     pre_transforms = None

    # # Load the training and validation datasets.
    # train_dir = os.path.join(config["data_dir"], "train")
    # val_dir = os.path.join(config["data_dir"], "val")

    # train_dataset = ParticleDynamicsDataset(
    #     root=train_dir, transform=augmentations, pre_transform=pre_transforms
    # )

    # val_dataset = ParticleDynamicsDataset(
    #     root=val_dir, pre_transform=pre_transforms
    # )

    # if config["quick_test"]:
    #     # Create smaller subsets of the datasets for quick testing.
    #     train_indices = list(range(config["train_batch_size"]))
    #     val_indices = list(range(config["val_batch_size"]))

    #     train_dataset = Subset(train_dataset, train_indices)
    #     val_dataset = Subset(val_dataset, val_indices)

    # # Initialise dataloaders.
    # train_loader = DataLoader(
    #     train_dataset, batch_size=config["train_batch_size"], shuffle=True
    # )

    # val_loader = DataLoader(
    #     val_dataset, batch_size=config["val_batch_size"], shuffle=False
    # )

    # # Load model.
    # model = model_factory(config["model"], config["model_params"])

    # # Load state dict if specified in the config.
    # if config["model_state_dict"]:
    #     model.load_state_dict(torch.load(config["model_state_dict"]))
    #     print("[INFO] Loaded model state dict...")
    # ---------------------------------------------------------
    import numpy as np
    from models import OGN, varOGN, get_edge_index
    from sklearn.model_selection import train_test_split
    from torch_geometric.data import Data, DataLoader
    from icecream import ic

    # Load the data - change this to data of the right shape.
    data = np.load(
        "simulations/data.npy"
    )
    accel_data = np.load(
        "simulations/accel_data.npy"
    )

    # Hyper params
    # ---------------
    sim = "spring"
    aggr = "add"
    hidden = 300
    test = "_kl_"
    msg_dim = 100
    n_f = data.shape[3]
    n = data.shape[2]
    init_lr = 1e-3
    total_epochs = 100
    dim = 2

    # Split the data into train and val.
    X = torch.from_numpy(
        np.concatenate([data[:, i] for i in range(0, data.shape[1], 5)])
    )
    y = torch.from_numpy(
        np.concatenate([accel_data[:, i] for i in range(0, data.shape[1], 5)])
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, random_state=42)

    X_train = X_train[:64]
    y_train = y_train[:64]
    X_test = X_test[:64]
    y_test = y_test[:64]

    edge_index = get_edge_index(n, "r2")  # doesnt matter unless string or ball.

    batch = int(64 * (4 / n) ** 2)

    train_loader = DataLoader(
        [
            Data(X_train[i], edge_index=edge_index, y=y_train[i])
            for i in range(len(y_train))
        ],
        batch_size=batch,
        shuffle=False,
    )

    val_loader = DataLoader(
        [
            Data(X_test[i], edge_index=edge_index, y=y_test[i])
            for i in range(len(y_test))
        ],
        batch_size=1024,
        shuffle=True,
    )

    model = varOGN(
        n_f,
        msg_dim,
        dim,
        dt=0.1,
        hidden=hidden,
        edge_index=get_edge_index(n, sim),
        aggr=aggr,
    ).to(device)

    # Load initial starter weights
    # ---------------------------------------------------------
    
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
        for graph in train_loader_iter:
            optim.zero_grad()
            pred = model(graph.x, graph.edge_index)
            loss = loss_fn(graph, pred, model)

            accelerator.backward(loss)
            optim.step()
            sched.step()

            total_train_loss += loss.item()
            num_train_items += 1

            avg_train_loss = total_train_loss / num_train_items

            if config["tqdm"]:
                train_loader_iter.set_postfix(avg_train_loss=avg_train_loss)
            exit(1)
        print(f"Average training loss for Epoch: {avg_train_loss}")

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

                val_loss = loss_fn(graph, pred, model).item()

                total_val_loss += val_loss
                num_val_items += 1
                avg_val_loss = total_val_loss / num_val_items
                if config["tqdm"]:
                    val_loader_iter.set_postfix(avg_val_loss=avg_val_loss)

            print(f"Average validation loss for Epoch: {avg_val_loss}")

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
                while msgs_recorded < 10000:
                    df = get_node_message_info_df(
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
    parser = argparse.ArgumentParser(description="Train a DDPM model.")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the yaml train config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
