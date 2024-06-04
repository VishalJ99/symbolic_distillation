import yaml
import os
import argparse

# Define the experiments, strategies, and dimensions
experiments = ["spring", "r1", "r2", "charge"]
strategies = ["standard", "l1", "kl", "bottleneck"]
dimensions = ["2d", "3d"]

# Hyperparameters and settings
seed = 42
wandb = True
wandb_project = "mphil_project"
save_messages = True
quick_test = False
tqdm = False
debug = False
epochs = 100
model_state_path = None
prune_graphs = 99.99
train_batch_size = 64
val_batch_size = 1024
lr = 0.001
weight_decay = 1e-8
save_every_n_epochs = 5
max_lr = 0.001
final_div_factor = 1e5
translate_scale = 3


l1_weight = 1.0e-2
kl_weight = 1


# Function to create a config dictionary
def create_config(experiment, strategy, dim, paths):
    num_features = 6 if dim == "2d" else 8
    msg_dim = 100 if strategy != "bottleneck" else int(dim[0])
    hidden_size = 300
    aggr_method = "add"
    model_type = "vargnn" if strategy == "kl" else "gnn"
    loss_type = "maeloss+l1reg" if strategy != "kl" else "maeloss+klreg"
    reg_weight = (
        kl_weight
        if strategy == "kl"
        else (l1_weight if strategy == "l1" else 0)
    )

    # Create the loss_params dictionary conditionally
    loss_params = {"reg_weight": reg_weight}

    config = {
        "seed": seed,
        "output_dir": os.path.join(
            paths["base_output_dir"], f"{experiment}_{dim}", strategy
        ),
        "wandb": wandb,
        "wandb_project": wandb_project,
        "save_messages": save_messages,
        "tqdm": tqdm,
        "debug": debug,
        "data_dir": os.path.join(paths["base_data_dir"], f"{experiment}_{dim}"),
        "prune_graphs": prune_graphs,
        "quick_test": quick_test,
        "model_state_path": model_state_path,
        "model": model_type,
        "model_params": {
            "n_f": num_features,
            "msg_dim": msg_dim,
            "ndim": int(dim[0]),
            "hidden": hidden_size,
            "aggr": aggr_method,
        },
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "save_every_n_epochs": save_every_n_epochs,
        "scheduler_params": {
            "max_lr": max_lr,
            "final_div_factor": final_div_factor,
        },
        "loss": loss_type,
        "loss_params": loss_params,
        "augmentations": {
            "random_translate": {
                "scale": translate_scale,
                "dims": list(range(int(dim[0]))),
            }
        },
    }
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training configuration files."
    )
    parser.add_argument(
        "root_config_dir_path",
        help="Root directory path to store generated config files.",
    )
    parser.add_argument(
        "base_output_dir",
        help="Base output directory path for storing training results.",
    )
    parser.add_argument(
        "base_data_dir", help="Base data directory path for storing datasets."
    )
    args = parser.parse_args()

    paths = {
        "base_output_dir": args.base_output_dir,
        "base_data_dir": args.base_data_dir,
    }

    # Loop through all combinations and write YAML files
    for experiment in experiments:
        for strategy in strategies:
            for dim in dimensions:
                config = create_config(experiment, strategy, dim, paths)
                dir_path = os.path.join(args.root_config_dir_path, experiment)
                os.makedirs(dir_path, exist_ok=True)
                file_path = os.path.join(dir_path, f"{strategy}_{dim}.yaml")

                with open(file_path, "w") as file:
                    yaml.dump(
                        config, file, sort_keys=False, default_flow_style=False
                    )

    print(
        f"[SUCCESS] Train config files created at {args.root_config_dir_path}"
    )
