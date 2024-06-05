import yaml
import os
import argparse

prune_outliers = 0
test_batch_size = 1024
hidden_size = 300
message_save_limit = 100000
quick_test = False


def create_config(experiment, strategy, dim, paths):
    model_type = "gnn" if strategy != "kl" else "vargnn"
    msg_dim_strategy = int(dim[0]) if strategy == "bottleneck" else 100
    aggr_method = "add"
    loss_type = "maeloss+l1reg" if strategy != "kl" else "maeloss+klreg"
    reg_weight = (
        1.0 if strategy == "kl" else (1.0e-2 if strategy == "l1" else 0)
    )

    loss_params = {"reg_weight": reg_weight}

    config = {
        "output_dir": os.path.join(
            paths["output_base_dir"], f"{experiment}_{dim}", strategy
        ),
        "tqdm": False,
        "quick_test": quick_test,
        "save_messages": True,
        "message_save_limit": message_save_limit,
        "data_dir": os.path.join(paths["data_base_dir"], f"{experiment}_{dim}"),
        "prune_outliers": prune_outliers,
        "test_batch_size": test_batch_size,
        "model_weights_path": os.path.join(
            paths["train_base_dir"],
            f"{experiment}_{dim}",
            strategy,
            "model_weights",
            "best_model.pt",
        ),
        "model": model_type,
        "model_params": {
            "n_f": (2 * int(dim[0]) + 2),
            "msg_dim": msg_dim_strategy,
            "ndim": int(dim[0]),
            "hidden": hidden_size,
            "aggr": aggr_method,
        },
        "loss": loss_type,
        "loss_params": loss_params,
    }
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training configuration files for experiments."
    )
    parser.add_argument(
        "root_config_dir_path",
        help="Base directory path to store generated config files.",
    )
    parser.add_argument(
        "output_base_dir", help="Base directory for output results."
    )
    parser.add_argument("data_base_dir", help="Base directory for input data.")
    parser.add_argument(
        "train_base_dir", help="Base directory for trained models."
    )
    args = parser.parse_args()

    paths = {
        "output_base_dir": args.output_base_dir,
        "data_base_dir": args.data_base_dir,
        "train_base_dir": args.train_base_dir,
    }

    experiments = ["spring", "r1", "r2", "charge"]
    strategies = ["standard", "l1", "kl", "bottleneck"]
    dimensions = ["2d", "3d"]

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

    print(f"[SUCCESS] Testing config created at {args.root_config_dir_path}")
