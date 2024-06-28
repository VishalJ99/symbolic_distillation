import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eval messages for all experiments"
    )

    parser.add_argument(
        "root_test_dir", type=str, help="Base directory for processed data"
    )

    args = parser.parse_args()

    root_test_dir = args.root_test_dir

    experiments = ["spring", "r1", "r2", "charge"]
    strategies = ["standard", "bottleneck", "l1", "kl"]
    dims = ["2d", "3d"]

    for experiment in experiments:
        for dim in dims:
            for strategy in strategies:
                msg_csv = os.path.join(
                    root_test_dir,
                    f"{experiment}_{dim}/{strategy}/symbolic_regression_csvs",
                    "edge_messages.csv",
                )
                node_csv = os.path.join(
                    root_test_dir,
                    f"{experiment}_{dim}/{strategy}/symbolic_regression_csvs",
                    "node_accels.csv",
                )

                eval_node_dir = os.path.join(
                    root_test_dir, f"{experiment}_{dim}/{strategy}/node_eval"
                )
                cmd = f"python src/eval_node_model.py\
                    {msg_csv} {node_csv} {eval_node_dir} --samples 1000"
                os.system(cmd)

    print("[SUCCESS] All message evaluations complete.")
