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
                eval_msg_dir = os.path.join(
                    root_test_dir, f"{experiment}_{dim}/{strategy}/msg_eval_accels"
                )
                cmd = f"python src/eval_msgs.py\
                    {msg_csv} {eval_msg_dir} {experiment} --samples 1000 --no_sr"
                os.system(cmd)

    print("[SUCCESS] All message evaluations complete.")
