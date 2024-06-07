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
                print(f"[INFO] Creating GIFs for {experiment}_{dim}/{strategy}")
                msg_dir = os.path.join(
                    root_test_dir,
                    f"{experiment}_{dim}/{strategy}/train_messages",
                )
                gif_dir = os.path.join(
                    root_test_dir, f"{experiment}_{dim}/{strategy}/msg_gifs"
                )
                cmd = (
                    "python src/plots.py "
                    f"{msg_dir} {gif_dir} {experiment} --delete_frames"
                )

                os.system(cmd)
                cmd = (
                    "python src/plots.py"
                    f"{msg_dir} {gif_dir} {experiment}"
                    "--plot_sparsity --delete_frames"
                )
                os.system(cmd)

    print("[SUCCESS] All GIFs created.")
