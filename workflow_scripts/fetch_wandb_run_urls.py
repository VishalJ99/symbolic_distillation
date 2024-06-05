import argparse
import os
import yaml


def fetch_urls(root_train_dir):
    for root, dirs, files in os.walk(root_train_dir):
        for file in files:
            if file.endswith(".yaml"):
                with open(os.path.join(root, file), "r") as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                    print(f"{root}: {config['run_url']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_train_runs_dir", help="dir which contains all the training runs"
    )
    args = parser.parse_args()
    root_train_runs_dir = args.root_train_runs_dir

    fetch_urls(root_train_runs_dir)
