import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse


def calculate_stats(data):
    stats = {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "max": float(np.max(data)),
        "min": float(np.min(data)),
        "uq": float(np.percentile(data, 75)),
        "lq": float(np.percentile(data, 25)),
    }
    return stats


def plot_histograms(vel, accel):
    flat_velocities = vel.flatten()
    flat_accelerations = accel.flatten()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(flat_velocities, bins=100)
    ax[0].set_title("Velocity")
    ax[0].set_xlabel("Velocity")
    ax[0].set_ylabel("Frequency")

    ax[1].hist(flat_accelerations, bins=100)
    ax[1].set_title("Acceleration")
    ax[1].set_xlabel("Acceleration")
    ax[1].set_ylabel("Frequency")

    return plt


def velocity_analysis(vel, accel, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vel_stats = calculate_stats(vel)
    vel_stats_path = os.path.join(output_dir, "vel_stats.json")
    with open(vel_stats_path, "w+") as f:
        json.dump(vel_stats, f)

    accel_stats = calculate_stats(accel)
    accel_stats_path = os.path.join(output_dir, "accel_stats.json")
    with open(accel_stats_path, "w+") as f:
        json.dump(accel_stats, f)

    plt = plot_histograms(vel, accel)
    plt.savefig(os.path.join(output_dir, "vel_accel_dist.png"))


def load_data(data_dir, use_raw=False):
    data_dir = os.path.join(data_dir, "processed" if not use_raw else "raw")

    if use_raw:
        raw_files = os.listdir(data_dir)
        for f in raw_files:
            if "accel" in f:
                y = np.load(os.path.join(data_dir, f))
            else:
                x = np.load(os.path.join(data_dir, f))
    else:
        data_fname = [f for f in os.listdir(data_dir) if "process" in f][0]
        graph_path = os.path.join(data_dir, data_fname)
        graph = torch.load(graph_path)[0]
        x = graph.x.numpy()
        y = graph.y.numpy()

    dim = (x.shape[-1] - 2) // 2
    vel = x[:, dim:-2]
    accel = y

    return vel, accel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarise velocities and accelerations"
    )
    parser.add_argument(
        "root_data_dir", type=str, help="Base directory for processed data"
    )
    parser.add_argument("save_dir", type=str, help="Directory to save output")
    parser.add_argument(
        "--use_raw",
        action="store_true",
        help="Use raw data instead of processed data",
    )
    args = parser.parse_args()
    root_data_dir = args.root_data_dir
    save_dir = args.save_dir
    use_raw = args.use_raw

    experiments = ["spring", "r1", "r2", "charge"]
    dims = [2, 3]
    splits = ["train", "val", "test"]

    for experiment in experiments:
        for dim in dims:
            for split in splits:
                data_dir = os.path.join(
                    root_data_dir, f"{experiment}_{dim}d/{split}/"
                )
                vel, accel = load_data(data_dir, use_raw=use_raw)

                output_dir = os.path.join(
                    save_dir, f"{experiment}_{dim}", f"{split}"
                )

                print(f"[INFO] running analysis for {experiment}-{dim}-{split}")
                velocity_analysis(vel, accel, output_dir)
