import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.optimize import minimize
import os
import imageio
import pandas as pd

plot_force_components = False
plot_sparsity = True


# Load the df. (replace with the pckled file and take the last df)
# with open("sampled_kl_messages_over_time_batch_per_epoch_5k.pkl", "rb") as f:
#     messages_over_time = pkl.load(f)

msgs_dir = '../rds/hpc-work/train_runs/kl_vj_experiment/training_messages/'
messages_over_time = []
csv_files = sorted(os.listdir(msgs_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
for csv in csv_files:
    if csv.endswith('.csv'):
        print(csv)
        messages_over_time.append(pd.read_csv(
            os.path.join(msgs_dir, csv)))

print(len(messages_over_time))


def percentile_sum(x):
    x = x.ravel()
    bot = x.min()
    top = np.percentile(x, 90)
    msk = (x >= bot) & (x <= top)
    frac_good = (msk).sum() / len(x)
    return x[msk].sum() / frac_good


def linear_transformation_2d(alpha, X, Y):
    lincomb1 = (alpha[0] * X[:, 0] + alpha[1] * X[:, 1]) + alpha[2]
    lincomb2 = (alpha[3] * X[:, 0] + alpha[4] * X[:, 1]) + alpha[5]

    # Avoid influence by outliers by only using MSEs within the 90th percentile.
    score = (
        percentile_sum(np.square(Y[:, 0] - lincomb1))
        + percentile_sum(np.square(Y[:, 1] - lincomb2))
    ) / 2.0

    return score


def linear_transformation_3d(alpha, X, Y):
    lincomb1 = (
        alpha[0] * X[:, 0] + alpha[1] * X[:, 1] + alpha[2] * X[:, 2]
    ) + alpha[3]
    lincomb2 = (
        alpha[4] * X[:, 0] + alpha[5] * X[:, 1] + alpha[6] * X[:, 2]
    ) + alpha[7]
    lincomb3 = (
        alpha[8] * X[:, 0] + alpha[9] * X[:, 1] + alpha[10] * X[:, 2]
    ) + alpha[11]

    # Avoid influence by outliers by only using MSEs within the 90th percentile.
    score = (
        percentile_sum(np.square(Y[:, 0] - lincomb1))
        + percentile_sum(np.square(Y[:, 1] - lincomb2))
        + percentile_sum(np.square(Y[:, 2] - lincomb3))
    ) / 3.0

    return score


def out_linear_transformation_2d(alpha, X):
    """Should Y be tranposed?"""
    lincomb1 = (alpha[0] * X[:, 0] + alpha[1] * X[:, 1]) + alpha[2]
    lincomb2 = (alpha[3] * X[:, 0] + alpha[4] * X[:, 1]) + alpha[5]

    Y = np.asarray([lincomb1, lincomb2]).T
    return Y


def out_linear_transformation_3d(alpha, X):
    lincomb1 = (
        alpha[0] * X[:, 0] + alpha[1] * X[:, 1] + alpha[2] * X[:, 2]
    ) + alpha[3]
    lincomb2 = (
        alpha[4] * X[:, 0] + alpha[5] * X[:, 1] + alpha[6] * X[:, 2]
    ) + alpha[7]
    lincomb3 = (
        alpha[8] * X[:, 0] + alpha[9] * X[:, 1] + alpha[10] * X[:, 2]
    ) + alpha[11]

    Y = np.asarray([lincomb1, lincomb2, lincomb3]).T
    return Y


filenames = []  # List to store filenames of the plots

for idx in range(0, len(messages_over_time)):
    df = messages_over_time[idx]

    # Fetch the dim of the experiment by checking if z is present in columns.
    # TODO: fetch this from the initial data.
    dim = 3 if "z1" in df.columns else 2
    sim = "spring"
    eps = 1e-2
    pos_cols = ["dx", "dy"]
    if dim == 3:
        pos_cols.append("dz")

    force_fnc = (
        lambda msg: -(msg.r + eps - 1).to_numpy()[:, None]
        * np.array(msg[pos_cols])
        / (msg.r + eps).to_numpy()[:, None]
    )

    # Fetch the message columns.
    msg_columns = [col for col in df.columns if "e" in col]

    # Fetch the message array.
    msg_array = np.array(df[msg_columns])

    # Select only top dim features by standard deviation.
    msg_importance = msg_array.std(axis=0)

    if plot_sparsity:
        fig, ax = plt.subplots(1, 1)
        ax.pcolormesh(
            msg_importance[np.argsort(msg_importance)[::-1][None, :15]],
            cmap="gray_r",
            edgecolors="k",
        )
        # Write std under the plot.
        for i, std in enumerate(msg_importance[np.argsort(msg_importance)[::-1][:15]]):
            ax.text(i+0.5, -0.5, "%.2f" % std, ha="center", va="center")
            
        plt.axis("off")
        plt.grid(True)
        ax.set_aspect("equal")
        plt.text(15.5, 0.5, "...", fontsize=30)
        plt.tight_layout()

    # Select the dim most important messages.
    most_important_idxs = np.argsort(msg_importance)[-dim:]
    msgs_to_compare = msg_array[:, most_important_idxs]

    # Standardise the messages.
    msgs_to_compare = (
        msgs_to_compare - np.average(msgs_to_compare, axis=0)
    ) / np.std(msgs_to_compare, axis=0)

    expected_forces = force_fnc(df)
    if dim == 2:
        min_result = minimize(
            linear_transformation_2d,
            x0=np.ones(dim**2 + dim),
            args=(expected_forces, msgs_to_compare),
            method="Powell",
        )
    if dim == 3:
        min_result = minimize(
            linear_transformation_3d,
            x0=np.ones(dim**2 + dim),
            args=(expected_forces, msgs_to_compare),
            method="Powell",
        )

    alpha = min_result.x
    if plot_force_components:
        fig, ax = plt.subplots(1, dim, figsize=(4 * dim, 4))
        if dim == 3:
            transformed_forces = out_linear_transformation_3d(
                alpha, expected_forces
            )
        else:
            transformed_forces = out_linear_transformation_2d(
                alpha, expected_forces
            )
        for i in range(dim):
            ax[i].scatter(
                transformed_forces[:, i],
                msgs_to_compare[:, i],
                alpha=0.1,
                s=0.1,
                color="k",
            )
            ax[i].set_xlabel("Linear combination of forces")
            ax[i].set_ylabel("Message Element %d" % (i + 1))

            xlim = np.array(
                [np.percentile(transformed_forces[:, i], q) for q in [10, 90]]
            )
            ylim = np.array(
                [np.percentile(msgs_to_compare[:, i], q) for q in [10, 90]]
            )

            xlim[0], xlim[1] = (
                xlim[0] - (xlim[1] - xlim[0]) * 0.05,
                xlim[1] + (xlim[1] - xlim[0]) * 0.05,
            )
            ylim[0], ylim[1] = (
                ylim[0] - (ylim[1] - ylim[0]) * 0.05,
                ylim[1] + (ylim[1] - ylim[0]) * 0.05,
            )

            ax[i].set_xlim(xlim)
            ax[i].set_ylim(ylim)

            plt.tight_layout()

    filename = f"frame_{idx}.png"
    print("save", filename)
    fig.savefig(filename)  # Save the plot to a file
    plt.close()
    filenames.append(filename)

# Create the GIF
with imageio.get_writer("force.gif", mode="I", duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# for filename in set(filenames):
#     os.remove(filename)  # Remove the files
