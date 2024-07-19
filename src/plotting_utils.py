"""
Scrappy code used for generating the plots in src/eval_msgs.py and
src/eval_node_model.py. Code modified from:
https://github.com/MilesCranmer/
symbolic_deep_learning/blob/master/GN_Demo_Colab.ipynb
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


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


def make_sparsity_plot(
    msgs_array, dim, top_n=15, cmap="gray_r", show_grid=True
):
    """Generates a sparsity plot for the most variable messages in a dataset.

    Args:
        msgs_array (np.ndarray): The source array from which to generate the
        plot.
        dim (int): The number of most significant features to select.
        top_n (int, optional): Number of top features to display in the plot.
        cmap (str, optional): The colormap for the plot.
        show_grid (bool, optional): Flag to show grid lines on the plot.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    # Standard deviation across messages
    msgs_std = msgs_array.std(axis=0)
    top_msgs_std = msgs_std[np.argsort(msgs_std)[::-1][None, :top_n]]
    fig, ax = plt.subplots(1, 1)

    ax.pcolormesh(
        top_msgs_std,
        cmap=cmap,
        edgecolors="k",
    )

    # Write std under the plot.
    for i, std in enumerate(top_msgs_std.squeeze()):
        x_pos = i if top_msgs_std.shape[1] == 15 else i + 0.5
        y_pos = -0.75 if top_msgs_std.shape[1] == 15 else -0.25

        # Normalise stds by the sum of the stds to make them comparable.
        std /= msgs_std.sum()
        ax.text(
            x_pos,
            y_pos,
            f"{std: .1e}",
            ha="center",
            va="center",
            rotation=45,
        )

    plt.axis("off")
    plt.grid(True)
    ax.set_aspect("equal")
    plt.text(15.5, 0.5, "...", fontsize=30)
    return fig, ax


def make_force_edge_msg_scatter(transformed_forces, most_important_msgs, dim=2):
    fig, axes = plt.subplots(1, dim, figsize=(4 * dim, 4))
    R2_stats = []
    for i in range(dim):
        # Fit a linear regression model to get the R^2 value.
        slope, intercept, r_value, p_value, stderr = linregress(
            transformed_forces[:, i], most_important_msgs[:, i]
        )

        R2 = r_value**2
        R2_stats.append(R2)

        # Scatter plot of transformed forces vs. message components.
        ax = axes[i] if dim > 1 else axes
        ax.scatter(
            transformed_forces[:, i],
            most_important_msgs[:, i],
            alpha=0.1,
            s=0.1,
            c="black",
        )
        ax.set_xlabel("Transformed Accelerations", fontsize=16)
        ax.set_ylabel(f"Message Component {i+1}", fontsize=16)
        ax.title.set_text(f"Component {i+1} R^2: {R2: .2f}")
        ax.title.set_fontsize(16)

    plt.tight_layout()
    return fig, axes, R2_stats
