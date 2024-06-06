import numpy as np

# TODO: Generalise these transformations functions.


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
