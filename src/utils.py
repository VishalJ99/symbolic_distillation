import torch
import numpy as np
import os
import random
import torch
from sklearn.model_selection import train_test_split


def get_edge_index(n, sim):
    """
    Code taken from:
    https://github.com/MilesCranmer/symbolic_deep_learning/blob/master/models.py
    """
    if sim in ["string", "string_ball"]:
        # Should just be along it.
        top = torch.arange(0, n - 1)
        bottom = torch.arange(1, n)
        edge_index = torch.cat(
            (
                torch.cat((top, bottom))[None],
                torch.cat((bottom, top))[None],
            ),
            dim=0,
        )
    else:
        adj = (np.ones((n, n)) - np.eye(n)).astype(int)
        edge_index = torch.from_numpy(np.array(np.where(adj)))

    return edge_index


def get_train_val_test_split(X, Y, train_val_test_split, shuffle=True, seed=42):
    """
    Split the data into train, val and test sets
    """
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=1 - train_val_test_split[0], random_state=seed
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp,
        Y_temp,
        test_size=train_val_test_split[-1]
        / (train_val_test_split[-2] + train_val_test_split[-1]),
        random_state=seed,
    )

    train_tuple = (X_train, Y_train)
    val_data = (X_val, Y_val)
    test_data = (X_test, Y_test)

    return train_tuple, val_data, test_data


def seed_everything(seed):
    # Set `PYTHONHASHSEED` environment variable to the seed.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set seed for all packages with built-in pseudo-random generators.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 5. If using CUDA, set also the below for determinism.
    if torch.cuda.is_available():
        # Sets the seed for generating random numbers for the current GPU.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU.

        # Ensures that the CUDA convolution uses deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
