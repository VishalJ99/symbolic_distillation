import numpy as np
import os
import random
import torch
from sklearn.model_selection import train_test_split
from models import GNN, VarGNN
from losses import MAELossWithL1MessageReg, MAELossWithKLMessageReg
from transforms import RandomTranslate
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import pandas as pd


def get_edge_index(sim_fname):
    """
    Code modified from:
    https://github.com/MilesCranmer/symbolic_deep_learning/blob/master/models.py
    """
    # Load sim hyper params from fname. Assumes format used by `run_sims.py`.
    sim_fname = sim_fname.split(".")[0]
    sim_hyperparams_dict = {}
    for kv in sim_fname.split("_"):
        if "=" in kv:
            k, v = kv.split("=")
            sim_hyperparams_dict[k] = v

    sim = sim_hyperparams_dict["sim"]
    n = int(sim_hyperparams_dict["body"])

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


def make_dir(dir_path):
    """
    Makes a directory if it does not exist. Otherwise, logs a message and exits.
    """
    try:
        os.makedirs(dir_path)
    except OSError:
        print(f"Directory: {dir_path} already exists..." " Exiting.")
        exit(1)


def transforms_factory(transform_key, transform_params):
    """
    Takes in a dictionary with keys as the transform names and values as the
    transform parameters and returns a transforms.Compose object.
    """
    transforms_dict = {
        "random_translate": RandomTranslate,
    }

    transform = transforms_dict[transform_key](**transform_params)
    return transform


def loss_factory(loss, loss_params):
    """
    Takes in a dictionary with keys as the loss names and values as the
    loss parameters and returns a loss function.
    """
    loss_dict = {
        "loss+l1reg": MAELossWithL1MessageReg,
        "loss+klreg": MAELossWithKLMessageReg,
    }

    loss_fn = loss_dict[loss](**loss_params)
    return loss_fn


def model_factory(model, model_params):
    """
    Takes in a dictionary with keys as the model names and values as the
    model parameters and returns a model object.
    """
    model_dict = {
        "gnn": GNN,
        "vargnn": VarGNN,
    }
    model = model_dict[model](**model_params)
    return model


def get_node_message_info_df(graph: Data, model: MessagePassing, dim: int):
    s = graph.x[graph.edge_index[0]]  # sending nodes
    r = graph.x[graph.edge_index[1]]  # recieving nodes
    msg = model.message(s, r)

    all_info = torch.cat((s, r, msg), dim=1)

    # Add node feature columns.
    if dim == 2:
        columns = [
            elem % (k)
            for k in range(1, 3)
            for elem in "x%d y%d vx%d vy%d q%d m%d".split(" ")
        ]
    elif dim == 3:
        columns = [
            elem % (k)
            for k in range(1, 3)
            for elem in "x%d y%d z%d vx%d vy%d vz%d q%d m%d".split(" ")
        ]

    # Add the message columns.
    columns += ["e%d" % (k,) for k in range(msg.shape[-1])]

    df = pd.DataFrame(data=all_info.cpu().detach().numpy(), columns=columns)

    # Create columns for the distance between the nodes.
    # Useful for performing symbolic regression.
    df["dx"] = df.x1 - df.x2
    df["dy"] = df.y1 - df.y2

    if dim == 2:
        df["r"] = np.sqrt((df.dx) ** 2 + (df.dy) ** 2)

    elif dim == 3:
        df["dz"] = df.z1 - df.z2
        df["r"] = np.sqrt((df.dx) ** 2 + (df.dy) ** 2 + (df.dz) ** 2)

    return df
