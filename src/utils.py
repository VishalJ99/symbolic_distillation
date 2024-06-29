import numpy as np
import os
import random
import torch
from models import GNN, VarGNN
from losses import MAELossWithL1MessageReg, MAELossWithKLMessageReg
from transforms import RandomTranslate
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import pandas as pd
from force_funcs import spring_force, r1_force, r2_force, charge_force
from typing import List, Callable


def get_edge_index(sim_fname: str) -> torch.Tensor:
    """
    Code modified from:
    https://github.com/MilesCranmer/symbolic_deep_learning/blob/master/models.py

    Constructs the edge index array needed to create a torch_geometric.data.Data
    object.

    Parameters
    ----------
    sim_fname : str
        The name of the simulation file.

    Returns
    -------
    torch.Tensor
        The edge index array.


    Notes
    -----
    Takes as input the simulation file name follows convention used in
    `simulations/run_sims.py`.
    Eg. sim=r2_ns=2500_seed=2_n_body=8_dim=3_nt=500_dt=1e-03_data.npy
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


def seed_everything(seed: int) -> None:
    """
    Seeds the random number generators for reproducibility.
    Note, this may slow down code as enforcing determinism could disable
    some optimisations.

    Parameters
    ----------
    seed : int
        The seed value to use.

    Returns
    -------
    None
    """
    # Set `PYTHONHASHSEED` environment variable to the seed.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set seed for all packages with built-in pseudo-random generators.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If using CUDA, set also the below for determinism.
    if torch.cuda.is_available():
        # Sets the seed for generating random numbers for the current GPU.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU.

        # Ensures that the CUDA convolution uses deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_dir(dir_path: str) -> None:
    """
    Makes a directory if it does not exist. Otherwise, logs a message and exits.

    Parameters
    ----------
    dir_path : str
        The path to the directory to create.

    Returns
    -------
    None
    """
    try:
        os.makedirs(dir_path)
    except OSError:
        print(f"Directory: {dir_path} already exists..." " Exiting.")
        exit(1)


def transforms_factory(
    transform_key: str, transform_params: dict
) -> torch.nn.Module:
    """
    Takes in a dictionary with keys as the transform names and values as the
    transform parameters and returns a transforms.Compose object.

    Parameters
    ----------
    transform_key : str
        The key for the transform to use.
    transform_params : dict
        The parameters for the transform.

    Returns
    -------
    torch.nn.Module
        The transform object.
    """
    transforms_dict = {
        "random_translate": RandomTranslate,
    }

    transform = transforms_dict[transform_key](**transform_params)
    return transform


def loss_factory(loss: str, loss_params: dict) -> torch.nn.Module:
    """
    Takes in a dictionary with keys as the loss names and values as the
    loss parameters and returns a loss function.

    Parameters
    ----------
    loss : str
        The name of the loss function to use.
    loss_params : dict
        The parameters for the loss function.

    Returns
    -------
    torch.nn.Module
        The loss function object.
    """
    loss_dict = {
        "maeloss+l1reg": MAELossWithL1MessageReg,
        "maeloss+klreg": MAELossWithKLMessageReg,
    }

    loss_fn = loss_dict[loss](**loss_params)
    return loss_fn


def model_factory(model: str, model_params: dict) -> torch.nn.Module:
    """
    Takes in a dictionary with keys as the model names and values as the
    model parameters and returns a model object.

    Parameters
    ----------
    model : str
        The name of the model to use.

    model_params : dict
        The parameters for the model.

    Returns
    -------
    torch.nn.Module
        The model object.
    """
    model_dict = {
        "gnn": GNN,
        "vargnn": VarGNN,
    }
    model = model_dict[model](**model_params)
    return model


def force_factory(sim: str) -> Callable:
    """
    TODO: Modify this to contain logic which swaps force to acceleration.
    Returns the force function for a given simulation.
    Used by src/eval_msgs.py when creating the edge message and transformed
    force correlation plots.

    Parameters
    ----------
    sim : str
        The name of the simulation.

    Returns
    -------
    function
        The force function.
    """
    
    force_dict = {
        "spring": spring_force,
        "r1": r1_force,
        "r2": r2_force,
        "charge": charge_force,
    }
    try:
        force_fnc = force_dict[sim]

    except KeyError:
        print(
            f"Force function for simulation: {sim} not found. "
            "Please choose from: "
            f"{list(force_dict.keys())}\n"
            "Alternatively, add the force function to force_funcs.py and update "
            "the force_dict in this utils.force_factory function.\n"
            "Exiting..."
        )
        exit(1)
    return force_fnc


def get_node_message_info_dfs(
    graph: Data, model: MessagePassing, dim: int
) -> List[pd.DataFrame]:
    """
    Extracts the node features and messages for each edge in the graph.
    Creates dataframes that are used by src/eval_msgs.py and
    src/eval_node_model.py to symbolically regress the edge and node models.
    Adds extra columns for the relative distance between the nodes for
    convenience.

    Based on the code in src/models.py, the graph networks are initialised with
    the default value for the flow parameter, therefore, bodies enumerated by
    the subscript 1 are the recieving nodes and bodies enumerated by the
    subscript 2 are the sending nodes. Relative distances point from the sending
    to the recieving node.

    Parameters
    ----------
    graph : Data
        The graph object containing the sample edge data to construct the
        dataframes.

    model : MessagePassing
        The message passing model.

    dim : int
        The dimension of the simulation.

    Returns
    -------
    pd.DataFrame
        The dataframe containing the node features and messages for each edge.

    pd.DataFrame
        The dataframe containing the output of the node models.
    """
    # Get node features.
    s = graph.x[graph.edge_index[0]]  # sending nodes
    r = graph.x[graph.edge_index[1]]  # receiving nodes

    # Calculate the edge messages.
    msg = model.message(r, s)

    # Calculate the predicted accelerations.
    pred = model(graph)

    # Concatenate the node features and messages for each edge.
    all_x_info = torch.cat((r, s, msg), dim=1)

    # Add node feature columns.
    if dim == 2:
        x_columns = [
            elem % (k)
            for k in range(1, 3)
            for elem in "x%d y%d vx%d vy%d q%d m%d".split(" ")
        ]
    elif dim == 3:
        x_columns = [
            elem % (k)
            for k in range(1, 3)
            for elem in "x%d y%d z%d vx%d vy%d vz%d q%d m%d".split(" ")
        ]

    # Add the message columns.
    x_columns += ["e%d" % (k,) for k in range(msg.shape[-1])]

    # Add the label columns.
    y_columns = ["a%d" % (k,) for k in range(1, pred.shape[-1] + 1)]

    # Save dataframe containing all edge information.
    df_x = pd.DataFrame(
        data=all_x_info.cpu().detach().numpy(), columns=x_columns
    )

    # Save dataframe containing the output of the node models.
    df_y = pd.DataFrame(data=pred.cpu().detach().numpy(), columns=y_columns)

    # Add rel dist between the nodes as extra columns for symbolic regression.
    # (Note: points towards the recieving node)
    df_x["dx"] = df_x.x1 - df_x.x2
    df_x["dy"] = df_x.y1 - df_x.y2

    if dim == 2:
        df_x["r"] = np.sqrt((df_x.dx) ** 2 + (df_x.dy) ** 2)

    elif dim == 3:
        df_x["dz"] = df_x.z1 - df_x.z2
        df_x["r"] = np.sqrt((df_x.dx) ** 2 + (df_x.dy) ** 2 + (df_x.dz) ** 2)

    return df_x, df_y


def debug_logs(
    graph: Data,
    pred: torch.Tensor,
    train_loss_components_dict: dict,
    loss: torch.Tensor,
) -> None:
    """
    Used in the training loop for debugging.
    Prints the first graph in the input, labels and predictions along with
    the various loss terms.

    Parameters
    ----------
    graph : Data
        The graph data object returned by the DataLoader in the training loop.

    pred : torch.Tensor
        The model predictions.

    train_loss_components_dict : dict
        The dictionary containing the loss components.

    loss : torch.Tensor
        The total loss.

    Returns
    -------
    None
    """
    print("Node Data for First Graph in Batch:")
    print(graph.x[: graph.ptr[1]])

    print("Labels for First Graph in Batch:")
    print(graph.y[: graph.ptr[1]])

    print("Pred for First Graph in Batch:")
    print(pred[: graph.ptr[1]])

    print("Loss Components for Entire Batch:")
    print(train_loss_components_dict)

    print("Loss for Entire Batch:")
    print(loss.item())


def calc_summary_stats(data: np.ndarray) -> dict:
    """
    Calculates the summary statistics for the input data.

    Parameters
    ----------
    data : np.ndarray
        The input data.

    Returns
    -------
    dict
        The summary statistics.
    """

    summary_stats = {
        "Mean": float(np.mean(data)),
        "Standard Deviation": float(np.std(data)),
        "Median": float(np.median(data)),
        "Lower Quartile": float(np.percentile(data, 25)),
        "Upper Quartile": float(np.percentile(data, 75)),
        "Min": float(np.min(data)),
        "Max": float(np.max(data)),
    }
    return summary_stats
