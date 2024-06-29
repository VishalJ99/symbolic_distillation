"""
Implementation of force laws used in the various simulations.
These functions are used by src/eval_msgs.py to calculate the expected forces
when making the message vs force scatter plots and calculate the R2 values.

To add custom force law, define the function here with the same argument
signature. Then add to the force_factory function in src/utils.py.

TODO:
Modify these functions to accept a flag which swaps force to acceleration.
As the edge message can learn the acceleration, so we would like to compare
the message to the acceleration instead of the force.
"""
import numpy as np
import pandas as pd


def spring_force(df: pd.DataFrame, eps: float = 1e-2) -> np.ndarray:
    """
    Spring force law with centre at r=1.
    Assumes structure of dataframe returned by the first output of the
    `get_node_message_info_dfs` function in src/utils.py.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the node features and edge messages for
        edges in the graph.

    eps : float
        Small value to prevent division by zero.

    Returns
    -------
    np.ndarray
        The forces.
    """
    pos_cols = ["dx", "dy"] + (["dz"] if "dz" in df.columns else [])
    dr = df[pos_cols].to_numpy()
    r = (df.r + eps).to_numpy()[:, None]
    return -((r - 1)) * (dr / r)


def r1_force(df: pd.DataFrame, eps: float = 1e-2) -> np.ndarray:
    """
    1/r Orbital force law.

    Assumes structure of dataframe returned by the first output of the
    `get_node_message_info_dfs` function in src/utils.py.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the node features and edge messages for
        edges in the graph.

    eps : float
        Small value to prevent division by zero.

    Returns
    -------
    np.ndarray
        The forces.
    """
    pos_cols = ["dx", "dy"] + (["dz"] if "dz" in df.columns else [])
    dr = np.array(df[pos_cols])
    r = (df.r + eps).to_numpy()[:, None]
    m1 = df.m1.to_numpy()[:, None]
    m2 = df.m2.to_numpy()[:, None]
    return -((m1 * m2) / r) * (dr / r)


def r2_force(df: pd.DataFrame, eps: float = 1e-2) -> np.ndarray:
    """
    1/r^2 Orbital force law.

    Assumes structure of dataframe returned by the first output of the
    `get_node_message_info_dfs` function in src/utils.py.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the node features and edge messages for
        edges in the graph.

    eps : float
        Small value to prevent division by zero.

    Returns
    -------
    np.ndarray
        The forces.
    """
    pos_cols = ["dx", "dy"] + (["dz"] if "dz" in df.columns else [])
    dr = np.array(df[pos_cols])
    r = (df.r + eps).to_numpy()[:, None]
    m1 = df.m1.to_numpy()[:, None]
    m2 = df.m2.to_numpy()[:, None]
    return -((m1 * m2) / r**2) * (dr / r)


def charge_force(df: pd.DataFrame, eps: float = 1e-2) -> np.ndarray:
    """
    Coulomb force law.

    Assumes structure of dataframe returned by the first output of
    the `get_node_message_info_dfs` function in src/utils.py.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the node features and edge messages
        for edges in the graph.

    eps : float
        Small value to prevent division by zero.

    Returns
    -------
    np.ndarray
        The forces.
    """
    pos_cols = ["dx", "dy"] + (["dz"] if "dz" in df.columns else [])
    dr = np.array(df[pos_cols])
    r = (df.r + eps).to_numpy()[:, None]
    q1 = df.q1.to_numpy()[:, None]
    q2 = df.q2.to_numpy()[:, None]
    return -((q1 * q2) / r**2) * (dr / r)
