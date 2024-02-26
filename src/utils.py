import torch
import numpy as np


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
            (torch.cat((top, bottom))[None], torch.cat((bottom, top))[None]),
            dim=0,
        )
    else:
        adj = (np.ones((n, n)) - np.eye(n)).astype(int)
        edge_index = torch.from_numpy(np.array(np.where(adj)))

    return edge_index
