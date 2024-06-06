# TODO: generalise edge and node model architecture / abstract it to config
# Add typing hints
import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, ReLU
import pickle as pkl
import numpy as np


class GNN(MessagePassing):
    def __init__(
        self,
        n_f,
        msg_dim,
        ndim,
        hidden=300,
        aggr="add",
        symbolic_edge_pkl_path=None,
    ):
        super(GNN, self).__init__(aggr=aggr)
        self.ndim = ndim
        self.msg_dim = msg_dim

        self.symbolic_edge_pkl_path = symbolic_edge_pkl_path

        if symbolic_edge_pkl_path:
            with open(symbolic_edge_pkl_path, "rb") as f:
                symbolic_edge_pkl = pkl.load(f)

            self.symbolic_edge_models = symbolic_edge_pkl["models"]
            self.important_msg_indices = symbolic_edge_pkl["important_msg_idxs"]

        self.edge_model = Sequential(
            Linear(2 * n_f, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, msg_dim),
        )

        self.node_model = Sequential(
            Linear(msg_dim + n_f, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, ndim),
        )

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        if self.symbolic_edge_pkl_path:
            # Construct input to pysr model: (dr_ij, r_ij, q_i, q_j, m_i, m_j).
            # This order must match X_cols (defined in eval_msgs.py).
            dr_ij = x_i[:, : self.ndim] - x_j[:, : self.ndim]
            r_ij = torch.linalg.norm(dr_ij, axis=1).unsqueeze(-1)

            q_i = x_i[:, -2].unsqueeze(-1)
            m_i = x_i[:, -1].unsqueeze(-1)

            q_j = x_j[:, -2].unsqueeze(-1)
            m_j = x_j[:, -1].unsqueeze(-1)

            x = torch.concatenate([dr_ij, r_ij, q_i, q_j, m_i, m_j], axis=1)

            # Calc most important message components using pysr edge models.
            important_msg_components = torch.tensor(
                np.asarray(
                    [model.predict(x) for model in self.symbolic_edge_models]
                )
            ).T.to(x_i.device, x_i.dtype)

            # Construct the full message.
            msg = torch.zeros(
                (x_i.shape[0], self.msg_dim), device=x_i.device, dtype=x_i.dtype
            )
            msg[:, self.important_msg_indices] = important_msg_components

        else:
            x = torch.cat([x_i, x_j], dim=1)
            msg = self.edge_model(x)

        return msg

    def update(self, aggr_out, x=None):
        x = torch.cat([x, aggr_out], dim=1)
        x_new = self.node_model(x)
        return x_new


class VarGNN(GNN):
    def __init__(self, n_f, msg_dim, ndim, hidden=300, aggr="add"):
        super(VarGNN, self).__init__(n_f, msg_dim, ndim, hidden, aggr)
        self.msg_dim = msg_dim
        self.edge_model = Sequential(
            Linear(2 * n_f, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, 2 * msg_dim),
        )

    def forward(self, graph, sample=True):
        x = graph.x
        edge_index = graph.edge_index
        return self.propagate(
            edge_index, size=(x.size(0), x.size(0)), x=x, sample=sample
        )

    def message(self, x_i, x_j, sample=False):
        # Sample False by default since util.get_node_message_info_df
        # Doesnt take custom arguments and just calls message method.
        # Want to just take message as mean for plotting.
        x = torch.cat([x_i, x_j], dim=1)
        param_msg = self.edge_model(x)

        # Unpack message parameters.
        mu = param_msg[:, ::2]
        logvar = param_msg[:, 1::2]
        std = torch.exp(0.5 * logvar)

        # Sample message.
        msg = mu

        if sample:
            msg += std * torch.randn_like(mu)

        return msg
