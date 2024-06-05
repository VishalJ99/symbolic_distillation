# TODO: generalise edge and node model architecture / abstract it to config
# Add typing hints
import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, ReLU
from pysr import PySRRegressor
from icecream import ic


class GNN(MessagePassing):
    def __init__(
        self,
        n_f,
        msg_dim,
        ndim,
        hidden=300,
        aggr="add",
        symbolic_edge_pkl_1=False,
        symbolic_edge_pkl_2=False,
    ):
        super(GNN, self).__init__(aggr=aggr)
        self.ndim = ndim
        self.msg_dim = msg_dim

        self.symbolic_edge_pkl_1 = symbolic_edge_pkl_1
        self.symbolic_edge_pkl_2 = symbolic_edge_pkl_2

        if symbolic_edge_pkl_2 and symbolic_edge_pkl_2:
            self.edge_model_1 = PySRRegressor.from_file(symbolic_edge_pkl_1)
            self.edge_model_2 = PySRRegressor.from_file(symbolic_edge_pkl_2)

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
        if self.symbolic_edge_pkl_1 and self.symbolic_edge_pkl_2:
            # Input to symbolic model is a dr_ij, r_ij, q_i, q_j, m_i, m_j.
            # This order must match order in X_cols defined in eval_msgs.py.
            dr_ij = x_i[:, : self.ndim] - x_j[:, : self.ndim]
            r_ij = torch.linalg.norm(dr_ij, axis=1).unsqueeze(-1)

            q_i = x_i[:, -2].unsqueeze(-1)
            m_i = x_i[:, -1].unsqueeze(-1)

            q_j = x_j[:, -2].unsqueeze(-1)
            m_j = x_j[:, -1].unsqueeze(-1)

            x = torch.concatenate([dr_ij, r_ij, q_i, q_j, m_i, m_j], axis=1)
            m1 = torch.tensor(self.edge_model_1.predict(x))
            m2 = torch.tensor(self.edge_model_2.predict(x))

            # Pad msg with zeros to create a tensor of shape (N_edges, msg_dim).
            msg = torch.zeros((x_i.shape[0], self.msg_dim), device=x_i.device)
            msg[:, 57] = m1
            msg[:, 64] = m2

            ic(m1[0], m2[0])
            # print('First two symbolic msg comps
            # :', msg[0, 57], msg[0, 64])
            x = torch.cat([x_i, x_j], dim=1)
            msg_learned = self.edge_model(x)
            ic(msg_learned[0, :])
            print(
                "First two learned msg components:",
                msg_learned[0, 57],
                msg_learned[0, 64],
            )
        else:
            x = torch.cat([x_i, x_j], dim=1)
            msg = self.edge_model(x)
            print("First two learned msg components:", msg[0, :2])
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
