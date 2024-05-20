# TODO: generalise edge and node model architecture / abstract it to config
# Add typing hints
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, ReLU


class GNN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, hidden=300, aggr="add"):
        super(GNN, self).__init__(aggr=aggr)  # "Add" aggregation.
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
        x = torch.cat([x_i, x_j], dim=1)
        msg = self.edge_model(x)
        return msg

    def update(self, aggr_out, x=None):
        x = torch.cat([x, aggr_out], dim=1)
        x_new = self.node_model(x)
        return x_new


# def make_packer(n, n_f):
#     def pack(x):
#         return x.reshape(-1, n_f * n)

#     return pack


# def make_unpacker(n, n_f):
#     def unpack(x):
#         return x.reshape(-1, n, n_f)

#     return unpack


class GN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, hidden=300, aggr="add"):
        super(GN, self).__init__(aggr=aggr)  # "Add" aggregation.
        self.msg_fnc = Seq(
            Lin(2 * n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            ##(Can turn on or off this layer:)
            #             Lin(hidden, hidden),
            #             ReLU(),
            Lin(hidden, msg_dim),
        )

        self.node_fnc = Seq(
            Lin(msg_dim + n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            #             Lin(hidden, hidden),
            #             ReLU(),
            Lin(hidden, ndim),
        )

    # [docs]
    def forward(self, x, edge_index):
        # x is [n, n_f]
        x = x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        # x_i has shape [n_e, n_f]; x_j has shape [n_e, n_f]
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.msg_fnc(tmp)

    def update(self, aggr_out, x=None):
        # aggr_out has shape [n, msg_dim]

        tmp = torch.cat([x, aggr_out], dim=1)
        return self.node_fnc(tmp)  # [n, nupdate]


class OGN(GN):
    def __init__(
        self, n_f, msg_dim, ndim, dt, edge_index, aggr="add", hidden=300, nt=1
    ):
        super(OGN, self).__init__(n_f, msg_dim, ndim, hidden=hidden, aggr=aggr)
        self.dt = dt
        self.nt = nt
        self.edge_index = edge_index
        self.ndim = ndim

    def just_derivative(self, g, augment=False, augmentation=3):
        # x is [n, n_f]
        x = g.x
        ndim = self.ndim
        if augment:
            augmentation = torch.randn(1, ndim) * augmentation
            augmentation = augmentation.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)

        edge_index = g.edge_index

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def loss(self, g, augment=True, square=False, augmentation=3, **kwargs):
        if square:
            return torch.sum(
                (
                    g.y
                    - self.just_derivative(
                        g, augment=augment, augmentation=augmentation
                    )
                )
                ** 2
            )
        else:
            return torch.sum(
                torch.abs(g.y - self.just_derivative(g, augment=augment))
            )
