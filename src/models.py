# TODO: generalise edge and node model architecture / abstract it to config
# Add typing hints
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing


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
