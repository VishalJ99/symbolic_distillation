"""
Script to define custom torch geometric message passing models.
To add a custom model, define a class here with the following signature:

```
class CustomGNN(MessagePassing):
    def __init__(self, *args, **kwargs):
        super(CustomGNN, self).__init__()
        # Define any parameters here.
        self.param = param
        self.edge_model = Sequential(*args, **kwargs)
        self.node_model = Sequential(*args, **kwargs)

    def message(self, x_i, x_j):
        # Define the message function here.
        return msg

    def update(self, aggr_out, x):
        # Define the update function here.
        return y
```
After defining the model, add it to the `model_factory` function in
`src/utils.py` to be able to specify it in the configs and use it in the
training and testing scripts.

To add option to use symbolic models for edge and node models, can follow the
template below used in the GNN class below.

TODO:
- Training with symbolic models is not working as expected. The models are
static throughout training. They should be differentiable as the .pytorch()
method is used to convert them to pytorch models would expect them to be updated
during training. Need to investigate why this is happening.

- Generalise edge and node model architecture by allowing the user to specify
the number of layers and hidden units in the config file.
"""
# TODO:
import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, ReLU
import pickle as pkl


class GNN(MessagePassing):
    def __init__(
        self,
        n_f,
        msg_dim,
        ndim,
        hidden=300,
        aggr="add",
        symbolic_edge_pkl_path=None,
        symbolic_node_pkl_path=None,
    ):
        super(GNN, self).__init__(aggr=aggr)
        self.ndim = ndim
        self.msg_dim = msg_dim

        self.symbolic_edge_pkl_path = symbolic_edge_pkl_path
        self.symbolic_node_pkl_path = symbolic_node_pkl_path

        # Loads symbolic models as saved by eval_msgs.py and eval_node_model.py.
        if symbolic_edge_pkl_path:
            with open(symbolic_edge_pkl_path, "rb") as f:
                symbolic_edge_pkl = pkl.load(f)
                self.symbolic_edge = symbolic_edge_pkl["model"]
                self.symbolic_edge_models = self.symbolic_edge.pytorch()
                self.important_msg_indices = symbolic_edge_pkl[
                    "important_msg_idxs"
                ]

        if symbolic_node_pkl_path:
            with open(symbolic_node_pkl_path, "rb") as f:
                symbolic_node_pkl = pkl.load(f)
                self.symbolic_node = symbolic_node_pkl["model"]
                self.symbolic_node_models = self.symbolic_node.pytorch()

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
            important_msg_components = torch.stack(
                [model(x) for model in self.symbolic_edge_models],
                dim=1,
            ).to(x_i.device, x_i.dtype)

            # Construct the full msg by padding non-important comps with zeros.
            msg = torch.zeros(
                (x_i.shape[0], self.msg_dim), device=x_i.device, dtype=x_i.dtype
            )
            msg[:, self.important_msg_indices] = important_msg_components

        else:
            x = torch.cat([x_i, x_j], dim=1)
            msg = self.edge_model(x)

        return msg

    def update(self, aggr_out, x=None):
        if self.symbolic_node_pkl_path:
            # Get the most important components of the aggregated messages.
            aggr_out_important = aggr_out[:, self.important_msg_indices]
            x = torch.cat([x, aggr_out_important], dim=1)
            y = torch.stack(
                [model(x) for model in self.symbolic_node_models],
                dim=1,
            ).to(x.device, x.dtype)
        else:
            x = torch.cat([x, aggr_out], dim=1)
            y = self.node_model(x)
        return y


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
        """
        Important! Keep sample=False by default since
        util.get_node_message_info_df doesnt take custom arguments and just
        calls message method passing x_i and x_j. In that function we want
        the messages to be sparse hence sample=False so that the mean
        message is returned which should be 0 for the unimportant messages.

        When training / testing, the forward method calls this method (via
        propagate) with sample=True.
        """
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
