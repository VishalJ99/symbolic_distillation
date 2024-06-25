import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing


class MAELossWithL1MessageReg(nn.Module):
    def __init__(self, reg_weight=1e-2):
        super(MAELossWithL1MessageReg, self).__init__()
        self.reg_weight = reg_weight

    def forward(
        self,
        input: Batch,
        target: torch.tensor,
        model: MessagePassing = None,
    ):
        total_loss = 0

        # Calculate the MAE.
        base_loss = torch.sum(torch.abs(input.y - target))

        # Divide by the number of nodes in the graph.
        base_loss /= input.y.shape[0]

        # Update the total loss.
        total_loss += base_loss

        if self.reg_weight:
            # Compute summed L1 norm of all the messages.
            s = input.x[input.edge_index[0]]  # sending / source nodes
            r = input.x[input.edge_index[1]]  # recieving / target nodes

            messages = model.message(s, r)
            l1_reg = torch.sum(torch.abs(messages))

            # Divide by the number of edges in the graph.
            l1_reg /= messages.shape[0]

            # Update the total loss.
            total_loss += self.reg_weight * l1_reg

            params = {
                "base_loss": base_loss,
                "l1_reg": l1_reg,
            }
        else:
            params = {}

        return total_loss, params


class MAELossWithKLMessageReg(nn.Module):
    def __init__(self, reg_weight=1):
        super(MAELossWithKLMessageReg, self).__init__()
        self.reg_weight = reg_weight

    def forward(
        self,
        input: Batch,
        target: torch.tensor,
        model: MessagePassing = None,
    ):
        total_loss = 0

        # Calculate the MAE.
        base_loss = torch.sum(torch.abs(input.y - target))

        # Divide by the number of nodes in the graph.
        base_loss /= input.y.shape[0]

        # Update the total loss.
        total_loss += base_loss

        if self.reg_weight:
            # TODO: is this the forward / backward KL divergence?
            # Compute the KL div of the messages with a standard normal dist.
            s = input.x[input.edge_index[0]]  # sending / source nodes
            r = input.x[input.edge_index[1]]  # recieving / target nodes
            e = torch.cat([s, r], dim=1)

            messages = model.edge_model(e)
            mu = messages[:, ::2]
            logvar = messages[:, 1::2]

            # TODO: Squash logvar to some range.
            kl_reg = torch.sum(0.5 * (mu**2 + torch.exp(logvar) - logvar - 1))

            # Divide by the number of edges in the graph.
            kl_reg /= messages.shape[0]

            # Update the loss.
            total_loss += self.reg_weight * kl_reg

            # create params dict of max min values of all quantities
            params = {"kl_reg": kl_reg, "base_loss": base_loss}

        else:
            params = {}

        return total_loss, params
