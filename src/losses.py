import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from icecream import ic


class LossWithL1MessageReg(nn.Module):
    def __init__(self, square=False, l1_weight=1e-2):
        super(LossWithL1MessageReg, self).__init__()
        self.square = square
        self.l1_weight = l1_weight

    def forward(
        self,
        input: Batch,
        target: torch.tensor,
        model: MessagePassing = None,
    ):
        if self.square:
            base_loss = torch.sum((input.y - target) ** 2)
        else:
            base_loss = torch.sum(torch.abs(input.y - target))

        print("base loss", base_loss)
        # Divide by the number of nodes in the graph.
        base_loss /= input.y.shape[0]
        print("base loss final contribution", base_loss)

        total_loss = base_loss

        if self.l1_weight:
            s = input.x[input.edge_index[0]]  # sending / source nodes
            r = input.x[input.edge_index[1]]  # recieving / target nodes

            # Compute all the messages
            messages = model.message(s, r)
            l1_reg = torch.sum(torch.abs(messages))

            print("summed abs edge msg", l1_reg)

            # Divide by the number of edges in the graph.
            l1_reg /= input.edge_index.shape[1]
            print("reg loss final contribution", self.l1_weight * l1_reg)
            total_loss += self.l1_weight * l1_reg

        print("total loss", total_loss)
        return total_loss
