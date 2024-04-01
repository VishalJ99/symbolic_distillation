from datasets import ParticleDynamicsDataset
from torch_geometric.data import DataLoader
from models import OGN
from utils import get_edge_index, seed_everything, seed_worker
from tqdm import tqdm
import torch
import numpy as np
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from icecream import ic
from accelerate import Accelerator
import wandb


wandb.init(project="symbolic_distillation")
# Set random seed for reproducibility
seed_everything(42)
acclerator = Accelerator()
device = acclerator.device
print(f"Using device: {device}")

# load data
train_set = ParticleDynamicsDataset(
    root="data/particle_dynamics/reduced_spring",
    split="train",
    train_val_test_split=[0.375, 0.125, 0.5],
)
val_set = ParticleDynamicsDataset(
    root="data/particle_dynamics/reduced_spring", split="val"
)
new_val_set = ParticleDynamicsDataset(
    root="data/particle_dynamics/reduced_spring", split="val"
)
print("loaded  data")
ic(len(train_set), len(val_set))

# initialise data loaders
batch = 64
g = torch.Generator()
g.manual_seed(0)

train_loader = DataLoader(
    train_set,
    batch_size=batch,
    shuffle=True,
)

val_loader = DataLoader(val_set, batch_size=1024, shuffle=False)
newtestloader = DataLoader(
    [Data(X_test[i], edge_index=edge_index, y=y_test[i]) for i in test_idxes],
    batch_size=len(X_test),
    shuffle=False,
)
print("loaded data loaders")
ic(len(train_loader), len(val_loader))

# define training params
n_f = 6  # number of node features
msg_dim = 100  # message dimension
dim = 2  # spatial dimensions
hidden = 300  # number of units in hidden layers
aggr = "add"  # aggregation method
n = 4  # number of bodies
sim = "spring"  # simulation type
epoch = 0  # epoch number
total_epochs = 3  # total number of epochs
batch_per_epoch = int(1000 * 10 / (batch / 32.0))  # number of batches per epoch
init_lr = 1e-3  # initial learning rate
test = "_l1_"  # loss function


def new_loss(self, g, augment=False, square=False):
    if square:
        return torch.sum((g.y - self.just_derivative(g, augment=augment)) ** 2)
    else:
        base_loss = torch.sum(
            torch.abs(g.y - self.just_derivative(g, augment=augment))
        )
        if test in ["_l1_", "_kl_"]:
            s1 = g.x[self.edge_index[0]]
            s2 = g.x[self.edge_index[1]]
            m12 = self.message(s1, s2)
            regularization = 1e-2
            # Want one loss value per row of g.y:
            normalized_l05 = torch.sum(torch.abs(m12))
            return (
                base_loss,
                regularization * batch * normalized_l05 / n**2 * n,
            )

        return base_loss


def get_messages(ogn):
    def get_message_info(tmp):
        ogn.cpu()

        s1 = tmp.x[tmp.edge_index[0]]
        s2 = tmp.x[tmp.edge_index[1]]
        tmp = torch.cat([s1, s2], dim=1)  # tmp has shape [E, 2 * in_channels]
        if test == "_kl_":
            raw_msg = ogn.msg_fnc(tmp)
            mu = raw_msg[:, 0::2]
            logvar = raw_msg[:, 1::2]

            m12 = mu
        else:
            m12 = ogn.msg_fnc(tmp)

        all_messages = torch.cat((s1, s2, m12), dim=1)
        if dim == 2:
            columns = [
                elem % (k)
                for k in range(1, 3)
                for elem in "x%d y%d vx%d vy%d q%d m%d".split(" ")
            ]
            columns += ["e%d" % (k,) for k in range(msg_dim)]
        elif dim == 3:
            columns = [
                elem % (k)
                for k in range(1, 3)
                for elem in "x%d y%d z%d vx%d vy%d vz%d q%d m%d".split(" ")
            ]
            columns += ["e%d" % (k,) for k in range(msg_dim)]

        return pd.DataFrame(
            data=all_messages.cpu().detach().numpy(), columns=columns
        )

    msg_info = []
    for i, g in enumerate(newtestloader):
        msg_info.append(get_message_info(g))

    msg_info = pd.concat(msg_info)
    msg_info["dx"] = msg_info.x1 - msg_info.x2
    msg_info["dy"] = msg_info.y1 - msg_info.y2
    if dim == 2:
        msg_info["r"] = np.sqrt((msg_info.dx) ** 2 + (msg_info.dy) ** 2)
    elif dim == 3:
        msg_info["dz"] = msg_info.z1 - msg_info.z2
        msg_info["r"] = np.sqrt(
            (msg_info.dx) ** 2 + (msg_info.dy) ** 2 + (msg_info.dz) ** 2
        )

    return msg_info


# initialise model, optimiser and scheduler
ogn = OGN(
    n_f,
    msg_dim,
    dim,
    dt=0.1,
    hidden=hidden,
    edge_index=get_edge_index(n, sim),
    aggr=aggr,
).to(device)
opt = torch.optim.Adam(ogn.parameters(), lr=init_lr, weight_decay=1e-8)
sched = OneCycleLR(
    opt,
    max_lr=init_lr,
    steps_per_epoch=batch_per_epoch,  # len(trainloader)
    epochs=total_epochs,
    final_div_factor=1e5,
)

# training loop
for epoch in tqdm(range(epoch, total_epochs)):
    total_loss = 0.0
    i = 0
    num_items = 0
    for ginput in train_loader:
        opt.zero_grad()
        ginput.x = ginput.x.to(device)
        ginput.y = ginput.y.to(device)
        ginput.edge_index = ginput.edge_index.to(device)
        ginput.batch = ginput.batch.to(device)
        loss, reg = new_loss(ogn, ginput, square=False)
        ((loss + reg) / int(ginput.batch[-1] + 1)).backward()
        opt.step()
        # sched.step()

        total_loss += loss.item()
        i += 1
        num_items += int(ginput.batch[-1] + 1)

    cur_loss = total_loss / num_items
    print(cur_loss)

    cur_loss = total_loss / num_items
    print(cur_loss)
    cur_msgs = get_messages(ogn)
    cur_msgs["epoch"] = epoch
    cur_msgs["loss"] = cur_loss
    messages_over_time.append(cur_msgs)

    ogn.cpu()
    from copy import deepcopy as copy

    recorded_models.append(ogn.state_dict())
