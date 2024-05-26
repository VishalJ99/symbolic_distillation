import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing
from sklearn.model_selection import train_test_split
from models import OGN, varOGN
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
import pandas as pd
import pickle as pkl
from copy import deepcopy as copy
from accelerate import Accelerator

accelerate = Accelerator()
device = accelerate.device


# Custom funcs
def get_edge_index(n, sim):
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


def new_loss(self, g, augment=True, square=False):
    if square:
        return torch.sum((g.y - self.just_derivative(g, augment=augment)) ** 2)
    else:
        base_loss = torch.sum(
            torch.abs(g.y - self.just_derivative(g, augment=augment))
        )
        if test in ["_l1_", "_kl_"]:
            s1 = g.x[self.edge_index[0]]
            s2 = g.x[self.edge_index[1]]
            if test == "_l1_":
                m12 = self.message(s1, s2)
                regularization = 1e-2
                # Want one loss value per row of g.y:
                normalized_l05 = torch.sum(torch.abs(m12))
                return (
                    base_loss,
                    regularization * batch * normalized_l05 / n**2 * n,
                )
            elif test == "_kl_":
                regularization = 1
                # Want one loss value per row of g.y:
                tmp = torch.cat(
                    [s1, s2], dim=1
                )  # tmp has shape [E, 2 * in_channels]
                raw_msg = self.msg_fnc(tmp)
                mu = raw_msg[:, 0::2]
                logvar = raw_msg[:, 1::2]
                full_kl = torch.sum(torch.exp(logvar) + mu**2 - logvar) / 2.0
                return base_loss, regularization * batch * full_kl / n**2 * n
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


# Load the data - change this to data of the right shape.
data = np.load("simulations/data.npy")
accel_data = np.load("simulations/accel_data.npy")

# Hyper params
# ---------------
sim = "spring"
aggr = "add"
hidden = 300
test = "_kl_"
msg_dim = 100
n_f = data.shape[3]
n = data.shape[2]
init_lr = 1e-3
total_epochs = 100
dim = 2


# Split the data into train and val.
X = torch.from_numpy(
    np.concatenate([data[:, i] for i in range(0, data.shape[1], 1)])
)
y = torch.from_numpy(
    np.concatenate([accel_data[:, i] for i in range(0, data.shape[1], 1)])
)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
edge_index = get_edge_index(n, "r2")  # doesnt matter unless string or ball.

batch = int(64 * (4 / n) ** 2)
batch_per_epoch = int(1000 * 10 / (batch / 32.0))

trainloader = DataLoader(
    [
        Data(X_train[i], edge_index=edge_index, y=y_train[i])
        for i in range(len(y_train))
    ],
    batch_size=batch,
    shuffle=True,
)

testloader = DataLoader(
    [
        Data(X_test[i], edge_index=edge_index, y=y_test[i])
        for i in range(len(y_test))
    ],
    batch_size=1024,
    shuffle=True,
)

# Record messages over test dataset here:
import numpy as onp

onp.random.seed(0)
test_idxes = onp.random.randint(0, len(X_test), 1000)
newtestloader = DataLoader(
    [Data(X_test[i], edge_index=edge_index, y=y_test[i]) for i in test_idxes],
    batch_size=len(X_test),
    shuffle=False,
)

if test == "_kl_":
    ogn = varOGN(
        n_f,
        msg_dim,
        dim,
        dt=0.1,
        hidden=hidden,
        edge_index=get_edge_index(n, sim),
        aggr=aggr,
    ).to(device)
else:
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
    steps_per_epoch=len(trainloader),  # len(trainloader),
    epochs=total_epochs,
    final_div_factor=1e5,
)

recorded_models = []
messages_over_time = []
print("using device ", device)
for epoch in range(1, total_epochs + 1):
    ogn.to(device)
    total_loss = 0.0
    i = 0
    num_items = 0
    for ginput in tqdm(trainloader):
        opt.zero_grad()
        ginput.x = ginput.x.to(device)
        ginput.y = ginput.y.to(device)
        ginput.edge_index = ginput.edge_index.to(device)
        ginput.batch = ginput.batch.to(device)
        if test in ["_l1_", "_kl_"]:
            loss, reg = new_loss(ogn, ginput, square=False)
            ((loss + reg) / int(ginput.batch[-1] + 1)).backward()
        else:
            loss = ogn.loss(ginput, square=False)
            (loss / int(ginput.batch[-1] + 1)).backward()
        opt.step()
        sched.step()

        total_loss += loss.item()
        i += 1
        num_items += int(ginput.batch[-1] + 1)
    cur_loss = total_loss / num_items
    print(cur_loss)
    cur_msgs = get_messages(ogn)
    cur_msgs["epoch"] = epoch
    cur_msgs["loss"] = cur_loss
    messages_over_time.append(cur_msgs)

    ogn.cpu()
    recorded_models.append(ogn.state_dict())

    pkl.dump(messages_over_time, open("../rds/hpc-work/kl_messages_over_time.pkl", "wb"))

    pkl.dump(recorded_models, open("../rds/hpc-work/kl_models_over_time.pkl", "wb"))
