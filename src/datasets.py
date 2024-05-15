import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from utils import get_edge_index


class ParticleDynamicsDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
    ):
        self.root = root
        super(ParticleDynamicsDataset, self).__init__(
            root, transform, pre_transform
        )
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        # Placeholders for the raw files names, logic to find files in process.
        return ["X.npy", "y.npy"]

    @property
    def processed_file_names(self):
        return ["processed.pt"]

    def download(self):
        # Assuming data is locally available, so download is not required.
        pass

    def _load_data(self, fpath):
        """
        Loads a npy file into a PyTorch tensor of the appropriate shape
        to input into the PyG Data object.
        Merges the simulation and timestep dimensions into a single batch.
        """
        # Load the data.
        src_array = np.load(fpath)

        # Reshape the tensors to combine sim and time dimensions into
        # a single batch dimension.
        src_array = np.concatenate(
            [src_array[:, i] for i in range(0, src_array.shape[1], 1)]
        )

        # Convert to PyTorch tensors.
        t = torch.from_numpy(src_array)

        return t

    def process(self):
        # NOTE assumes file name follows format used by `run_sims.py`.
        # Load data from the raw files.
        for f in os.listdir(self.raw_dir):
            if "accel" in f:
                accel_fpath = os.path.join(self.raw_dir, f)
            else:
                pos_vel_charge_mass_fpath = os.path.join(self.raw_dir, f)

        # Load the data.
        pos_vel_charge_mass_t = self._load_data(pos_vel_charge_mass_fpath)
        acceleration_t = self._load_data(accel_fpath)

        # Get edge indices.
        edge_index = get_edge_index(f)

        data_list = [
            Data(
                x=pos_vel_charge_mass_t[k],
                y=acceleration_t[k],
                edge_index=edge_index,
            )
            for k in range(pos_vel_charge_mass_t.size(0))
        ]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save the splits.
        data, slices = self.collate([data for data in data_list])
        torch.save((data, slices), self.processed_paths[0])
