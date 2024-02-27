import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from utils import get_edge_index


class ParticleDynamicsDataset(InMemoryDataset):
    """
    TODO:
    - Figure out if seperate split transforms are needed
    - Handle scope of random seed setting for reproducibility
    - Add docstrings
    """

    def __init__(
        self,
        root,
        split,
        transform=None,
        pre_transform=None,
        train_val_test_split=[0.8, 0.1, 0.1],
    ):
        self.root = root
        self.split = split
        self.train_val_test_split = train_val_test_split
        super(ParticleDynamicsDataset, self).__init__(
            root, transform, pre_transform
        )

        # Load the appropriate split based on 'split' attribute.
        path = self.processed_paths[{"train": 0, "val": 1, "test": 2}[split]]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        # Dummy names, actual logic to identify files is in `process`.
        return ["pos_vel_charge_mass.npy", "acceleration.npy"]

    @property
    def processed_file_names(self):
        # Names of the processed files.
        return ["train.pt", "val.pt", "test.pt"]

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

        # Reshape the tensors from n_sims, timesteps , n_bodies, feats
        # to n_sims*timesteps (batch), n_bodies (nodes), feats
        # NOTE: Lose time series nature of the data at this step.
        # NOTE: Old method of reshaping the tensor - replace with view.

        src_array = np.concatenate(
            [src_array[:, i] for i in range(0, src_array.shape[1], 1)]
        )

        # Convert to PyTorch tensors.
        t = torch.from_numpy(src_array)

        # TODO: Figure out if this reshaping works fine too.
        # t = t.view(
        #     -1,
        #     t.shape[-2],
        #     t.shape[-1],
        # )

        return t

    def process(self):
        # Load data from the raw files.
        for f in os.listdir(self.raw_dir):
            if "accel" in f:
                accel_fpath = os.path.join(self.raw_dir, f)
            else:
                pos_vel_charge_mass_fpath = os.path.join(self.raw_dir, f)

        # Get sim name from the file name.
        sim = f.split("_")[0].split("=")[1]

        pos_vel_charge_mass_tensor = self._load_data(pos_vel_charge_mass_fpath)
        acceleration_tensor = self._load_data(accel_fpath)

        # Get edge indices.
        edge_index = get_edge_index(pos_vel_charge_mass_tensor.shape[0], sim)

        # Wrap data in PyG Data objects.
        data_list = [
            Data(
                x=pos_vel_charge_mass_tensor[i],
                y=acceleration_tensor[i],
                edge_index=edge_index,
            )
            for i in range(pos_vel_charge_mass_tensor.size(0))
        ]

        # Optionally apply pre_transforms here.
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Randomly shuffle data_list. TODO: Handle seed for reproducibility!
        data_list = [data_list[i] for i in torch.randperm(len(data_list))]

        # Calculate split sizes.
        train_size = int(len(data_list) * self.train_val_test_split[0])
        val_size = int(len(data_list) * self.train_val_test_split[1])
        test_size = len(data_list) - train_size - val_size

        # Split data.
        train_data, remaining_data = torch.utils.data.random_split(
            data_list, [train_size, len(data_list) - train_size]
        )
        val_data, test_data = torch.utils.data.random_split(
            remaining_data, [val_size, test_size]
        )

        # Save the splits
        self._save_split(
            [data_list[i] for i in train_data.indices], self.processed_paths[0]
        )
        self._save_split(
            [data_list[i] for i in val_data.indices], self.processed_paths[1]
        )
        self._save_split(
            [data_list[i] for i in test_data.indices], self.processed_paths[2]
        )

    def _save_split(self, split_data, path):
        data, slices = self.collate([data for data in split_data])
        torch.save((data, slices), path)
