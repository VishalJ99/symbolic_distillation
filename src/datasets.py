import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from utils import get_train_val_test_split, get_edge_index
from sklearn.model_selection import train_test_split
from icecream import ic


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
        src_array = np.concatenate(
            [src_array[:, i] for i in range(0, src_array.shape[1], 1)]
        )

        # Convert to PyTorch tensors.
        t = torch.from_numpy(src_array)

        return t

    def process(self):
        # Load data from the raw files.
        for f in os.listdir(self.raw_dir):
            if "accel" in f:
                accel_fpath = os.path.join(self.raw_dir, f)
            else:
                pos_vel_charge_mass_fpath = os.path.join(self.raw_dir, f)

        # Get sim name from the file name.
        # NOTE assumes file name follows format used by `simulations/run_sims.py`.
        sim = f.split("_")[0].split("=")[1]

        # Load the data.
        pos_vel_charge_mass_t = self._load_data(pos_vel_charge_mass_fpath)
        acceleration_t = self._load_data(accel_fpath)

        # Get edge indices.
        edge_index = get_edge_index(pos_vel_charge_mass_t.shape[-2], sim)

        # Split the data into train, val and test sets.
        split_tuples = get_train_val_test_split(
            pos_vel_charge_mass_t,
            acceleration_t,
            self.train_val_test_split,
            shuffle=False,
            seed=42,  # make this a parameter
        )

        # Wrap data in PyG Data objects for each split.
        split_data_lists = []
        for i in range(3):
            X_split, Y_split = split_tuples[i][0], split_tuples[i][1]
            data_list = [
                Data(
                    x=X_split[k],
                    y=Y_split[k],
                    edge_index=edge_index,
                )
                for k in range(X_split.size(0))
            ]
            split_data_lists.append(data_list)

        for idx, data_list in enumerate(split_data_lists):
            # Optionally apply pre_transforms here.
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            # Save the splits.
            self._save_split(data_list, self.processed_paths[idx])

    def _save_split(self, split_data, path):
        data, slices = self.collate([data for data in split_data])
        torch.save((data, slices), path)
