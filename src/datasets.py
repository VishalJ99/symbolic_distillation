import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.dataset import _repr, files_exist

from utils import get_edge_index
import sys
import warnings
import shutil


class ParticleDynamicsDataset(InMemoryDataset):
    def __init__(
        self, root, transform=None, pre_transform=None, prune_outliers=False
    ):
        # These must be set before calling the parent class constructor.
        self.root = root
        self.prune_outliers = prune_outliers

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

    def _process(self):
        """
        Over write torch_geometric.data.dataset.Dataset class method.
        Now checks if prune_outliers attribute has been changed since processing
        the data and if so, deletes the processed data and calls process method.
        """
        f = os.path.join(self.processed_dir, "pre_transform.pt")
        if os.path.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first"
            )

        f = os.path.join(self.processed_dir, "pre_filter.pt")
        if os.path.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, make sure to "
                "delete '{self.processed_dir}' first"
            )

        # Check if prune_outliers float is different to current attribute.
        f = os.path.join(self.processed_dir, "prune_outliers.pt")
        if os.path.exists(f) and torch.load(f) != self.prune_outliers:
            warnings.warn(
                "The `prune_outliers` argument differs from the one used to "
                "create the processed version of this dataset. "
                f"deleting {self.processed_dir} and re processing..."
            )
            shutil.rmtree(self.processed_dir)

        if files_exist(self.processed_paths):  # pragma: no cover
            return

        if self.log and "pytest" not in sys.modules:
            print("Processing...", file=sys.stderr)

        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        # Save arguments relevant to generating the processed data.
        path = os.path.join(self.processed_dir, "pre_transform.pt")
        torch.save(_repr(self.pre_transform), path)
        path = os.path.join(self.processed_dir, "pre_filter.pt")
        torch.save(_repr(self.pre_filter), path)
        path = os.path.join(self.processed_dir, "prune_outliers.pt")
        torch.save(self.prune_outliers, path)

        if self.log and "pytest" not in sys.modules:
            print("Done!", file=sys.stderr)

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

    def _prune_outlier_graphs(self, x, y):
        """
        Prunes the outliers from the data.
        """
        dim = (x.shape[-1] - 2) // 2
        velocities = x[:, :, dim:-2]
        accelerations = y

        # Define upper and lower vel and accel bounds for pruning.
        upper_val_v = np.percentile(velocities, self.prune_outliers)
        lower_val_v = np.percentile(velocities, 100 - self.prune_outliers)
        upper_val_a = np.percentile(accelerations, self.prune_outliers)
        lower_val_a = np.percentile(accelerations, 100 - self.prune_outliers)

        # Generate masks using bounds.
        velocity_mask = (velocities < upper_val_v) & (velocities > lower_val_v)
        accel_mask = (accelerations < upper_val_a) & (
            accelerations > lower_val_a
        )

        # Combine masks.
        mask = velocity_mask & accel_mask

        # Use all() over last two dims to keep graphs where all nodes are valid.
        mask = mask.all(axis=(-2, -1))

        # Prune data.
        x = x[mask]
        y = y[mask]

        # Calculate the percentage of graphs pruned.
        outlier_percent = 100 * (1 - sum(mask) / mask.size(0))
        return x, y, outlier_percent

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

        # Prune outliers if necessary.
        if self.prune_outliers:
            (
                pos_vel_charge_mass_t,
                acceleration_t,
                outlier_percent,
            ) = self._prune_outlier_graphs(
                pos_vel_charge_mass_t, acceleration_t
            )
            # Insert comment about why this is here.
            processed_prune_path = os.path.join(
                self.processed_dir, "prune_outliers.pt"
            )
            torch.save(self.prune_outliers, processed_prune_path)
            print(
                f"[INFO] Pruned {outlier_percent: .2f}% of graphs "
                f"in {self.raw_dir}"
            )

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
            # Get rid of None values returned by ThresholdOutliers transform.
            data_list = [data for data in data_list if data is not None]

        # Save the splits.
        data, slices = self.collate([data for data in data_list])
        torch.save((data, slices), self.processed_paths[0])
