from typing import Tuple
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

# TODO: Add random rotation as force laws based on distance between particles
# are also invariant wrt to rotations as well as translations.


class RandomTranslate(BaseTransform):
    """
    TODO: Update so it applies a different translation to nodes for each graph
    in the batch. Currently applies the same random translation to all graphs in
    a batch.

    Applies a random translation to node features along specified dimensions.
    The translation offsets for each dimension are sampled from a normal dist
    centered at zero with a standard deviation defined by the 'scale' parameter.

    Attributes:
        scale (float):  The standard deviation of the normal distribution from
                        which the translation offsets are sampled.

        dims (Tuple[int, int]): The dimensions (indices of features) along which
                                the translation is applied.
                                Default is (0, 1) for 2D spatial data.
    """

    def __init__(self, scale: float, dims: Tuple[int, int] = (0, 1)) -> None:
        """
        Sets the class attributes for the random translation transformation.

        Parameters:
        ----------
            scale (float):  The standard deviation of the normal distribution
                            from which the translation offsets are sampled.

            dims (Tuple[int, int]): The dimensions (indices of features) along
                                    which the translation is applied.
                                    Default is (0, 1) for 2D spatial data.
        """
        self.scale = scale
        self.dims = dims

    def forward(self, data: Data) -> Data:
        """
        Applies the random translation transformation to the node features.
        Assumes that dims correctly indexes the spatial dimensions of the
        node features.

        Parameters:
            data (Data): A graph data object from PyTorch Geometric.
            with node features as the 'x' attribute.


        Returns:
            Data: The modified graph data object with translated node features.
        """
        x = data.x

        # Sample the random translation vector.
        translation = torch.randn(1, len(self.dims)) * self.scale

        # Repeat the translation vector for each node in the graph.
        translation = translation.repeat(len(x), 1).to(x.device)

        # Apply the translation to the specified node feature dimensions.
        x = x.index_add(1, torch.tensor(self.dims), translation)

        # Update the node features in the data object
        data.x = x

        return data
