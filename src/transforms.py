"""
Script defining custom transforms for PyTorch Geometric data objects.
To add a custom transform, define a class here with the following signature:
```
class CustomTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super(CustomTransform, self).__init__()
        # Define any parameters here.
        self.param = param

    def forward(self, data_obj: Data) -> Data:
        # Define the transformation here.
        return transformed_data_object

After defining the transform, add it to the `transform_factory` function in
`src/utils.py` to be able to specify it in the configs and use it in the
training and testing scripts.
"""
from typing import Tuple
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RandomTranslate(BaseTransform):
    """
    TODO: Update so it applies a different translation to nodes for each graph
    in the batch. Currently applies the same random translation to all graphs in
    a batch.

    Applies a random translation to node features along specified dimensions.
    The translation offsets for each dimension are sampled from a normal dist
    centered at zero with a standard deviation defined by the 'scale' parameter.

    Code modified from:
    https://github.com/MilesCranmer/symbolic_deep_learning/blob/master/models.py
    """

    def __init__(self, scale: float, dims: Tuple[int, ...] = (0, 1)) -> None:
        """
        Sets the scale of the random translation and the dimensions to be
        translated.

        Parameters:
        ----------
        scale: float
            The standard deviation of the normal distribution used to sample
            the translation offsets.

        dims: Tuple[int, ...]
            The dimensions of the node features to be translated.
        """
        self.scale = scale
        self.dims = dims

    def forward(self, data: Data) -> Data:
        """
        Applies the random translation transformation to the node features.

        Parameters:
        ----------
        data: Data
            The graph data object containing the node features to be translated.

        Returns:
        -------
        Data
            The transformed graph data object.
        """
        x = data.x

        # Sample the random translation vector.
        translation = torch.randn(1, len(self.dims)) * self.scale

        # Repeat the translation vector for each node in the graph.
        translation = translation.repeat(len(x), 1).to(x.device)

        # Apply the translation to the specified node feature dimensions.
        x = x.index_add(1, torch.tensor(self.dims), translation)

        # Update the node features in the data object.
        data.x = x

        return data
