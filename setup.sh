#!/bin/bash
# Update or create the base conda environment.
conda env update --file environment.yml

# Activate the conda environment.
source activate vj279_project_env

# Set the PyTorch version and the TORCH environment variable for compatibility with PyTorch Geometric
python -c "import torch; import os; version_nums = torch.__version__.split('.'); version_nums[-1] = '0' + version_nums[-1][1:]; torch_version = '.'.join(version_nums); os.environ['TORCH'] = torch_version; print('TORCH version set for PyTorch Geometric:', os.environ['TORCH'])"

# Store the environment variable TORCH in the shell's environment
export TORCH=$(python -c "import os; print(os.environ['TORCH'])")

# Use the TORCH variable to install the necessary packages
pip install --upgrade torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install --upgrade torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install --upgrade torch-geometric
