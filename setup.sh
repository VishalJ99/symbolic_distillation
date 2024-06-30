#!/bin/bash
# Update or create the base conda environment.
conda env update --file environment.yml

# # Activate the conda environment.
source activate vj279_project_env

# # Store the environment variable TORCH in the shell's environment
export TORCH=$(python -c "import torch; version_nums = torch.__version__.split('.'); version_nums[-1] = '0' + version_nums[-1][1:]; print('.'.join(version_nums))")

# # Use the TORCH variable to install the necessary packages
pip install --upgrade torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install --upgrade torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install --upgrade torch-geometric

# Run a script which imports julia call to install the necessary Julia packages
python -c "import juliacall;"

