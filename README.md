# Discovering Symbolic Models from Deep Learning with Inductive Biases - A Reproducibility Project

This project is aimed at reproducing the work presented in [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287), as part of the Data Intensive Science MPhil at the University of Cambridge.

## Pre-requisites
Ensure conda is installed on your system. If not, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Note, the docker commands do not need conda to be installed on your system. However, do require docker to be installed. If docker is not installed, follow the instructions [here](https://docs.docker.com/get-docker/). Also commands run in docker are not configured to access the GPU, so the training time will be significantly longer. It is advised to run the training and testing scripts locally in the conda environment if the local hardware has a GPU.


## Set up
```
# Clone the repository.
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/projects/vj279.git

# Set up the environment locally.
cd vj279
source setup.sh

# Build the docker image.
docker build -t vj279_project .
```
## Scripts
The following scripts are available in the `src` directory:
- `simulations/run_sims.py`:

    Runs a simulation and saves the data to a specified directory.

    Usage:
    ```
    python simulations/run_sims.py <sim> <output_dir> [--dim] [--nt] [--ns] [--n] [--seed]
    ```
    Arguments:
    - `sim`: The force law used in the simulation for example `spring`. See `simuations/simulate.py` or `sim_sets` dictionary for valid options.
    - `output_dir`: Path to the output directory where the data will be saved.
    - `--n`: The number of bodies in the simulation. Default is 4.
    - `--dim`: The dimensionality of the simulation. Default is 2.
    - `--nt`: The number of time steps in the simulation. Default is 1000.
    - `--ns`: The number of simulations to run. Default is 10000.
    - `--seed`: The seed for the random number generator. Default is 1.

    The output of the script will be two npy files saved in the output directory. The first data file will contain the positions, velocities, charges and mass of the particles of all the simulations at each time step and the second file will contain the corresponding accelerations. The data file will be of shape (ns, nt, n, 2*dim+2) and the acceleration file will be of shape (ns, nt, n, dim).

    Note, this is just a wrapper script to call `simulations/simulate.py` with the specified arguments. The `simulate.py` script can be used directly to run a single simulation, it is the same code as was used to generate the data for the experiments in the original paper and is taken from the [original repository](https://github.com/MilesCranmer/symbolic_deep_learning/blob/master/simulate.py)

- `src/train.py`: 

    Trains a message passing neural network.

    Usage:
    ```
    python src/train.py <config_file>
    ```
    Arguments:
    - `config_file`: Path to the config file, for example `configs/hello_world/train_config.yaml`. The config file specifies the dataset, model, training strategy, and other hyperparameters. 

    The output of the train script will be a directory of the model weights, a copy of the config with the git hash and wandb run url added to it. If `save_messages` is set to true in the config file, the edge messages will also be saved to a subdirectory `train_messages` in the output directory. The messages will be saved according to the value specified for the `save_every_n_epochs`, which also determines the number of epochs between saving the model weights. The `save_message_limit` decides the number of messages to save. The best model weights are saved as `best_model.pt` in the model weights subdirectory of the output directory.

- `src/test.py`:

    Tests a trained message passing neural network.

    Usage:
    ```
    python src/test.py <config_file>
    ```
    Arguments:
    - `config_file`: Path to the config file, for example `configs/hello_world/test_config.yaml`. The config file specifies the dataset, model, loss and other hyperparameters. 

    The output of the test script will be a directory containing the summary statistics of the models loss on the test set, a copy of the test config with the git hash added to it. Further, a sub directory called `symbolic_regression_csvs` will be created if the `save_messages` parameter is set to true in the config file. This directory will contain the edge messages and node accelerations for the test set for a sample of the data. The number of samples used is determed by the `message_save_limit` which sets the number of edges after which to stop saving messages. This also sets the limit for the number of samples the node accelerations are saved for.

- `src/eval_msgs.py`:
    Evaluates the edge model of a trained message passing neural network.
    Quantifies edge message sparsity, calculates the R2 scores between the edge messages and a linear transformation of the true underlying force law. Also distills the edge model into a symbolic representation.

    Usage:
    ```
    python src/eval_msgs.py <input_csv> <output_dir> <sim> [--samples] [--eps] [--no_sr]
    ```
    Arguments:
    - `edge_message_csv`: Path to the edge messages csv file created by the test script.
    - `output_dir`: Path to the output directory where the results will be saved.
    - `sim` : The force law used in the simulation for example `spring`. See `utils.force_factory` for valid options.
    - `--samples`: Number of samples to use for fitting the symbolic model.
    - `--eps`: The epsilon value to add to the distance between particles when calculating the forces. This is used to prevent division by zero. Default is 1e-2.
    - `--no_sr`: Skip the symbolic regression step.


    The output of the eval_msgs script will be a directory containing:
    1) `messages_vs_transformed_force.png` plot showing the scatter plot of the edge message components and the transformed force law.
    2) `R2_stats.txt` containing the R2 scores for each edge message component against the closest linear transformation of the true force law.
    3) `sparsity_plot.png` plot showing visualising the sparsity of the top 15 most important edge messages components. Shows the fraction of the total standard deviation each component contains.
    4) `top_msgs_std.txt` txt file containing the total fraction of the standard deviation contained in dim most important edge messages components. Where dim is the dimensionality of the problem, determined by the columns in the `input_csv` file.
    5) `symbolic_edge.pkl` pickle file of a dictionary with keys: model, var_names and important_msg_idxs. The model is the pysr symbolic regression model, var_names is the names of the variables used in the symbolic regression and important_msg_idxs is the indices of the most important edge messages.
    6) `nn_msg_symbolic_msg_diff.json` json file containing the summary statistics of the difference between the edge messages and the output of the best symbolic model. 
    7) `nn_msg_symbolic_msg_diff.png` plot showing the scatter of the the edge message components and the output components of the best symbolic model.

- `src/eval_node_model.py`:
    Symbolically distills the node model of a trained message passing neural network, visualises the correlation between the node accelerations and the symbolic model output and outputs the summary statistics of the difference between the node accelerations and the symbolic model output.

    Usage:
    ```
    python src/eval_node_model.py <edge_message_csv> <node_output_csv> <output_dir> [--samples]
    ```
    Arguments:
    - `edge_message_csv`: Path to the edge messages csv file created by the test script.
    - `node_output_csv`: Path to the node accelerations csv file created by the test script.
    - `output_dir`: Path to the output directory where the results will be saved.
    - `--samples`: Number of samples to use for fitting the symbolic model.


    The output of the eval_node_model script will be a directory containing:
    1) `nn_a_vs_symbolic.png` plot showing the scatter plot of the node accelerations and the output of the symbolic model.
    2) `nn_a_vs_symbolic_a_diff.json` json file containing the summary statistics of the difference between the node accelerations and the output of the symbolic model.
    3) `symbolic_node.pkl` pickle file of a dictionary with keys: model, var_names, important_msg_idxs. The model is the pysr symbolic regression model and var_names is the names of the variables used in the symbolic regression, important_msg_idxs is the indices of the most important components of the aggregated edge message.

- `src/view_symbolic_eq.py`:
    Displays the equation of the symbolic equation discovered by the symbolic regression model using the variable names as opposed to the standard sympy representation.

    Usage:
    ```
    python src/view_symbolic_eq.py <symbolic_eq_pkl>
    ```
    Arguments:
    - `symbolic_eq_pkl`: Path to the symbolic equation pickle file created by the eval_msgs or eval_node_model script.

    Logs to the console the symbolic equation in a more human readable format.

- `src/create_msg_r2_sparsity_gif.py`
    Creates a gif of the sparsity of the edge messages or a gif of the scatter plot of the edge messages and the transformed force law from the saved edge messages csv files.

    Usage:
    ```
    python src/create_msg_r2_sparsity_gif.py <edge_message_dir> <output_dir> <sim> [--plot_sparsity] [--delete_frames] [--eps]
    ```
    Arguments:
    - `edge_message_dir`: Path to the directory containing the edge messages csv files created by the train script.
    - `output_dir`: Path to the output directory where the gif will be saved.
    - `sim` : The force law used in the simulation for example `spring`. See `utils.force_factory` for valid options.
    - `--plot_sparsity`: If set, the gif will be of the sparsity of the edge messages, otherwise it will be of the scatter plot of the edge messages and the transformed force law.
    - `--delete_frames`: If set, the individual frames of the gif will be deleted after the gif is created.
    - `--eps`: The epsilon value to add to the distance between particles when calculating the forces. This is used to prevent division by zero. Default is 1e-2.

    Useful for visualising the evolution of the edge messages during training.




## Hello World Example
For an outline of this project please read the report in the `report` directory. 
The following is an example of how to generate the results for a single simulation and training strategy.
Specifically, we will train a model on a tiny spring 2d data under the L1 training strategy and distill it. The same pipeline can easily be used for other datasets and training strategies.

### Generate Data
First the data must be generated. To avoid exessive computation time, a very small dataset is generated here, however the full dataset can be generated by changing the `--nt` to a larger value (1000 in the original paper).



Generate an example dataset:
```
docker run -it -v $(pwd):/app vj279_project /bin/bash -c "source activate vj279_project_env && python simulations/run_sims.py spring data/spring_2d/train/raw --dim 2 --nt 10 --ns 10000 --n 4 --seed 1"
docker run -it -v $(pwd):/app vj279_project /bin/bash -c "source activate vj279_project_env && python simulations/run_sims.py spring data/spring_2d/val/raw --dim 2 --nt 2 --ns 10000 --n 4 --seed 2"
docker run -it -v $(pwd):/app vj279_project /bin/bash -c "source activate vj279_project_env && python simulations/run_sims.py spring data/spring_2d/test/raw --dim 2 --nt 10 --ns 10000 --n 4 --seed 3"
```

### Train Model
Next, the model is trained for 30 epochs (100-200 epochs is recommended). This can be done by running the following command:

For CPU training (60-90 mins - docker version currently does not support GPU training):
```
docker run -it -v $(pwd):/app vj279_project /bin/bash -c "source activate vj279_project_env && python src/train.py configs/hello_world/train_config.yaml"
```
Alternatively, to train the model locally and make use of accelerated training with a GPU (<5 mins) or MPS (10-20 mins):
```
# Make sure the conda environment is activated.
python src/train.py configs/hello_world/train_config.yaml
```

### Visualise Training Progress
To visualise the training progress over the epochs, one can create a gif of the sparsity of the edge messages and the scatter plot of the edge messages and the transformed force law. This can be done by running the following command:

```
docker run -it -v $(pwd):/app vj279_project /bin/bash -c "source activate vj279_project_env && python src/create_msg_r2_sparsity_gif.py train_runs/spring_2d_l1/train_messages train_runs/spring_2d_l1/gif spring --plot_sparsity --delete_frames"
docker run -it -v $(pwd):/app vj279_project /bin/bash -c "source activate vj279_project_env && python src/create_msg_r2_sparsity_gif.py train_runs/spring_2d_l1/train_messages train_runs/spring_2d_l1/gif spring --delete_frames"
```

The gifs will be saved in the `train_runs/spring_2d_l1/gif` directory.

### Test Model
Next test the model, or use the pretrained model in the `pretrained_models` directory by using the config file `configs/hello_world/test_config_pretrained.yaml`. 

Note: The first test run will take longer as julia needs to be compiled, subsequent runs will be faster.


Docker:
```
docker run -it -v $(pwd):/app vj279_project /bin/bash -c "source activate vj279_project_env && python src/test.py configs/hello_world/test_config.yaml"
```

Local:
```
# Make sure the conda environment is activated.
python src/test.py configs/hello_world/test_config.yaml
```

### Distill the Graph Neural Network
Distill the edge model:

```
docker run -it -v $(pwd):/app vj279_project /bin/bash -c "source activate vj279_project_env && python src/eval_msgs.py test_runs/spring_2d_l1/symbolic_regression_csvs/edge_messages.csv  test_runs/spring_2d_l1/msg_eval spring --samples 1000"
```

Distill the node model:
```
docker run -it -v $(pwd):/app vj279_project /bin/bash -c "source activate vj279_project_env && python src/eval_node_model.py test_runs/spring_2d_l1/symbolic_regression_csvs/edge_messages.csv test_runs/spring_2d_l1/symbolic_regression_csvs/node_accels.csv test_runs/spring_2d_l1/node_eval --samples 1000"
```

Visualise the discovered equations:
```
docker run -it -v $(pwd):/app vj279_project /bin/bash -c "source activate vj279_project_env && python src/view_symbolic_eq.py test_runs/spring_2d_l1/msg_eval/symbolic_edge.pkl"
docker run -it -v $(pwd):/app vj279_project /bin/bash -c "source activate vj279_project_env && python src/view_symbolic_eq.py test_runs/spring_2d_l1/node_eval/symbolic_node.pkl"
```

### Evaluate the Distilled Model
Finally, evaluate the distilled model:

Local:
```
# Make sure the conda environment is activated.
python src/test.py configs/hello_world/test_symbolic_config.yaml
```

Docker:
```
docker run -it -v $(pwd):/app vj279_project /bin/bash -c "source activate vj279_project_env && python src/test.py configs/hello_world/test_symbolic_config.yaml"
```



## Running the full pipeline for other experiments.

To run the experiments described in the report, simply repeat the steps above and create full sized datasets as described in the report. Currently, the following arguments for the `sim` parameter are supported: `spring`, `r1`, `r2` and `charge`. The report used the argument for the seed parameter as follows: `seed=1` for the training set, `seed=2` for the validation set and `seed=3` for the test set. 

It is best to use the commands above to generate the data to guarantee the correct structure and naming. Simply change the arguments to the `run_sims.py` script to generate the desired dataset, however as long as the following structure is maintained, the training and testing scripts will work with any dataset.

### Dataset structure
The train and testing scripts expect the data dir passed to follow a specific structure and naming convention shown below. The data directory should contain subdirectories for the train, val and test sets. Each of these directories should contain a `raw` directory which contains the data and acceleration files. The data files should be named in the following format:

```
data
└── spring_2d
    └── train
    |    └── raw
    |        ├── sim=r2_ns=2500_seed=2_n_body=8_dim=3_nt=500_dt=1e-03_data.npy
    |        ├── sim=r2_ns=2500_seed=2_n_body=8_dim=3_nt=500_dt=1e-03_accel_data.npy
    └── val
    |    └── raw
    |        ...
    └── test
        └── raw
            ...
```

The *_data.npy file contains a 4D array of the data, with the first dimension being the number of simulations, the second being the number of time steps, and the third being the number of bodies and the last dimension containing the features of the bodies. The *_accel_data.npy file contains the acceleration data for the same simulations and is of the same shape, except the last dimension which will be 2 or 3 depending on the dimension of the data.

### Train and Test Configs
Depending on which strategy one wishes to train a model under `standard`, `bottleneck`, `l1` or `kl`, for 2 or 3 dimensions, the premade config files from `config/template_train_configs` and `config/template_test_configs` can be used. Simply fill in the paths to the datasets, the output paths and the model weight paths where indicated.
