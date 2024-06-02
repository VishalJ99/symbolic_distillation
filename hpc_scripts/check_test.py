import yaml
import os

# Define the experiments, strategies, and dimensions
experiments = ['spring', 'r1', 'r2', 'charge']
strategies = ['standard', 'l1', 'kl', 'bottleneck']
dimensions = ['2d', '3d']

# Function to generate a config dictionary
def generate_config(experiment, strategy, dim):
    config = {
        'output_dir': f'../rds/hpc-work/test_runs/half_size/{experiment}_{dim}/{strategy}',
        'tqdm': False,
        'save_messages': True,
        'message_save_limit': 100000,
        'data_dir': f'../rds/hpc-work/test_data/{experiment}_{dim}',
        'test_batch_size': 1024,
        'model_weights_dir': f'../rds/hpc-work/train_runs/half_size/{experiment}_{dim}/{strategy}/model_weights',
        'model': 'gnn' if strategy != 'kl' else 'vargnn',
        'model_params': {
            'n_f': (2 * int(dim[0]) + 2),
            'msg_dim': int(dim[0]) if strategy == 'bottleneck' else 3,
            'ndim': int(dim[0]),
            'hidden': 300,
            'aggr': 'add'
        },
        'loss': 'loss+l1reg' if strategy != 'kl' else 'loss+klreg',
        'loss_params': {
            'reg_weight': 1 if strategy == 'kl' else (1e-2 if strategy == 'l1' else 0)
        }
    }
    return config

# Create directories and files
for experiment in experiments:
    for strategy in strategies:
        for dim in dimensions:
            # Set the directory path
            dir_path = f'configs/test_configs/{experiment}'
            os.makedirs(dir_path, exist_ok=True)
            
            # Set the file path
            file_path = f'{dir_path}/{strategy}_{dim}.yaml'
            
            # Generate the config
            config = generate_config(experiment, strategy, dim)
            
            # Write the config to a YAML file
            with open(file_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

print("Configuration files have been created.")
