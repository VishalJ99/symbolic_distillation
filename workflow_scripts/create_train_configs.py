import yaml
import os

# Define the experiments, strategies, and dimensions
experiments = ['spring', 'r1', 'r2', 'charge']
strategies = ['standard', 'l1', 'kl', 'bottleneck']
dimensions = ['2d', '3d']

# Hyperparameters and settings
seed = 42
wandb = True
wandb_project = 'mphil_project'
save_messages = True
tqdm = False
epochs = 100
train_batch_size = 64
val_batch_size = 1024
lr = 0.001
weight_decay = 1e-8
save_every_n_epochs = 5
max_lr = 0.001
final_div_factor = 1e5
translate_scale = 3

l1_weight = 1.0e-2
kl_weight = 1

# Base directory paths
base_output_dir = '../rds/hpc-work/train_runs/half_size'
base_data_dir = '../rds/hpc-work/test_data'

# Function to create a config dictionary
def create_config(experiment, strategy, dim):
    
    num_features = 4 if dim == '2d' else 6
    msg_dim = 100 if strategy != 'bottleneck' else int(dim[0]) # Extract '2' or '3' from '2d' or '3d'
    node_dim = 2*int(dim[0]) + 2  
    hidden_size = 300
    aggr_method = 'add'
    model_type = 'vargnn' if strategy == 'kl' else 'gnn'
    loss_type = 'maeloss+l1reg' if strategy != 'kl' else 'maeloss+klreg'
    reg_weight = kl_weight if strategy == 'kl' else (l1_weight if strategy == 'l1' else 0)
    
    # Create the loss_params dictionary conditionally
    loss_params = {'reg_weight': reg_weight}
    if strategy == 'kl':
        loss_params['msg_dim'] = msg_dim

    config = {
        'seed': seed,
        'output_dir': f'{base_output_dir}/{experiment}_{dim}/{strategy}',
        'wandb': wandb,
        'wandb_project': wandb_project,
        'save_messages': save_messages,
        'tqdm': tqdm,
        'data_dir': f'{base_data_dir}/{experiment}_{dim}',
        'quick_test': False,
        'model_state_dict': None,
        'model': model_type,
        'model_params': {
            'n_f': num_features,
            'msg_dim': msg_dim,
            'ndim': node_dim,
            'hidden': hidden_size,
            'aggr': aggr_method
        },
        'epochs': epochs,
        'train_batch_size': train_batch_size,
        'val_batch_size': val_batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'save_every_n_epochs': save_every_n_epochs,
        'scheduler_params': {
            'max_lr': max_lr,
            'final_div_factor': final_div_factor
        },
        'loss': loss_type,
        'loss_params': loss_params,
        'augmentations': {
            'random_translate': {
                'scale': translate_scale,
                'dims': list(range(int(dim[0])))
            }
        }
    }
    return config

# Loop through all combinations and write YAML files
for experiment in experiments:
    for strategy in strategies:
        for dim in dimensions:
            config = create_config(experiment, strategy, dim)
            dir_path = f'configs/train_configs/{experiment}'
            os.makedirs(dir_path, exist_ok=True)
            file_path = f'{dir_path}/{strategy}_{dim}.yaml'
            
            with open(file_path, 'w') as file:
                yaml.dump(config, file, sort_keys=False, default_flow_style=False)

print("Training configuration files have been created.")
