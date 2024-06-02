import yaml
import os

# Define the experiments, strategies, and dimensions.
experiments = ['spring', 'r1', 'r2', 'charge']
strategies = ['standard', 'l1', 'kl', 'bottleneck']
dimensions = ['2d', '3d']

# Configurable parameters
hidden = 300
msg_dim = 100
msg_save_limit = 100000 
test_batch_size = 1024

l1_weight = 1.0e-2
kl_weight = 1.0
output_base_dir = '../rds/hpc-work/test_runs/half_size'
data_base_dir = '../rds/hpc-work/test_data'
train_base_dir = '../rds/hpc-work/train_runs/half_size'
configs_base_dir = 'configs/test_configs'
tqdm_enabled = False
save_messages = True

# Function to generate a config dictionary.
def generate_config(experiment, strategy, dim):
    model_type = 'gnn' if strategy != 'kl' else 'vargnn'
    msg_dim_strategy = int(dim[0]) if strategy == 'bottleneck' else msg_dim
    aggr_method = 'add'
    
    loss_type = 'loss+l1reg' if strategy != 'kl' else 'loss+klreg'
    reg_weight = kl_weight if strategy == 'kl' else (l1_weight if strategy == 'l1' else 0)
    loss_params = {'reg_weight': reg_weight}
    if strategy == 'kl':
        loss_params['msg_dim'] = msg_dim
      
    config = {
        'output_dir': f'{output_base_dir}/{experiment}_{dim}/{strategy}',
        'tqdm': tqdm_enabled,
        'save_messages': save_messages,
        'message_save_limit': msg_save_limit,
        'data_dir': f'{data_base_dir}/{experiment}_{dim}',
        'test_batch_size': test_batch_size,
        'model_weights_path': f'{train_base_dir}/{experiment}_{dim}/{strategy}/model_weights/best_model.pt',
        'model': model_type,
        'model_params': {
            'n_f': (2 * int(dim[0]) + 2),
            'msg_dim': msg_dim_strategy,
            'ndim': int(dim[0]),
            'hidden': hidden,
            'aggr': aggr_method
        },
        'loss': loss_type,
        'loss_params': loss_params
    }
    return config

# Create directories and files.
for experiment in experiments:
    for strategy in strategies:
        for dim in dimensions:
            dir_path = f'{configs_base_dir}/{experiment}'
            os.makedirs(dir_path, exist_ok=True)
            
            file_path = f'{dir_path}/{strategy}_{dim}.yaml'
            
            config = generate_config(experiment, strategy, dim)
            
            with open(file_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

print("Configuration files have been created.")
