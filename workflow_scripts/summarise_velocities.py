import os
import json
import numpy as np
import matplotlib.pyplot as plt

def velocity_analysis(x, output_dir,  dim = 2, winsorise = False):
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot histogram of the velocities
    velocities = x[:, :, :, dim:-2]
        
    flat_velocities = velocities.flatten()
    if winsorise:
        upper_val = np.percentile(flat_velocities, winsorise)
        lower_val = np.percentile(flat_velocities, 100 - winsorise)
        # Remove elements if they are above the 95th percentile
        flat_velocities = flat_velocities[(flat_velocities < upper_val) & (flat_velocities > lower_val)]
    
    mean = np.mean(flat_velocities)
    median = np.median(flat_velocities)
    std = np.std(flat_velocities)
    max_ = np.max(flat_velocities)
    min_ = np.min(flat_velocities)
    uq = np.percentile(flat_velocities, 75)
    lq = np.percentile(flat_velocities, 25)
    
    
    summary_stats = {
        'mean': float(mean),
        'median': float(median),
        'std': float(std),
        'max': float(max_),
        'min': float(min_),
        'uq': float(uq),
        'lq': float(lq)
        }

    plt.hist(flat_velocities.flatten(), bins = 100)
    plt.title('Histogram of all velocity comps')
    plt.savefig(os.path.join(output_dir,'hist.png'))
    plt.close()
    with open(os.path.join(output_dir,'stats.json'), 'w+') as f:
        json.dump(summary_stats, f)


base_dir="data/test_data_hpc_reproduce"
save_dir = "./vel_summaries__winsorise_99-99"
winsorise = 99.99

experiments = ['spring', 'r1', 'r2', 'charge']
dims = [2,3]
splits = ['train', 'val', 'test']

for experiment in experiments:
    for dim in dims:
        for split in splits:
            data_dir = os.path.join(base_dir,f"{experiment}_{dim}d/{split}/raw/")
            data_fname = [f for f in os.listdir(data_dir) if 'accel' not in f][0]
            data_path = os.path.join(data_dir,data_fname)
            output_dir = os.path.join(save_dir, f"{experiment}_{dim}", f"{split}")
            x = np.load(data_path)
            print(f"[INFO] running velocity analysis for {experiment} - {dim} - {split}")
            velocity_analysis(x, output_dir, dim, winsorise)