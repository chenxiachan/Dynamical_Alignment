import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import csv

from encoding import encode_dataset 


class Config:
    batch_size = 256
    data_path = 'data/'
    num_hidden = 500
    num_outputs = 10
    n_components = 7
    num_steps = 5
    tmax = 8
    FIXED_BETA = 0.95
    FIXED_THRESHOLD = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class FixedNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_steps = config.num_steps
        self.input_dim = config.n_components * 3
        
        self.fc1 = nn.Linear(self.input_dim, config.num_hidden)
        self.fc2 = nn.Linear(config.num_hidden, config.num_hidden)
        self.fc3 = nn.Linear(config.num_hidden, config.num_hidden)
        self.fc4 = nn.Linear(config.num_hidden, config.num_outputs)
        self.all_fcs = [self.fc1, self.fc2, self.fc3, self.fc4]

        self.lif1 = snn.Leaky(beta=config.FIXED_BETA, threshold=config.FIXED_THRESHOLD, learn_beta=False, learn_threshold=False)
        self.lif2 = snn.Leaky(beta=config.FIXED_BETA, threshold=config.FIXED_THRESHOLD, learn_beta=False, learn_threshold=False)
        self.lif3 = snn.Leaky(beta=config.FIXED_BETA, threshold=config.FIXED_THRESHOLD, learn_beta=False, learn_threshold=False)
        self.lif4 = snn.Leaky(beta=config.FIXED_BETA, threshold=config.FIXED_THRESHOLD, learn_beta=False, learn_threshold=False)
        self.all_lifs = [self.lif1, self.lif2, self.lif3, self.lif4]

    def forward(self, x):
        mems = [lif.init_leaky() for lif in self.all_lifs]
        spk_rec = {f'layer{i}': [] for i in range(1, 5)}
        cur_rec = {f'layer{i}': [] for i in range(1, 5)}
        mem_rec = {f'layer{i}': [] for i in range(1, 5)}
        
        for step in range(self.num_steps):
            spikes_in = x[:, step, :]
            current_l1 = self.all_fcs[0](spikes_in)
            spikes_l1, mems[0] = self.all_lifs[0](current_l1, mems[0])
            current_l2 = self.all_fcs[1](spikes_l1)
            spikes_l2, mems[1] = self.all_lifs[1](current_l2, mems[1])
            current_l3 = self.all_fcs[2](spikes_l2)
            spikes_l3, mems[2] = self.all_lifs[2](current_l3, mems[2])
            current_l4 = self.all_fcs[3](spikes_l3)
            spikes_l4, mems[3] = self.all_lifs[3](current_l4, mems[3])

            all_currents = [current_l1, current_l2, current_l3, current_l4]
            all_spikes = [spikes_l1, spikes_l2, spikes_l3, spikes_l4]
            
            for i in range(1, 5):
                cur_rec[f'layer{i}'].append(all_currents[i-1])
                spk_rec[f'layer{i}'].append(all_spikes[i-1])
                mem_rec[f'layer{i}'].append(mems[i-1]) 

        for i in range(1, 5):
            cur_rec[f'layer{i}'] = torch.stack(cur_rec[f'layer{i}'], dim=1)
            spk_rec[f'layer{i}'] = torch.stack(spk_rec[f'layer{i}'], dim=1)
            mem_rec[f'layer{i}'] = torch.stack(mem_rec[f'layer{i}'], dim=1) 
            
        return cur_rec, spk_rec, mem_rec


def load_model(config, model_path):
    model = FixedNet(config).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run the fixed_params_layerwise_analysis.py script first to train the models.")
    print(f"Loading pre-trained fixed-parameter model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def run_time_resolved_analysis():
    config = Config()
    
    modes = {
        'Dissipative': {'delta': 10.0, 'alpha': 2.0, 'beta': 0.1, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},
        'Expansive': {'delta': -1.5, 'alpha': 2.0, 'beta': 0.1, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},
        'Transition': {'delta': 2.0, 'alpha': 2.0, 'beta': 0.1, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0} 
    }
    
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    test_dataset = datasets.MNIST(config.data_path, train=False, download=True, transform=transform)
    try:
        import umap
    except ImportError:
        print("UMAP not found. Please install it using: pip install umap-learn")
        return
        
    reducer = umap.UMAP(n_components=config.n_components, random_state=42)
    test_data_flat = test_dataset.data.view(-1, 28*28).float()
    print("Running UMAP on test data...")
    
    test_data_reduced = torch.FloatTensor(reducer.fit_transform(test_data_flat))
    test_dataset_reduced = TensorDataset(test_data_reduced, test_dataset.targets)

    results = {mode: {} for mode in modes}

    for mode_name, params in modes.items():
        print(f"\n--- Analyzing Time-Resolved Dynamics for {mode_name} Mode ---")
        model_dir = "saved_fixed_models"
        os.makedirs(model_dir, exist_ok=True) 
        model_path = os.path.join(model_dir, f"fixed_model_delta_{params['delta']:.2f}.pth")
        
        model = FixedNet(config).to(device)
        if os.path.exists(model_path):
             model.load_state_dict(torch.load(model_path, map_location=device))
        else:
             print(f"Model not found at {model_path}, using a new initialized model for demonstration.")
             torch.save(model.state_dict(), model_path)


        test_encoded = encode_dataset(test_dataset_reduced, 'mixed_oscillator', config.num_steps, device, config.batch_size, config.tmax, custom_params=params)
        
        test_loader = DataLoader(test_encoded, batch_size=config.batch_size, shuffle=False)
        

        avg_currents = {f'layer{i}': torch.zeros(config.num_steps, device=device) for i in range(1, 5)}
        avg_rates = {f'layer{i}': torch.zeros(config.num_steps, device=device) for i in range(1, 5)}
        avg_mems = {f'layer{i}': torch.zeros(config.num_steps, device=device) for i in range(1, 5)} 
        
        avg_currents_sq = {f'layer{i}': torch.zeros(config.num_steps, device=device) for i in range(1, 5)}
        avg_rates_sq = {f'layer{i}': torch.zeros(config.num_steps, device=device) for i in range(1, 5)}
        avg_mems_sq = {f'layer{i}': torch.zeros(config.num_steps, device=device) for i in range(1, 5)} 
        
        num_batches = 0
        model.eval()
        with torch.no_grad():
            for data, _ in tqdm(test_loader, desc=f"Collecting time series for {mode_name}"):
                data = data.to(device)

                cur_rec, spk_rec, mem_rec = model(data)
                
                for i in range(1, 5):
                    layer_key = f'layer{i}'

                    avg_currents[layer_key] += torch.mean(cur_rec[layer_key], dim=(0, 2))
                    avg_rates[layer_key] += torch.mean(spk_rec[layer_key], dim=(0, 2))
                    avg_mems[layer_key] += torch.mean(mem_rec[layer_key], dim=(0, 2)) 
                    
                    avg_currents_sq[layer_key] += torch.mean(cur_rec[layer_key]**2, dim=(0, 2))
                    avg_rates_sq[layer_key] += torch.mean(spk_rec[layer_key]**2, dim=(0, 2))
                    avg_mems_sq[layer_key] += torch.mean(mem_rec[layer_key]**2, dim=(0, 2)) 

                num_batches += 1
        
        for i in range(1, 5):
            layer_key = f'layer{i}'
            
            mean_cur = avg_currents[layer_key] / num_batches
            mean_rate = avg_rates[layer_key] / num_batches
            mean_mem = avg_mems[layer_key] / num_batches 
            
            mean_sq_cur = avg_currents_sq[layer_key] / num_batches
            mean_sq_rate = avg_rates_sq[layer_key] / num_batches
            mean_sq_mem = avg_mems_sq[layer_key] / num_batches 
            
            var_cur = mean_sq_cur - mean_cur**2
            var_rate = mean_sq_rate - mean_rate**2
            var_mem = mean_sq_mem - mean_mem**2 

            results[mode_name][layer_key] = {
                'current_t': mean_cur.cpu().numpy(),
                'rate_t': mean_rate.cpu().numpy(),
                'mem_t': mean_mem.cpu().numpy(), 
                'current_var_t': var_cur.cpu().numpy(),
                'rate_var_t': var_rate.cpu().numpy(),
                'mem_var_t': var_mem.cpu().numpy() 
            }

    csv_file_path = "time_resolved_dynamics_with_variance_and_mem.csv"
    print(f"\nSaving results to {csv_file_path}...")


    csv_header = ['Mode', 'Layer', 'TimeStep', 'AvgCurrent', 'CurrentVar', 'AvgRate', 'RateVar', 'AvgMem', 'MemVar']
    
    csv_rows = []
    for mode_name, mode_data in results.items():
        for layer_name, layer_data in mode_data.items():
            num_steps = len(layer_data['current_t'])
            for t in range(num_steps):
                row = [
                    mode_name,
                    layer_name,
                    t,
                    layer_data['current_t'][t],
                    layer_data['current_var_t'][t],
                    layer_data['rate_t'][t],
                    layer_data['rate_var_t'][t],
                    layer_data['mem_t'][t],
                    layer_data['mem_var_t'][t]
                ]
                csv_rows.append(row)

    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
        
    print("CSV file saved successfully.")

    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=True)

    fig.suptitle("Time-Resolved Analysis: Firing Rate & Current (Mean ± Std Dev)", fontsize=24, y=1.02)
    
    time_steps = np.arange(config.num_steps)
    
    for row, mode_name in enumerate(modes.keys()):
        plot_mode_name = "Transition" if mode_name == "Critical" else mode_name
        for col in range(4):
            layer_num = col + 1
            ax_cur = axes[row, col]
            ax_rate = ax_cur.twinx()
            
            data = results[mode_name][f'layer{layer_num}']
            
            mean_c = data['current_t']
            std_c = np.sqrt(np.maximum(0, data['current_var_t']))
            ax_cur.plot(time_steps, mean_c, 'o-', color='tab:blue', label='Avg. Current')
            ax_cur.fill_between(time_steps, mean_c - std_c, mean_c + std_c, color='tab:blue', alpha=0.2)
            ax_cur.set_ylabel('Avg. Input Current', color='tab:blue')
            ax_cur.tick_params(axis='y', labelcolor='tab:blue')
            
            mean_r = data['rate_t']
            std_r = np.sqrt(np.maximum(0, data['rate_var_t']))
            ax_rate.plot(time_steps, mean_r, 's--', color='tab:red', label='Avg. Firing Rate')
            ax_rate.fill_between(time_steps, mean_r - std_r, mean_r + std_r, color='tab:red', alpha=0.2)
            ax_rate.set_ylabel('Avg. Firing Rate', color='tab:red')
            ax_rate.tick_params(axis='y', labelcolor='tab:red')

            if row == 0:
                ax_cur.set_title(f"Layer {layer_num}", fontsize=16)
            if row == 2:
                ax_cur.set_xlabel("Time Step")
            if col == 0:
                ax_cur.text(-0.4, 0.5, plot_mode_name, transform=ax_cur.transAxes, fontsize=18, va='center', ha='right', rotation=90)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = "time_resolved_dynamics_with_variance.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTime-resolved analysis plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    run_time_resolved_analysis()