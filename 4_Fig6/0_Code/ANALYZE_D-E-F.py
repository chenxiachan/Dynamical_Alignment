import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from base_Attactors import Config
import snntorch as snn
import torch.nn as nn

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
config = Config() #
config.num_steps = 5


## Net with cur_rec return
class Net(nn.Module):
    def __init__(self, config, encoding):
        super().__init__()
        self.encoding = encoding
        self.num_steps = config.num_steps
        self.config = config

        # Dynamically calculate input dimension
        if encoding in ['lorenz', 'chen', 'rossler', 'aizawa', 'nose_hoover', 'sprott', 'chua','mixed_oscillator']:
            self.input_dim = config.n_components * 3
        elif encoding == 'default':
            self.input_dim = config.num_inputs
        elif encoding == 'umap':
            self.input_dim = config.n_components
        else:
            self.input_dim = config.num_inputs

        self.fc1 = nn.Linear(self.input_dim, config.num_hidden)
        self.fc2 = nn.Linear(config.num_hidden, config.num_hidden)
        self.fc3 = nn.Linear(config.num_hidden, config.num_hidden)
        self.fc4 = nn.Linear(config.num_hidden, config.num_outputs)

        self.lif1 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)
        self.lif2 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)
        self.lif3 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)
        self.lif4 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)

    def forward(self, x):
        # Initialize recordings
        spk_rec_dict, mem_rec_dict, cur_rec_dict = {}, {}, {}
        mem_state = {
             'layer1': self.lif1.init_leaky(), 'layer2': self.lif2.init_leaky(),
             'layer3': self.lif3.init_leaky(), 'layer4': self.lif4.init_leaky()
        }
        lif_layers = {'layer1': self.lif1, 'layer2': self.lif2, 'layer3': self.lif3, 'layer4': self.lif4}
        fc_layers = {'layer1': self.fc1, 'layer2': self.fc2, 'layer3': self.fc3, 'layer4': self.fc4}
        layer_order = ['layer1', 'layer2', 'layer3', 'layer4']

        # Temporal processing
        for step in range(self.num_steps):
            # --- Input processing ---
            # Handle different encodings and input dimensions
            if self.encoding in ['default', 'umap']:
                # These encodings typically lack a time dimension or are only used at step=0
                if step == 0:
                    current_input = x.view(x.size(0), -1)
                else:
                    if 'layer4' in spk_rec_dict and spk_rec_dict['layer4']:
                         current_input = spk_rec_dict['layer4'][-1]
                    else:
                         current_input = torch.zeros_like(fc_layers['layer1'].weight[:,0]).unsqueeze(0).repeat(x.size(0),1).to(x.device)

            elif x.dim() == 3 and x.shape[1] == self.num_steps:
                 # Standard temporal input [batch, time, features]
                 current_input = x[:, step, :]
            elif x.dim() == 2:
                 # Static input, potentially already encoded, fed to the first layer
                 current_input = x
            else:
                 raise ValueError(f"Unexpected input shape: {x.shape}")


            # --- Process through layers ---
            layer_input = current_input
            for i, layer_name in enumerate(layer_order):
                 fc_layer = fc_layers[layer_name]
                 lif_layer = lif_layers[layer_name]

                 # Check for dimension mismatch
                 expected_dim = fc_layer.weight.shape[1]
                 if layer_input.shape[-1] != expected_dim:
                      raise ValueError(f"Dimension mismatch for {layer_name}: input has {layer_input.shape[-1]} features, layer expects {expected_dim}")

                 current_val = fc_layer(layer_input)
                 spk_val, mem_val = lif_layer(current_val, mem_state[layer_name])
                 mem_state[layer_name] = mem_val

                 # --- Record activities ---
                 if layer_name not in spk_rec_dict: spk_rec_dict[layer_name] = []
                 if layer_name not in mem_rec_dict: mem_rec_dict[layer_name] = []
                 if layer_name not in cur_rec_dict: cur_rec_dict[layer_name] = []

                 spk_rec_dict[layer_name].append(spk_val)
                 mem_rec_dict[layer_name].append(mem_val)
                 cur_rec_dict[layer_name].append(current_val)

                 layer_input = spk_val

        # --- Stack temporal sequences ---
        try:
            spk_rec = {k: torch.stack(v, dim=0) for k, v in spk_rec_dict.items() if v}
            mem_rec = {k: torch.stack(v, dim=0) for k, v in mem_rec_dict.items() if v}
            cur_rec = {k: torch.stack(v, dim=0) for k, v in cur_rec_dict.items() if v}
        except Exception as e:
            print(f"Error stacking recordings: {e}")
            return {}, {}, {}

        return spk_rec, mem_rec, cur_rec


def load_saved_test_data(path):
    """Load saved encoded test data"""
    try:
        saved_data = torch.load(path, map_location='cpu')
        data_tensor = saved_data['data'] if isinstance(saved_data['data'], torch.Tensor) else torch.tensor(saved_data['data'])
        labels_tensor = saved_data['labels'] if isinstance(saved_data['labels'], torch.Tensor) else torch.tensor(saved_data['labels'])
        test_dataset = TensorDataset(data_tensor, labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=min(128, config.batch_size), shuffle=False)
        return test_loader
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return None
    except Exception as e:
        print(f"Error loading data from {path}: {e}")
        return None


def compute_neuron_correlations(spikes):
    """Calculates the firing correlation between neurons."""
    # spikes shape: [time, batch, neurons]
    if spikes.numel() == 0: # Handle empty tensor
        return torch.tensor(np.nan)

    num_time, num_batch, num_neurons = spikes.shape

    # At least 2 neurons are needed to calculate correlation
    if num_neurons < 2:
        return torch.tensor(np.nan)

    # Merge time and batch dimensions: [time*batch, neurons]
    spikes_flat = spikes.float().reshape(-1, num_neurons).cpu()
    n_total_samples = spikes_flat.shape[0]

    # Calculate the standard deviation of each neuron to find those that are always 0 (never fired)
    std_dev = spikes_flat.std(dim=0)
    active_neurons_mask = std_dev > 1e-8 # Increase tolerance
    num_active_neurons = active_neurons_mask.sum().item()

    # At least two active neurons are needed
    if num_active_neurons < 2:
        return torch.tensor(0.0) # If there is only one or no active neurons, the correlation is 0 or NaN

    # Select only active neurons for calculation
    active_spikes = spikes_flat[:, active_neurons_mask]

    # Re-check standard deviation (theoretically should not be zero, but just in case)
    active_std_dev = active_spikes.std(dim=0)
    if torch.any(active_std_dev <= 1e-8):
        print("Warning: Zero standard deviation found even after filtering active neurons.")
        # In this rare case, return NaN or 0
        return torch.tensor(np.nan)


    # Calculate the correlation matrix (Pearson correlation coefficient)
    mean = active_spikes.mean(dim=0)
    std = active_spikes.std(dim=0)
    # Normalize
    normalized_spikes = (active_spikes - mean) / std

    # Use torch.corrcoef more directly
    try:
         # transpose() for corrcoef which expects [features, samples]
        corr_matrix = torch.corrcoef(normalized_spikes.transpose(0, 1))
    except Exception as e:
         print(f"Error calculating corrcoef: {e}")
         return torch.tensor(np.nan)


    # Calculate the mean absolute correlation (excluding the diagonal)
    # Create a mask without the diagonal
    mask = ~torch.eye


def analyze_collective_activity(model, data_loader, device):
    """Analyze the collective activity patterns of neurons in each layer of the model"""
    model.eval()
    metrics_accumulator = {
        layer: {'sync_cv': [], 'mean_correlation': []}
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']
    }
    layers_to_analyze = ['layer1', 'layer2', 'layer3', 'layer4']

    # Process a small number of batches to get an estimate (adjustable)
    num_batches_to_process = 5
    processed_batches = 0

    with torch.no_grad():
        for data, _ in data_loader:
            if processed_batches >= num_batches_to_process:
                break
            data = data.to(device)

            # Run model
            try:
                spk_rec, _, _ = model(data)
            except Exception as e:
                print(f"Error in model forward pass during collective analysis: {e}")
                continue

            # Analyze each layer
            for layer_name in layers_to_analyze:
                if layer_name in spk_rec and isinstance(spk_rec[layer_name], torch.Tensor) and spk_rec[layer_name].numel() > 0:
                    spikes = spk_rec[layer_name] # [time, batch, neurons]

                    # 1. Synchronous Coefficient of Variation (Sync CV)
                    sync_rate_per_sample_per_time = []
                    if spikes.shape[2] > 0:
                        for t in range(spikes.shape[0]):
                            # Calculate the proportion of active neurons for each sample at time t [batch]
                            active_ratio_batch = spikes[t].float().mean(dim=1)
                            sync_rate_per_sample_per_time.append(active_ratio_batch)

                        if sync_rate_per_sample_per_time:
                            # Shape: [time, batch]
                            sync_rate_t_b = torch.stack(sync_rate_per_sample_per_time)
                            # Calculate the time-averaged synchrony rate for each sample [batch]
                            mean_sync_rate_b = sync_rate_t_b.mean(dim=0)
                            # Calculate the time standard deviation for each sample [batch]
                            std_sync_rate_b = sync_rate_t_b.std(dim=0)
                            # Calculate the CV for each sample [batch]
                            # Add epsilon to prevent division by zero
                            sync_cv_b = std_sync_rate_b / (mean_sync_rate_b + 1e-8)
                            # Calculate the batch-averaged CV
                            batch_avg_sync_cv = sync_cv_b.mean().item()
                            metrics_accumulator[layer_name]['sync_cv'].append(batch_avg_sync_cv)
                        else:
                             metrics_accumulator[layer_name]['sync_cv'].append(np.nan)
                    else:
                         metrics_accumulator[layer_name]['sync_cv'].append(np.nan)


                    # 2. Mean Neuron Correlation
                    batch_mean_corr = compute_neuron_correlations(spikes)
                    metrics_accumulator[layer_name]['mean_correlation'].append(batch_mean_corr.item())

                else:
                    metrics_accumulator[layer_name]['sync_cv'].append(np.nan)
                    metrics_accumulator[layer_name]['mean_correlation'].append(np.nan)

            processed_batches += 1

    # --- Calculate final average metrics (averaged across batches) ---
    final_metrics = {}
    for layer_name in layers_to_analyze:
        final_metrics[layer_name] = {}
        for metric in ['sync_cv', 'mean_correlation']:
            valid_values = [v for v in metrics_accumulator[layer_name][metric] if not np.isnan(v)]
            if valid_values:
                final_metrics[layer_name][metric] = np.mean(valid_values)
            else:
                final_metrics[layer_name][metric] = np.nan

    return final_metrics


def analyze_models_multiple_runs(num_runs=10):
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    config.num_steps = 5
    config.tmax = 8

    # Analysis conditions
    analysis_conditions = [
        {
            'name': 'Expansive_Peak',
            'base_name': 'Expansive_Peak',
            'delta': -1.5,
        },
        {
            'name': 'Dissipative_Peak',
            'base_name': 'Dissipative_Peak',
            'delta': 10.0,
        },
    ]

    all_run_data = []

    # --- Loop through conditions and runs ---
    for condition in analysis_conditions:
        condition_base_name = condition['base_name']
        delta_val = condition['delta']
        print(f"\nAnalyzing Condition: {condition['name']} (Delta: {delta_val}) across {num_runs} runs...")

        for run_idx in range(num_runs):
            print(f"  Processing Run {run_idx}...")

            # Construct file paths
            model_path = f'mixed_oscillator_results/saved_models/model_delta_{delta_val:.2f}_run{run_idx}.pth'
            test_dataset_path = f'mixed_oscillator_results/saved_models/encoded_test_delta_{delta_val:.2f}_run{run_idx}.pt'

            # Check if files exist
            if not os.path.exists(model_path) or not os.path.exists(test_dataset_path):
                print(f"    Skipping Run {run_idx}: Model or Data file not found.")
                # Add NaN results for missing runs to maintain data integrity
                for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                     all_run_data.append({
                         'Condition': condition['name'], 'Run': run_idx, 'Layer': layer_name,
                         'sync_cv': np.nan, 'mean_correlation': np.nan, 'Delta': delta_val
                     })
                continue

            # Load test data
            test_loader = load_saved_test_data(test_dataset_path)
            if test_loader is None:
                 print(f"    Skipping Run {run_idx}: Failed to load test data.")
                 # Add NaN results
                 for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                      all_run_data.append({
                          'Condition': condition['name'], 'Run': run_idx, 'Layer': layer_name,
                          'sync_cv': np.nan, 'mean_correlation': np.nan, 'Delta': delta_val
                      })
                 continue

            # Load model
            model = Net(config, encoding='mixed_oscillator').to(device)
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
            except Exception as e:
                print(f"    Skipping Run {run_idx}: Failed to load model state dict: {e}")
                # Add NaN results
                for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                     all_run_data.append({
                         'Condition': condition['name'], 'Run': run_idx, 'Layer': layer_name,
                         'sync_cv': np.nan, 'mean_correlation': np.nan, 'Delta': delta_val
                     })
                continue

            # Analyze collective activity
            try:
                metrics = analyze_collective_activity(model, test_loader, device)
                # Store results for this run
                for layer_name, layer_metrics in metrics.items():
                    all_run_data.append({
                        'Condition': condition['name'],
                        'Run': run_idx,
                        'Layer': layer_name,
                        'sync_cv': layer_metrics.get('sync_cv', np.nan),
                        'mean_correlation': layer_metrics.get('mean_correlation', np.nan),
                        'Delta': delta_val
                    })
                print(f"    Run {run_idx} analysis complete.")
            except Exception as e:
                print(f"    Error during collective activity analysis for run {run_idx}: {e}")
                # Add NaN results
                for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                     all_run_data.append({
                         'Condition': condition['name'], 'Run': run_idx, 'Layer': layer_name,
                         'sync_cv': np.nan, 'mean_correlation': np.nan, 'Delta': delta_val
                     })

    # --- Data aggregation and saving ---
    if not all_run_data:
        print("\nNo data collected from any runs. Exiting.")
        return None

    results_df = pd.DataFrame(all_run_data)

    # Save raw run data (optional)
    raw_save_path = os.path.join("mixed_oscillator_results", "collective_activity", "raw_collective_metrics_all_runs.csv")
    os.makedirs(os.path.dirname(raw_save_path), exist_ok=True)
    results_df.to_csv(raw_save_path, index=False)
    print(f"\nRaw results for all runs saved to {raw_save_path}")

    # Aggregate: calculate Mean and Std for each Condition and Layer
    aggregated_results = results_df.groupby(['Condition', 'Layer']).agg(
        sync_cv_mean=('sync_cv', 'mean'),
        sync_cv_std=('sync_cv', 'std'),
        mean_correlation_mean=('mean_correlation', 'mean'),
        mean_correlation_std=('mean_correlation', 'std'),
        run_count=('Run', 'count')
    ).reset_index()

    # Save aggregated results
    agg_save_path = os.path.join("mixed_oscillator_results", "collective_activity", "aggregated_collective_metrics.csv")
    aggregated_results.to_csv(agg_save_path, index=False)
    print(f"\nAggregated results saved to {agg_save_path}")
    print(aggregated_results)

    # --- Visualize aggregated results ---
    visualize_aggregated_collective_metrics(aggregated_results)

    return aggregated_results


def visualize_aggregated_collective_metrics(agg_df):
    """Visualize aggregated collective activity metrics (with error bars)"""
    if agg_df is None or agg_df.empty:
        print("No aggregated data to visualize.")
        return

    save_path = os.path.join("mixed_oscillator_results", "collective_activity")
    os.makedirs(save_path, exist_ok=True)

    conditions = agg_df['Condition'].unique()
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    metrics_agg = {'sync_cv': ('sync_cv_mean', 'sync_cv_std'),
                   'mean_correlation': ('mean_correlation_mean', 'mean_correlation_std')}

    num_conditions = len(conditions)
    bar_width = 0.8 / num_conditions
    index = np.arange(len(layers))

    # Create plots for two metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Collective Activity Metrics (Mean ± Std across runs)', fontsize=14)

    # Define colors (consistent with main plot)
    color_map = {'Expansive_Peak': 'blue', 'Dissipative_Peak': 'orange', 'Trough_Region SNN': 'green'}

    metric_titles = {'sync_cv': 'Sync. Coefficient of Variation',
                     'mean_correlation': 'Mean Neuron Correlation'}

    for i, (metric_key, (mean_col, std_col)) in enumerate(metrics_agg.items()):
        ax = axes[i]
        for j, condition in enumerate(conditions):
            condition_data = agg_df[(agg_df['Condition'] == condition)].set_index('Layer').reindex(layers)

            means = condition_data[mean_col].values
            stds = condition_data[std_col].values

            # Replace NaN with 0 to avoid plotting errors
            stds = np.nan_to_num(stds)

            ax.bar(index + j * bar_width, means, bar_width,
                   yerr=stds, label=condition, capsize=4,
                   color=color_map.get(condition, 'gray'))

        ax.set_xlabel('LIF Layer')
        ax.set_ylabel('Mean Index Value')
        ax.set_title(metric_titles[metric_key])
        ax.set_xticks(index + bar_width * (num_conditions - 1) / 2)
        ax.set_xticklabels(layers)
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    # Update saved file name
    plt.savefig(os.path.join(save_path, 'aggregated_collective_metrics_summary.png'))
    plt.close()
    print("Aggregated visualization with error bars saved.")


if __name__ == "__main__":
    aggregated_collective_results = analyze_models_multiple_runs(num_runs=10)

