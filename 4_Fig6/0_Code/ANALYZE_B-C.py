import snntorch as snn
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from joblib import load
from torch.utils.data import DataLoader, TensorDataset


from base_Attactors import Config, load_data_with_encoding
# 导入更新后的 utils
from utils import visualize_representation, evaluate_linear_separability



# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
config = Config()
config.num_steps = 5
config.tmax = 8

# --- Net with record ---
class Net(nn.Module):
    def __init__(self, config, encoding):
        super().__init__()
        self.encoding = encoding
        self.num_steps = config.num_steps

        # Dynamically calculate input dimension
        if encoding in ['lorenz', 'chen', 'rossler', 'aizawa', 'nose_hoover', 'sprott', 'chua','mixed_oscillator']:
            self.input_dim = config.n_components * 3
        elif encoding == 'default':
            self.input_dim = config.num_inputs
        elif encoding == 'umap':
            self.input_dim = config.n_components
        else:
            # For all time-based encodings, the input dimension per time step is the number of input features
            self.input_dim = config.num_inputs

        self.fc1 = nn.Linear(self.input_dim, config.num_hidden)

        # Hidden layers
        self.fc2 = nn.Linear(config.num_hidden, config.num_hidden)
        self.fc3 = nn.Linear(config.num_hidden, config.num_hidden)
        self.fc4 = nn.Linear(config.num_hidden, config.num_outputs)

        # LIF neurons
        self.lif1 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)
        self.lif2 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)
        self.lif3 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)
        self.lif4 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)

    def forward(self, x):
        # Initialize spike, membrane potential, and current records
        spk_rec_dict, mem_rec_dict, cur_rec_dict = {}, {}, {} # Use a dictionary for initialization

        # Initialize membrane potentials
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
            if self.encoding in ['default', 'umap']:
                 current_input = x if step == 0 else spk_rec_dict['layer4'][-1] # Or handle differently
            elif x.dim() == 3 and x.shape[1] == self.num_steps:
                 current_input = x[:, step, :]
            elif x.dim() == 2:
                 current_input = x
            else:
                 raise ValueError(f"Unexpected input shape: {x.shape}")

            # --- Process through layers ---
            layer_input = current_input # Input to layer1
            for i, layer_name in enumerate(layer_order):
                 # Calculate current
                 fc_layer = fc_layers[layer_name]
                 current_val = fc_layer(layer_input) # Input current to LIF

                 # Process LIF
                 lif_layer = lif_layers[layer_name]
                 spk_val, mem_val = lif_layer(current_val, mem_state[layer_name])

                 # Update state
                 mem_state[layer_name] = mem_val # Update membrane state for next step

                 # --- Record activities ---
                 if layer_name not in spk_rec_dict: spk_rec_dict[layer_name] = []
                 if layer_name not in mem_rec_dict: mem_rec_dict[layer_name] = []
                 if layer_name not in cur_rec_dict: cur_rec_dict[layer_name] = []

                 spk_rec_dict[layer_name].append(spk_val)
                 mem_rec_dict[layer_name].append(mem_val)
                 cur_rec_dict[layer_name].append(current_val) # Record input current

                 # Output spikes become input for the next layer
                 layer_input = spk_val


        # --- Stack temporal sequences ---
        spk_rec = {k: torch.stack(v, dim=0) for k, v in spk_rec_dict.items() if v}
        mem_rec = {k: torch.stack(v, dim=0) for k, v in mem_rec_dict.items() if v}
        cur_rec = {k: torch.stack(v, dim=0) for k, v in cur_rec_dict.items() if v}

        return spk_rec, mem_rec, cur_rec # Return all records


def extract_fc_parameters(model):
    """
    Extracts the weight and bias parameters for each fully connected layer (fc1-fc4) in the model.

    Args:
        model (torch.nn.Module): The trained SNN model.

    Returns:
        dict: A dictionary containing the weights and biases for each layer.
    """
    params = {
        'fc1': {
            'weight': model.fc1.weight.detach().cpu(),
            'bias': model.fc1.bias.detach().cpu() if model.fc1.bias is not None else None
        },
        'fc2': {
            'weight': model.fc2.weight.detach().cpu(),
            'bias': model.fc2.bias.detach().cpu() if model.fc2.bias is not None else None
        },
        'fc3': {
            'weight': model.fc3.weight.detach().cpu(),
            'bias': model.fc3.bias.detach().cpu() if model.fc3.bias is not None else None
        },
        'fc4': {
            'weight': model.fc4.weight.detach().cpu(),
            'bias': model.fc4.bias.detach().cpu() if model.fc4.bias is not None else None
        }
    }
    return params


def visualize_fc_parameters(params, save_path='.', condition_name=''):
    """
    Visualizes the distribution of weights and biases for each fully connected layer.

    Args:
        params (dict): A dictionary containing weight and bias values.
        save_path (str): The path to save the visualization results.
        condition_name (str): The condition name for the plot title.
    """
    num_layers = len(params)
    plt.figure(figsize=(15, 4 * num_layers))

    for i, (layer_name, layer_params) in enumerate(params.items()):
        weight_values = layer_params['weight'].numpy().flatten()

        # Plot weight distribution
        plt.subplot(num_layers, 2, 2 * i + 1)
        plt.hist(weight_values, bins=50, alpha=0.7)
        plt.axvline(weight_values.mean(), color='r', linestyle='--',
                    label=f'Avg.: {weight_values.mean():.4f}\nStd.: {weight_values.std():.4f}')
        plt.title(f'{condition_name} - {layer_name} weight dist.')
        plt.xlabel('Weight')
        plt.ylabel('Num')
        plt.legend()

        # Plot bias distribution (if it exists)
        if layer_params['bias'] is not None:
            bias_values = layer_params['bias'].numpy().flatten()
            plt.subplot(num_layers, 2, 2 * i + 2)
            plt.hist(bias_values, bins=30, alpha=0.7)
            plt.axvline(bias_values.mean(), color='r', linestyle='--',
                        label=f'Avg.: {bias_values.mean():.4f}\nStd.: {bias_values.std():.4f}')
            plt.title(f'{condition_name} - {layer_name} bias dist.')
            plt.xlabel('Bias')
            plt.ylabel('Num')
            plt.legend()

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{condition_name}_fc_parameters.png'), dpi=150)
    plt.close()


def print_fc_parameter_summary(params, condition_name=''):
    """
    Prints a summary of the weight and bias statistics for each fully connected layer.

    Args:
        params (dict): A dictionary containing weight and bias values.
        condition_name (str): The name of the condition.
    """
    print(f"\n--- {condition_name} Summary of fc layer parameters ---")
    print(
        f"{'Layer':<10} | {'Weight Avg.':<12} | {'Weight Std':<12} | {'Weight. Min':<12} | {'Weight Max':<12} | {'Bias Avg.':<12} | {'Bias Std.':<12}")
    print(f"{'-' * 10} | {'-' * 12} | {'-' * 12} | {'-' * 12} | {'-' * 12} | {'-' * 12} | {'-' * 12}")

    for layer_name, layer_params in params.items():
        weight_values = layer_params['weight'].numpy().flatten()

        if layer_params['bias'] is not None:
            bias_values = layer_params['bias'].numpy().flatten()
            bias_mean = bias_values.mean()
            bias_std = bias_values.std()
        else:
            bias_mean = "N/A"
            bias_std = "N/A"

        print(f"{layer_name:<10} | {weight_values.mean():<12.6f} | {weight_values.std():<12.6f} | "
              f"{weight_values.min():<12.6f} | {weight_values.max():<12.6f} | "
              f"{bias_mean if isinstance(bias_mean, str) else bias_mean:<12.6f} | "
              f"{bias_std if isinstance(bias_std, str) else bias_std:<12.6f}")


##### Leaky #####
def extract_neuron_parameters(model):
    """
    Extracts the beta and threshold values of each LIF neuron layer in the model.

    Args:
        model (torch.nn.Module): The trained SNN model.

    Returns:
        dict: A dictionary containing the beta and threshold values for each layer.
    """
    params = {
        'layer1': {'beta': model.lif1.beta.detach().cpu(), 'threshold': model.lif1.threshold.detach().cpu()},
        'layer2': {'beta': model.lif2.beta.detach().cpu(), 'threshold': model.lif2.threshold.detach().cpu()},
        'layer3': {'beta': model.lif3.beta.detach().cpu(), 'threshold': model.lif3.threshold.detach().cpu()},
        'layer4': {'beta': model.lif4.beta.detach().cpu(), 'threshold': model.lif4.threshold.detach().cpu()}
    }
    return params


def visualize_neuron_parameters(params, save_path='.', condition_name=''):
    """
    Visualizes the beta and threshold distributions for each layer.

    Args:
        params (dict): A dictionary containing beta and threshold values.
        save_path (str): The path to save the visualization results.
        condition_name (str): The condition name for the plot title.
    """
    num_layers = len(params)
    plt.figure(figsize=(15, 4 * num_layers))

    for i, (layer_name, layer_params) in enumerate(params.items()):
        beta_values = layer_params['beta'].numpy().flatten()
        threshold_values = layer_params['threshold'].numpy().flatten()

        # Plot beta distribution
        plt.subplot(num_layers, 2, 2 * i + 1)
        plt.hist(beta_values, bins=30, alpha=0.7)
        plt.axvline(beta_values.mean(), color='r', linestyle='--', label=f'Avg.: {beta_values.mean():.4f}')
        plt.title(f'{condition_name} - {layer_name} Beta Dist.')
        plt.xlabel('Beta')
        plt.ylabel('Count')
        plt.legend()

        # Plot threshold distribution
        plt.subplot(num_layers, 2, 2 * i + 2)
        plt.hist(threshold_values, bins=30, alpha=0.7)
        plt.axvline(threshold_values.mean(), color='r', linestyle='--', label=f'Avg.: {threshold_values.mean():.4f}')
        plt.title(f'{condition_name} - {layer_name} Threshold Dist.')
        plt.xlabel('Threshold')
        plt.ylabel('Count')
        plt.legend()

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{condition_name}_neuron_parameters.png'), dpi=150)
    plt.close()


def print_neuron_parameter_summary(params, condition_name=''):
    """
    Prints a summary of the beta and threshold statistics for each layer.

    Args:
        params (dict): A dictionary containing beta and threshold values.
        condition_name (str): The name of the condition.
    """
    print(f"\n--- {condition_name} Summary of Neuron Parameters ---")
    print(f"{'Layer':<10} | {'Beta Avg.':<12} | {'Beta Std.':<12} | {'Threshold Avg.':<15} | {'Threshold Std.':<15}")
    print(f"{'-' * 10} | {'-' * 12} | {'-' * 12} | {'-' * 15} | {'-' * 15}")

    for layer_name, layer_params in params.items():
        beta_values = layer_params['beta'].numpy().flatten()
        threshold_values = layer_params['threshold'].numpy().flatten()

        print(f"{layer_name:<10} | {beta_values.mean():<12.6f} | {beta_values.std():<12.6f} | "
              f"{threshold_values.mean():<15.6f} | {threshold_values.std():<15.6f}")

    print("\n")


def compare_neuron_parameters_across_conditions(condition_params, save_path='.'):
    """
    Compares the differences in neuron parameters across different conditions.

    Args:
        condition_params (dict): A dictionary where keys are condition names and values are
                                 dictionaries of neuron parameters.
        save_path (str): The directory to save the plots.
    """
    # Ensure there are at least two conditions for comparison
    if len(condition_params) < 2:
        print("At least two conditions are required to compare.")
        return

    conditions = list(condition_params.keys())
    layers = list(condition_params[conditions[0]].keys())

    # Create a plot for each layer to compare beta and threshold
    for layer in layers:
        plt.figure(figsize=(12, 5))

        # Compare beta
        plt.subplot(1, 2, 1)
        for condition in conditions:
            beta_values = condition_params[condition][layer]['beta'].numpy().mean()
            plt.bar(condition, beta_values, alpha=0.7)
        plt.title(f'{layer} - Beta Comparison')
        plt.ylabel('Avg. Beta')
        plt.xticks(rotation=45)

        # Compare threshold
        plt.subplot(1, 2, 2)
        for condition in conditions:
            threshold_values = condition_params[condition][layer]['threshold'].numpy().mean()
            plt.bar(condition, threshold_values, alpha=0.7)
        plt.title(f'{layer} - Threshold Comparison')
        plt.ylabel('Avg. Threshold')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{layer}_parameter_comparison.png'), dpi=150)
        plt.close()


@torch.no_grad()
def extract_layer_representations(model, data_loader, layer_name, config, device, representation_type='spike_count'):
    """
    Extracts representation vectors from a specified layer.

    Args:
        ... (parameters unchanged)
        representation_type (str): The type of representation ('spike_count', 'avg_rate', 'last_mem', 'avg_mem', 'input_current_sum', 'input_current_mean').

    Returns:
        ... (return values unchanged)
    """
    model.eval()
    all_representations = []
    all_labels = []

    print(f"Extracting '{representation_type}' representations from {layer_name}...")
    for data, targets in data_loader:
        data = data.to(device)
        targets = targets.to(device)

        # Modify this to receive the third return value cur_rec
        spk_rec, mem_rec, cur_rec = model(data) # <-- Receive cur_rec

        # Check if the requested layer exists in the corresponding record dictionary
        layer_exists = False
        if representation_type in ['spike_count', 'avg_rate']:
            layer_exists = layer_name in spk_rec
        elif representation_type in ['last_mem', 'avg_mem']:
            layer_exists = layer_name in mem_rec
        elif representation_type in ['input_current_sum', 'input_current_mean']: # <-- New check for current types
            layer_exists = layer_name in cur_rec

        if not layer_exists:
             print(f"Warning: Layer '{layer_name}' or its required recording type ('{representation_type}') not found in model outputs. Skipping batch.")
             continue


        batch_representation = None
        try: # Add a try-except block for enhanced robustness
            if representation_type == 'spike_count':
                batch_representation = spk_rec[layer_name].sum(dim=0).cpu()
            elif representation_type == 'avg_rate':
                batch_representation = spk_rec[layer_name].float().mean(dim=0).cpu()
            elif representation_type == 'last_mem':
                batch_representation = mem_rec[layer_name][-1].cpu()
            elif representation_type == 'avg_mem':
                batch_representation = mem_rec[layer_name].mean(dim=0).cpu()
            # --- New addition for input current handling ---
            elif representation_type == 'input_current_sum':
                batch_representation = cur_rec[layer_name].sum(dim=0).cpu()
            elif representation_type == 'input_current_mean':
                batch_representation = cur_rec[layer_name].mean(dim=0).cpu()
            # --- End new addition ---
            else:
                raise ValueError(f"Unknown representation_type: {representation_type}")
        except KeyError:
             print(f"Error: Layer '{layer_name}' key not found in the corresponding record dictionary for type '{representation_type}'. Skipping batch.")
             continue
        except Exception as e:
             print(f"Error processing batch for layer '{layer_name}', type '{representation_type}': {e}. Skipping batch.")
             continue


        if batch_representation is not None:
             all_representations.append(batch_representation)
             all_labels.append(targets.cpu())

    if not all_representations:
         print("Error: No representations were extracted.")
         return None, None

    all_representations = torch.cat(all_representations, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"Finished extraction. Representation shape: {all_representations.shape}, Labels shape: {all_labels.shape}")
    return all_representations, all_labels


@torch.no_grad()
def extract_temporal_representation(model, data_loader, layer_name, config, device, record_type='spk'):
    """
    Extracts and flattens the complete temporal representation (e.g., spike or current sequences)
    from a specified layer.

    Args:
        model (torch.nn.Module): The trained SNN model.
        data_loader (DataLoader): The data loader.
        layer_name (str): The name of the layer (e.g., 'layer1', 'layer3').
        config (Config): The configuration object.
        device (torch.device): The device.
        record_type (str): The type of record to extract ('spk', 'mem', 'cur').

    Returns:
        torch.Tensor: The flattened temporal representation [num_samples, num_neurons * num_steps].
        torch.Tensor: The corresponding ground truth labels [num_samples].
    """
    model.eval()
    all_representations_flat = []
    all_labels = []

    print(f"Extracting flattened temporal '{record_type}' representations from {layer_name}...")
    for data, targets in data_loader:
        data = data.to(device)
        targets = targets.to(device)

        # Assume model returns spk_rec, mem_rec, cur_rec
        spk_rec, mem_rec, cur_rec = model(data)

        rec_dict = None
        if record_type == 'spk':
            rec_dict = spk_rec
        elif record_type == 'mem':
            rec_dict = mem_rec
        elif record_type == 'cur':
            rec_dict = cur_rec
        else:
            raise ValueError(f"Unknown record_type: {record_type}")

        if layer_name not in rec_dict:
            print(f"Warning: Layer '{layer_name}' not found in {record_type}_rec. Skipping batch.")
            continue

        # Get the temporal data: [time, batch, neurons]
        temporal_data = rec_dict[layer_name]
        num_steps = temporal_data.shape[0]
        batch_size = temporal_data.shape[1]
        num_neurons = temporal_data.shape[2]

        # Reshape to [batch, time * neurons] and then flatten to [batch, time * neurons]
        # Permute to [batch, time, neurons] first
        batch_repr_temporal = temporal_data.permute(1, 0, 2).contiguous()
        # Flatten the time and neuron dimensions
        batch_repr_flat = batch_repr_temporal.view(batch_size, -1).cpu() # [batch, num_steps * num_neurons]


        all_representations_flat.append(batch_repr_flat)
        all_labels.append(targets.cpu())

    if not all_representations_flat:
         print("Error: No temporal representations were extracted.")
         return None, None

    # Concatenate all batches
    all_representations_flat = torch.cat(all_representations_flat, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"Finished extraction. Flattened temporal representation shape: {all_representations_flat.shape}, Labels shape: {all_labels.shape}")
    return all_representations_flat, all_labels


def visualize_activation_distribution(representations, labels, title='Activation Distribution', save_path='.', filename='activation_hist.png', bins=50):
    """Plots a distribution histogram of activations (e.g., input current or spike counts), distinguished by class"""
    if representations is None or labels is None:
        print("Cannot visualize distribution: Representations or labels are None.")
        return

    if isinstance(representations, torch.Tensor):
        representations = representations.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    # Check for NaN/Inf values
    if np.any(np.isnan(representations)) or np.any(np.isinf(representations)):
        print("Warning: NaN or Inf values found in representations. Replacing with 0 for histogram.")
        representations = np.nan_to_num(representations)


    num_classes = len(np.unique(labels))
    num_neurons = representations.shape[1] if representations.ndim > 1 else 1


    plt.figure(figsize=(9, 2.5))

    # Plot the overall distribution of all neurons
    plt.subplot(1, 2, 1)
    plt.hist(representations.flatten(), bins=bins, density=True, alpha=0.7)
    plt.title(f"{title} - Overall")
    plt.xlabel("Activation Value")
    plt.ylabel("Density")

    # Plot distribution by class (selecting a subset of neurons to avoid crowding)
    plt.subplot(1, 2, 2)
    num_neurons_to_plot = 1 if representations.ndim == 1 else min(10, num_neurons)
    plot_indices = np.random.choice(num_neurons, num_neurons_to_plot, replace=False) if representations.ndim > 1 else [0]


    for i in np.unique(labels): # Iterate through unique labels present
        class_repr = representations[labels == i]
        if class_repr.ndim > 1:
             class_repr_subset = class_repr[:, plot_indices]
        else:
             class_repr_subset = class_repr

        if class_repr_subset.size > 0: # Check if there is data for this class
            plt.hist(class_repr_subset.flatten(), bins=bins, density=True, alpha=0.5, label=f'Class {i}')
        else:
            print(f"Warning: No data to plot for Class {i}")


    plt.title("Dist. by Class")
    plt.xlabel("Activation Value")
    plt.ylabel("Density")

    plt.tight_layout()
    full_save_path = os.path.join(save_path, filename)
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
    plt.savefig(full_save_path, dpi=150)
    print(f"Activation distribution plot saved to {full_save_path}")
    plt.close()


def visualize_fc_weight_comparison(all_fc_params, save_path='.'):
    """
    Compares the statistical properties of fully-connected layer weights under different conditions.

    Args:
        all_fc_params (dict): Dictionary where keys are condition names and values are dicts of fc parameters.
        save_path (str): Path to save the plots.
    """
    conditions = list(all_fc_params.keys())
    if len(conditions) < 2:
        print("At least two conditions are required to compare。")
        return

    fc_layers = list(all_fc_params[conditions[0]].keys())

    # Create a comparison plot for each fc layer
    for layer in fc_layers:
        plt.figure(figsize=(15, 6))

        # Compare weight mean
        plt.subplot(1, 3, 1)
        means = [all_fc_params[cond][layer]['weight'].mean().item() for cond in conditions]
        plt.bar(conditions, means)
        plt.title(f'{layer} - Weight mean comparison')
        plt.xticks(rotation=45)

        # Compare weight Std.
        plt.subplot(1, 3, 2)
        stds = [all_fc_params[cond][layer]['weight'].std().item() for cond in conditions]
        plt.bar(conditions, stds)
        plt.title(f'{layer} - Weight Std. comparison')
        plt.xticks(rotation=45)

        # Compare weight distribution
        plt.subplot(1, 3, 3)
        for cond in conditions:
            weights = all_fc_params[cond][layer]['weight'].numpy().flatten()
            plt.hist(weights, bins=30, alpha=0.5, label=cond, density=True)
        plt.title(f'{layer} - Weight dist. comparison')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{layer}_weight_comparison.png'), dpi=150)
        plt.close()


def visualize_temporal_representation(model, data_loader, layer_name, config, device, record_type='spk', method='umap', title_prefix='', save_path='.'):
    """
    Extracts temporal representations, reduces their dimensionality, and visualizes them.

    Args:
        model: Trained model.
        data_loader: Data loader.
        layer_name: Layer name.
        config: Configuration object.
        device: Device.
        record_type: 'spk', 'mem', or 'cur'.
        method: 'umap', 'tsne', or 'pca'.
        title_prefix: Prefix for the plot title.
        save_path: Path to save the plot.
    """
    # 1. Extract flattened temporal representation
    temporal_repr_flat, labels = extract_temporal_representation(
        model, data_loader, layer_name, config, device, record_type=record_type
    )

    if temporal_repr_flat is None:
        print(f"Skipping temporal visualization for {layer_name} ({record_type}) due to extraction error.")
        return

    # 2. Visualize
    visualize_representation(
        temporal_repr_flat,
        labels,
        method=method,
        title=f"{title_prefix} - {layer_name} Flattened '{record_type}' ( {method.upper()} )",
        save_path=save_path,
        filename=f"{layer_name}_flat_{record_type}_{method}.png"
    )


# --- Select the model and conditions to analyze ---
analysis_conditions = [
        {
            'name': 'Expansive_Peak',
            'delta': -1.5,
            'model_path': 'mixed_oscillator_results/saved_models/model_delta_-1.50_run0.pth',
            'reducer_path': 'mixed_oscillator_results/saved_models/umap_transformer_delta_-1.50_run0.pth',
            'test_dataset_path': 'mixed_oscillator_results/saved_models/encoded_test_delta_-1.50_run0.pt'
        },
        {
            'name': 'Dissipative_Peak',
            'delta': 10.0,
            'model_path': 'mixed_oscillator_results/saved_models/model_delta_10.00_run0.pth',
            'reducer_path': 'mixed_oscillator_results/saved_models/umap_transformer_delta_10.00_run0.pth',
            'test_dataset_path': 'mixed_oscillator_results/saved_models/encoded_test_delta_10.00_run0.pt'
        },
        {
            'name': 'Trough_Region',
            'delta': 2.0,
            'model_path': 'mixed_oscillator_results/saved_models/model_delta_2.00_run0.pth',
            'reducer_path': 'mixed_oscillator_results/saved_models/umap_transformer_delta_2.00_run0.pth',
            'test_dataset_path': 'mixed_oscillator_results/saved_models/encoded_test_delta_2.00_run0.pt'
        }
    ]

def load_saved_test_data(path):
    saved_data = torch.load(path)
    test_dataset = TensorDataset(saved_data['data'], saved_data['labels'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return test_loader


results_list = []
all_condition_params = {}
all_fc_params = {}

for condition in analysis_conditions:
    exp_name = condition['name']
    model_path = condition['model_path']
    delta_val = condition['delta']
    # encoder_params = {'delta': delta_val, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0}

    reducer = load(condition['reducer_path'])

    print("Loading test data...")
    data_path = condition['test_dataset_path']
    test_loader = load_saved_test_data(data_path)


    print(f"\n{'='*20} Analyzing Condition: {exp_name} {'='*20}")

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}. Skipping.")
        continue

    model = Net(config, encoding='mixed_oscillator').to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state dict from {model_path}: {e}")
        continue # Skip if model cannot be loaded
    model.eval()


    analysis_save_path = os.path.join("mixed_oscillator_results", "representation_analysis", exp_name)
    os.makedirs(analysis_save_path, exist_ok=True)

    neuron_params = extract_neuron_parameters(model)
    all_condition_params[exp_name] = neuron_params


    print_neuron_parameter_summary(neuron_params, condition_name=exp_name)
    visualize_neuron_parameters(neuron_params, save_path=analysis_save_path, condition_name=exp_name)

    fc_params = extract_fc_parameters(model)
    print_fc_parameter_summary(fc_params, condition_name=exp_name)
    visualize_fc_parameters(fc_params, save_path=analysis_save_path, condition_name=exp_name)

    all_fc_params[exp_name] = fc_params
    layer_transitions = [
        {'name': 'Layer 1 Membrane (Avg)', 'layer': 'layer1', 'type': 'avg_mem'},
        {'name': 'Layer 1 Output Spikes', 'layer': 'layer1', 'type': 'spike_count'},
        {'name': 'Layer 2 Membrane (Avg)', 'layer': 'layer2', 'type': 'avg_mem'},
        {'name': 'Layer 2 Output Spikes', 'layer': 'layer2', 'type': 'spike_count'},
        {'name': 'Layer 3 Membrane (Avg)', 'layer': 'layer3', 'type': 'avg_mem'},
        {'name': 'Layer 3 Output Spikes', 'layer': 'layer3', 'type': 'spike_count'},
        {'name': 'Layer 4 Membrane (Avg)', 'layer': 'layer4', 'type': 'avg_mem'},
        {'name': 'Layer 4 Output Spikes', 'layer': 'layer4', 'type': 'spike_count'},

    ]

    for item in layer_transitions:
        print(f"\n--- Analyzing: {item['name']} ---")
        representations, labels = extract_layer_representations(
            model, test_loader, item['layer'], config, device, representation_type=item['type']
        )

        if representations is None: continue

        # # UMAP
        # visualize_representation(
        #     representations, labels, method='umap',
        #     title=f"{exp_name} - {item['name']} (UMAP)",
        #     save_path=analysis_save_path,
        #     filename=f"{item['layer']}_{item['type']}_umap.png"
        # )

        linear_accuracy = evaluate_linear_separability(representations, labels)
        results_list.append({
            'Condition': exp_name, 'Analysis Point': item['name'],
            'Layer': item['layer'], 'Repr Type': item['type'],
            'Linear Separability': linear_accuracy, 'Delta': delta_val
        })

        visualize_activation_distribution(
            representations, labels, # representations is the sum here
            title=f"{exp_name} - {item['name']} Agg. Dist.",
            save_path=analysis_save_path,
            filename=f"{item['layer']}_{item['type']}_distribution.png"
        )

    print(f"\n--- Analyzing: Layer 4 Flattened Spike Train ---")
    visualize_temporal_representation(
        model, data_loader=test_loader, layer_name='layer1', config=config, device=device,
        record_type='spk',
        method='umap',
        title_prefix=exp_name,
        save_path=analysis_save_path
    )
    visualize_temporal_representation(
        model, data_loader=test_loader, layer_name='layer2', config=config, device=device,
        record_type='spk',
        method='umap',
        title_prefix=exp_name,
        save_path=analysis_save_path
    )
    visualize_temporal_representation(
        model, data_loader=test_loader, layer_name='layer3', config=config, device=device,
        record_type='spk',
        method='umap',
        title_prefix=exp_name,
        save_path=analysis_save_path
    )
    visualize_temporal_representation(
        model, data_loader=test_loader, layer_name='layer4', config=config, device=device,
        record_type='spk',
        method='umap',
        title_prefix=exp_name,
        save_path=analysis_save_path
    )
compare_neuron_parameters_across_conditions(all_condition_params,
                                            save_path=os.path.join("mixed_oscillator_results",
                                                                   "representation_analysis"))
visualize_fc_weight_comparison(all_fc_params,
                               save_path=os.path.join("mixed_oscillator_results", "representation_analysis"))


results_df = pd.DataFrame(results_list)

os.makedirs(os.path.join("mixed_oscillator_results", "representation_analysis"), exist_ok=True)
results_df.to_csv(os.path.join("mixed_oscillator_results", "representation_analysis", "layer_transition_linearity.csv"), index=False)
print("\n--- Layer Transition Analysis Summary ---")
print(results_df)


print("\nLayer transition analysis complete.")


