import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mutual_info_score

from base_Attactors import Config, load_data_with_encoding
from utils import extract_layer_representations, visualize_representation, evaluate_linear_separability, evaluate_model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

import snntorch as snn
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, config, encoding):
        super().__init__()
        self.encoding = encoding
        self.num_steps = config.num_steps

        # Dynamically calculate input dimension
        if encoding in ['lorenz', 'chen', 'rossler', 'aizawa', 'nose_hoover', 'sprott', 'chua', 'mixed_oscillator']:
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
        spk_rec_dict, mem_rec_dict, cur_rec_dict = {}, {}, {}

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
                 current_input = x if step == 0 else spk_rec_dict['layer4'][-1]
            elif x.dim() == 3 and x.shape[1] == self.num_steps:
                 current_input = x[:, step, :]
            elif x.dim() == 2:
                 current_input = x
            else:
                 raise ValueError(f"Unexpected input shape: {x.shape}")

            # --- Process through layers ---
            layer_input = current_input
            for i, layer_name in enumerate(layer_order):
                 # Calculate current
                 fc_layer = fc_layers[layer_name]
                 current_val = fc_layer(layer_input)

                 # Process LIF
                 lif_layer = lif_layers[layer_name]
                 spk_val, mem_val = lif_layer(current_val, mem_state[layer_name])

                 # Update state
                 mem_state[layer_name] = mem_val

                 # --- Record activities ---
                 if layer_name not in spk_rec_dict: spk_rec_dict[layer_name] = []
                 if layer_name not in mem_rec_dict: mem_rec_dict[layer_name] = []
                 if layer_name not in cur_rec_dict: cur_rec_dict[layer_name] = []

                 spk_rec_dict[layer_name].append(spk_val)
                 mem_rec_dict[layer_name].append(mem_val)
                 cur_rec_dict[layer_name].append(current_val)

                 # Output spikes become input for the next layer
                 layer_input = spk_val


        # --- Stack temporal sequences ---
        spk_rec = {k: torch.stack(v, dim=0) for k, v in spk_rec_dict.items() if v}
        mem_rec = {k: torch.stack(v, dim=0) for k, v in mem_rec_dict.items() if v}
        cur_rec = {k: torch.stack(v, dim=0) for k, v in cur_rec_dict.items() if v}

        return spk_rec, mem_rec, cur_rec # Return all records


@torch.no_grad()
def extract_temporal_representation(model, data_loader, layer_name, config, device, record_type='spk'):
    """
    Extracts the full temporal representation (e.g., spike, membrane, or current sequences)
    from a specified layer, and flattens it to the format [num_samples, num_steps * num_neurons].

    Args:
        model (torch.nn.Module): The trained SNN model.
        data_loader (DataLoader): The data loader used for extraction (e.g., test set).
        layer_name (str): The name of the layer to extract representation from (e.g., 'layer1', 'layer4').
        config (Config): The configuration object (primarily for num_steps).
        device (torch.device): The device.
        record_type (str): The type of record to extract ('spk', 'mem', or 'cur').

    Returns:
        torch.Tensor: The flattened temporal representation vector for all samples in the layer.
                      Shape: [num_samples, num_steps * num_neurons].
        torch.Tensor: The corresponding ground truth labels [num_samples].
                      Returns None, None if extraction fails.
    """
    model.eval()
    all_representations_flat = []
    all_labels = []
    num_steps = config.num_steps

    print(f"Extracting flattened temporal '{record_type}' representations from {layer_name}...")
    for data, targets in data_loader:
        data = data.to(device)
        targets = targets.to(device)

        # --- Run the model ---
        try:
            spk_rec, mem_rec, cur_rec = model(data)
        except ValueError:
            print("Warning: Model forward pass did not return three values. Assuming spk_rec, mem_rec.")
            spk_rec, mem_rec = model(data)
            cur_rec = {}
        except Exception as e:
            print(f"Error during model forward pass: {e}. Skipping batch.")
            continue

        # --- Select the record dictionary to use ---
        rec_dict = None
        if record_type == 'spk':
            rec_dict = spk_rec
        elif record_type == 'mem':
            rec_dict = mem_rec
        elif record_type == 'cur':
            if not cur_rec:
                 print(f"Warning: Input current record ('cur') requested but not available from model. Skipping batch.")
                 continue
            rec_dict = cur_rec
        else:
            raise ValueError(f"Unknown record_type: {record_type}")

        # --- Check if layer exists ---
        if layer_name not in rec_dict:
             print(f"Warning: Layer '{layer_name}' not found in {record_type}_rec. Skipping batch.")
             continue

        # --- Extract temporal data ---
        temporal_data = rec_dict[layer_name]

        # Validate the time step dimension
        if temporal_data.shape[0] != num_steps:
            print(f"Warning: Extracted data for {layer_name} has {temporal_data.shape[0]} time steps, expected {num_steps}. Check config/model. Using extracted steps.")
            current_num_steps = temporal_data.shape[0]
        else:
            current_num_steps = num_steps

        if temporal_data.dim() != 3 or temporal_data.shape[1] != data.size(0):
            print(f"Error: Unexpected shape for {layer_name} {record_type}_rec: {temporal_data.shape}. Expected [time, batch, neurons]. Skipping batch.")
            continue

        batch_size = temporal_data.shape[1]
        num_neurons = temporal_data.shape[2]

        # --- Reshape and flatten ---
        # 1. Permute to [batch, time, neurons]
        batch_repr_temporal = temporal_data.permute(1, 0, 2).contiguous()

        # 2. Flatten the time and neuron dimensions: [batch, time * neurons]
        batch_repr_flat = batch_repr_temporal.view(batch_size, current_num_steps * num_neurons).cpu()

        # --- Collect results ---
        all_representations_flat.append(batch_repr_flat)
        all_labels.append(targets.cpu())

    # --- Handle cases where no data was extracted ---
    if not all_representations_flat:
         print(f"Error: No temporal representations were extracted for layer '{layer_name}' type '{record_type}'.")
         return None, None

    # --- Concatenate all batches ---
    all_representations_flat = torch.cat(all_representations_flat, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"Finished extraction. Flattened temporal representation shape: {all_representations_flat.shape}, Labels shape: {all_labels.shape}")
    return all_representations_flat, all_labels


@torch.no_grad()
def calculate_information_plane_metrics(model, data_loader, config, device, bins=20):
    """
    Calculates the mutual information required for the information plane: MI(Layer, Input) and MI(Layer, Label).
    Uses aggregated spike counts as the layer representation.

    Args:
        model (torch.nn.Module): The trained SNN model.
        data_loader (DataLoader): DataLoader containing encoded inputs (X) and labels (Y).
        config (Config): The configuration object.
        device (torch.device): The device.
        bins (int): The number of bins for mutual information calculation.

    Returns:
        dict: A dictionary containing MI(Layer_i, Input) and MI(Layer_i, Label).
              e.g., {'L1_Input_MI': mi, 'L1_Label_MI': mi, ...}
    """
    model.eval()
    all_layer_repr = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
    all_inputs = []
    all_labels = []
    layers_to_process = ['layer1', 'layer2', 'layer3', 'layer4']

    print("Collecting representations for Information Plane analysis...")
    for batch_data, batch_labels in data_loader:
        batch_data_device = batch_data.to(device)

        # --- Run model ---
        try:
            outputs = model(batch_data_device)
            spk_rec = outputs[0]
        except Exception as e:
            print(f"Error during model forward pass: {e}. Skipping batch.")
            continue

        # --- Collect inputs, labels, and layer representations (using aggregated spike counts) ---
        all_inputs.append(batch_data.cpu())
        all_labels.append(batch_labels.cpu())

        for layer_name in layers_to_process:
            if layer_name in spk_rec:
                layer_repr = spk_rec[layer_name].sum(dim=0).cpu()
                all_layer_repr[layer_name].append(layer_repr)
            else:
                print(f"Warning: Spike record for {layer_name} not found in this batch.")
                pass

    # --- Concatenate all batches ---
    try:
        all_inputs = torch.cat(all_inputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        for layer_name in layers_to_process:
            if all_layer_repr[layer_name]:
                all_layer_repr[layer_name] = torch.cat(all_layer_repr[layer_name], dim=0)
            else:
                print(f"Error: No representations collected for {layer_name}. Cannot calculate MI.")
                return None
    except Exception as e:
        print(f"Error concatenating data: {e}")
        return None

    # --- Calculate mutual information ---
    print("Calculating Mutual Information for Information Plane...")
    mi_results = {}
    num_samples = all_inputs.shape[0]
    input_repr_flat = all_inputs.view(num_samples, -1)

    for layer_name in layers_to_process:
        if not isinstance(all_layer_repr[layer_name], torch.Tensor):
             print(f"Skipping MI calculation for {layer_name} due to missing data.")
             mi_results[f'{layer_name}_Input_MI'] = np.nan
             mi_results[f'{layer_name}_Label_MI'] = np.nan
             continue

        layer_repr = all_layer_repr[layer_name]

        # Calculate MI(Layer_i, Input)
        try:
            mi_input = calculate_mutual_information(layer_repr, input_repr_flat, bins=bins)
            mi_results[f'{layer_name}_Input_MI'] = mi_input
            print(f"  MI({layer_name}, Input): {mi_input:.4f}")
        except Exception as e:
            print(f"  Error calculating MI({layer_name}, Input): {e}")
            mi_results[f'{layer_name}_Input_MI'] = np.nan

        # Calculate MI(Layer_i, Label)
        try:
            mi_label = calculate_mutual_information(layer_repr, all_labels, bins=bins)
            mi_results[f'{layer_name}_Label_MI'] = mi_label
            print(f"  MI({layer_name}, Label): {mi_label:.4f}")
        except Exception as e:
            print(f"  Error calculating MI({layer_name}, Label): {e}")
            mi_results[f'{layer_name}_Label_MI'] = np.nan

    return mi_results


def plot_information_plane(mi_data, condition_name, save_path):
    """
    Plots the Information Plane, MI(Layer, Label) vs MI(Layer, Input).

    Args:
        mi_data (dict): A dictionary containing MI(L_i, Input) and MI(L_i, Label).
                        Keys should be in the format 'layerX_Input_MI' and 'layerX_Label_MI'.
        condition_name (str): The name of the current condition, used for the title.
        save_path (str): The path to save the plot.
    """
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    mi_input = [mi_data.get(f'{layer}_Input_MI', np.nan) for layer in layers]
    mi_label = [mi_data.get(f'{layer}_Label_MI', np.nan) for layer in layers]

    # Check for valid MI data points
    valid_points = ~ (np.isnan(mi_input) | np.isnan(mi_label))
    if not np.any(valid_points):
        print(f"No valid MI data to plot for Information Plane for {condition_name}.")
        return

    plt.figure(figsize=(7, 6))
    plt.plot(np.array(mi_input)[valid_points], np.array(mi_label)[valid_points], 'o-', markersize=8, linewidth=2)

    # Add layer labels
    for i, layer in enumerate(layers):
        if valid_points[i]:
            plt.text(mi_input[i] * 1.01, mi_label[i] * 1.01, layer, fontsize=10)

    plt.xlabel('MI(Layer, Input) - Compression')
    plt.ylabel('MI(Layer, Label) - Prediction')
    plt.title(f'Information Plane - {condition_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Ensure directory exists
    plot_dir = os.path.join(save_path, "information_plane")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'info_plane_{condition_name}.png'), dpi=150)
    plt.close()
    print(f"Information Plane plot saved for {condition_name}.")


def calculate_mutual_information(x, y, bins=20):
    """Calculates mutual information between two variables using binning."""
    from sklearn.metrics import mutual_info_score
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.decomposition import PCA
    import torch
    import numpy as np

    # Convert to numpy arrays
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # Handle multi-dimensional data
    if x.ndim > 1 and x.shape[1] > 1:
        # Use PCA for dimensionality reduction to speed up computation
        pca_dim = min(10, x.shape[1])
        x_pca = PCA(n_components=pca_dim).fit_transform(x)
        x = x_pca

    if y.ndim > 1 and y.shape[1] > 1:
        pca_dim = min(10, y.shape[1])
        y_pca = PCA(n_components=pca_dim).fit_transform(y)
        y = y_pca

    # Ensure x and y are 2D arrays
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = y.reshape(-1, 1) if y.ndim == 1 else y

    # Discretize continuous variables
    x_discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    y_discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')

    x_discrete = x_discretizer.fit_transform(x).astype(int)
    y_discrete = y_discretizer.fit_transform(y).astype(int)

    # Handle multi-dimensional discretized data
    if x_discrete.shape[1] > 1:
        x_discrete = np.ravel_multi_index(
            [x_discrete[:, i] for i in range(x_discrete.shape[1])],
            [bins] * x_discrete.shape[1]
        )
    else:
        x_discrete = x_discrete.ravel()

    if y_discrete.shape[1] > 1:
        y_discrete = np.ravel_multi_index(
            [y_discrete[:, i] for i in range(y_discrete.shape[1])],
            [bins] * y_discrete.shape[1]
        )
    else:
        y_discrete = y_discrete.ravel()

    # Calculate mutual information
    mi = mutual_info_score(x_discrete, y_discrete)

    return mi


def calculate_layer_mutual_information(model, data_loader, config, device):
    all_repr = {}
    all_labels = []

    print("Collecting network representation data...")
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)

            # Run model to get activations from each layer
            spk_rec, mem_rec = model(data)

            # Collect spike representations (sum over time)
            for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                if layer_name not in all_repr:
                    all_repr[layer_name] = []
                all_repr[layer_name].append(spk_rec[layer_name].sum(dim=0).cpu())

            # Collect labels
            all_labels.append(labels.cpu())

    # Concatenate all batches
    for layer_name in all_repr:
        all_repr[layer_name] = torch.cat(all_repr[layer_name], dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Calculate mutual information between layers
    print("Calculating mutual information between layers...")
    mi_results = {
        'layer1_layer2': calculate_mutual_information(all_repr['layer1'], all_repr['layer2']),
        'layer2_layer3': calculate_mutual_information(all_repr['layer2'], all_repr['layer3']),
        'layer3_layer4': calculate_mutual_information(all_repr['layer3'], all_repr['layer4']),
        'layer1_labels': calculate_mutual_information(all_repr['layer1'], all_labels),
        'layer2_labels': calculate_mutual_information(all_repr['layer2'], all_labels),
        'layer3_labels': calculate_mutual_information(all_repr['layer3'], all_labels),
        'layer4_labels': calculate_mutual_information(all_repr['layer4'], all_labels)
    }

    # Calculate information preservation ratio
    info_ratio = {
        'layer1_to_layer2': mi_results['layer2_labels'] / mi_results['layer1_labels'] if mi_results[
                                                                                             'layer1_labels'] > 0 else float(
            'nan'),
        'layer2_to_layer3': mi_results['layer3_labels'] / mi_results['layer2_labels'] if mi_results[
                                                                                             'layer2_labels'] > 0 else float(
            'nan'),
        'layer3_to_layer4': mi_results['layer4_labels'] / mi_results['layer3_labels'] if mi_results[
                                                                                             'layer3_labels'] > 0 else float(
            'nan')
    }

    # Print results
    print("\nMutual Information Results:")
    for key, value in mi_results.items():
        print(f"  {key}: {value:.4f}")

    print("\nInformation Preservation Ratio:")
    for key, value in info_ratio.items():
        print(f"  {key}: {value:.4f}")

    # Merge results into a single dictionary
    result_dict = {**mi_results, **info_ratio}

    return result_dict


def estimate_intrinsic_dim_PCA(X, plot=False):
    """Estimate intrinsic dimension using PCA variance analysis"""
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt

    # Compute PCA
    pca = PCA().fit(X)

    # Get explained variance ratio
    var_ratio = pca.explained_variance_ratio_

    # Compute cumulative explained variance
    cum_var_ratio = np.cumsum(var_ratio)

    # Estimate dimension based on 95% variance explained
    dim_95 = np.argmax(cum_var_ratio >= 0.95) + 1

    # Estimate dimension based on the elbow method
    second_deriv = np.diff(np.diff(var_ratio, prepend=0), prepend=0)
    elbow_idx = np.argmax(second_deriv[1:]) + 1

    if plot:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(cum_var_ratio, 'o-')
        plt.axhline(y=0.95, color='r', linestyle='--')
        plt.axvline(x=dim_95 - 0.5, color='g', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Dimension Estimation based on 95% Variance')

        plt.subplot(1, 2, 2)
        plt.plot(var_ratio, 'o-')
        plt.axvline(x=elbow_idx - 0.5, color='g', linestyle='--')
        plt.xlabel('Component Index')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Dimension Estimation based on Elbow Method')

        plt.tight_layout()
        plt.show()

    return {
        'dim_95_variance': dim_95,
        'dim_elbow': elbow_idx,
        'explained_variance_ratio': var_ratio
    }


def load_saved_test_data(path, config):
    saved_data = torch.load(path)
    test_dataset = TensorDataset(saved_data['data'], saved_data['labels'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return test_loader


def plot_aggregated_information_plane(agg_df, save_path):
    """Plot aggregated information plane trajectories for comparison (with error bars)"""
    if agg_df is None or agg_df.empty:
        print("No aggregated data to visualize for information plane.")
        return

    plt.figure(figsize=(8, 7))
    conditions = agg_df['Condition'].unique()
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    color_map = {'Expansive_Peak': 'blue', 'Dissipative_Peak': 'orange', 'Trough_Region': 'green'}
    marker_map = {'layer1': 'o', 'layer2': 's', 'layer3': '^', 'layer4': 'D'}

    for condition in conditions:
        condition_data = agg_df[agg_df['Condition'] == condition].set_index('Layer').reindex(layers)

        mi_input_mean = condition_data['MI_Input_mean'].values
        mi_input_std = condition_data['MI_Input_std'].values
        mi_label_mean = condition_data['MI_Label_mean'].values
        mi_label_std = condition_data['MI_Label_std'].values

        # Handle potential NaNs (if not enough runs to calculate std)
        mi_input_std = np.nan_to_num(mi_input_std)
        mi_label_std = np.nan_to_num(mi_label_std)

        # Check for valid data points
        valid_points = ~ (np.isnan(mi_input_mean) | np.isnan(mi_label_mean))
        if not np.any(valid_points):
            continue

        line_color = color_map.get(condition, 'gray')

        # Plot trajectory line
        plt.plot(mi_input_mean[valid_points], mi_label_mean[valid_points], '-',
                 color=line_color, linewidth=1.5, alpha=0.6, label=f'_{condition}_line')

        # Plot points with error bars
        for i in range(len(layers)):
            if valid_points[i]:
                plt.errorbar(mi_input_mean[i], mi_label_mean[i],
                             xerr=mi_input_std[i], yerr=mi_label_std[i],
                             fmt=marker_map.get(layers[i], 'o'),
                             color=line_color, markersize=8, capsize=4,
                             label=f'{condition} L{i+1}' if i==0 else f'_{condition} L{i+1}')
                # (Optional) label layer number next to the point
                # plt.text(mi_input_mean[i]*1.02, mi_label_mean[i]*1.02, f"L{i+1}", fontsize=9, color=line_color)


    plt.xlabel('MI(Layer, Input) - Compression Axis')
    plt.ylabel('MI(Layer, Label) - Prediction Axis')
    plt.title('Aggregated Information Plane Comparison (Mean ± Std)')
    plt.legend(title='Condition & Layer', bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(os.path.join(save_path, "aggregated_information_plane_comparison.png"), bbox_inches='tight')
    plt.close()
    print("Aggregated Information Plane comparison plot saved.")


def plot_aggregated_id_comparison(agg_df, id_metric_name, save_path):
    """Plots aggregated intrinsic dimension comparison with error bars."""
    if agg_df is None or agg_df.empty or f'{id_metric_name}_mean' not in agg_df.columns:
        print(f"No aggregated data or missing {id_metric_name} data to visualize.")
        return

    plt.figure(figsize=(8, 6))
    conditions = agg_df['Condition'].unique()
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    color_map = {'Expansive_Peak': 'blue', 'Dissipative_Peak': 'orange', 'Trough_Region': 'green'}
    mean_col = f'{id_metric_name}_mean'
    std_col = f'{id_metric_name}_std'

    for condition in conditions:
        condition_data = agg_df[agg_df['Condition'] == condition].set_index('Layer').reindex(layers)
        means = condition_data[mean_col].values
        stds = condition_data[std_col].values
        stds = np.nan_to_num(stds)

        valid_points = ~np.isnan(means)
        if not np.any(valid_points): continue

        plt.errorbar(np.array(layers)[valid_points], means[valid_points], yerr=stds[valid_points],
                     fmt='o-', label=condition, color=color_map.get(condition, 'gray'),
                     linewidth=2, markersize=6, capsize=4)

    plt.xlabel('Network Layer')
    plt.ylabel(f'Estimated Intrinsic Dimension ({id_metric_name})')
    plt.title(f'Intrinsic Dimension Comparison ({id_metric_name})')
    plt.legend(title='Condition')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"aggregated_{id_metric_name}_comparison.png"))
    plt.close()
    print(f"Aggregated {id_metric_name} comparison plot saved.")


def analyze_mi_id_multiple_runs(num_runs=10, process_id=True, process_mi=True):
    """Main function to run multiple experiments and analyze MI and ID"""
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    config = Config()
    config.num_steps = 5
    config.tmax = 8

    # Analysis conditions
    analysis_conditions = [
        { 'name': 'Expansive_Peak', 'delta': -1.5 },
        { 'name': 'Dissipative_Peak', 'delta': 10.0 },
        # { 'name': 'Trough_Region', 'delta': 2.0 }
    ]

    all_run_data = []
    layers_to_analyze = ['layer1', 'layer2', 'layer3', 'layer4']

    # --- Loop through conditions and runs ---
    for condition in analysis_conditions:
        exp_name = condition['name']
        delta_val = condition['delta']
        print(f"\nAnalyzing Condition: {exp_name} (Delta: {delta_val}) across {num_runs} runs...")

        for run_idx in range(num_runs):
            print(f"  Processing Run {run_idx}...")

            # Build file paths
            model_path = f'mixed_oscillator_results/saved_models/model_delta_{delta_val:.2f}_run{run_idx}.pth'
            test_dataset_path = f'mixed_oscillator_results/saved_models/encoded_test_delta_{delta_val:.2f}_run{run_idx}.pt'

            # Check if files exist
            if not os.path.exists(model_path) or not os.path.exists(test_dataset_path):
                print(f"    Skipping Run {run_idx}: Model or Data file not found.")
                # Add placeholders
                for layer_name in layers_to_analyze:
                    all_run_data.append({
                        'Condition': exp_name, 'Run': run_idx, 'Layer': layer_name, 'Delta': delta_val,
                        'MI_Input': np.nan, 'MI_Label': np.nan, 'ID_PCA_95': np.nan
                    })
                continue

            # Load data and model
            test_loader = load_saved_test_data(test_dataset_path, config)
            if test_loader is None:
                 print(f"    Skipping Run {run_idx}: Failed to load test data.")
                 for layer_name in layers_to_analyze: all_run_data.append({'Condition': exp_name, 'Run': run_idx, 'Layer': layer_name, 'Delta': delta_val, 'MI_Input': np.nan, 'MI_Label': np.nan, 'ID_PCA_95': np.nan})
                 continue

            model = Net(config, encoding='mixed_oscillator').to(device)
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
            except Exception as e:
                print(f"    Skipping Run {run_idx}: Failed to load model: {e}")
                for layer_name in layers_to_analyze: all_run_data.append({'Condition': exp_name, 'Run': run_idx, 'Layer': layer_name, 'Delta': delta_val, 'MI_Input': np.nan, 'MI_Label': np.nan, 'ID_PCA_95': np.nan})
                continue

            # --- Calculate MI ---
            info_plane_mi = None
            if process_mi:
                print(f"    Calculating MI for Run {run_idx}...")
                info_plane_mi = calculate_information_plane_metrics(model, test_loader, config, device)
                if info_plane_mi is None:
                    print(f"    MI calculation failed for Run {run_idx}.")
            else:
                 print(f"    Skipping MI calculation for Run {run_idx}.")


            # --- Calculate ID ---
            run_id_results = {}
            if process_id:
                print(f"    Calculating ID for Run {run_idx}...")
                for layer_name in layers_to_analyze:
                    print(f"      Processing Layer {layer_name} for ID...")
                    id_pca_95, id_pca_elbow, id_two_nn = np.nan, np.nan, np.nan
                    try:
                        repr_flat, _ = extract_temporal_representation(
                            model, test_loader, layer_name, config, device, record_type='spk'
                        )
                        if repr_flat is not None and repr_flat.shape[0] > 1:
                            repr_np = repr_flat.cpu().numpy()
                            # PCA ID
                            if not np.allclose(np.var(repr_np, axis=0), 0):
                                pca_results = estimate_intrinsic_dim_PCA(repr_np, plot=False)
                                id_pca_95 = pca_results.get('dim_95_variance', np.nan)

                            else: print(f"      Skipping PCA ID for {layer_name} (zero variance).")
                        else: print(f"      Skipping ID for {layer_name} (extraction failed or insufficient samples).")

                    except Exception as e:
                        print(f"      Error calculating ID for {layer_name}: {e}")
                    finally:
                        run_id_results[layer_name] = {'ID_PCA_95': id_pca_95}
                        print(f"      ID results for {layer_name}: PCA95={id_pca_95}")


            # --- Store aggregated results for the current run ---
            for layer_name in layers_to_analyze:
                 layer_mi_input = info_plane_mi.get(f'{layer_name}_Input_MI', np.nan) if info_plane_mi else np.nan
                 layer_mi_label = info_plane_mi.get(f'{layer_name}_Label_MI', np.nan) if info_plane_mi else np.nan
                 ids = run_id_results.get(layer_name, {'ID_PCA_95': np.nan})

                 all_run_data.append({
                     'Condition': exp_name, 'Run': run_idx, 'Layer': layer_name, 'Delta': delta_val,
                     'MI_Input': layer_mi_input, 'MI_Label': layer_mi_label,
                     'ID_PCA_95': ids['ID_PCA_95']
                 })


    # --- Aggregate and save data ---
    if not all_run_data:
        print("\nNo data collected from any runs. Exiting.")
        return

    results_df = pd.DataFrame(all_run_data)

    # Save raw run data
    output_dir = os.path.join("mixed_oscillator_results", "mi_id_analysis")
    os.makedirs(output_dir, exist_ok=True)
    raw_save_path = os.path.join(output_dir, "raw_mi_id_metrics_all_runs.csv")
    results_df.to_csv(raw_save_path, index=False)
    print(f"\nRaw MI & ID results for all runs saved to {raw_save_path}")

    # Aggregate by computing Mean and Std
    aggregated_results = results_df.groupby(['Condition', 'Layer']).agg(
        MI_Input_mean=('MI_Input', 'mean'), MI_Input_std=('MI_Input', 'std'),
        MI_Label_mean=('MI_Label', 'mean'), MI_Label_std=('MI_Label', 'std'),
        ID_PCA_95_mean=('ID_PCA_95', 'mean'), ID_PCA_95_std=('ID_PCA_95', 'std'),
        run_count=('Run', 'count')
    ).reset_index()

    # Save aggregated results
    agg_save_path = os.path.join(output_dir, "aggregated_mi_id_metrics.csv")
    aggregated_results.to_csv(agg_save_path, index=False)
    print(f"\nAggregated MI & ID results saved to {agg_save_path}")
    print(aggregated_results)

    # --- Visualize aggregated results ---
    plot_aggregated_information_plane(aggregated_results, output_dir)
    plot_aggregated_id_comparison(aggregated_results, 'ID_PCA_95', output_dir)


    print("\n=== MI and ID Analysis Complete ===")


if __name__ == "__main__":
    analyze_mi_id_multiple_runs(num_runs=10, process_id=True, process_mi=True)
