import snntorch as snn
import torch.nn as nn

import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from joblib import load

from base_Attactors import Config



class Net(nn.Module):
    def __init__(self, config, encoding):
        super().__init__()
        self.encoding = encoding
        self.num_steps = config.num_steps
        self.config = config

        # --- Input dimension calculation ---
        if encoding in ['lorenz', 'chen', 'rossler', 'aizawa', 'nose_hoover', 'sprott', 'chua','mixed_oscillator']:
            self.input_dim = config.n_components * 3
        elif encoding == 'default':
            self.input_dim = config.num_inputs
        elif encoding == 'umap':
            self.input_dim = config.n_components
        else:
            self.input_dim = config.num_inputs

        # --- Fully connected layers ---
        self.fc1 = nn.Linear(self.input_dim, config.num_hidden)
        self.fc2 = nn.Linear(config.num_hidden, config.num_hidden)
        self.fc3 = nn.Linear(config.num_hidden, config.num_hidden)
        self.fc4 = nn.Linear(config.num_hidden, config.num_outputs)

        # --- LIF neurons ---
        self.lif1 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)
        self.lif2 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)
        self.lif3 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)
        self.lif4 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)

    def forward(self, x, perturb_type='none', perturb_rate=0.0):
        """
        Args:
            x: Input data
            perturb_type (str): Perturbation type: 'none', 'add_remove_spikes'.
            perturb_rate (float): Perturbation probability (e.g., 0.01 means 1% probability to add or remove a spike).
        """
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
            if self.encoding in ['default', 'umap']:
                 current_input = x.view(x.size(0), -1) if step == 0 else torch.zeros(x.size(0), self.config.num_hidden, device=x.device) # Handle non-temporal input
            elif x.dim() == 3 and x.shape[1] == self.num_steps:
                 current_input = x[:, step, :]
            elif x.dim() == 2: # Handle potentially pre-processed static input
                 current_input = x if step == 0 else torch.zeros(x.size(0), self.config.num_hidden, device=x.device)
            else:
                 raise ValueError(f"Unexpected input shape: {x.shape}")

            # --- Process through layers ---
            layer_input = current_input
            for i, layer_name in enumerate(layer_order):
                 fc_layer = fc_layers[layer_name]
                 lif_layer = lif_layers[layer_name]

                 # Check dimension
                 expected_dim = fc_layer.weight.shape[1]
                 if layer_input.shape[-1] != expected_dim:
                      # Try to handle dimension compression for batch_size=1
                      if layer_input.ndim == 1 and expected_dim == layer_input.shape[0]:
                           layer_input = layer_input.unsqueeze(0) # Add batch dimension
                      # Handle possible flattening (e.g. previous layer only has one neuron)
                      elif layer_input.numel() == layer_input.shape[0]*expected_dim :
                           try:
                               layer_input = layer_input.view(layer_input.shape[0], expected_dim)
                           except RuntimeError:
                               raise ValueError(f"Dimension mismatch for {layer_name}: input shape {layer_input.shape}, layer expects {expected_dim}")
                      elif layer_input.shape[-1] != expected_dim:
                            raise ValueError(f"Dimension mismatch for {layer_name}: input shape {layer_input.shape}, layer expects {expected_dim}")


                 current_val = fc_layer(layer_input)
                 spk_val, mem_val = lif_layer(current_val, mem_state[layer_name])
                 mem_state[layer_name] = mem_val

                 # --- New: Apply spike perturbation (in hidden layers L1, L2, L3) ---
                 perturbed_spk_val = spk_val # No change by default
                 if perturb_type == 'add_remove_spikes' and perturb_rate > 0 and layer_name != 'layer4':
                     # Apply perturbation to spk_val
                     noise_mask = torch.rand_like(spk_val)
                     # Remove spikes: original 1s become 0 if random number is < p
                     remove_mask = (spk_val == 1) & (noise_mask < perturb_rate)
                     perturbed_spk_val = torch.where(remove_mask, torch.zeros_like(spk_val), spk_val)
                     # Add spikes: original 0s become 1 if random number is < p
                     add_mask = (spk_val == 0) & (noise_mask < perturb_rate)
                     perturbed_spk_val = torch.where(add_mask, torch.ones_like(spk_val), perturbed_spk_val)

                 # --- Record activities (record perturbed spikes) ---
                 if layer_name not in spk_rec_dict: spk_rec_dict[layer_name] = []
                 if layer_name not in mem_rec_dict: mem_rec_dict[layer_name] = []
                 if layer_name not in cur_rec_dict: cur_rec_dict[layer_name] = []

                 spk_rec_dict[layer_name].append(perturbed_spk_val) # <-- Record perturbed spikes
                 mem_rec_dict[layer_name].append(mem_val)
                 cur_rec_dict[layer_name].append(current_val)

                 layer_input = perturbed_spk_val # <-- Use perturbed spikes as input for next layer

        # --- Stack temporal sequences ---
        try:
            spk_rec = {k: torch.stack(v, dim=0) for k, v in spk_rec_dict.items() if v}
            mem_rec = {k: torch.stack(v, dim=0) for k, v in mem_rec_dict.items() if v}
            cur_rec = {k: torch.stack(v, dim=0) for k, v in cur_rec_dict.items() if v}
        except Exception as e:
            print(f"Error stacking recordings: {e}")
            return {}, {}, {}

        return spk_rec, mem_rec, cur_rec



def evaluate_model_with_perturbation(model, data_loader, criterion, config, model_type, device,
                                      perturb_type='none', perturb_rate=0.0):
    """
    Evaluates model performance, allowing perturbation parameters to be passed to the SNN during evaluation.

    Args:
        ... (original evaluate_model parameters) ...
        perturb_type (str): Perturbation type passed to the SNN forward pass.
        perturb_rate (float): Perturbation rate passed to the SNN forward pass.

    Returns:
        avg_loss (float): The average loss.
        accuracy (float): The accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    is_snn = (model_type == 'SNN')

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)

            # --- Model forward pass ---
            if is_snn:
                try:
                    # Call SNN's forward with perturbation parameters
                    spk_rec, mem_rec, _ = model(data,
                                                perturb_type=perturb_type,
                                                perturb_rate=perturb_rate)
                except TypeError as e:
                     print(f"Error calling SNN forward with perturbation args: {e}. Maybe model definition is outdated?")
                     # Try calling without parameters for a baseline evaluation
                     try:
                          spk_rec, mem_rec, _ = model(data)
                          if perturb_rate > 0: print("Warning: Perturbation args ignored by model.")
                     except Exception as e_inner:
                          print(f"Error calling SNN forward without args either: {e_inner}")
                          # If evaluation fails, set invalid results
                          loss = torch.tensor(float('nan'))
                          predicted = torch.zeros_like(targets) - 1
                          # Skip further computation for this batch
                          if not torch.isnan(loss): test_loss += loss.item() * data.size(0)
                          total += targets.size(0) # Still count total, but invalid predictions will be excluded from accuracy
                          correct += predicted.eq(targets).sum().item() # Will be 0
                          continue # Jump to next batch

                except Exception as e:
                    print(f"Unexpected error during SNN forward: {e}")
                    loss = torch.tensor(float('nan'))
                    predicted = torch.zeros_like(targets) - 1
                    if not torch.isnan(loss): test_loss += loss.item() * data.size(0)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    continue

                # Calculate SNN loss and predictions (similar to original evaluate_model)
                if 'layer4' in mem_rec and 'layer4' in spk_rec and \
                   mem_rec['layer4'] is not None and spk_rec['layer4'] is not None and \
                   mem_rec['layer4'].numel() > 0 and spk_rec['layer4'].numel() > 0:
                    # Check if number of steps matches
                    steps_available = mem_rec['layer4'].shape[0]
                    if steps_available == config.num_steps:
                         loss = torch.stack([criterion(mem_rec['layer4'][step], targets)
                                             for step in range(config.num_steps)]).mean()
                         _, predicted = spk_rec['layer4'].sum(dim=0).max(1)
                    else:
                         print(f"Warning: Mismatched steps in layer4 record ({steps_available} vs {config.num_steps}). Cannot compute loss/preds reliably.")
                         loss = torch.tensor(float('nan'))
                         predicted = torch.zeros_like(targets) - 1
                else:
                    print(f"Warning: Missing or empty layer4 record in SNN output.")
                    loss = torch.tensor(float('nan'))
                    predicted = torch.zeros_like(targets) - 1

            else: # ANN
                # ANN models typically do not accept perturbation parameters
                try:
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    _, predicted = outputs.max(1)
                except Exception as e:
                    print(f"Error during ANN forward pass: {e}")
                    loss = torch.tensor(float('nan'))
                    predicted = torch.zeros_like(targets) - 1


            # Update statistics (add check to ensure predicted and targets shapes match)
            valid_preds_mask = (predicted != -1)
            if valid_preds_mask.shape == targets.shape:
                valid_targets = targets[valid_preds_mask]
                valid_predicted = predicted[valid_preds_mask]
                current_total = valid_targets.size(0)
                current_correct = valid_predicted.eq(valid_targets).sum().item()
                total += current_total
                correct += current_correct
            else:
                print(f"Warning: Shape mismatch between predicted ({predicted.shape}) and targets ({targets.shape}) after masking.")


            if not torch.isnan(loss):
                # Use the number of valid samples to weight the loss (if current_total was calculated)
                test_loss += loss.item() * current_total if 'current_total' in locals() and current_total > 0 else 0

    # Calculate final metrics
    avg_loss = test_loss / total if total > 0 else float('nan')
    accuracy = 100. * correct / total if total > 0 else float('nan')

    return avg_loss, accuracy


def test_spike_perturbation_robustness(model, data_loader, criterion, config, device,
                                       perturb_type='add_remove_spikes',
                                       perturb_levels=[0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]):
    """Tests the sensitivity of an SNN model to internal spike addition/removal perturbations."""
    results = []

    # Evaluate baseline accuracy (perturb_rate = 0)
    print(f"  Evaluating baseline accuracy (Perturb Rate=0.0)...")
    try:
        original_loss, original_accuracy = evaluate_model_with_perturbation(
            model, data_loader, criterion, config, 'SNN', device,
            perturb_type='none', perturb_rate=0.0
        )
        if np.isnan(original_accuracy): raise ValueError("Baseline accuracy is NaN")
        print(f"  Baseline Accuracy: {original_accuracy:.2f}%")
    except Exception as e:
        print(f"  Error evaluating baseline accuracy: {e}")
        return None

    results.append({
        'perturb_rate': 0.0, 'avg_accuracy': original_accuracy, 'std_accuracy': 0.0,
        'accuracy_drop': 0.0, 'relative_drop': 0.0
    })

    # Test different non-zero perturbation levels
    for p_rate in perturb_levels:
        if p_rate == 0.0: continue

        print(f"  Testing Perturb Rate: {p_rate}")
        run_accuracies = []
        # Run multiple times to get an average, as the perturbation is stochastic
        num_eval_runs = 10
        for i in range(num_eval_runs):
            try:
                _, accuracy = evaluate_model_with_perturbation(
                    model, data_loader, criterion, config, 'SNN', device,
                    perturb_type=perturb_type, perturb_rate=p_rate
                )
                if np.isnan(accuracy): raise ValueError("Accuracy is NaN")
                run_accuracies.append(accuracy)
            except Exception as e:
                print(f"    Evaluation error (Run {i+1}, p={p_rate}): {e}")
                # Record NaN so subsequent processing knows this evaluation failed
                run_accuracies.append(np.nan)

        # Calculate statistics (using only valid run results)
        valid_accuracies = [acc for acc in run_accuracies if not np.isnan(acc)]
        if not valid_accuracies:
             print(f"    All evaluation runs failed for p={p_rate}")
             avg_accuracy, std_accuracy, accuracy_drop, relative_drop = np.nan, np.nan, np.nan, np.nan
        else:
             avg_accuracy = np.mean(valid_accuracies)
             std_accuracy = np.std(valid_accuracies)
             accuracy_drop = original_accuracy - avg_accuracy
             relative_drop = (accuracy_drop / original_accuracy) if original_accuracy > 0 else 0.0

        results.append({
            'perturb_rate': p_rate,
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'accuracy_drop': accuracy_drop,
            'relative_drop': relative_drop
        })

        if not np.isnan(avg_accuracy):
             print(f"    Average Accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
             print(f"    Accuracy Drop: {accuracy_drop:.2f}% ({relative_drop * 100:.2f}% Relative Drop)")
        else:
             print(f"    Failed to calculate valid results for p={p_rate}.")

    return results


def load_saved_test_data(path):
    """Loads saved encoded test data"""
    try:
        saved_data = torch.load(path, map_location='cpu')
        data_tensor = saved_data['data'] if isinstance(saved_data['data'], torch.Tensor) else torch.tensor(saved_data['data'])
        labels_tensor = saved_data['labels'] if isinstance(saved_data['labels'], torch.Tensor) else torch.tensor(saved_data['labels'])
        test_dataset = TensorDataset(data_tensor, labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        return test_loader
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return None
    except Exception as e:
        print(f"Error loading data from {path}: {e}")
        return None


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    config = Config()
    config.num_steps = 5
    config.tmax = 8

    analysis_conditions = [
        {
            'name': 'Expansive_Peak', 'delta': -1.5,
            'snn_model_path': 'mixed_oscillator_results/saved_models/model_delta_-1.50_run0.pth',
            'test_dataset_path': 'mixed_oscillator_results/saved_models/encoded_test_delta_-1.50_run0.pt'
        },
        {
            'name': 'Dissipative_Peak', 'delta': 10.0,
            'snn_model_path': 'mixed_oscillator_results/saved_models/model_delta_10.00_run0.pth',
            'test_dataset_path': 'mixed_oscillator_results/saved_models/encoded_test_delta_10.00_run0.pt'
        },
    ]

    spike_perturbation_results = []

    criterion = torch.nn.CrossEntropyLoss()

    for condition in analysis_conditions:
        exp_name = condition['name']
        snn_model_path = condition['snn_model_path']
        delta_val = condition['delta']

        print(f"\n=== Analyzing Condition: {exp_name} (Spike Perturbation Robustness) ===")


        data_path = condition['test_dataset_path']
        if not os.path.exists(data_path):
            print(f"  Test data file not found: {data_path}. Skipping.")
            continue
        test_loader = load_saved_test_data(data_path)
        if test_loader is None: continue

        # --- Load SNN model ---
        snn_model = None
        if os.path.exists(snn_model_path):
            try:
                snn_model = Net(config, encoding='mixed_oscillator').to(device)
                snn_model.load_state_dict(torch.load(snn_model_path, map_location=device))
                print(f"  SNN model loaded: {snn_model_path}")

                print("\n  Testing SNN internal spike perturbation robustness...")
                # *** Call the new robustness test function ***
                sensitivity_result = test_spike_perturbation_robustness(
                    snn_model, test_loader, criterion, config, device,
                    perturb_type='add_remove_spikes',
                    perturb_levels=[0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                )

                if sensitivity_result is not None:
                    for result in sensitivity_result:
                        result['Condition'] = exp_name
                        result['Delta'] = delta_val
                        result['ModelType'] = 'SNN'
                        spike_perturbation_results.append(result)
                else:
                    print("  SNN spike perturbation robustness analysis failed.")

            except Exception as e:
                print(f"  Error loading or analyzing SNN model: {e}")
        else:
            print(f"  SNN model file not found: {snn_model_path}.")


    # --- Save spike perturbation robustness results ---
    if spike_perturbation_results:
        robustness_df = pd.DataFrame(spike_perturbation_results)
        output_dir = os.path.join("mixed_oscillator_results", "robustness_analysis")
        os.makedirs(output_dir, exist_ok=True)
        # Update filename
        robustness_df.to_csv(os.path.join(output_dir, "spike_perturbation_robustness_summary.csv"), index=False)
        print(f"\nInternal spike perturbation robustness results saved to {output_dir}")

        # --- Plot spike perturbation robustness comparison ---
        print("Plotting internal spike perturbation robustness comparison...")
        plt.figure(figsize=(8, 6))
        color_map = {'Expansive_Peak': 'blue', 'Dissipative_Peak': 'orange', 'Trough_Region': 'green'}
        linestyle_map = {'SNN': '-'}
        marker_map = {'SNN': 'o'}

        # Plot grouped by Condition
        grouped = robustness_df[robustness_df['ModelType']=='SNN'].groupby('Condition')
        for name, group in grouped:
            condition_name = name
            if group['accuracy_drop'].notna().any():
                plt.plot(group['perturb_rate'], group['accuracy_drop'],
                         label=f"{condition_name}",
                         color=color_map.get(condition_name, 'gray'),
                         linestyle=linestyle_map['SNN'],
                         marker=marker_map['SNN'],
                         linewidth=2)

        plt.title('SNN Robustness to Internal Spike Perturbation')
        plt.xlabel('Spike Perturbation Rate (p)')
        plt.ylabel('Accuracy Drop (%)')
        plt.legend(title='Condition', bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.grid(alpha=0.3)
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        # Update filename
        plt.savefig(os.path.join(output_dir, "spike_perturbation_robustness_comparison.png"))
        plt.close()
        print("Internal spike perturbation robustness comparison plot saved.")
    else:
        print("\nNo valid internal spike perturbation robustness results to save or plot.")

    print("\n=== Spike Perturbation Robustness Analysis Complete ===")
