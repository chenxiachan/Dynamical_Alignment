import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import umap
from typing import Dict, List, Tuple, Optional


try:
    from encoding import mixed_oscillator_encode
    from nns import Net, ANNNet
    from utils import calculate_spike_metrics, extract_layer_representations, visualize_representation, \
        evaluate_linear_separability, analyze_internal_dynamics, save_raster_plots, save_spike_counts_hist, \
        save_membrane_potential_traces, plot_layer_comparison_rasters

    SNN_AVAILABLE = True
    ANN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SNN modules not fully available: {e}")
    try:
        from nns import ANNNet

        ANN_AVAILABLE = True
        SNN_AVAILABLE = False
    except ImportError:
        print("Will use fallback MLP implementation.")
        ANN_AVAILABLE = False
        SNN_AVAILABLE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 1. Unified Configuration Class ---
class UnifiedConfig:
    """Unified experiment configuration to ensure SNN and ANN use the same basic settings"""

    # Dataset parameters
    num_samples = 5000
    feature_dim_a = 500
    feature_dim_b = 500
    noise_level = 0.25

    # Preprocessing parameters
    apply_umap = True
    n_components_umap = 64

    # Network parameters - compatible with nns.py
    num_hidden = 64
    num_outputs = 2
    num_inputs = 1000

    # SNN-specific parameters
    beta = 0.95
    num_steps = 5
    tmax_encoding = 8

    # Dynamical encoding parameters
    osc_alpha = 2.0
    osc_beta_osc = 0.1
    osc_gamma = 0.1
    osc_omega = 1.0
    osc_drive = 0.0
    osc_dt = 0.05

    # Training parameters
    batch_size = 64
    learning_rate = 1e-4
    num_epochs = 100
    early_stopping_patience = 15

    # Experiment parameters
    test_size = 0.2
    random_state = 42

    def get_unified_config(self):
        # Dynamically set n_components
        self.n_components = self.n_components_umap
        return self


# --- 2. Data Generation ---
def generate_binding_dataset(num_samples: int, dim_a: int, dim_b: int,
                             noise_level: float = 0.05, random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a synthetic feature binding dataset, inspired by the cognitive task described in the paper.
    It simulates the challenge of binding two distinct feature patterns in a noisy, high-dimensional space.

    Args:
        num_samples (int): The total number of samples to generate.
        dim_a (int): The dimensionality of the first feature pattern.
        dim_b (int): The dimensionality of the second feature pattern.
        noise_level (float, optional): The standard deviation of the Gaussian noise added to the features.
                                       Defaults to 0.05. The paper uses a noise level of 0.25.
        random_state (int, optional): A random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the feature tensor (X) and the label tensor (y).
    """

    # Set random seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Define the target patterns for the two features
    # These patterns are sparse and localized within their respective dimensions.
    target_pattern_a = torch.zeros(dim_a)
    target_pattern_a[dim_a // 4: dim_a // 2] = 1
    target_pattern_b = torch.zeros(dim_b)
    target_pattern_b[dim_b // 2: 3 * dim_b // 4] = 1

    X = []
    y = []

    num_positive_needed = num_samples // 2
    num_negative_needed = num_samples - num_positive_needed

    # Generate positive samples (both target patterns present)
    for _ in range(num_positive_needed):
        pattern_a = target_pattern_a.clone()
        pattern_b = target_pattern_b.clone()
        label = 1

        # Combine patterns and add Gaussian noise
        combined_features = torch.cat((pattern_a, pattern_b)).float()
        noise = torch.randn_like(combined_features) * noise_level
        combined_features += noise
        combined_features.clamp_(0, 1)
        X.append(combined_features)
        y.append(label)

    # Generate negative samples (at least one target pattern is missing)
    for _ in range(num_negative_needed):
        while True:
            # Generate random binary patterns that are not the target patterns
            pattern_a = torch.rand(dim_a) > 0.5
            pattern_b = torch.rand(dim_b) > 0.5
            if not (torch.equal(pattern_a, target_pattern_a) and torch.equal(pattern_b, target_pattern_b)):
                break
        label = 0

        combined_features = torch.cat((pattern_a, pattern_b)).float()
        noise = torch.randn_like(combined_features) * noise_level
        combined_features += noise
        combined_features.clamp_(0, 1)
        X.append(combined_features)
        y.append(label)

    # Shuffle the dataset to ensure sample independence
    indices = torch.randperm(num_samples)
    X_tensor = torch.stack(X)[indices]
    y_tensor = torch.tensor(y, dtype=torch.long)[indices]

    print(f"Generated dataset: X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")
    print(f"Positive samples (label 1): {y_tensor.sum().item()} / {num_samples}")

    return X_tensor, y_tensor


# --- 3. Data Preprocessing Function ---
def preprocess_data(X_raw: torch.Tensor, y_raw: torch.Tensor, config: UnifiedConfig,
                    random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """A unified data preprocessing function"""

    X_processed = X_raw.cpu()

    if config.apply_umap:
        print(f"Applying UMAP, reducing to {config.n_components_umap} components...")
        if X_raw.shape[0] < 5 or X_raw.shape[0] <= config.n_components_umap:
            print("Warning: Too few samples for UMAP. Skipping UMAP.")
        elif X_raw.shape[0] < 15:
            reducer = umap.UMAP(n_components=config.n_components_umap,
                                n_neighbors=max(2, X_raw.shape[0] - 1),
                                min_dist=0.1, random_state=random_state, n_jobs=1)
            try:
                X_reduced_np = reducer.fit_transform(X_raw.cpu().numpy())
                X_processed = torch.from_numpy(X_reduced_np).float()
                print(f"UMAP complete. New shape: {X_processed.shape}")
            except Exception as e:
                print(f"UMAP failed: {e}. Using raw features.")
        else:
            reducer = umap.UMAP(n_components=config.n_components_umap,
                                n_neighbors=15, min_dist=0.1, random_state=random_state, n_jobs=1)
            try:
                X_reduced_np = reducer.fit_transform(X_raw.cpu().numpy())
                X_processed = torch.from_numpy(X_reduced_np).float()
                print(f"UMAP complete. New shape: {X_processed.shape}")
            except Exception as e:
                print(f"UMAP failed: {e}. Using raw features.")

    return X_processed, y_raw.cpu()

# --- 4. SNN Encoding Function ---
def encode_snn_data(X_processed: torch.Tensor, y_processed: torch.Tensor,
                    config: UnifiedConfig, dynamic_params: Dict) -> TensorDataset:
    """SNN dynamical encoding function"""

    if not SNN_AVAILABLE:
        raise RuntimeError("SNN modules not available")

    print(f"Starting SNN encoding with dynamic_params: {dynamic_params}")
    X_to_encode = X_processed.to(device)

    osc_params = {
        'alpha': config.osc_alpha,
        'beta': config.osc_beta_osc,
        'delta': dynamic_params['delta'],
        'gamma': config.osc_gamma,
        'omega': config.osc_omega,
        'drive': config.osc_drive
    }

    encoded_X_list = []
    temp_loader = DataLoader(TensorDataset(X_to_encode), batch_size=config.batch_size, shuffle=False)

    for batch_x_tuple in temp_loader:
        batch_x = batch_x_tuple[0]
        encoded_batch = mixed_oscillator_encode(batch_x,
                                                num_steps=config.num_steps,
                                                tmax=config.tmax_encoding,
                                                params=osc_params)
        encoded_X_list.append(encoded_batch.cpu())

    encoded_X = torch.cat(encoded_X_list, dim=0)
    print(f"SNN encoding complete. Encoded X shape: {encoded_X.shape}")
    return TensorDataset(encoded_X, y_processed)

# --- 5. Train & Evaluation ---
def train_ann_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                    config: UnifiedConfig, model_name: str) -> Tuple[nn.Module, float, Dict]:
    """Trains an ANN model (using ANNNet from nns.py)"""

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    print(f"\n--- Starting {model_name} Training ---")
    for epoch in range(config.num_epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        epoch_start_time = time.time()

        for i, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = 100. * train_correct / train_total

        # Evaluation phase
        test_loss, test_acc = evaluate_ann_model(model, test_loader, criterion)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        epoch_duration = time.time() - epoch_start_time
        print(f"{model_name} Epoch [{epoch + 1}/{config.num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"Time: {epoch_duration:.2f}s")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch + 1}.")
            break

    print(f"--- {model_name} Training finished. Best Test Acc: {best_test_acc:.2f}% ---")
    return model, best_test_acc, history


def evaluate_ann_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    """Evaluate ANN model"""
    model.eval()
    total_test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    avg_test_acc = 100. * test_correct / test_total
    return avg_test_loss, avg_test_acc


def train_snn_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                    config: UnifiedConfig, model_name: str) -> Tuple[nn.Module, float, Dict]:
    """Trains an SNN model."""

    if not SNN_AVAILABLE:
        raise RuntimeError("SNN modules not available")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'spikes': []}

    print(f"\n--- Starting {model_name} Training ---")
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        epoch_start_time = time.time()

        for i, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            spk_rec, mem_rec = model(data)

            loss = torch.tensor(0.0, device=device)
            if isinstance(mem_rec, dict) and 'layer4' in mem_rec and mem_rec['layer4'] is not None:
                loss = torch.stack([criterion(mem_rec['layer4'][step], targets)
                                    for step in range(mem_rec['layer4'].shape[0])]).mean()
            elif isinstance(mem_rec, torch.Tensor):
                loss = torch.stack([criterion(mem_rec[step], targets)
                                    for step in range(mem_rec.shape[0])]).mean()
            else:
                print("Warning: mem_rec structure not recognized")
                continue

            if torch.is_tensor(loss) and loss.requires_grad:
                loss.backward()
                optimizer.step()
            else:
                continue

            # Accuracy calculation
            output_for_acc = None
            if isinstance(spk_rec, dict) and 'layer4' in spk_rec and spk_rec['layer4'] is not None:
                summed_spikes = spk_rec['layer4'].sum(dim=0)
                _, predicted = summed_spikes.max(1)
                output_for_acc = predicted
            elif isinstance(mem_rec, dict) and 'layer4' in mem_rec and mem_rec['layer4'] is not None:
                summed_mem = mem_rec['layer4'].sum(dim=0)
                _, predicted = summed_mem.max(1)
                output_for_acc = predicted

            if output_for_acc is not None:
                train_total += targets.size(0)
                train_correct += (output_for_acc == targets).sum().item()

            if torch.is_tensor(loss):
                total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_train_acc = 100. * train_correct / train_total if train_total > 0 else 0

        test_loss, test_acc, avg_spikes = evaluate_snn_model(model, test_loader, criterion, config)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['spikes'].append(avg_spikes)

        epoch_duration = time.time() - epoch_start_time
        print(f"{model_name} Epoch [{epoch + 1}/{config.num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"Avg Spikes: {avg_spikes:.2f} | Time: {epoch_duration:.2f}s")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch + 1}.")
            break

    print(f"--- {model_name} Training finished. Best Test Acc: {best_test_acc:.2f}% ---")
    return model, best_test_acc, history


def evaluate_snn_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module,
                       config: UnifiedConfig) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_spikes_epoch = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            spk_rec, mem_rec = model(data)

            loss = torch.tensor(0.0, device=device)
            if isinstance(mem_rec, dict) and 'layer4' in mem_rec and mem_rec['layer4'] is not None:
                loss = torch.stack([criterion(mem_rec['layer4'][step], targets)
                                    for step in range(mem_rec['layer4'].shape[0])]).mean()
            elif isinstance(mem_rec, torch.Tensor):
                loss = torch.stack([criterion(mem_rec[step], targets)
                                    for step in range(mem_rec.shape[0])]).mean()

            if torch.is_tensor(loss):
                total_loss += loss.item()

            output_for_acc = None
            if isinstance(spk_rec, dict) and 'layer4' in spk_rec and spk_rec['layer4'] is not None:
                summed_spikes = spk_rec['layer4'].sum(dim=0)
                _, predicted = summed_spikes.max(1)
                output_for_acc = predicted
            elif isinstance(mem_rec, dict) and 'layer4' in mem_rec and mem_rec['layer4'] is not None:
                summed_mem = mem_rec['layer4'].sum(dim=0)
                _, predicted = summed_mem.max(1)
                output_for_acc = predicted

            if output_for_acc is not None:
                total += targets.size(0)
                correct += (output_for_acc == targets).sum().item()

            # Calculate spike count
            try:
                batch_spikes, _, _ = calculate_spike_metrics(spk_rec, data.size(0))
                total_spikes_epoch += batch_spikes
            except:
                pass

    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0
    avg_spikes = total_spikes_epoch / total if total > 0 else 0

    return avg_loss, accuracy, avg_spikes

# --- 6. Run Single Experiment ---
def run_single_repetition_experiment(repetition_idx: int, config: UnifiedConfig) -> pd.DataFrame:
    """Runs a single repetition, ensuring SNN and MLP use the same data"""

    print(f"\n{'#' * 10} Starting Unified Repetition {repetition_idx} {'#' * 10}")

    # Create results directory
    base_repetition_dir = f"unified_feature_binding_results/repetition_{repetition_idx}"
    os.makedirs(base_repetition_dir, exist_ok=True)

    # Generate dataset - use a fixed seed to ensure the same base data is generated for each repetition
    repetition_seed = config.random_state + repetition_idx
    print(f"--- Repetition {repetition_idx}: Generating Dataset (seed={repetition_seed}) ---")
    X_raw, y_raw = generate_binding_dataset(
        config.num_samples, config.feature_dim_a, config.feature_dim_b,
        config.noise_level, random_state=repetition_seed
    )

    # Data splitting - use the same random seed
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=config.test_size, random_state=repetition_seed, stratify=y_raw
    )
    print(f"Repetition {repetition_idx}: Data split: Train {X_train_raw.shape[0]}, Test {X_test_raw.shape[0]}")

    # Preprocess data
    print(f"--- Repetition {repetition_idx}: Preprocessing Data ---")
    X_train_processed, y_train_processed = preprocess_data(X_train_raw, y_train_raw, config, repetition_seed)
    X_test_processed, y_test_processed = preprocess_data(X_test_raw, y_test_raw, config, repetition_seed)

    # Update config to match the actual input dimension
    unified_config = config.get_unified_config()
    print(f"Processed data - Input dimension: {X_train_processed.shape[1]}")

    repetition_summary_list = []

    # 1. Train ANN baseline model (using ANNNet)
    if ANN_AVAILABLE:
        print(f"\n{'=' * 20} Repetition {repetition_idx}: ANN Baseline {'=' * 20}")
        ann_train_dataset = TensorDataset(X_train_processed, y_train_processed)
        ann_test_dataset = TensorDataset(X_test_processed, y_test_processed)

        ann_train_loader = DataLoader(ann_train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        ann_test_loader = DataLoader(ann_test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

        # Use ANNNet class
        ann_model = ANNNet(unified_config, encoding='umap')
        trained_ann, ann_best_acc, ann_history = train_ann_model(
            ann_model, ann_train_loader, ann_test_loader, config, "ANN"
        )

        ann_summary = {
            "repetition": repetition_idx,
            "model_type": "ANN",
            "mode": "baseline",
            "delta": None,
            "best_test_accuracy": ann_best_acc,
            "final_train_loss": ann_history['train_loss'][-1] if ann_history['train_loss'] else None,
            "final_test_loss": ann_history['test_loss'][-1] if ann_history['test_loss'] else None,
            "final_avg_spikes": 0,
            "layer4_linearity_spike_count": None,
            "internal_metrics": None
        }
        repetition_summary_list.append(ann_summary)

        # Save ANN training history
        ann_results_dir = os.path.join(base_repetition_dir, "ann_baseline")
        os.makedirs(ann_results_dir, exist_ok=True)
        with open(os.path.join(ann_results_dir, f"training_history_ann_rep{repetition_idx}.json"), 'w') as f:
            json.dump(ann_history, f, indent=4)
    else:
        print("ANN experiments skipped - ANNNet not available")

    # 2. Train SNN models (if available)
    if SNN_AVAILABLE:
        encoding_modes = {
            "expansive_delta_-1.5": {'delta': -1.5},
            "tough_delta_2.0": {'delta': 2.0},
            "dissipative_delta_10.0": {'delta': 10.0}
        }

        for mode_name, dynamic_params in encoding_modes.items():
            print(f"\n{'=' * 20} Repetition {repetition_idx}, Mode: {mode_name} {'=' * 20}")

            # SNN encoding
            print(f"Encoding training data for {mode_name}...")
            snn_train_dataset = encode_snn_data(X_train_processed.clone(), y_train_processed.clone(), config,
                                                dynamic_params)
            print(f"Encoding test data for {mode_name}...")
            snn_test_dataset = encode_snn_data(X_test_processed.clone(), y_test_processed.clone(), config,
                                               dynamic_params)

            snn_train_loader = DataLoader(snn_train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
            snn_test_loader = DataLoader(snn_test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

            # Train SNN - use Net class and mixed_oscillator encoding
            snn_model = Net(unified_config, encoding='mixed_oscillator').to(device)
            trained_snn, snn_best_acc, snn_history = train_snn_model(
                snn_model, snn_train_loader, snn_test_loader, config, f"SNN-{mode_name}"
            )

            snn_summary = {
                "repetition": repetition_idx,
                "model_type": "SNN",
                "mode": mode_name,
                "delta": dynamic_params['delta'],
                "best_test_accuracy": snn_best_acc,
                "final_train_loss": snn_history['train_loss'][-1] if snn_history['train_loss'] else None,
                "final_test_loss": snn_history['test_loss'][-1] if snn_history['test_loss'] else None,
                "final_avg_spikes": snn_history['spikes'][-1] if snn_history['spikes'] else None,
                "layer4_linearity_spike_count": None,
                "internal_metrics": None
            }

            # Detailed analysis (optional)
            try:
                print(f"Performing detailed analysis for {mode_name}...")
                mode_results_dir = os.path.join(base_repetition_dir, mode_name)
                mode_plots_dir = os.path.join(mode_results_dir, "plots")
                os.makedirs(mode_plots_dir, exist_ok=True)

                # Extract representations and analyze
                representations, labels = extract_layer_representations(
                    trained_snn, snn_test_loader, 'layer4', unified_config, device, representation_type='spike_count'
                )
                if representations is not None and labels is not None and len(representations) > 0:
                    linearity_score = evaluate_linear_separability(representations, labels)
                    snn_summary['layer4_linearity_spike_count'] = linearity_score
                    print(f"Linear separability for {mode_name}: {linearity_score:.4f}")

                # Analyze internal dynamics
                internal_metrics, sample_spk_rec, sample_mem_rec = analyze_internal_dynamics(
                    trained_snn, snn_test_loader, unified_config, device
                )
                snn_summary['internal_metrics'] = internal_metrics

            except Exception as e:
                print(f"Detailed analysis failed for {mode_name}: {e}")

            repetition_summary_list.append(snn_summary)

            # Save SNN training history
            snn_results_dir = os.path.join(base_repetition_dir, mode_name)
            os.makedirs(snn_results_dir, exist_ok=True)
            with open(os.path.join(snn_results_dir, f"training_history_{mode_name}_rep{repetition_idx}.json"),
                      'w') as f:
                json.dump(snn_history, f, indent=4)

    else:
        print("SNN experiments skipped - modules not available")

    # Save summary for this repetition
    repetition_summary_df = pd.DataFrame(repetition_summary_list)
    summary_csv_path = os.path.join(base_repetition_dir, f"unified_experiment_summary_repetition_{repetition_idx}.csv")
    repetition_summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n--- Repetition {repetition_idx} Complete. Summary saved to {summary_csv_path} ---")
    print(repetition_summary_df[['model_type', 'mode', 'best_test_accuracy', 'final_avg_spikes']])

    return repetition_summary_df


# --- 7. Main Experiment Function ---
def run_unified_experiment(num_repetitions: int = 20):
    """Run a unified multi-repetition experiment"""

    config = UnifiedConfig()
    all_repetitions_summaries = []
    base_output_dir = "unified_feature_binding_results_final"
    os.makedirs(base_output_dir, exist_ok=True)

    print(f"Starting unified experiment with {num_repetitions} repetitions")
    print(f"ANN modules available: {ANN_AVAILABLE}")
    print(f"SNN modules available: {SNN_AVAILABLE}")

    if not ANN_AVAILABLE and not SNN_AVAILABLE:
        print("Neither ANN nor SNN modules are available. Cannot run experiments.")
        return

    start_time_total = time.time()

    for i in range(num_repetitions):
        try:
            repetition_df = run_single_repetition_experiment(repetition_idx=i, config=config)
            if repetition_df is not None and not repetition_df.empty:
                all_repetitions_summaries.append(repetition_df)
            else:
                print(f"Warning: Repetition {i} returned empty results")
        except Exception as e:
            print(f"Error in repetition {i}: {e}")
            continue

    if not all_repetitions_summaries:
        print("No valid data collected from any repetition")
        return

    # Concatenate all results
    full_summary_df = pd.concat(all_repetitions_summaries, ignore_index=True)
    full_summary_path = os.path.join(base_output_dir, "all_unified_repetitions_summary.csv")
    full_summary_df.to_csv(full_summary_path, index=False)

    print(f"\n--- All {num_repetitions} Repetitions Complete ---")
    print(f"Combined summary saved to {full_summary_path}")
    print("\nSummary Statistics:")

    # Calculate statistics
    numeric_cols = ['best_test_accuracy', 'final_train_loss', 'final_test_loss', 'final_avg_spikes']
    for col in numeric_cols:
        if col in full_summary_df.columns:
            full_summary_df[col] = pd.to_numeric(full_summary_df[col], errors='coerce')

    # Group and aggregate statistics by model type and mode
    if 'model_type' in full_summary_df.columns and 'best_test_accuracy' in full_summary_df.columns:
        stats_summary = full_summary_df.groupby(['model_type', 'mode'])['best_test_accuracy'].agg(
            ['mean', 'std', 'count'])
        print("\nAccuracy Statistics by Model Type and Mode:")
        print(stats_summary)

        # Save statistics results
        stats_path = os.path.join(base_output_dir, "statistical_summary.csv")
        stats_summary.to_csv(stats_path)
        print(f"Statistical summary saved to {stats_path}")

    # Generate comparison plot
    try:
        plt.figure(figsize=(12, 8))

        if 'model_type' in full_summary_df.columns and 'best_test_accuracy' in full_summary_df.columns:
            # Create box plot
            model_types = full_summary_df['model_type'].unique()
            plot_data = []
            plot_labels = []

            for model_type in model_types:
                model_data = full_summary_df[full_summary_df['model_type'] == model_type]
                if model_type == 'ANN':
                    plot_data.append(model_data['best_test_accuracy'].dropna().values)
                    plot_labels.append('ANN')
                else:
                    for mode in model_data['mode'].unique():
                        mode_data = model_data[model_data['mode'] == mode]['best_test_accuracy'].dropna().values
                        if len(mode_data) > 0:
                            plot_data.append(mode_data)
                            plot_labels.append(f'SNN-{mode}')

            if plot_data:
                plt.boxplot(plot_data, labels=plot_labels, patch_artist=True)
                plt.title(f'Performance Comparison across {num_repetitions} Repetitions')
                plt.ylabel('Best Test Accuracy (%)')
                plt.xlabel('Model Configuration')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                plot_path = os.path.join(base_output_dir, "unified_performance_comparison.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Performance comparison plot saved to {plot_path}")
                plt.close()

    except Exception as e:
        print(f"Error generating plots: {e}")

    end_time_total = time.time()
    print(f"\nTotal execution time: {(end_time_total - start_time_total) / 60:.2f} minutes")


if __name__ == "__main__":
    num_repetitions_to_run = 30

    # num_repetitions_to_run = 3

    run_unified_experiment(num_repetitions=num_repetitions_to_run)