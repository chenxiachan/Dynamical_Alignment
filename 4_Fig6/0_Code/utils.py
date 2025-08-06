import torch
from scipy import stats
import seaborn as sns
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader # 确保导入 DataLoader
import time

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import snntorch.spikeplot as splt


device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

def evaluate_model(model, test_loader, criterion, config, model_type, device=device):
    """Evaluate model performance"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)

            if model_type == 'SNN':
                spk_rec, mem_rec = model(data)
                loss = torch.stack([criterion(mem_rec['layer4'][step], targets)
                                    for step in range(config.num_steps)]).mean()
                _, predicted = spk_rec['layer4'].sum(dim=0).max(1)
            else:  # ANN
                outputs = model(data)
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            test_loss += loss.item()

    return test_loss / len(test_loader), 100. * correct / total


def analyze_and_visualize_results(all_results, save_path, num_runs):
    stats_data = []

    detailed_data = []

    for exp_name in all_results:
        for model_type in all_results[exp_name]:
            results = all_results[exp_name][model_type]

            acc_mean = np.mean(results['best_acc'])
            acc_std = np.std(results['best_acc'])
            conv_mean = np.mean(results['convergence_epoch'])
            conv_std = np.std(results['convergence_epoch'])

            stats_data.append({
                'Experiment': exp_name,
                'Model': model_type,
                'Accuracy Mean': acc_mean,
                'Accuracy Std': acc_std,
                'Convergence Mean': conv_mean,
                'Convergence Std': conv_std
            })

            for run in range(num_runs):
                detailed_data.append({
                    'Experiment': exp_name,
                    'Model': model_type,
                    'Run': run + 1,
                    'Best Accuracy': results['best_acc'][run],
                    'Convergence Epoch': results['convergence_epoch'][run],
                    'Final Loss': results['final_loss'][run]
                })

    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(save_path, 'statistical_summary.csv'), index=False)

    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(os.path.join(save_path, 'detailed_results.csv'), index=False)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    data_to_plot = []
    labels = []
    for exp_name in all_results:
        for model_type in all_results[exp_name]:
            data_to_plot.append(all_results[exp_name][model_type]['best_acc'])
            labels.append(f"{exp_name}\n{model_type}")

    plt.boxplot(data_to_plot, labels=labels)
    plt.title(f'Accuracy Distribution Over {num_runs} Runs')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    data_to_plot = []
    for exp_name in all_results:
        for model_type in all_results[exp_name]:
            data_to_plot.append(all_results[exp_name][model_type]['convergence_epoch'])

    plt.boxplot(data_to_plot, labels=labels)
    plt.title('Convergence Speed Distribution')
    plt.ylabel('Convergence Epoch')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'statistical_analysis.png'))
    plt.close()

    plt.figure(figsize=(15, 8))
    data_to_plot = []
    for exp_name in all_results:
        for model_type in all_results[exp_name]:
            data_to_plot.append(all_results[exp_name][model_type]['best_acc'])

    plt.violinplot(data_to_plot)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
    plt.title(f'Accuracy Distribution (Violin Plot) Over {num_runs} Runs')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'accuracy_distribution_violin.png'))
    plt.close()

    print("\nStatistical Summary:")
    print(stats_df.to_string(index=False))

    print("\nDetailed Results (first few rows):")
    print(detailed_df.head().to_string())


class TrainingTracker:
    """Track and store training metrics for each epoch"""

    def __init__(self, experiment_name, model_type):
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []
        self.epochs = []
        self.spike_counts = []  # For SNN energy metrics
        self.current_epoch = 0
        self.encoding_time = None
        self.training_time = 0
        self.epoch_times = []

    def update(self, train_loss, train_acc, test_loss, test_acc, spike_count=None, epoch=None, epoch_time=None):
        """Update metrics for the current epoch"""
        current_epoch = epoch if epoch is not None else self.current_epoch

        self.epochs.append(current_epoch)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.test_loss.append(test_loss)
        self.test_acc.append(test_acc)

        if spike_count is not None:
            self.spike_counts.append(spike_count)

        if epoch_time is not None:
            self.epoch_times.append(epoch_time)

        if epoch is None:
            self.current_epoch += 1

    def set_encoding_time(self, encoding_time):
        self.encoding_time = encoding_time

    def add_training_time(self, time_delta):
        self.training_time += time_delta

    def get_metrics_df(self):
        """Convert tracking data to DataFrame"""
        rows = []

        for i in range(len(self.epochs)):
            row = {
                'Experiment': self.experiment_name,
                'Model': self.model_type,
                'Epoch': self.epochs[i],
                'Train Loss': self.train_loss[i],
                'Train Accuracy': self.train_acc[i],
                'Test Loss': self.test_loss[i],
                'Test Accuracy': self.test_acc[i],
            }

            if i < len(self.spike_counts):
                row['Spike Count'] = self.spike_counts[i]

            if i < len(self.epoch_times):
                row['Epoch Time'] = self.epoch_times[i]

            if self.encoding_time is not None:
                row['Encoding Time'] = self.encoding_time

            row['Training Time'] = self.training_time

            rows.append(row)

        return pd.DataFrame(rows)


def calculate_spike_metrics(spk_rec, batch_size):
    """
    Calculate spike-based energy metrics

    Parameters:
    - spk_rec: Dictionary of spike recordings from SNN layers
    - batch_size: Size of the batch

    Returns:
    - total_spikes: Total number of spikes across all layers and timesteps
    - avg_spikes_per_neuron: Average number of spikes per neuron
    - efficiency: Efficiency metric (can be used to compare models)
    """
    total_spikes = 0
    neuron_count = 0

    # Sum spikes across all layers
    for layer_name, spikes in spk_rec.items():
        # Sum across time dimension (dim=0) and count True/1 values
        layer_spikes = spikes.sum().sum().item()
        total_spikes += layer_spikes

        # Count neurons in this layer
        # For spikes shape: [time_steps, batch_size, neurons]
        neuron_count += spikes.shape[0] * spikes.shape[2] if len(spikes.shape) > 2 else spikes.shape[1]

    # Average spikes per neuron
    avg_spikes_per_neuron = total_spikes / (neuron_count * batch_size) if neuron_count > 0 else 0

    # Simple efficiency metric (can be refined)
    efficiency = 1.0 / (total_spikes + 1e-10)  # Add small constant to avoid division by zero

    return total_spikes, avg_spikes_per_neuron, efficiency


def visualize_convergence(all_trackers, save_path):
    """
    Create comprehensive convergence visualizations

    Parameters:
    - all_trackers: List of TrainingTracker objects
    - save_path: Directory to save visualizations

    Returns:
    - conv_df: DataFrame with convergence data
    """
    # Check if we have any trackers
    if not all_trackers:
        print("Warning: No tracking data available for visualization.")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame({
            'Experiment': [],
            'Model': [],
            'Convergence Epoch': [],
            'Max Accuracy': []
        })

    # Convert all trackers to a single DataFrame
    all_data = pd.concat([tracker.get_metrics_df() for tracker in all_trackers])

    if all_data.empty:
        print("Warning: No data in trackers for visualization.")
        return pd.DataFrame({
            'Experiment': [],
            'Model': [],
            'Convergence Epoch': [],
            'Max Accuracy': []
        })

    # Create experiment-specific colors for consistent plotting
    experiments = all_data['Experiment'].unique()
    color_map = {}
    cmap = plt.cm.get_cmap('tab10', max(len(experiments), 1))
    for i, exp in enumerate(experiments):
        color_map[exp] = cmap(i)

    # 1. Test Accuracy vs Epoch for all experiments
    plt.figure(figsize=(12, 7))

    # Check if we have any grouped data
    has_groups = False
    for name, group in all_data.groupby(['Experiment', 'Model']):
        has_groups = True
        exp_name, model_type = name
        plt.plot(group['Epoch'], group['Test Accuracy'],
                 marker='o', linestyle='-', alpha=0.7,
                 label=f"{exp_name} ({model_type})",
                 color=color_map[exp_name])

    if not has_groups:
        print("Warning: No experiment groups found for plotting.")
        plt.close()
        return pd.DataFrame({
            'Experiment': [],
            'Model': [],
            'Convergence Epoch': [],
            'Max Accuracy': []
        })

    plt.title('Test Accuracy vs. Epoch for Different Encoding Methods', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'test_accuracy_vs_epoch.png'), dpi=300)
    plt.close()

    # 2. Time to convergence comparison
    # Define convergence as reaching 90% of maximum accuracy
    convergence_data = []

    for name, group in all_data.groupby(['Experiment', 'Model']):
        exp_name, model_type = name
        if group.empty:
            continue

        max_acc = group['Test Accuracy'].max()
        convergence_threshold = 0.9 * max_acc

        # Find first epoch where accuracy exceeds threshold
        convergence_rows = group[group['Test Accuracy'] >= convergence_threshold]
        if convergence_rows.empty:
            # If convergence threshold never reached, use last epoch
            conv_epoch = group['Epoch'].max()
        else:
            conv_epoch = convergence_rows['Epoch'].iloc[0]

        convergence_data.append({
            'Experiment': exp_name,
            'Model': model_type,
            'Convergence Epoch': conv_epoch,
            'Max Accuracy': max_acc
        })

    # If no convergence data, return early
    if not convergence_data:
        print("Warning: No convergence data could be calculated.")
        return pd.DataFrame({
            'Experiment': [],
            'Model': [],
            'Convergence Epoch': [],
            'Max Accuracy': []
        })

    conv_df = pd.DataFrame(convergence_data)

    # Create convergence comparison bar chart
    plt.figure(figsize=(14, 8))

    try:
        ax = sns.barplot(data=conv_df, x='Experiment', y='Convergence Epoch', hue='Model', palette='Set2')

        # Add max accuracy as text on bars, with safety checks
        for i, p in enumerate(ax.patches):
            if i < len(conv_df):  # Safety check for index
                try:
                    accuracy = conv_df.iloc[i]['Max Accuracy']
                    ax.annotate(f"{accuracy:.1f}%",
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom',
                                fontsize=9, color='black')
                except (IndexError, KeyError) as e:
                    print(f"Warning: Error annotating bar {i}: {str(e)}")
    except Exception as e:
        print(f"Warning: Error creating barplot: {str(e)}")

    plt.title('Epochs to Convergence for Different Encoding Methods', fontsize=14)
    plt.xlabel('Experiment', fontsize=12)
    plt.ylabel('Convergence Epoch', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'convergence_comparison.png'), dpi=300)
    plt.close()

    # 3. Statistical comparison with error bars
    plt.figure(figsize=(14, 8))

    try:
        # Group by experiment and model, then compute stats
        stat_data = all_data.groupby(['Experiment', 'Model', 'Epoch'])['Test Accuracy'].agg(
            ['mean', 'std']).reset_index()

        # Plot learning curves with error bands
        for name, group in stat_data.groupby(['Experiment', 'Model']):
            exp_name, model_type = name
            plt.plot(group['Epoch'], group['mean'],
                     label=f"{exp_name} ({model_type})",
                     color=color_map[exp_name])
            plt.fill_between(group['Epoch'],
                             group['mean'] - group['std'],
                             group['mean'] + group['std'],
                             alpha=0.2, color=color_map[exp_name])
    except Exception as e:
        print(f"Warning: Error creating statistical plots: {str(e)}")

    plt.title('Learning Curves with Standard Deviation', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'learning_curves_with_std.png'), dpi=300)
    plt.close()

    return conv_df  # Return convergence data for further analysis


def visualize_learning_curves(trackers, save_path):
    """Visualize learning curves and save as images and CSV"""
    # Group by experiment type
    by_experiment = {}
    for tracker in trackers:
        key = f"{tracker.experiment_name}"
        if key not in by_experiment:
            by_experiment[key] = []
        by_experiment[key].append(tracker)

    # Create separate learning curve plots for each experiment
    for exp_name, exp_trackers in by_experiment.items():
        # Training loss curve
        plt.figure(figsize=(12, 8))
        for tracker in exp_trackers:
            plt.plot(tracker.epochs, tracker.train_loss,
                     label=f"{tracker.model_type} (Best: {max(tracker.test_acc):.2f}%)")
        plt.title(f"Training Loss - {exp_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f"{exp_name}_train_loss.png"))
        plt.close()

        # Test accuracy curve
        plt.figure(figsize=(12, 8))
        for tracker in exp_trackers:
            plt.plot(tracker.epochs, tracker.test_acc,
                     label=f"{tracker.model_type} (Best: {max(tracker.test_acc):.2f}%)")
        plt.title(f"Test Accuracy - {exp_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f"{exp_name}_test_accuracy.png"))
        plt.close()

        # Save CSV for each tracker
        for tracker in exp_trackers:
            df = tracker.get_metrics_df()
            csv_path = os.path.join(save_path, f"{tracker.experiment_name}_{tracker.model_type}_metrics.csv")
            df.to_csv(csv_path, index=False)


def analyze_energy_efficiency(all_trackers, conv_df, save_path):
    """
    Analyze and visualize energy efficiency metrics

    Parameters:
    - all_trackers: List of TrainingTracker objects
    - conv_df: DataFrame with convergence data
    - save_path: Directory to save visualizations
    """
    # Error checking - make sure we have trackers and convergence data
    if not all_trackers:
        print("Warning: No trackers provided for energy efficiency analysis.")
        return

    if conv_df.empty:
        print("Warning: Empty convergence data provided for energy efficiency analysis.")
        return

    # Filter trackers that have spike data (SNN models)
    snn_trackers = [t for t in all_trackers if hasattr(t, 'spike_counts') and
                    t.spike_counts and t.model_type == 'SNN']

    if not snn_trackers:
        print("Warning: No SNN trackers with spike data found.")
        return

    # Create DataFrame for energy analysis
    energy_data = []

    for tracker in snn_trackers:
        # Get best accuracy achieved
        best_acc = max(tracker.test_acc) if tracker.test_acc else 0

        # Get total spikes at convergence
        convergence_rows = conv_df[(conv_df['Experiment'] == tracker.experiment_name) &
                                   (conv_df['Model'] == tracker.model_type)]

        if convergence_rows.empty:
            print(f"Warning: No convergence data for {tracker.experiment_name} ({tracker.model_type})")
            continue

        conv_epoch = convergence_rows.iloc[0]['Convergence Epoch']

        # Get spike count at or nearest to convergence epoch
        if not tracker.spike_counts:
            print(f"Warning: No spike counts for {tracker.experiment_name} ({tracker.model_type})")
            continue

        if conv_epoch <= len(tracker.spike_counts) - 1:
            spikes_at_convergence = tracker.spike_counts[int(conv_epoch)]
        else:
            spikes_at_convergence = tracker.spike_counts[-1]

        # Calculate energy efficiency metrics
        efficiency = best_acc / spikes_at_convergence if spikes_at_convergence > 0 else 0

        energy_data.append({
            'Experiment': tracker.experiment_name,
            'Model': tracker.model_type,
            'Best Accuracy': best_acc,
            'Spikes at Convergence': spikes_at_convergence,
            'Efficiency (Acc/Spike)': efficiency * 100  # Scale for readability
        })

    if not energy_data:
        print("Warning: No energy data could be compiled.")
        return

    energy_df = pd.DataFrame(energy_data)

    # Save energy metrics to CSV
    energy_df.to_csv(os.path.join(save_path, 'energy_efficiency_metrics.csv'), index=False)

    try:
        # Visualize efficiency metrics
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=energy_df, x='Experiment', y='Efficiency (Acc/Spike)')

        # Add accuracy as text on bars
        for i, p in enumerate(ax.patches):
            if i < len(energy_df):  # Safety check
                accuracy = energy_df.iloc[i]['Best Accuracy']
                spikes = energy_df.iloc[i]['Spikes at Convergence']
                ax.annotate(f"Acc: {accuracy:.1f}%\nSpikes: {spikes:.0f}",
                            (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                            ha='center', va='center',
                            fontsize=9, color='black')

        plt.title('Energy Efficiency (Accuracy per Spike) for SNN Models', fontsize=14)
        plt.xlabel('Encoding Method', fontsize=12)
        plt.ylabel('Efficiency (Accuracy % / Spike Count) × 100', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'energy_efficiency_comparison.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Error creating efficiency barplot: {str(e)}")

    try:
        # Scatter plot: Accuracy vs Spike Count
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=energy_df, x='Spikes at Convergence', y='Best Accuracy',
                        hue='Experiment', size='Efficiency (Acc/Spike)',
                        sizes=(50, 250), alpha=0.7)

        plt.title('Accuracy vs Spike Count for Different Encoding Methods', fontsize=14)
        plt.xlabel('Number of Spikes at Convergence', fontsize=12)
        plt.ylabel('Best Accuracy (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'accuracy_vs_spikes.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Error creating accuracy vs spikes plot: {str(e)}")

    # Visualize spike count over epochs for each method
    for tracker in snn_trackers:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(tracker.epochs, tracker.spike_counts, marker='o')
            plt.title(f'Spike Count vs Epoch: {tracker.experiment_name}', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Total Spike Count', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'spike_count_{tracker.experiment_name}.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Warning: Error creating spike count plot for {tracker.experiment_name}: {str(e)}")

    return energy_df


def perform_statistical_tests(all_trackers, save_path):
    """
    Perform statistical tests to evaluate significance of results
    """
    if not all_trackers:
        print("Warning: No trackers provided for statistical testing.")
        return

    # Group trackers by experiment name
    experiment_groups = {}
    for tracker in all_trackers:
        if tracker.experiment_name not in experiment_groups:
            experiment_groups[tracker.experiment_name] = []
        experiment_groups[tracker.experiment_name].append(tracker)

    if not experiment_groups:
        print("Warning: No experiment groups found for statistical testing.")
        return

    # Focus on accuracy and convergence speed
    accuracy_data = {}
    convergence_data = {}

    for exp_name, trackers in experiment_groups.items():
        if not trackers:
            continue

        if not all(hasattr(t, 'test_acc') and t.test_acc for t in trackers):
            print(f"Warning: Missing test accuracy data for {exp_name}")
            continue

        accuracy_data[exp_name] = [max(tracker.test_acc) for tracker in trackers if tracker.test_acc]

        # Define convergence as first epoch to reach 90% of max accuracy
        convergence_epochs = []
        for tracker in trackers:
            if not tracker.test_acc:
                continue

            max_acc = max(tracker.test_acc)
            threshold = 0.9 * max_acc
            try:
                conv_epoch = next((i for i, acc in enumerate(tracker.test_acc) if acc >= threshold),
                                  len(tracker.test_acc) - 1)
                convergence_epochs.append(conv_epoch)
            except (StopIteration, IndexError):
                # If no convergence found, use the last epoch
                if tracker.epochs:
                    convergence_epochs.append(tracker.epochs[-1])

        if convergence_epochs:
            convergence_data[exp_name] = convergence_epochs

    if not accuracy_data or not convergence_data:
        print("Warning: Insufficient data for statistical tests.")
        return

    # Prepare for statistical tests
    experiment_names = list(accuracy_data.keys())

    # Create summary dataframes
    accuracy_summary = []
    convergence_summary = []

    # Fill summary dataframes
    for exp_name in experiment_names:
        if exp_name in accuracy_data and accuracy_data[exp_name]:
            accuracy_summary.append({
                'Experiment': exp_name,
                'Mean Accuracy': np.mean(accuracy_data[exp_name]),
                'Std Dev': np.std(accuracy_data[exp_name]),
                'Min': np.min(accuracy_data[exp_name]),
                'Max': np.max(accuracy_data[exp_name])
            })

        if exp_name in convergence_data and convergence_data[exp_name]:
            convergence_summary.append({
                'Experiment': exp_name,
                'Mean Epochs to Converge': np.mean(convergence_data[exp_name]),
                'Std Dev': np.std(convergence_data[exp_name]),
                'Min': np.min(convergence_data[exp_name]),
                'Max': np.max(convergence_data[exp_name])
            })

    # Convert to DataFrame
    accuracy_summary_df = pd.DataFrame(accuracy_summary)
    convergence_summary_df = pd.DataFrame(convergence_summary)

    # Save summaries if not empty
    if not accuracy_summary_df.empty:
        accuracy_summary_df.to_csv(os.path.join(save_path, 'accuracy_summary.csv'), index=False)

    if not convergence_summary_df.empty:
        convergence_summary_df.to_csv(os.path.join(save_path, 'convergence_summary.csv'), index=False)

    # Perform t-tests between Lorenz-SNN and other methods
    if 'lorenz-SNN' in accuracy_data and len(accuracy_data['lorenz-SNN']) > 1:
        lorenz_accuracy = accuracy_data['lorenz-SNN']

        accuracy_ttest_results = []
        convergence_ttest_results = []

        for exp_name in experiment_names:
            if exp_name != 'lorenz-SNN' and exp_name in accuracy_data and len(accuracy_data[exp_name]) > 1:
                try:
                    # Accuracy t-test
                    t_stat, p_value = stats.ttest_ind(
                        lorenz_accuracy,
                        accuracy_data[exp_name],
                        equal_var=False  # Welch's t-test for unequal variances
                    )

                    accuracy_ttest_results.append({
                        'Comparison': f'lorenz-SNN vs {exp_name}',
                        'Metric': 'Accuracy',
                        'T-statistic': t_stat,
                        'P-value': p_value,
                        'Significant': p_value < 0.05
                    })
                except Exception as e:
                    print(f"Warning: Error in accuracy t-test for {exp_name}: {str(e)}")

        if 'lorenz-SNN' in convergence_data and len(convergence_data['lorenz-SNN']) > 1:
            lorenz_convergence = convergence_data['lorenz-SNN']

            for exp_name in experiment_names:
                if exp_name != 'lorenz-SNN' and exp_name in convergence_data and len(convergence_data[exp_name]) > 1:
                    try:
                        # Convergence t-test
                        t_stat, p_value = stats.ttest_ind(
                            lorenz_convergence,
                            convergence_data[exp_name],
                            equal_var=False
                        )

                        convergence_ttest_results.append({
                            'Comparison': f'lorenz-SNN vs {exp_name}',
                            'Metric': 'Convergence Epoch',
                            'T-statistic': t_stat,
                            'P-value': p_value,
                            'Significant': p_value < 0.05
                        })
                    except Exception as e:
                        print(f"Warning: Error in convergence t-test for {exp_name}: {str(e)}")

        # Save t-test results if not empty
        if accuracy_ttest_results:
            pd.DataFrame(accuracy_ttest_results).to_csv(
                os.path.join(save_path, 'accuracy_ttest_results.csv'), index=False)

        if convergence_ttest_results:
            pd.DataFrame(convergence_ttest_results).to_csv(
                os.path.join(save_path, 'convergence_ttest_results.csv'), index=False)

        # Print t-test results
        print("\nStatistical Test Results (Lorenz-SNN Comparisons):")

        if accuracy_ttest_results:
            print("\nAccuracy T-Tests:")
            for result in accuracy_ttest_results:
                significance = "SIGNIFICANT" if result['Significant'] else "not significant"
                print(f"{result['Comparison']}: p={result['P-value']:.4f} ({significance})")

        if convergence_ttest_results:
            print("\nConvergence T-Tests:")
            for result in convergence_ttest_results:
                significance = "SIGNIFICANT" if result['Significant'] else "not significant"
                print(f"{result['Comparison']}: p={result['P-value']:.4f} ({significance})")

    # Create boxplot visualizations of the comparisons
    try:
        plt.figure(figsize=(14, 7))

        # Accuracy comparison
        if accuracy_data:
            plt.subplot(1, 2, 1)
            data_to_plot = [accuracy_data[exp] for exp in experiment_names if exp in accuracy_data]
            labels = [exp for exp in experiment_names if exp in accuracy_data]
            if data_to_plot and labels:
                plt.boxplot(data_to_plot, labels=labels)
                plt.title('Classification Accuracy Comparison')
                plt.ylabel('Accuracy (%)')
                plt.xticks(rotation=45)

        # Convergence comparison
        if convergence_data:
            plt.subplot(1, 2, 2)
            data_to_plot = [convergence_data[exp] for exp in experiment_names if exp in convergence_data]
            labels = [exp for exp in experiment_names if exp in convergence_data]
            if data_to_plot and labels:
                plt.boxplot(data_to_plot, labels=labels)
                plt.title('Convergence Speed Comparison')
                plt.ylabel('Epochs to Converge')
                plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'statistical_comparisons.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Error creating statistical comparison plots: {str(e)}")


def calculate_spike_metrics(spk_rec, batch_size):
    """Calculate SNN spike metrics"""
    total_spikes = 0
    spike_rate = 0

    for layer_name, spikes in spk_rec.items():
        if isinstance(spikes, torch.Tensor):
            layer_spikes = torch.sum(spikes).item()
            total_spikes += layer_spikes
            # Calculate average firing rate per neuron per sample
            layer_size = spikes.shape[-1] if len(spikes.shape) > 2 else spikes.shape[1]
            spike_rate += layer_spikes / (batch_size * layer_size)

    # Calculate average energy efficiency (spikes per classification decision)
    efficiency = total_spikes / batch_size if batch_size > 0 else 0

    return total_spikes, spike_rate, efficiency

#####################################################################

@torch.no_grad()
def analyze_internal_dynamics(model, data_loader, config, device):
    """
    Analyzes internal SNN dynamics (firing rates, sparsity) for a trained model.

    Args:
        model (torch.nn.Module): The trained SNN model.
        data_loader (DataLoader): DataLoader for the dataset (e.g., test set).
        config (Config): Configuration object.
        device (torch.device): Device to run computation on.

    Returns:
        dict: A dictionary containing aggregated internal metrics per layer.
              Example: {'layer1_rate_mean': 0.1, 'layer1_sparsity_mean': 0.9, ...}
        dict: Spike recordings for a sample batch (for visualization).
              Example: {'layer1': tensor, 'layer2': tensor, ...}
        dict: Membrane potential recordings for a sample batch (for visualization). <-- New Return
              Example: {'layer1': tensor, 'layer2': tensor, ...}
    """
    model.eval() # Set model to evaluation mode

    # Use lists to accumulate metrics across batches
    layer_metrics = {}
    sample_spk_rec = None # To store spikes from one batch for visualization
    sample_mem_rec = None # <-- New: To store membrane potentials from one batch

    # Determine number of layers from model (assuming fc1-4 and lif1-4 naming)
    num_layers = 4 # Based on Net class in nns.py

    for layer_idx in range(1, num_layers + 1):
        layer_metrics[f'layer{layer_idx}_rates'] = []
        layer_metrics[f'layer{layer_idx}_sparsities'] = []

    total_samples = 0

    for i, (data, _) in enumerate(data_loader):
        data = data.to(device)
        batch_size = data.size(0)
        total_samples += batch_size

        # Run the model
        spk_rec, mem_rec = model(data) # Get both spike and membrane recordings

        # Store spikes and membrane potentials from the first batch for visualization
        if i == 0:
             # Detach and clone necessary tensors to avoid holding references
            sample_spk_rec = {key: val.detach().clone().cpu() for key, val in spk_rec.items()}
            sample_mem_rec = {key: val.detach().clone().cpu() for key, val in mem_rec.items()}


        # Calculate metrics per layer for this batch
        for layer_idx in range(1, num_layers + 1):
            layer_name = f'layer{layer_idx}'
            if layer_name in spk_rec:
                spikes = spk_rec[layer_name] # Shape: [num_steps, batch_size, num_neurons]

                # Firing Rate: Avg spikes per neuron per time step
                rate = spikes.float().mean() # Global average firing rate
                layer_metrics[f'{layer_name}_rates'].append(rate.item() * batch_size) # Store weighted by batch size

                # Sparsity: Fraction of (neuron, time step) pairs *without* a spike
                sparsity = 1.0 - spikes.float().mean()
                layer_metrics[f'{layer_name}_sparsities'].append(sparsity.item() * batch_size) # Store weighted by batch size

    # Aggregate metrics across all batches
    aggregated_metrics = {}
    if total_samples > 0:
        for layer_idx in range(1, num_layers + 1):
            layer_name = f'layer{layer_idx}'
            # Calculate weighted average
            aggregated_metrics[f'{layer_name}_rate_mean'] = np.sum(layer_metrics[f'{layer_name}_rates']) / total_samples
            aggregated_metrics[f'{layer_name}_sparsity_mean'] = np.sum(layer_metrics[f'{layer_name}_sparsities']) / total_samples
    else: # Handle case with no data
         for layer_idx in range(1, num_layers + 1):
            layer_name = f'layer{layer_idx}'
            aggregated_metrics[f'{layer_name}_rate_mean'] = np.nan
            aggregated_metrics[f'{layer_name}_sparsity_mean'] = np.nan


    # Return aggregated metrics, sample spikes, AND sample membrane potentials
    return aggregated_metrics, sample_spk_rec, sample_mem_rec


def visualize_internal_metrics(merged_df, save_path):
    """
    Visualizes internal SNN metrics against Lyapunov sum.

    Args:
        merged_df (pd.DataFrame): DataFrame containing aggregated results including internal metrics.
        save_path (str): Path to save the plots.
    """
    num_layers = 4 # Assuming 4 layers as in your Net model

    # Plot Layer-wise Firing Rates vs Lyapunov Sum
    plt.figure(figsize=(15, 5 * num_layers // 2))
    for i in range(1, num_layers + 1):
        plt.subplot(num_layers // 2, 2, i)
        metric_col = f'layer{i}_rate_mean'
        if metric_col in merged_df.columns:
            plt.scatter(merged_df['lyapunov_sum'], merged_df[metric_col], alpha=0.7)

             # Add regression line (e.g., linear or quadratic)
            try:
                valid_data = merged_df.dropna(subset=['lyapunov_sum', metric_col])
                if len(valid_data) > 1:
                    z = np.polyfit(valid_data['lyapunov_sum'], valid_data[metric_col], 1) # Linear fit
                    p = np.poly1d(z)
                    x_range = np.linspace(valid_data['lyapunov_sum'].min(), valid_data['lyapunov_sum'].max(), 100)
                    plt.plot(x_range, p(x_range), "r--", alpha=0.6)
            except Exception as e:
                print(f"Could not fit regression line for {metric_col}: {e}")


            plt.xlabel('Sum of Lyapunov Exponents (Σλᵢ)')
            plt.ylabel('Mean Firing Rate')
            plt.title(f'Layer {i} Firing Rate vs Lyapunov Sum')
            plt.grid(True, alpha=0.3)
    plt.suptitle('Internal Firing Rates vs Encoder Dynamics', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(os.path.join(save_path, 'internal_firing_rates.png'), dpi=300)
    plt.close()

    # Plot Layer-wise Sparsity vs Lyapunov Sum
    plt.figure(figsize=(15, 5 * num_layers // 2))
    for i in range(1, num_layers + 1):
        plt.subplot(num_layers // 2, 2, i)
        metric_col = f'layer{i}_sparsity_mean'
        if metric_col in merged_df.columns:
            plt.scatter(merged_df['lyapunov_sum'], merged_df[metric_col], alpha=0.7)

            # Add regression line (e.g., linear or quadratic)
            try:
                valid_data = merged_df.dropna(subset=['lyapunov_sum', metric_col])
                if len(valid_data) > 1:
                    z = np.polyfit(valid_data['lyapunov_sum'], valid_data[metric_col], 1) # Linear fit
                    p = np.poly1d(z)
                    x_range = np.linspace(valid_data['lyapunov_sum'].min(), valid_data['lyapunov_sum'].max(), 100)
                    plt.plot(x_range, p(x_range), "r--", alpha=0.6)
            except Exception as e:
                print(f"Could not fit regression line for {metric_col}: {e}")

            plt.xlabel('Sum of Lyapunov Exponents (Σλᵢ)')
            plt.ylabel('Mean Sparsity (1 - Firing Rate)')
            plt.title(f'Layer {i} Sparsity vs Lyapunov Sum')
            plt.grid(True, alpha=0.3)
    plt.suptitle('Internal Sparsity vs Encoder Dynamics', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, 'internal_sparsity.png'), dpi=300)
    plt.close()


def save_raster_plots(sample_spk_rec, exp_name, save_path):
    """
    Generates and saves raster plots for a sample batch using snntorch.

    Args:
        sample_spk_rec (dict): Dictionary of spike recordings for one batch.
                               Keys are layer names, values are tensors [time, batch, neurons].
        exp_name (str): Experiment name for file naming.
        save_path (str): Directory to save the plots.
    """
    if sample_spk_rec is None:
        print(f"No sample spike recordings to plot for {exp_name}.")
        return

    num_layers = len(sample_spk_rec)
    if num_layers == 0:
        print(f"Empty spike recordings for {exp_name}.")
        return

    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3 * num_layers), sharex=True) # Adjusted figsize
    if num_layers == 1: # Handle case with only one layer
        axes = [axes]

    layer_names = sorted(sample_spk_rec.keys()) # Ensure consistent order

    for idx, layer_name in enumerate(layer_names):
        spk_tensor = sample_spk_rec[layer_name] # Shape: [time, batch, neurons]
        if spk_tensor is None or spk_tensor.numel() == 0:
            print(f"Skipping empty spike tensor for layer {layer_name} in {exp_name}")
            axes[idx].set_title(f'{exp_name} - {layer_name} Raster Plot (No Spikes)')
            axes[idx].text(0.5, 0.5, 'No Spikes Recorded', ha='center', va='center')
            continue


        # Select spikes for the first sample in the batch for clarity
        # Permute to [batch, time, neurons] for spikeplot if needed by splt.raster
        # Check splt.raster documentation for expected input shape
        # Assuming splt.raster expects [time, neurons] for a single sample:
        spk_sample_time_neuron = spk_tensor[:, 0, :] # Shape: [time, neurons] for sample 0

        # Generate raster plot for this layer sample
        # Note: splt.raster might expect different shape, adjust accordingly
        # If it expects [batch, time, neuron] use:
        # spk_sample_batch_time_neuron = spk_tensor.permute(1, 0, 2)[0].unsqueeze(0) # [1, time, neurons]
        # splt.raster(spk_sample_batch_time_neuron, axes[idx], s=5, c='black')
        splt.raster(spk_sample_time_neuron, axes[idx], s=15, c='black') # Increased size 's'
        axes[idx].set_title(f'{exp_name} - {layer_name} Raster Plot (Sample 0)')
        axes[idx].set_ylabel("Neuron Index")
        axes[idx].locator_params(axis='y', nbins=5) # Limit y-axis ticks
        if idx == num_layers - 1:
             axes[idx].set_xlabel("Time Step")
        else:
             axes[idx].set_xlabel("") # Remove x-label for non-bottom plots


    plt.tight_layout()
    # Ensure the directory exists
    plot_dir = os.path.join(save_path, "raster_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = os.path.join(plot_dir, f'{exp_name}_raster.png')
    plt.savefig(plot_filename, dpi=150)
    print(f"Raster plot saved to {plot_filename}")
    plt.close(fig)


def save_spike_counts_hist(sample_spk_rec, exp_name, save_path):
    """
    Generates and saves histograms of spike counts per neuron for each layer.

    Args:
        sample_spk_rec (dict): Dictionary of spike recordings for one batch.
                               Keys are layer names, values are tensors [time, batch, neurons].
        exp_name (str): Experiment name for file naming.
        save_path (str): Directory to save the plots.
    """
    if sample_spk_rec is None:
        print(f"No sample spike recordings for spike count histograms for {exp_name}.")
        return

    num_layers = len(sample_spk_rec)
    if num_layers == 0:
        print(f"Empty spike recordings for {exp_name}.")
        return

    fig, axes = plt.subplots(num_layers, 1, figsize=(8, 3 * num_layers)) # Adjusted figsize
    if num_layers == 1:
        axes = [axes]

    layer_names = sorted(sample_spk_rec.keys())

    for idx, layer_name in enumerate(layer_names):
        spk_tensor = sample_spk_rec[layer_name] # Shape: [time, batch, neurons]
        if spk_tensor is None or spk_tensor.numel() == 0:
            print(f"Skipping empty spike tensor for layer {layer_name} in {exp_name}")
            axes[idx].set_title(f'{exp_name} - {layer_name} Spike Count (No Spikes)')
            axes[idx].text(0.5, 0.5, 'No Spikes Recorded', ha='center', va='center')
            continue


        # Calculate spike count per neuron over time for the first sample
        spike_counts_per_neuron = spk_tensor[:, 0, :].sum(dim=0).numpy() # Sum over time dim

        axes[idx].hist(spike_counts_per_neuron, bins=max(1, int(spike_counts_per_neuron.max()) + 1) , density=True)
        axes[idx].set_title(f'{exp_name} - {layer_name} Spike Count Distribution (Sample 0)')
        axes[idx].set_ylabel("Density")
        if idx == num_layers - 1:
            axes[idx].set_xlabel("Spikes per Neuron")
        else:
            axes[idx].set_xlabel("")


    plt.tight_layout()
    plot_dir = os.path.join(save_path, "spike_counts")
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = os.path.join(plot_dir, f'{exp_name}_spike_counts.png')
    plt.savefig(plot_filename, dpi=150)
    print(f"Spike count histogram saved to {plot_filename}")
    plt.close(fig)


def save_membrane_potential_traces(sample_mem_rec, exp_name, save_path, num_neurons_to_plot=5):
    """
    Generates and saves membrane potential traces for a few sample neurons.

    Args:
        sample_mem_rec (dict): Dictionary of membrane potential recordings for one batch.
                               Keys are layer names, values are tensors [time, batch, neurons].
        exp_name (str): Experiment name for file naming.
        save_path (str): Directory to save the plots.
        num_neurons_to_plot (int): Number of sample neurons to plot per layer.
    """
    if sample_mem_rec is None:
        print(f"No sample membrane recordings to plot for {exp_name}.")
        return

    num_layers = len(sample_mem_rec)
    if num_layers == 0:
        print(f"Empty membrane recordings for {exp_name}.")
        return

    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3 * num_layers), sharex=True)
    if num_layers == 1:
        axes = [axes]

    layer_names = sorted(sample_mem_rec.keys())

    for idx, layer_name in enumerate(layer_names):
        mem_tensor = sample_mem_rec[layer_name] # Shape: [time, batch, neurons]
        if mem_tensor is None or mem_tensor.numel() == 0:
             print(f"Skipping empty membrane tensor for layer {layer_name} in {exp_name}")
             axes[idx].set_title(f'{exp_name} - {layer_name} Membrane Potential (No Data)')
             axes[idx].text(0.5, 0.5, 'No Membrane Data', ha='center', va='center')
             continue


        num_neurons = mem_tensor.shape[2]
        neurons_to_plot = min(num_neurons_to_plot, num_neurons)
        neuron_indices = np.random.choice(num_neurons, neurons_to_plot, replace=False)

        # Plot traces for selected neurons from the first sample
        for neuron_idx in neuron_indices:
            trace = mem_tensor[:, 0, neuron_idx].numpy() # Get trace for sample 0, neuron neuron_idx
            axes[idx].plot(trace, label=f'Neuron {neuron_idx}')

        axes[idx].set_title(f'{exp_name} - {layer_name} Membrane Potential Traces (Sample 0)')
        axes[idx].set_ylabel("Potential (U)")
        axes[idx].legend(loc='upper right', fontsize='small')
        axes[idx].grid(True, alpha=0.3)
        if idx == num_layers - 1:
            axes[idx].set_xlabel("Time Step")
        else:
            axes[idx].set_xlabel("")


    plt.tight_layout()
    plot_dir = os.path.join(save_path, "membrane_traces")
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = os.path.join(plot_dir, f'{exp_name}_membrane_traces.png')
    plt.savefig(plot_filename, dpi=150)
    print(f"Membrane potential traces saved to {plot_filename}")
    plt.close(fig)

###############################################



@torch.no_grad()
def extract_layer_representations(model, data_loader, layer_name, config, device, representation_type='spike_count'):

    model.eval()
    all_representations = []
    all_labels = []

    print(f"Extracting '{representation_type}' representations from {layer_name}...")
    for data, targets in data_loader:
        data = data.to(device)
        targets = targets.to(device)

        spk_rec, mem_rec = model(data)

        if layer_name not in spk_rec and layer_name not in mem_rec:
             print(f"Warning: Layer '{layer_name}' not found in model outputs. Skipping batch.")
             continue


        batch_representation = None
        if representation_type == 'spike_count':
            if layer_name in spk_rec:
                 # Sum spikes over time dimension (dim=0)
                 # Shape: [batch_size, num_neurons]
                batch_representation = spk_rec[layer_name].sum(dim=0).cpu()
            else:
                 print(f"Warning: Spike recording for layer '{layer_name}' not found. Skipping batch.")
                 continue
        elif representation_type == 'avg_rate':
             if layer_name in spk_rec:
                  # Mean spikes over time dimension (dim=0)
                  batch_representation = spk_rec[layer_name].float().mean(dim=0).cpu()
             else:
                 print(f"Warning: Spike recording for layer '{layer_name}' not found. Skipping batch.")
                 continue
        elif representation_type == 'last_mem':
             if layer_name in mem_rec:
                 # Membrane potential at the last time step
                 batch_representation = mem_rec[layer_name][-1].cpu() # Last time step
             else:
                 print(f"Warning: Membrane recording for layer '{layer_name}' not found. Skipping batch.")
                 continue

        elif representation_type == 'avg_mem':
             if layer_name in mem_rec:
                  # Average membrane potential over time
                  batch_representation = mem_rec[layer_name].mean(dim=0).cpu()
             else:
                 print(f"Warning: Membrane recording for layer '{layer_name}' not found. Skipping batch.")
                 continue
        else:
            raise ValueError(f"Unknown representation_type: {representation_type}")

        if batch_representation is not None:
             all_representations.append(batch_representation)
             all_labels.append(targets.cpu())

    if not all_representations:
         print("Error: No representations were extracted.")
         return None, None


    # Concatenate all batches
    all_representations = torch.cat(all_representations, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"Finished extraction. Representation shape: {all_representations.shape}, Labels shape: {all_labels.shape}")
    return all_representations, all_labels


def visualize_representation(representations, labels, method='umap', title='Layer Representation', save_path='.',
                             filename='representation.png'):

    if representations is None or labels is None:
        print("Cannot visualize: Representations or labels are None.")
        return

    if isinstance(representations, torch.Tensor):
        representations = representations.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    print(f"Visualizing using {method.upper()}...")
    start_time = time.time()

    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42, init='pca', learning_rate='auto')
        embedding = reducer.fit_transform(representations)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2)
        embedding = reducer.fit_transform(representations)
    elif method == 'pca':
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(representations)
    else:
        raise ValueError("Method must be 'tsne', 'umap', or 'pca'")

    end_time = time.time()
    print(f"{method.upper()} took {end_time - start_time:.2f} seconds.")

    plt.figure(figsize=(1.5, 1.35))

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7
    })

    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1],
        c=labels,
        cmap='tab10',
        s=3,
        alpha=0.8,
        edgecolors='none'
    )

    plt.xlabel(f'{method.upper()} 1', labelpad=4)
    plt.ylabel(f'{method.upper()} 2', labelpad=4)

    if title and not title in filename:
        filename = f"{title.replace(' ', '_')}_{filename}"

    plt.locator_params(axis='both', nbins=5)

    plt.grid(False)

    plt.tight_layout(pad=0.8)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_alpha(0.3)
    plt.gca().spines['left'].set_alpha(0.3)

    os.makedirs(os.path.dirname(os.path.join(save_path, filename)), exist_ok=True)

    plt.savefig(
        os.path.join(save_path, filename),
        dpi=300,
        bbox_inches='tight',
        transparent=False
    )

    print(f"Publication-style visualization saved to {os.path.join(save_path, filename)}")
    plt.close()


def evaluate_linear_separability(representations, labels, test_size=0.3, random_state=42):

    if representations is None or labels is None:
        print("Cannot evaluate linear separability: Representations or labels are None.")
        return np.nan

    if isinstance(representations, torch.Tensor):
        representations = representations.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    print("Evaluating linear separability...")
    start_time = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        representations, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    probe = LogisticRegression(max_iter=1000, random_state=random_state, solver='saga', multi_class='multinomial', C=1.0)
    probe.fit(X_train_scaled, y_train)

    y_pred = probe.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    end_time = time.time()
    print(f"Linear probe evaluation took {end_time - start_time:.2f} seconds. Accuracy: {accuracy:.4f}")

    return accuracy


def plot_layer_comparison_rasters(spk_rec, layer_name1, layer_name2, sample_index, title, save_path, filename, max_neurons_to_plot=100):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=14)

    for i, layer_name in enumerate([layer_name1, layer_name2]):
        ax = axes[i]
        if layer_name not in spk_rec or spk_rec[layer_name] is None:
            print(f"Warning: Spike data for layer '{layer_name}' not found.")
            ax.set_title(f'{layer_name} - No Data')
            ax.text(0.5, 0.5, 'No Spike Data', ha='center', va='center')
            continue

        spk_tensor = spk_rec[layer_name].cpu()

        if sample_index >= spk_tensor.shape[1]:
             print(f"Warning: sample_index {sample_index} is out of bounds for layer {layer_name} batch size {spk_tensor.shape[1]}. Skipping plot.")
             ax.set_title(f'{layer_name} - Index Error')
             ax.text(0.5, 0.5, 'Sample Index Error', ha='center', va='center')
             continue


        num_time_steps, _, num_neurons = spk_tensor.shape

        if num_neurons > max_neurons_to_plot:
            neuron_indices_to_plot = np.random.choice(num_neurons, max_neurons_to_plot, replace=False)
            neuron_indices_to_plot = np.sort(neuron_indices_to_plot)
            spk_subset = spk_tensor[:, sample_index, neuron_indices_to_plot] # Shape: [time, max_neurons_to_plot]
            y_labels = neuron_indices_to_plot
            ylabel_text = f"Neuron Index (Subset of {max_neurons_to_plot})"
        else:
            spk_subset = spk_tensor[:, sample_index, :] # Shape: [time, num_neurons]
            y_labels = np.arange(num_neurons)
            ylabel_text = "Neuron Index"

        if spk_subset.numel() == 0 or spk_subset.sum() == 0:
             ax.set_title(f'{layer_name} Raster Plot (No Spikes in Sample/Subset)')
             ax.text(0.5, 0.5, 'No Spikes', ha='center', va='center')
             ax.set_ylabel(ylabel_text)
             continue

        splt.raster(spk_subset, ax, s=5, c='black') # 使用较小的点

        if num_neurons > max_neurons_to_plot:
            ax.set_ylim(-0.5, max_neurons_to_plot - 0.5)
            tick_positions = np.linspace(0, max_neurons_to_plot - 1, num=min(5, max_neurons_to_plot))
            tick_labels = y_labels[tick_positions.astype(int)]
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)

        ax.set_title(f'Layer: {layer_name}')
        ax.set_ylabel(ylabel_text)
        if i == 1:
            ax.set_xlabel("Time Step")
        else:
            ax.set_xlabel("")


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    full_save_path = os.path.join(save_path, filename)
    plt.savefig(full_save_path, dpi=150)
    print(f"Layer comparison raster plot saved to {full_save_path}")
    plt.close(fig)


