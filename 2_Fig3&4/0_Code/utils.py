
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
from scipy import stats
import seaborn as sns
from matplotlib.ticker import MaxNLocator

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
    # Create DataFrame for statistical results
    stats_data = []

    # Create data for detailed results DataFrame
    detailed_data = []

    for exp_name in all_results:
        for model_type in all_results[exp_name]:
            results = all_results[exp_name][model_type]

            # Calculate statistics
            acc_mean = np.mean(results['best_acc'])
            acc_std = np.std(results['best_acc'])
            conv_mean = np.mean(results['convergence_epoch'])
            conv_std = np.std(results['convergence_epoch'])

            # Add summary statistics data
            stats_data.append({
                'Experiment': exp_name,
                'Model': model_type,
                'Accuracy Mean': acc_mean,
                'Accuracy Std': acc_std,
                'Convergence Mean': conv_mean,
                'Convergence Std': conv_std
            })

            # Add detailed data for each run
            for run in range(num_runs):
                detailed_data.append({
                    'Experiment': exp_name,
                    'Model': model_type,
                    'Run': run + 1,
                    'Best Accuracy': results['best_acc'][run],
                    'Convergence Epoch': results['convergence_epoch'][run],
                    'Final Loss': results['final_loss'][run]
                })

    # Save summary statistics
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(save_path, 'statistical_summary.csv'), index=False)

    # Save detailed results
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(os.path.join(save_path, 'detailed_results.csv'), index=False)

    # Plot accuracy distribution
    plt.figure(figsize=(15, 10))

    # Box plot for accuracy
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

    # Box plot for convergence epochs
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

    # Plot violin plot for accuracy distribution
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

    # Print statistical summary
    print("\nStatistical Summary:")
    print(stats_df.to_string(index=False))

    # Print the first few rows of detailed results
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
        """Record the encoding time"""
        self.encoding_time = encoding_time

    def add_training_time(self, time_delta):
        """Accumulate total training time"""
        self.training_time += time_delta

    def get_metrics_df(self):
        """Convert tracking data to a DataFrame"""
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


def estimate_intrinsic_dim_PCA(X, plot=False):
    """
    Estimate intrinsic dimensionality of data using PCA variance analysis.
    
    Args:
        X: The input data matrix.
        plot: If True, plots the explained variance and cumulative explained variance.
    
    Returns:
        A dictionary containing the estimated dimensions and explained variance ratios.
    """
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt

    # Fit PCA on the data
    pca = PCA().fit(X)

    # Get explained variance ratios
    var_ratio = pca.explained_variance_ratio_

    # Calculate cumulative explained variance
    cum_var_ratio = np.cumsum(var_ratio)

    # Estimate dimension based on 95% explained variance
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


def estimate_intrinsic_dim_TWO_NN(X, plot=False):
    """Estimate intrinsic dimension using the TWO-NN method."""
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    n_samples = X.shape[0]

    # Find the 3 nearest neighbors (including itself)
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    distances, _ = nbrs.kneighbors(X)

    # Calculate the log of the ratio of the second and first neighbor distances
    mu = np.log(distances[:, 2] / distances[:, 1])

    # Fit the empirical cumulative distribution
    sorted_mu = np.sort(mu)
    F_mu = np.arange(1, n_samples + 1) / float(n_samples)

    # Linear regression to estimate the dimension
    reg = stats.linregress(sorted_mu, np.log(F_mu))
    dim_estimate = reg.slope

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_mu, np.log(F_mu), 'o', alpha=0.5)
        plt.plot(sorted_mu, reg.intercept + reg.slope * sorted_mu, 'r')
        plt.xlabel('log(r2/r1)')
        plt.ylabel('log(F(μ))')
        plt.title(f'TWO-NN Method: Estimated Dimension = {dim_estimate:.2f}')
        plt.grid(True, alpha=0.3)
        plt.show()

    return dim_estimate