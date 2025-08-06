import os
import shutil
import time
from itertools import product
import torch
import numpy as np

from utils import TrainingTracker, visualize_learning_curves, visualize_convergence, analyze_energy_efficiency, \
    perform_statistical_tests, analyze_and_visualize_results, evaluate_model, calculate_spike_metrics
from nns import ANNNet, Net
from encoding import *
from Lyapunov import analyze_lyapunov_exponents, compare_lyapunov_with_encoding_performance
from base_Attactors import Config, load_data_with_encoding

# Define grid search parameters
TMAX_VALUES = [2, 4, 8, 16, 32]
NUM_STEPS_VALUES = [1, 3, 5, 10, 20]


ATTRACTORS = ['lorenz', 'rossler', 'aizawa', 'nose_hoover', 'sprott', 'chua']
NUM_RUNS = 10
SUBSAMPLE_SIZE = 0.1  

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")


def setup_experiment_folders(tmax, num_steps):
    """Create and setup experiment folders for the given parameters"""
    # Create main experiment folder
    experiment_folder = f"grid_search_results/tmax_{tmax}_steps_{num_steps}"
    os.makedirs(experiment_folder, exist_ok=True)

    # Create subfolders for different results
    statistical_folder = os.path.join(experiment_folder, "statistical_results")
    lyapunov_folder = os.path.join(experiment_folder, "lyapunov_results")
    info_capacity_folder = os.path.join(experiment_folder, "info_capacity_results")

    os.makedirs(statistical_folder, exist_ok=True)
    os.makedirs(lyapunov_folder, exist_ok=True)
    os.makedirs(info_capacity_folder, exist_ok=True)

    return experiment_folder, statistical_folder, lyapunov_folder, info_capacity_folder


def run_lyapunov_analysis(config, lyapunov_folder):
    """Run Lyapunov exponent analysis for the current configuration"""
    print("\n=== Analyzing Lyapunov Exponents ===")

    # Override the save path in the analyze_lyapunov_exponents function
    original_results_dir = 'lyapunov_results'

    # Analyze Lyapunov exponents
    lyapunov_df = analyze_lyapunov_exponents(config, results_dir=lyapunov_folder)

    # Compare with encoding performance
    compare_lyapunov_with_encoding_performance(config, results_dir=os.path.join(lyapunov_folder, 'comparison'))

    return lyapunov_df


def run_statistical_experiment(config, statistical_folder):
    """Run the main statistical experiments for all attractors"""
    print("\n=== Running Statistical Experiments ===")

    # Define experiments: one for each attractor
    experiments = [
        {'name': f'{attractor.capitalize()}-SNN', 'model_type': ['SNN'], 'encoding': attractor}
        for attractor in ATTRACTORS
    ]

    # Store results for all runs
    all_results = {
        exp['name']: {
            model_type: {
                'best_acc': [],
                'convergence_epoch': [],
                'final_loss': []
            }
            for model_type in exp['model_type']
        }
        for exp in experiments
    }

    # Store all trackers
    all_run_trackers = []

    for run in range(NUM_RUNS):
        print(f"\n=== Running experiment {run + 1}/{NUM_RUNS} ===")

        # Store trackers for this run
        run_trackers = []

        for experiment in experiments:
            print(f"\n--- {experiment['name']} ---")
            exp_name = experiment['name']

            # Load and encode data
            try:
                train_loader, test_loader, encoding_time = load_data_with_encoding(
                    config,
                    apply_umap=(experiment['encoding'] in ATTRACTORS),
                    encoding=experiment['encoding'],
                    n_components=config.n_components,
                    subsample_size=SUBSAMPLE_SIZE
                )
            except Exception as e:
                print(f"Error loading data for {exp_name}: {e}")
                continue

            for model_type in experiment['model_type']:
                print(f"Training {model_type}...")
                try:
                    if model_type == 'SNN':
                        model = Net(config, experiment['encoding']).to(device)
                    else:
                        model = ANNNet(config, experiment['encoding']).to(device)

                    criterion = torch.nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

                    # Create a tracker for this experiment
                    tracker = TrainingTracker(exp_name, model_type)
                    tracker.set_encoding_time(encoding_time)

                    best_acc = 0
                    convergence_epoch = config.num_epochs
                    patience_counter = 0

                    for epoch in range(config.num_epochs):
                        epoch_start_time = time.time()
                        model.train()
                        epoch_loss = 0
                        correct = 0
                        total = 0
                        epoch_spike_count = 0

                        for batch_idx, (data, targets) in enumerate(train_loader):
                            data, targets = data.to(device), targets.to(device)
                            optimizer.zero_grad()

                            if model_type == 'SNN':
                                spk_rec, mem_rec = model(data)
                                loss = torch.stack([
                                    criterion(mem_rec['layer4'][step], targets)
                                    for step in range(config.num_steps)
                                ]).mean()
                                _, predicted = spk_rec['layer4'].sum(dim=0).max(1)

                                # Calculate spike metrics
                                batch_spikes, _, _ = calculate_spike_metrics(spk_rec, data.size(0))
                                epoch_spike_count += batch_spikes
                            else:  # ANN
                                outputs = model(data)
                                loss = criterion(outputs, targets)
                                _, predicted = outputs.max(1)

                            loss.backward()
                            optimizer.step()

                            total += targets.size(0)
                            correct += predicted.eq(targets).sum().item()
                            epoch_loss += loss.item()

                        # Calculate time for this epoch
                        epoch_time = time.time() - epoch_start_time
                        tracker.add_training_time(epoch_time)

                        # End of epoch evaluation
                        avg_epoch_loss = epoch_loss / len(train_loader)
                        avg_epoch_acc = 100. * correct / total
                        test_loss, test_acc = evaluate_model(
                            model, test_loader, criterion, config,
                            model_type, device)

                        # Update tracker
                        if model_type == 'SNN':
                            tracker.update(avg_epoch_loss, avg_epoch_acc, test_loss, test_acc,
                                           epoch_spike_count, epoch, epoch_time)
                        else:
                            tracker.update(avg_epoch_loss, avg_epoch_acc, test_loss, test_acc,
                                           0, epoch, epoch_time)

                        print(f'Epoch {epoch} completed: '
                              f'Train Loss: {avg_epoch_loss:.4f}, '
                              f'Train Acc: {avg_epoch_acc:.2f}%, '
                              f'Test Loss: {test_loss:.4f}, '
                              f'Test Acc: {test_acc:.2f}%, '
                              f'Time: {epoch_time:.2f}s')

                        if model_type == 'SNN':
                            print(f'Total spike count: {epoch_spike_count}')

                        if test_acc > best_acc:
                            best_acc = test_acc
                            convergence_epoch = epoch
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= config.early_stopping_patience:
                            print(f"Early stopping triggered! Best accuracy: {best_acc:.2f}%")
                            break

                    # Record results
                    all_results[exp_name][model_type]['best_acc'].append(best_acc)
                    all_results[exp_name][model_type]['convergence_epoch'].append(convergence_epoch)
                    all_results[exp_name][model_type]['final_loss'].append(test_loss)

                    # Add tracker to the list
                    run_trackers.append(tracker)

                    print(f"\nFinished {model_type} training for {exp_name}")
                    print(f"Best accuracy: {best_acc:.2f}%")
                    print(f"Convergence epoch: {convergence_epoch}")
                    print(f"Final test loss: {test_loss:.4f}")
                    print(f"Total training time: {tracker.training_time:.2f}s")
                    print(f"Encoding time: {tracker.encoding_time:.2f}s")

                except Exception as e:
                    print(f"Error training {model_type} for {exp_name}: {e}")
                    continue

        # Store trackers for this run
        all_run_trackers.extend(run_trackers)

        # Save intermediate results after each run
        run_results_path = os.path.join(statistical_folder, f'run_{run + 1}_results')
        os.makedirs(run_results_path, exist_ok=True)

        try:
            # Create learning curve visualizations for this run
            visualize_learning_curves(run_trackers, run_results_path)

            # Create convergence visualizations for this run
            conv_df = visualize_convergence(run_trackers, run_results_path)

            # Analyze energy efficiency for this run
            analyze_energy_efficiency(run_trackers, conv_df, run_results_path)
        except Exception as e:
            print(f"Error creating visualizations for run {run + 1}: {e}")

    # Combine all runs for final analysis
    final_results_path = os.path.join(statistical_folder, 'final_results')
    os.makedirs(final_results_path, exist_ok=True)

    try:
        # Create final learning curve visualizations
        visualize_learning_curves(all_run_trackers, final_results_path)

        # Create final convergence visualizations
        final_conv_df = visualize_convergence(all_run_trackers, final_results_path)

        # Analyze energy efficiency across all runs
        analyze_energy_efficiency(all_run_trackers, final_conv_df, final_results_path)

        # Traditional statistical analysis
        analyze_and_visualize_results(all_results, final_results_path, NUM_RUNS)

        # Perform statistical tests
        perform_statistical_tests(all_run_trackers, final_results_path)
    except Exception as e:
        print(f"Error in final analysis: {e}")

    return all_results, all_run_trackers


def run_grid_search():
    """Run grid search over tmax and num_steps parameters"""

    # Create main grid search folder
    os.makedirs("grid_search_results", exist_ok=True)

    # Create a configuration object
    config = Config()

    # Set up the parameter grid
    parameter_grid = list(product(TMAX_VALUES, NUM_STEPS_VALUES))
    total_combinations = len(parameter_grid)

    print(f"Starting grid search with {total_combinations} parameter combinations")
    print(f"Each combination will run {NUM_RUNS} times for {len(ATTRACTORS)} attractors")
    print(f"Total experiments: {total_combinations * NUM_RUNS * len(ATTRACTORS)}")

    # Run experiments for each parameter combination
    for i, (tmax, num_steps) in enumerate(parameter_grid):
        print(f"\n{'=' * 80}")
        print(f"Running parameter combination {i + 1}/{total_combinations}: tmax={tmax}, num_steps={num_steps}")
        print(f"{'=' * 80}")

        # Update config with current parameters
        config.tmax = tmax
        config.num_steps = num_steps

        # Setup experiment folders
        experiment_folder, statistical_folder, lyapunov_folder, info_capacity_folder = setup_experiment_folders(tmax,
                                                                                                                num_steps)

        # Save current configuration to the experiment folder
        config_info = {
            'tmax': tmax,
            'num_steps': num_steps,
            'attractors': ATTRACTORS,
            'num_runs': NUM_RUNS,
            'batch_size': config.batch_size,
            'num_hidden': config.num_hidden,
            'learning_rate': config.learning_rate,
            'early_stopping_patience': config.early_stopping_patience,
            'beta': config.beta
        }

        with open(os.path.join(experiment_folder, 'config.txt'), 'w') as f:
            for key, value in config_info.items():
                f.write(f"{key}: {value}\n")

        # Run statistical experiments
        try:
            all_results, all_run_trackers = run_statistical_experiment(config, statistical_folder)
        except Exception as e:
            print(f"Error in statistical experiments: {e}")

        try:
            lyapunov_df = run_lyapunov_analysis(config, lyapunov_folder)
        except Exception as e:
            print(f"Error in Lyapunov analysis: {e}")


        print(f"\nCompleted parameter combination: tmax={tmax}, num_steps={num_steps}")


if __name__ == "__main__":
    print("Starting grid search experiment for chaos attractors")
    run_grid_search()
    print("Grid search complete!")