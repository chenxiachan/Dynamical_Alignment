import json
from joblib import dump
from utils import TrainingTracker, evaluate_model, calculate_spike_metrics, \
    analyze_internal_dynamics, visualize_internal_metrics,\
    save_raster_plots, save_spike_counts_hist, save_membrane_potential_traces
from nns import ANNNet, Net
from encoding import *
from Lyapunov import compute_lyapunov_exponent
from base_Attactors import Config, load_data_with_encoding


ATTRACTORS = ['lorenz', 'rossler', 'aizawa', 'nose_hoover', 'sprott', 'chua']

NUM_RUNS = 10
SUBSAMPLE_SIZE = 0.1  # Using 10% of the data for faster iteration

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")


def generate_experiment_name(params):
    delta = params['delta']
    beta = params['beta']
    gamma = params['gamma']

    exp_name = f"MixedOsc_d{delta:.2f}"

    if beta != 0.1:
        exp_name += f"_b{beta:.2f}"
    if gamma != 0.1:
        exp_name += f"_g{gamma:.2f}"
    if params['alpha'] != 2.0:
        exp_name += f"_a{params['alpha']:.1f}"
    if params['drive'] != 0.0:
        exp_name += f"_dr{params['drive']:.1f}"

    return exp_name

def run_mixed_oscillator_grid_search():
    """
    Run parameter grid searches on hybrid oscillator systems,
    recording impulse statistics and internal dynamics
    """

    os.makedirs("mixed_oscillator_results", exist_ok=True)
    model_save_dir = os.path.join("mixed_oscillator_results", "saved_models")
    os.makedirs(model_save_dir, exist_ok=True)

    # Inherit config from base_Attractors
    config = Config()
    config.num_steps = 5
    config.tmax = 8
    config.num_runs = 10 # Example: Use 3 runs for faster testing

    # Parameter sets
    parameter_sets = [
        # Example sets - use your full list here
        #  {'delta': -1.5, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0}, # Expansive
        #  {'delta': 0.0, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},  # Critical
        #  {'delta': 2.0, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},  # Mid-dissipative (Transition)
        #  {'delta': 10.0, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0} # Strong Dissipative

        {'delta': -1.5, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},

        {'delta': -1, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},

        {'delta': -0.6, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},

        {'delta': -0.3, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},

        {'delta': -0.15, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},

        {'delta': 0.0, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},

        {'delta': 0.15, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},

        {'delta': 0.3, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},

        {'delta': 0.6, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},

        {'delta': 1, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},

        {'delta': 1.5, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},

        {'delta': 5, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},

        {'delta': 10, 'beta': 0.1, 'alpha': 2.0, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0}
    ]

    # save results
    all_lyapunov_data = []
    all_performance_data = []
    all_spike_data = []
    all_internal_metrics_agg = []
    all_detailed_runs = []

    # Run experiments for each parameter combination
    for i, params in enumerate(parameter_sets):
        # (Keep the experiment naming and Lyapunov calculation part as before)
        exp_name = generate_experiment_name(params)
        print(f"\n{'=' * 50}")
        print(f"Running experiment: {exp_name}")
        print(f"Parameters: {params}")
        print(f"{'=' * 50}")

        # 1. Calculate Lyapunov exponent
        try:
            lyapunov_exponents = compute_lyapunov_exponent(
                'mixed_oscillator', steps=int(config.tmax * 100), dt=0.01,
                initial_conditions=(0.1, 0.1, 0.1), attractor_params=params, qr_interval=5
            )
            largest_lyapunov = lyapunov_exponents[0]
            lyapunov_sum = np.sum(lyapunov_exponents)

            all_lyapunov_data.append({
                'attractor': exp_name, 'largest_lyapunov': largest_lyapunov,
                'lyapunov_sum': lyapunov_sum,
                'all_lyapunov': ', '.join(map(str, lyapunov_exponents)),
                'delta': params['delta'], 'beta': params['beta'],
                'alpha': params['alpha'], 'gamma': params['gamma']
            })
            print(f"Lyapunov exponents: {lyapunov_exponents}, Sum: {lyapunov_sum:.4f}")
        except Exception as e:
            print(f"Error computing Lyapunov exponents for {exp_name}: {e}")
            all_lyapunov_data.append({
                'attractor': exp_name, 'largest_lyapunov': np.nan,
                'lyapunov_sum': np.nan, 'all_lyapunov': 'Error',
                'delta': params['delta'], 'beta': params['beta'],
                'alpha': params['alpha'], 'gamma': params['gamma']
            })
            # Decide if you want to continue to SNN training even if Lyapunov fails
            # continue # Option: skip SNN training if Lyapunov fails

        # 2. Run SNN performance tests and analyze internal mechanisms
        run_accuracies = []
        run_convergence_epochs = []
        run_spikes_at_convergence = []
        run_internal_metrics = [] # <-- Store internal metrics for each run
        sample_spikes_for_plot = None  # <-- Store spikes samples for plotting
        sample_mems_for_plot = None  # <-- Store membrane potential samples for plotting


        for run in range(config.num_runs):
            model_for_analysis = None # Store the model from the last epoch for analysis
            try:
                print(f"\nRun {run + 1}/{config.num_runs}")
                train_loader, test_loader, encoding_time, reducer = load_data_with_encoding(
                    config, apply_umap=True, encoding='mixed_oscillator',
                    custom_params=params, n_components=config.n_components,
                    subsample_size=0.1 # Use full dataset for analysis runs
                )

                # save the data
                model_filename = f"encoded_test_delta_{params['delta']:.2f}_run{run}.pt"
                model_save_path = os.path.join(model_save_dir, model_filename)
                encoded_test_data = torch.cat([data for data, _ in test_loader], dim=0)
                encoded_test_labels = torch.cat([labels for _, labels in test_loader], dim=0)
                torch.save({
                    'data': encoded_test_data,
                    'labels': encoded_test_labels
                }, model_save_path)


                # save the umap
                model_filename = f"umap_transformer_delta_{params['delta']:.2f}_run{run}.pth"
                model_save_path = os.path.join(model_save_dir, model_filename)
                dump(reducer, model_save_path)

                model = Net(config, 'mixed_oscillator').to(device)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
                tracker = TrainingTracker(exp_name, 'SNN')
                tracker.set_encoding_time(encoding_time)
                epoch_history = [] # For detailed run saving

                # --- Training Loop ---
                best_acc = 0
                convergence_epoch = config.num_epochs
                patience_counter = 0
                spike_count_at_convergence = 0
                best_model_state = None

                for epoch in range(config.num_epochs):
                    model.train()
                    epoch_loss = 0
                    correct = 0
                    total = 0
                    epoch_spike_count = 0
                    # ... (Inner training loop processing batches) ...
                    for batch_idx, (data, targets) in enumerate(train_loader):
                        data, targets = data.to(device), targets.to(device)
                        optimizer.zero_grad()
                        spk_rec, mem_rec = model(data)
                        loss = torch.stack([criterion(spk_rec['layer4'][step], targets)
                                            for step in range(config.num_steps)]).mean()
                        _, predicted = spk_rec['layer4'].sum(dim=0).max(1)
                        batch_spikes, _, _ = calculate_spike_metrics(spk_rec, data.size(0))
                        epoch_spike_count += batch_spikes
                        loss.backward()
                        optimizer.step()
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                        epoch_loss += loss.item()

                    avg_epoch_loss = epoch_loss / len(train_loader)
                    avg_epoch_acc = 100. * correct / total
                    test_loss, test_acc = evaluate_model(model, test_loader, criterion, config, 'SNN', device)
                    tracker.update(avg_epoch_loss, avg_epoch_acc, test_loss, test_acc, epoch_spike_count, epoch)
                    epoch_history.append({ # Store epoch details
                        'epoch': epoch, 'train_loss': avg_epoch_loss, 'train_acc': avg_epoch_acc,
                        'test_loss': test_loss, 'test_acc': test_acc, 'epoch_spike_count': epoch_spike_count
                    })
                    print(f'Epoch {epoch}: Train Acc={avg_epoch_acc:.2f}%, Test Acc={test_acc:.2f}%, Spikes={epoch_spike_count}')

                    if test_acc > best_acc:
                        best_acc = test_acc
                        convergence_epoch = epoch
                        spike_count_at_convergence = epoch_spike_count
                        patience_counter = 0
                        best_model_state = model.state_dict()
                        # Optional: Save best model state here if needed for analysis later
                        # torch.save(model.state_dict(), f"temp_best_model_{exp_name}_run{run}.pth")
                    else:
                        patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
                # --- End of Training Loop ---
                if best_model_state is not None:
                    model_filename = f"model_delta_{params['delta']:.2f}_run{run}.pth"
                    model_save_path = os.path.join(model_save_dir, model_filename)
                    torch.save(best_model_state, model_save_path)
                    print(f"Best model state for run {run} saved to {model_save_path}")
                else:
                    print(f"No best model state recorded for run {run}, model not saved.")

                # Store performance metrics for this run
                run_accuracies.append(best_acc)
                run_convergence_epochs.append(convergence_epoch)
                run_spikes_at_convergence.append(spike_count_at_convergence)
                # Store detailed history
                all_detailed_runs.append({
                    'experiment': exp_name, 'run': run, 'best_accuracy': best_acc,
                    'convergence_epoch': convergence_epoch, 'spikes_at_convergence': spike_count_at_convergence,
                    'delta': params['delta'], 'beta': params['beta'], 'alpha': params['alpha'], 'gamma': params['gamma'],
                    'lyapunov_sum': lyapunov_sum, 'largest_lyapunov': largest_lyapunov,
                    'epoch_history': epoch_history
                })

                # --- Internal Dynamics Analysis (New Part) ---
                print(f"Analyzing internal dynamics for run {run + 1}...")
                internal_metrics, sample_spikes, sample_mems = analyze_internal_dynamics(model, test_loader, config,
                                                                                         device)
                run_internal_metrics.append(internal_metrics)
                print("Internal metrics collected:", internal_metrics)

                # --- Save sample raster plot for the first run ---
                if run == 0:
                    sample_spikes_for_plot = sample_spikes
                    sample_mems_for_plot = sample_mems
                    print("Stored sample spikes and membrane potentials for plotting.")


            except Exception as e:
                print(f"Error in run {run + 1} for {exp_name}: {e}")
                # Append NaNs or skip if a run fails, ensure lists have consistent length or handle later
                run_accuracies.append(np.nan)
                run_convergence_epochs.append(np.nan)
                run_spikes_at_convergence.append(np.nan)
                run_internal_metrics.append({}) # Append empty dict or NaNs for metrics

        if sample_spikes_for_plot is not None or sample_mems_for_plot is not None:
            print(f"\nGenerating visualizations for {exp_name} (from run 1)...")
            plot_save_path = os.path.join("mixed_oscillator_results", "plots")
            os.makedirs(plot_save_path, exist_ok=True)

            save_raster_plots(sample_spikes_for_plot, exp_name, plot_save_path)
            save_spike_counts_hist(sample_spikes_for_plot, exp_name, plot_save_path)
            save_membrane_potential_traces(sample_mems_for_plot, exp_name, plot_save_path)
        else:
            print(f"No sample data available for plotting for {exp_name}.")


        # --- Aggregate results across runs for this parameter set ---
        if any(np.isfinite(run_accuracies)): # Check if there's at least one successful run
            accuracy_mean = np.nanmean(run_accuracies)
            accuracy_std = np.nanstd(run_accuracies)
            convergence_mean = np.nanmean(run_convergence_epochs)
            convergence_std = np.nanstd(run_convergence_epochs)
            spikes_mean = np.nanmean(run_spikes_at_convergence)
            spikes_std = np.nanstd(run_spikes_at_convergence)

             # Aggregate internal metrics (average across runs)
            aggregated_internal = {}
            if run_internal_metrics:
                # Get keys from the first valid dictionary
                first_valid_metrics = next((m for m in run_internal_metrics if m), {})
                metric_keys = first_valid_metrics.keys()
                for key in metric_keys:
                    values = [m.get(key, np.nan) for m in run_internal_metrics]
                    aggregated_internal[key + '_run_mean'] = np.nanmean(values)
                    aggregated_internal[key + '_run_std'] = np.nanstd(values)


            # Append aggregated data
            all_performance_data.append({
                'Experiment': exp_name, 'Model': 'SNN', 'Accuracy Mean': accuracy_mean, 'Accuracy Std': accuracy_std,
                'Convergence Mean': convergence_mean, 'Convergence Std': convergence_std,
                'delta': params['delta'], 'beta': params['beta'], 'alpha': params['alpha'], 'gamma': params['gamma']
            })
            all_spike_data.append({
                'Experiment': exp_name, 'Spikes Mean': spikes_mean, 'Spikes Std': spikes_std,
                'Accuracy Mean': accuracy_mean, 'Lyapunov Sum': lyapunov_sum, 'Largest Lyapunov': largest_lyapunov,
                'delta': params['delta'], 'beta': params['beta'], 'alpha': params['alpha'], 'gamma': params['gamma']
            })
            # Append aggregated internal metrics
            internal_data_row = {'Experiment': exp_name, 'Lyapunov Sum': lyapunov_sum, 'Largest Lyapunov': largest_lyapunov,
                                 'delta': params['delta'], 'beta': params['beta'], 'alpha': params['alpha'], 'gamma': params['gamma']}
            internal_data_row.update(aggregated_internal)
            all_internal_metrics_agg.append(internal_data_row)


            print(f"\nAggregated results for {exp_name} across {config.num_runs} runs:")
            print(f"  Accuracy: {accuracy_mean:.2f}% ± {accuracy_std:.2f}%")
            print(f"  Convergence: {convergence_mean:.1f} ± {convergence_std:.1f} epochs")
            print(f"  Spikes @ Convergence: {spikes_mean:.0f} ± {spikes_std:.0f}")
            # Print some internal metrics
            for k, v in aggregated_internal.items():
                 if '_run_mean' in k:
                     print(f"  {k}: {v:.4f}")

        else:
            print(f"No successful runs completed for {exp_name}. Skipping aggregation.")


    # --- Save all results after the loop ---
    lyapunov_df = pd.DataFrame(all_lyapunov_data)
    performance_df = pd.DataFrame(all_performance_data)
    spike_df = pd.DataFrame(all_spike_data)
    internal_metrics_df = pd.DataFrame(all_internal_metrics_agg) # <-- New DataFrame

    results_dir = "mixed_oscillator_results"
    lyapunov_df.to_csv(os.path.join(results_dir, "lyapunov_data.csv"), index=False)
    performance_df.to_csv(os.path.join(results_dir, "performance_data.csv"), index=False)
    spike_df.to_csv(os.path.join(results_dir, "spike_data.csv"), index=False)
    internal_metrics_df.to_csv(os.path.join(results_dir, "internal_metrics_data.csv"), index=False) # <-- Save new data

    # Save detailed runs (keep this part as before)
    detailed_runs_dir = os.path.join(results_dir, "detailed_runs")
    os.makedirs(detailed_runs_dir, exist_ok=True)
    for experiment in set([run['experiment'] for run in all_detailed_runs]):
         experiment_runs = [run for run in all_detailed_runs if run['experiment'] == experiment]
         # Separate epoch history before saving run data
         for run_data in experiment_runs:
             epoch_history = run_data.pop('epoch_history', None) # Remove history safely
             if epoch_history:
                 # Save epoch history separately
                 history_filename = f"{experiment}_run{run_data['run']}_epochs.json"
                 with open(os.path.join(detailed_runs_dir, history_filename), 'w') as f:
                     json.dump(epoch_history, f, indent=2)
         # Save main run data (without epoch history)
         runs_filename = f"{experiment}_runs.json"
         with open(os.path.join(detailed_runs_dir, runs_filename), 'w') as f:
             json.dump(experiment_runs, f, indent=2)


    # --- Visualization ---
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Merge all data for comprehensive plotting
    # Start with Lyapunov data
    merged_df = lyapunov_df.copy()
    # Merge performance data
    merged_df = pd.merge(merged_df, performance_df.drop(columns=['Model']),
                         left_on='attractor', right_on='Experiment', how='left', suffixes=('', '_perf'))
    # Merge spike data
    merged_df = pd.merge(merged_df, spike_df.drop(columns=['Accuracy Mean', 'Largest Lyapunov']),
                         left_on='attractor', right_on='Experiment', how='left', suffixes=('', '_spike'))
    # Merge internal metrics data
    merged_df = pd.merge(merged_df, internal_metrics_df.drop(columns=['Largest Lyapunov']),
                         left_on='attractor', right_on='Experiment', how='left', suffixes=('', '_internal'))

    # Clean up potential duplicate columns from merges if necessary (adjust based on actual column names)
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
    # Select only relevant columns or rename if needed
    # For example, if 'Experiment_perf' exists, drop it if 'attractor' is the primary key


    # Call existing visualization function (it needs to be updated to handle potential missing columns)
    try:
        visualize_mixed_oscillator_results(merged_df, plots_dir) # Pass the single merged dataframe
    except Exception as e:
        print(f"Error during standard visualization: {e}")
        print("Columns in DataFrame passed to visualize_mixed_oscillator_results:", merged_df.columns)


    # Call new visualization function for internal metrics
    try:
        visualize_internal_metrics(merged_df, plots_dir)
    except Exception as e:
        print(f"Error during internal metrics visualization: {e}")
        print("Columns in DataFrame passed to visualize_internal_metrics:", merged_df.columns)


    # Note: visualize_mixed_oscillator_results needs modification to use the merged_df properly
    # and handle potentially missing columns gracefully if some runs failed.
    # For example, accessing merged_df['Accuracy Mean'] should now work directly.


    print("\n--- Mixed Oscillator Grid Search with Internal Analysis Complete ---")
    # (Return statement might need adjustment based on what the calling context expects)
    # return lyapunov_df, performance_df, spike_df, internal_metrics_df

def visualize_mixed_oscillator_results(merged_df, save_path):
    """可视化混合振子系统实验结果 (Updated to use merged_df and add internal metrics)"""
    plots_dir = os.path.join(save_path) # Ensure plots_dir is defined correctly
    os.makedirs(plots_dir, exist_ok=True) # Create directory if it doesn't exist

    # --- Plot 1: Lyapunov Sum vs Performance & Spikes ---
    plt.figure(figsize=(20, 15))
    plt.suptitle("Performance and Energy vs Encoder Dynamics (Mixed Oscillator)", fontsize=16)


    # Define columns, checking for existence
    lyapunov_sum_col = 'lyapunov_sum'
    accuracy_col = 'Accuracy Mean'
    spikes_col = 'Spikes Mean'
    beta_col = 'beta' # Assuming 'beta' is the final column name after merge


    # 1.1 Lyapunov Sum vs Accuracy
    plt.subplot(2, 2, 1)
    if lyapunov_sum_col in merged_df.columns and accuracy_col in merged_df.columns and beta_col in merged_df.columns:
        valid_data = merged_df.dropna(subset=[lyapunov_sum_col, accuracy_col, beta_col])
        scatter = plt.scatter(valid_data[lyapunov_sum_col], valid_data[accuracy_col],
                              c=valid_data[beta_col], s=100, alpha=0.7, cmap='viridis')
        plt.colorbar(scatter, label='Beta (Nonlinearity)')
        # Add regression line (Optional: Fit based on expected curve, e.g., quadratic for double peak)
        if len(valid_data) > 2:
             z = np.polyfit(valid_data[lyapunov_sum_col], valid_data[accuracy_col], 4) # Higher order fit for potential double peak
             p = np.poly1d(z)
             x_range = np.linspace(valid_data[lyapunov_sum_col].min(), valid_data[lyapunov_sum_col].max(), 100)
             plt.plot(x_range, p(x_range), "r--", alpha=0.6)
    else:
        plt.text(0.5, 0.5, 'Data Missing', horizontalalignment='center', verticalalignment='center')
    plt.xlabel('Sum of Lyapunov Exponents (Σλᵢ)')
    plt.ylabel('Mean Accuracy (%)')
    plt.title('Accuracy vs Lyapunov Sum')
    plt.grid(True, alpha=0.3)


    # 1.2 Lyapunov Sum vs Spikes
    plt.subplot(2, 2, 2)
    if lyapunov_sum_col in merged_df.columns and spikes_col in merged_df.columns and beta_col in merged_df.columns:
        valid_data = merged_df.dropna(subset=[lyapunov_sum_col, spikes_col, beta_col])
        scatter = plt.scatter(valid_data[lyapunov_sum_col], valid_data[spikes_col],
                              c=valid_data[beta_col], s=100, alpha=0.7, cmap='viridis')
        plt.colorbar(scatter, label='Beta (Nonlinearity)')
         # Add regression line
        if len(valid_data) > 1:
            z = np.polyfit(valid_data[lyapunov_sum_col], valid_data[spikes_col], 1) # Linear fit often appropriate here
            p = np.poly1d(z)
            x_range = np.linspace(valid_data[lyapunov_sum_col].min(), valid_data[lyapunov_sum_col].max(), 100)
            plt.plot(x_range, p(x_range), "r--", alpha=0.6)
    else:
        plt.text(0.5, 0.5, 'Data Missing', horizontalalignment='center', verticalalignment='center')
    plt.xlabel('Sum of Lyapunov Exponents (Σλᵢ)')
    plt.ylabel('Mean Spikes at Convergence')
    plt.title('Spike Count vs Lyapunov Sum')
    plt.grid(True, alpha=0.3)

    # 1.3 Accuracy vs Spikes (Energy Efficiency View)
    plt.subplot(2, 2, 3)
    if accuracy_col in merged_df.columns and spikes_col in merged_df.columns and lyapunov_sum_col in merged_df.columns:
        valid_data = merged_df.dropna(subset=[accuracy_col, spikes_col, lyapunov_sum_col])
        scatter = plt.scatter(valid_data[spikes_col], valid_data[accuracy_col],
                              c=valid_data[lyapunov_sum_col], s=100, alpha=0.7, cmap='coolwarm')
        plt.colorbar(scatter, label='Lyapunov Sum (Σλᵢ)')
         # Add labels for experiments if needed (can get cluttered)
        # for i, txt in enumerate(valid_data['attractor']): # Assuming 'attractor' column exists
        #     plt.annotate(txt, (valid_data[spikes_col].iloc[i], valid_data[accuracy_col].iloc[i]), fontsize=8)
    else:
        plt.text(0.5, 0.5, 'Data Missing', horizontalalignment='center', verticalalignment='center')

    plt.xlabel('Mean Spikes at Convergence')
    plt.ylabel('Mean Accuracy (%)')
    plt.title('Accuracy vs Spike Count (Colored by Lyapunov Sum)')
    plt.grid(True, alpha=0.3)


    # 1.4 Efficiency vs Lyapunov Sum
    plt.subplot(2, 2, 4)
    if accuracy_col in merged_df.columns and spikes_col in merged_df.columns and lyapunov_sum_col in merged_df.columns and beta_col in merged_df.columns:
        # Calculate efficiency, handle potential division by zero or near-zero spikes
        merged_df['Efficiency'] = merged_df.apply(
            lambda row: (row[accuracy_col] / row[spikes_col]) * 1e5 if row[spikes_col] > 1 else 0,
            axis=1
        )
        valid_data = merged_df.dropna(subset=[lyapunov_sum_col, 'Efficiency', beta_col])

        scatter = plt.scatter(valid_data[lyapunov_sum_col], valid_data['Efficiency'],
                              c=valid_data[beta_col], s=100, alpha=0.7, cmap='viridis')
        plt.colorbar(scatter, label='Beta (Nonlinearity)')
        # Add regression line
        if len(valid_data) > 2:
             z = np.polyfit(valid_data[lyapunov_sum_col], valid_data['Efficiency'], 4) # Fit to capture potential peak
             p = np.poly1d(z)
             x_range = np.linspace(valid_data[lyapunov_sum_col].min(), valid_data[lyapunov_sum_col].max(), 100)
             plt.plot(x_range, p(x_range), "r--", alpha=0.6)

    else:
         plt.text(0.5, 0.5, 'Data Missing', horizontalalignment='center', verticalalignment='center')
    plt.xlabel('Sum of Lyapunov Exponents (Σλᵢ)')
    plt.ylabel('Efficiency (Acc/Spikes × 10⁵)')
    plt.title('Efficiency vs Lyapunov Sum')
    plt.grid(True, alpha=0.3)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    plt.savefig(os.path.join(plots_dir, "lyapunov_performance_spikes_summary.png"), dpi=300)
    plt.close()

    # --- Plot 2: 3D Visualization (Keep as before, ensure columns exist) ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    largest_lyapunov_col = 'largest_lyapunov'
    if lyapunov_sum_col in merged_df.columns and largest_lyapunov_col in merged_df.columns and \
       accuracy_col in merged_df.columns and spikes_col in merged_df.columns:
        valid_data = merged_df.dropna(subset=[lyapunov_sum_col, largest_lyapunov_col, accuracy_col, spikes_col])
        scatter = ax.scatter(valid_data[lyapunov_sum_col], valid_data[largest_lyapunov_col], valid_data[accuracy_col],
                             c=valid_data[spikes_col], s=70, alpha=0.7, cmap='plasma')
        plt.colorbar(scatter, ax=ax, label='Mean Spike Count')
        # Add labels if 'attractor' column is present
        # if 'attractor' in valid_data.columns:
        #      for i, txt in enumerate(valid_data['attractor']):
        #          ax.text(valid_data[lyapunov_sum_col].iloc[i], valid_data[largest_lyapunov_col].iloc[i], valid_data[accuracy_col].iloc[i], txt, size=8)
    else:
         ax.text(0.5, 0.5, 0.5, 'Data Missing', horizontalalignment='center', verticalalignment='center')

    ax.set_xlabel('Lyapunov Sum (Σλᵢ)')
    ax.set_ylabel('Largest Lyapunov (λ_max)')
    ax.set_zlabel('Accuracy (%)')
    ax.set_title('3D View: Lyapunov Metrics vs Accuracy (Color by Spikes)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "3d_visualization.png"), dpi=300)
    plt.close(fig)

    print("Standard visualizations saved.")


if __name__ == "__main__":
    print("Starting mixed oscillator grid search experiment")
    run_mixed_oscillator_grid_search()
    print("Mixed oscillator grid search complete!")
