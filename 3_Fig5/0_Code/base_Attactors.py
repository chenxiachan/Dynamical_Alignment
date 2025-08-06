from utils import TrainingTracker, calculate_spike_metrics, visualize_learning_curves, visualize_convergence, \
                analyze_energy_efficiency, perform_statistical_tests, analyze_and_visualize_results, evaluate_model
from nns import ANNNet, Net
from Lyapunov import *


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Config:
    batch_size = 32
    data_path = 'data/'
    num_inputs = 28 * 28
    num_hidden = 500
    num_outputs = 10
    n_components = 7
    num_steps = 5
    tmax = 8
    beta = 0.95
    num_epochs = 500  # Maximum number of epochs

    learning_rate = 5e-5
    early_stopping_patience = 5  # Number of evaluations to wait before early stopping


device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

def load_data_with_encoding(config, apply_umap=False, n_components=5, encoding='default', subsample_size=0.1,
                            custom_params=None, reducer=None):
    """
    Load data with consistent encoding and batch handling
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])

    # Load datasets
    train_dataset = datasets.MNIST(config.data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(config.data_path, train=False, download=True, transform=transform)

    # Subsample if needed
    if subsample_size > 0:
        if isinstance(subsample_size, float):
            train_size = int(len(train_dataset) * subsample_size)
            test_size = int(len(test_dataset) * subsample_size)
        else:
            train_size = min(subsample_size, len(train_dataset))
            test_size = min(subsample_size // 5, len(test_dataset))

        train_indices = torch.randperm(len(train_dataset))[:train_size]
        test_indices = torch.randperm(len(test_dataset))[:test_size]

        train_dataset.data = train_dataset.data[train_indices]
        train_dataset.targets = train_dataset.targets[train_indices]
        test_dataset.data = test_dataset.data[test_indices]
        test_dataset.targets = test_dataset.targets[test_indices]

    encoding_start_time = time.time()

    if encoding == 'umap':
        print("Applying UMAP as encoding...")
        all_data = torch.cat([train_dataset.data.float(), test_dataset.data.float()]).view(-1, 28 * 28)
        if reducer is None:
            reducer = umap.UMAP(n_components=n_components)
            all_data_reduced = reducer.fit_transform(all_data)
        else:
            print('Loaded Reducer Applied!')
            all_data_reduced = reducer.transform(all_data)


        train_data_reduced = torch.FloatTensor(all_data_reduced[:len(train_dataset)])
        test_data_reduced = torch.FloatTensor(all_data_reduced[len(train_dataset):])

        train_dataset = TensorDataset(train_data_reduced, train_dataset.targets)
        test_dataset = TensorDataset(test_data_reduced, test_dataset.targets)
        apply_umap = False

    # Apply UMAP if needed
    if apply_umap:
        print("Applying UMAP...")
        all_data = torch.cat([train_dataset.data.float(), test_dataset.data.float()]).view(-1, 28 * 28)
        if reducer is None:
            reducer = umap.UMAP(n_components=n_components)
            all_data_reduced = reducer.fit_transform(all_data)
        else:
            print('Loaded Reducer Applied!')
            all_data_reduced = reducer.transform(all_data)


        train_data_reduced = torch.FloatTensor(all_data_reduced[:len(train_dataset)])
        test_data_reduced = torch.FloatTensor(all_data_reduced[len(train_dataset):])

        train_dataset = TensorDataset(train_data_reduced, train_dataset.targets)
        test_dataset = TensorDataset(test_data_reduced, test_dataset.targets)

    # Encode datasets
    print(f"Encoding datasets with {encoding} encoding...")
    train_dataset = encode_dataset(train_dataset, encoding, config.num_steps, device,
                                   config.batch_size, config.tmax, custom_params)
    test_dataset = encode_dataset(test_dataset, encoding, config.num_steps, device,
                                  config.batch_size, config.tmax, custom_params)

    encoding_time = time.time() - encoding_start_time

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True
    )

    return train_loader, test_loader, encoding_time, reducer


def main_statistical():
    config = Config()
    save_path = 'statistical_results_attractors_grid'
    os.makedirs(save_path, exist_ok=True)

    experiments = [
        # {'name': 'Default-SNN', 'model_type': ['SNN'], 'encoding': 'default'},
        {'name': 'Rossler-SNN', 'model_type': ['SNN'], 'encoding': 'rossler'},
        {'name': 'Lorenz-SNN', 'model_type': ['SNN'], 'encoding': 'lorenz'},
        {'name': 'Aizawa-SNN', 'model_type': ['SNN'], 'encoding': 'aizawa'},
        {'name': 'NoseHoover-SNN', 'model_type': ['SNN'], 'encoding': 'nose_hoover'},
        {'name': 'Sprott-SNN', 'model_type': ['SNN'], 'encoding': 'sprott'},
        {'name': 'Chua-SNN', 'model_type': ['SNN'], 'encoding': 'chua'},
    ]

    num_runs = 10
    subsample_size = 0.1

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
    # Store all trackers for each run
    all_run_trackers = []

    for run in range(num_runs):
        print(f"\n=== Running experiment {run + 1}/{num_runs} ===")

        # Store trackers for this run
        run_trackers = []

        for experiment in experiments:
            print(f"\n--- {experiment['name']} ---")
            exp_name = experiment['name']

            # Load and encode data, measuring encoding time
            train_loader, test_loader, encoding_time, _ = load_data_with_encoding(
                config,
                apply_umap=(experiment['encoding'] in ['lorenz', 'chen', 'rossler', 'aizawa',
                                                       'nose_hoover', 'sprott', 'chua']),
                encoding=experiment['encoding'],
                n_components=config.n_components,
                subsample_size=subsample_size
            )

            for model_type in experiment['model_type']:
                print(f"Training {model_type}...")
                if model_type == 'SNN':
                    model = Net(config, experiment['encoding']).to(device)
                else:
                    model = ANNNet(config, experiment['encoding']).to(device)

                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

                # Create a tracker for this experiment
                tracker = TrainingTracker(exp_name, model_type)
                tracker.set_encoding_time(encoding_time)  # Set encoding time

                best_acc = 0
                convergence_epoch = config.num_epochs
                patience_counter = 0

                for epoch in range(config.num_epochs):
                    epoch_start_time = time.time()
                    model.train()
                    epoch_loss = 0
                    correct = 0
                    total = 0
                    epoch_spike_count = 0  # For SNN energy metrics

                    for batch_idx, (data, targets) in enumerate(train_loader):
                        data, targets = data.to(device), targets.to(device)
                        optimizer.zero_grad()

                        # Forward pass and loss calculation
                        if model_type == 'SNN':

                            spk_rec, mem_rec = model(data)
                            loss = torch.stack([
                                criterion(mem_rec['layer4'][step], targets)
                                for step in range(config.num_steps)
                            ]).mean()
                            _, predicted = spk_rec['layer4'].sum(dim=0).max(1)

                            # Calculate spike metrics for this batch
                            batch_spikes, _, _ = calculate_spike_metrics(spk_rec, data.size(0))
                            epoch_spike_count += batch_spikes

                        else:  # ANN
                            outputs = model(data)
                            loss = criterion(outputs, targets)
                            _, predicted = outputs.max(1)

                        # Backward pass
                        loss.backward()
                        optimizer.step()

                        # Update statistics
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

                    # Update tracker with epoch results
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

        # Store trackers for this run
        all_run_trackers.extend(run_trackers)

        # Save intermediate results after each run
        run_results_path = os.path.join(save_path, f'run_{run + 1}_results')
        os.makedirs(run_results_path, exist_ok=True)

        # Create learning curve visualizations for this run
        visualize_learning_curves(run_trackers, run_results_path)

        # Create convergence visualizations for this run
        conv_df = visualize_convergence(run_trackers, run_results_path)

        # Analyze energy efficiency for this run
        analyze_energy_efficiency(run_trackers, conv_df, run_results_path)

    # Combine all runs for final analysis
    final_results_path = os.path.join(save_path, 'final_results')
    os.makedirs(final_results_path, exist_ok=True)

    # Create final learning curve visualizations
    visualize_learning_curves(all_run_trackers, final_results_path)

    # Create final convergence visualizations
    final_conv_df = visualize_convergence(all_run_trackers, final_results_path)

    # Analyze energy efficiency across all runs
    analyze_energy_efficiency(all_run_trackers, final_conv_df, final_results_path)

    # Traditional statistical analysis using original functions
    analyze_and_visualize_results(all_results, final_results_path, num_runs)

    perform_statistical_tests(all_run_trackers, final_results_path)


def main_with_lyapunov_analysis():
    """
    Main function: Run Lyapunov exponent analysis and compare with performance
    """
    config = Config()

    # 1. Analyze Lyapunov exponents for different attractors
    print("\n=== Analyzing Lyapunov Exponents for Different Attractors ===")
    lyapunov_df = analyze_lyapunov_exponents(config)

    # 2. Compare Lyapunov exponents with encoding performance
    print("\n=== Comparing Lyapunov Exponents with Encoding Performance ===")
    compare_lyapunov_with_encoding_performance(config)

    print("\n=== Lyapunov Exponent Analysis Complete ===")


def main_with_information_dynamics_analysis():
    config = Config()
    save_path = 'information_dynamics_results'
    os.makedirs(save_path, exist_ok=True)

    attractors = [
        'lorenz', 'rossler', 'aizawa', 'nose_hoover', 'sprott', 'chua'
    ]

    results = []

    for attractor in attractors:
        print(f"\n--- Analyze {attractor} attracor ---")

        try:
            # 1. Generate trajectories
            trajectory = generate_attractor_trajectory(attractor, num_points=2000, tmax=config.tmax)

            # 2. Calculate Lyapunov exponent
            lyapunov_exponents = compute_attractor_lyapunov_exponents(attractor)

            # 3. Calculate the AIS for each dimension
            ais_x = calculate_active_info_storage(trajectory[:, 0])
            ais_y = calculate_active_info_storage(trajectory[:, 1])
            ais_z = calculate_active_info_storage(trajectory[:, 2])
            avg_ais = np.nanmean([ais_x, ais_y, ais_z])

            results.append({
                'attractor': attractor,
                'largest_lyapunov': lyapunov_exponents[0],
                'lyapunov_sum': sum(lyapunov_exponents),
                'ais_x': ais_x,
                'ais_y': ais_y,
                'ais_z': ais_z,
                'avg_ais': avg_ais,
            })

        except Exception as e:
            print(f" {attractor} Error: {str(e)}")
            results.append({
                'attractor': attractor,
                'error': str(e)
            })

    df = pd.DataFrame(results)
    csv_path = os.path.join(save_path, 'attractor_information_dynamics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nThe result has been saved to {csv_path}")

    return df


def generate_attractor_trajectory(attractor_type, num_points=2000, tmax=8.0, initial_cond=(0.1, 0.1, 0.1)):
    """
    Generates the trajectory of a specified attractor.

    Args:
        attractor_type (str): The type of attractor ('lorenz', 'rossler', etc.).
        num_points (int, optional): The number of points in the trajectory.
        tmax (float, optional): The maximum integration time.
        initial_cond (tuple, optional): The initial conditions (x0, y0, z0).

    Returns:
        np.ndarray: An array of shape [num_points, 3] containing the trajectory.
    """
    dt = tmax / num_points
    t = np.linspace(0, tmax, num_points)
    x0, y0, z0 = initial_cond

    # Select the corresponding system of differential equations
    if attractor_type == 'lorenz':
        system = lambda x, y, z: (
            10 * (y - x),
            x * (28 - z) - y,
            x * y - (8 / 3) * z
        )
    elif attractor_type == 'rossler':
        system = lambda x, y, z: (
            -y - z,
            x + 0.2 * y,
            0.2 + z * (x - 5.7)
        )
    elif attractor_type == 'aizawa':
        system = lambda x, y, z: (
            (z - 0.7) * x - 3.5 * y,
            3.5 * x + (z - 0.7) * y,
            0.6 + 0.95 * z - z ** 3 / 3 - (x ** 2 + y ** 2) * (1 + 0.25 * z) + 0.1 * z * x ** 3
        )
    elif attractor_type == 'nose_hoover':
        system = lambda x, y, z: (
            y,
            -x - y * z,
            y ** 2 - 1.0
        )
    elif attractor_type == 'sprott':
        system = lambda x, y, z: (
            y * z,
            x - y,
            1 - 3 * x ** 2
        )
    elif attractor_type == 'chua':
        def system(x, y, z):
            # Chua diode nonlinear function
            h_x = -0.714 * x + 0.5 * (0.143) * (abs(x + 1) - abs(x - 1))
            return (
                15.6 * (y - x - h_x),
                x - y + z,
                -28.58 * y
            )
    else:
        raise ValueError(f"Unsupported attractor type: {attractor_type}")

    # Initialize trajectory array
    trajectory = np.zeros((num_points, 3))
    trajectory[0] = [x0, y0, z0]

    # Use RK4 integration
    for i in range(1, num_points):
        x, y, z = trajectory[i - 1]

        # RK4 method
        k1x, k1y, k1z = system(x, y, z)
        k1x, k1y, k1z = dt * k1x, dt * k1y, dt * k1z

        k2x, k2y, k2z = system(x + k1x / 2, y + k1y / 2, z + k1z / 2)
        k2x, k2y, k2z = dt * k2x, dt * k2y, dt * k2z

        k3x, k3y, k3z = system(x + k2x / 2, y + k2y / 2, z + k2z / 2)
        k3x, k3y, k3z = dt * k3x, dt * k3y, dt * k3z

        k4x, k4y, k4z = system(x + k3x, y + k3y, z + k3z)
        k4x, k4y, k4z = dt * k4x, dt * k4y, dt * k4z

        # Update
        x_new = x + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        y_new = y + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        z_new = z + (k1z + 2 * k2z + 2 * k3z + k4z) / 6

        trajectory[i] = [x_new, y_new, z_new]

    return trajectory


def compute_attractor_lyapunov_exponents(attractor_type):
    lyapunov_values = {
        'lorenz': [0.37, 0.0, -14.04],
        'rossler': [0.099, 0.0, -5.59],
        'aizawa': [0.176, 0.0, -1.01],
        'nose_hoover': [0.203, 0.0, -0.203],
        'sprott': [0.352, 0.0, -1.352],
        'chua': [0.403, 0.0, -4.82]
    }

    if attractor_type in lyapunov_values:
        return lyapunov_values[attractor_type]
    else:
        return [0.1, 0.0, -0.1]





if __name__ == "__main__":
    main_statistical()
    main_with_lyapunov_analysis()
    main_with_information_dynamics_analysis()

