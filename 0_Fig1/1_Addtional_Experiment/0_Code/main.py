from utils import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import umap

class Config:
    batch_size = 32
    data_path = 'data/'
    num_inputs = 28 * 28
    num_hidden = 500
    num_outputs = 10
    n_components = 7
    num_steps = 8
    tmax = 2
    beta = 0.95
    num_epochs = 500
    learning_rate = 5e-5
    early_stopping_patience = 5


device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")


def load_data_with_encoding(config, apply_umap_before_encoding=False, n_components=5, encoding='default', subsample_size=0.1):
    """Load data with encoding and measure encoding time"""
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])

    train_dataset = datasets.MNIST(config.data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(config.data_path, train=False, download=True, transform=transform)

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

    if encoding == 'lorenz' or apply_umap_before_encoding:
        all_data = torch.cat([train_dataset.data.float(), test_dataset.data.float()]).view(-1, 28 * 28)
        print("Applying UMAP for dimensionality reduction...")
        reducer = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, metric='euclidean')
        all_data_reduced = reducer.fit_transform(all_data)

        # --- 新增的归一化步骤 ---
        min_val = np.min(all_data_reduced)
        max_val = np.max(all_data_reduced)
        # 防止分母为0
        if max_val - min_val > 1e-8:
            all_data_normalized = (all_data_reduced - min_val) / (max_val - min_val)
        else:
            all_data_normalized = np.zeros_like(all_data_reduced)

        train_data_reduced = torch.FloatTensor(all_data_normalized[:len(train_dataset)])
        test_data_reduced = torch.FloatTensor(all_data_normalized[len(train_dataset):])
        
        train_dataset = TensorDataset(train_data_reduced, train_dataset.targets)
        test_dataset = TensorDataset(test_data_reduced, test_dataset.targets)

        print(f"Encoding datasets with {encoding} encoding on UMAP-reduced data...")
        train_dataset = encode_dataset(train_dataset, encoding, config.num_steps, device, config.batch_size, config.tmax)
        test_dataset = encode_dataset(test_dataset, encoding, config.num_steps, device, config.batch_size, config.tmax)

    else:
        print(f"Encoding datasets with {encoding} encoding on raw data...")
        train_dataset = encode_dataset(train_dataset, encoding, config.num_steps, device, config.batch_size, config.tmax)
        test_dataset = encode_dataset(test_dataset, encoding, config.num_steps, device, config.batch_size, config.tmax)


    encoding_time = time.time() - encoding_start_time
    print(f"Encoding complete, time: {encoding_time:.2f} seconds")

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

    return train_loader, test_loader, encoding_time


def main_statistical():
    config = Config()
    save_path = 'statistical_results_UMAP_Fig1'
    os.makedirs(save_path, exist_ok=True)

    experiments = [
        # {'name': 'Default-SNN', 'model_type': ['SNN'], 'encoding': 'default', 'apply_umap': False},
        {'name': 'Lorenz-SNN', 'model_type': ['SNN'], 'encoding': 'lorenz', 'apply_umap': True},
        # {'name': 'Rate-SNN', 'model_type': ['SNN'], 'encoding': 'rate', 'apply_umap': False},
        # {'name': 'Phase-SNN', 'model_type': ['SNN'], 'encoding': 'phase', 'apply_umap': False},
        # {'name': 'Latency-SNN', 'model_type': ['SNN'], 'encoding': 'latency', 'apply_umap': False},
        # {'name': 'TTFS-SNN', 'model_type': ['SNN'], 'encoding': 'ttfs', 'apply_umap': False},
        # {'name': 'Burst-SNN', 'model_type': ['SNN'], 'encoding': 'burst', 'apply_umap': False},

        {'name': 'UMAP-Rate-SNN', 'model_type': ['SNN'], 'encoding': 'rate', 'apply_umap': True},
        {'name': 'UMAP-Phase-SNN', 'model_type': ['SNN'], 'encoding': 'phase', 'apply_umap': True},
        {'name': 'UMAP-Latency-SNN', 'model_type': ['SNN'], 'encoding': 'latency', 'apply_umap': True},
        {'name': 'UMAP-TTFS-SNN', 'model_type': ['SNN'], 'encoding': 'ttfs', 'apply_umap': True},
        {'name': 'UMAP-Burst-SNN', 'model_type': ['SNN'], 'encoding': 'burst', 'apply_umap': True},

        {'name': 'Default-MLP', 'model_type': ['ANN'], 'encoding': 'default', 'apply_umap': False},
        {'name': 'UMAP-MLP', 'model_type': ['ANN'], 'encoding': 'umap', 'apply_umap': True},
        {'name': 'Lorenz-MLP', 'model_type': ['ANN'], 'encoding': 'lorenz', 'apply_umap': True},
    ]
    num_runs = 5
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

    all_run_trackers = []

    for run in range(num_runs):
        print(f"\n=== Running experiment {run + 1}/{num_runs} ===")
        run_trackers = []

        for experiment in experiments:
            print(f"\n--- {experiment['name']} ---")
            exp_name = experiment['name']
            
            apply_umap_before_encoding = experiment['apply_umap']
            
            train_loader, test_loader, encoding_time = load_data_with_encoding(
                config,
                apply_umap_before_encoding=apply_umap_before_encoding,
                encoding=experiment['encoding'],
                n_components=config.n_components,
                subsample_size=subsample_size
            )

            for model_type in experiment['model_type']:
                print(f"Training {model_type}...")
                if model_type == 'SNN':
                    # Pass the flag to the Net class
                    model = Net(config, experiment['encoding'], is_umap_preprocessed=apply_umap_before_encoding).to(device)
                else:
                    model = ANNNet(config, experiment['encoding']).to(device)

                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

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
                            batch_spikes, _, _ = calculate_spike_metrics(spk_rec, data.size(0))
                            epoch_spike_count += batch_spikes
                        else:
                            outputs = model(data)
                            loss = criterion(outputs, targets)
                            _, predicted = outputs.max(1)

                        loss.backward()
                        optimizer.step()

                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                        epoch_loss += loss.item()

                    epoch_time = time.time() - epoch_start_time
                    tracker.add_training_time(epoch_time)

                    avg_epoch_loss = epoch_loss / len(train_loader)
                    avg_epoch_acc = 100. * correct / total
                    test_loss, test_acc = evaluate_model(
                        model, test_loader, criterion, config,
                        model_type, device)

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

                all_results[exp_name][model_type]['best_acc'].append(best_acc)
                all_results[exp_name][model_type]['convergence_epoch'].append(convergence_epoch)
                all_results[exp_name][model_type]['final_loss'].append(test_loss)

                run_trackers.append(tracker)

                print(f"\nFinished {model_type} training for {exp_name}")
                print(f"Best accuracy: {best_acc:.2f}%")
                print(f"Convergence epoch: {convergence_epoch}")
                print(f"Final test loss: {test_loss:.4f}")
                print(f"Total training time: {tracker.training_time:.2f}s")
                print(f"Encoding time: {tracker.encoding_time:.2f}s")

        all_run_trackers.extend(run_trackers)

        run_results_path = os.path.join(save_path, f'run_{run + 1}_results')
        os.makedirs(run_results_path, exist_ok=True)

        visualize_learning_curves(run_trackers, run_results_path)
        conv_df = visualize_convergence(run_trackers, run_results_path)
        analyze_energy_efficiency(run_trackers, conv_df, run_results_path)

    final_results_path = os.path.join(save_path, 'final_results')
    os.makedirs(final_results_path, exist_ok=True)

    visualize_learning_curves(all_run_trackers, final_results_path)
    final_conv_df = visualize_convergence(all_run_trackers, final_results_path)
    analyze_energy_efficiency(all_run_trackers, final_conv_df, final_results_path)
    analyze_and_visualize_results(all_results, final_results_path, num_runs)
    perform_statistical_tests(all_run_trackers, final_results_path)

    
if __name__ == "__main__":
    main_statistical()