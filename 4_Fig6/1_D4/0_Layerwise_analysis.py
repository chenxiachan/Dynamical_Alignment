import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import pandas as pd
import os
import time
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from encoding import encode_dataset



class Config:
    batch_size = 128
    data_path = 'data/'
    num_hidden = 500
    num_outputs = 10
    n_components = 7
    num_steps = 5
    tmax = 8

    FIXED_BETA = 0.95
    FIXED_THRESHOLD = 1.0

    num_epochs = 40
    learning_rate = 1e-4
    early_stopping_patience = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class FixedNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_steps = config.num_steps
        self.input_dim = config.n_components * 3

        # Weights are still learnable
        self.fc1 = nn.Linear(self.input_dim, config.num_hidden)
        self.fc2 = nn.Linear(config.num_hidden, config.num_hidden)
        self.fc3 = nn.Linear(config.num_hidden, config.num_hidden)
        self.fc4 = nn.Linear(config.num_hidden, config.num_outputs)

        # Crucially: set learn_beta=False and learn_threshold=False
        self.lif1 = snn.Leaky(beta=config.FIXED_BETA, threshold=config.FIXED_THRESHOLD, learn_beta=False,
                              learn_threshold=False)
        self.lif2 = snn.Leaky(beta=config.FIXED_BETA, threshold=config.FIXED_THRESHOLD, learn_beta=False,
                              learn_threshold=False)
        self.lif3 = snn.Leaky(beta=config.FIXED_BETA, threshold=config.FIXED_THRESHOLD, learn_beta=False,
                              learn_threshold=False)
        self.lif4 = snn.Leaky(beta=config.FIXED_BETA, threshold=config.FIXED_THRESHOLD, learn_beta=False,
                              learn_threshold=False)
        self.all_lifs = [self.lif1, self.lif2, self.lif3, self.lif4]

    def forward(self, x):
        mems = [lif.init_leaky() for lif in self.all_lifs]

        spk_rec = {f'layer{i}': [] for i in range(1, 5)}
        mem_rec = {f'layer{i}': [] for i in range(1, 5)}
        cur_rec = {f'layer{i}': [] for i in range(1, 5)}

        for step in range(self.num_steps):
            cur1 = self.fc1(x[:, step, :])
            spk1, mems[0] = self.lif1(cur1, mems[0])
            cur2 = self.fc2(spk1)
            spk2, mems[1] = self.lif2(cur2, mems[1])
            cur3 = self.fc3(spk2)
            spk3, mems[2] = self.lif3(cur3, mems[2])
            cur4 = self.fc4(spk3)
            spk4, mems[3] = self.lif4(cur4, mems[3])

            currents = [cur1, cur2, cur3, cur4]
            spikes = [spk1, spk2, spk3, spk4]

            for i in range(1, 5):
                cur_rec[f'layer{i}'].append(currents[i - 1])
                spk_rec[f'layer{i}'].append(spikes[i - 1])
                mem_rec[f'layer{i}'].append(mems[i - 1])

        for i in range(1, 5):
            cur_rec[f'layer{i}'] = torch.stack(cur_rec[f'layer{i}'], dim=0)
            mem_rec[f'layer{i}'] = torch.stack(mem_rec[f'layer{i}'], dim=0)

        return cur_rec, mem_rec


def get_data_loaders(config, encoding_params):
    """Load and encode data"""
    transform = transforms.Compose([
        transforms.Resize((28, 28)), transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0,), (1,))
    ])

    train_dataset = datasets.MNIST(config.data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(config.data_path, train=False, download=True, transform=transform)

    train_indices = torch.randperm(len(train_dataset))[:10000]

    import umap
    reducer = umap.UMAP(n_components=config.n_components, random_state=42)

    # Use only the training set to fit UMAP, then transform the test set
    train_data_flat = train_dataset.data[train_indices].view(-1, 28 * 28).float()
    test_data_flat = test_dataset.data.view(-1, 28 * 28).float()

    print("Fitting UMAP on training data...")
    train_data_reduced = torch.FloatTensor(reducer.fit_transform(train_data_flat))
    print("Transforming test data with fitted UMAP...")
    test_data_reduced = torch.FloatTensor(reducer.transform(test_data_flat))

    train_dataset_reduced = TensorDataset(train_data_reduced, train_dataset.targets[train_indices])
    test_dataset_reduced = TensorDataset(test_data_reduced, test_dataset.targets)

    print(f"Encoding datasets for delta={encoding_params['delta']}...")
    train_encoded = encode_dataset(train_dataset_reduced, 'mixed_oscillator', config.num_steps, device, config.batch_size, config.tmax, custom_params=encoding_params)
    test_encoded = encode_dataset(test_dataset_reduced, 'mixed_oscillator', config.num_steps, device, config.batch_size, config.tmax, custom_params=encoding_params)

    train_loader = DataLoader(train_encoded, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_encoded, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader


def train_or_load_fixed_model(config, encoding_params, model_path):
    model = FixedNet(config).to(device)
    if os.path.exists(model_path):
        print(f"Loading pre-trained fixed-parameter model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    print(f"Model not found. Training a new fixed-parameter model for delta={encoding_params['delta']}...")
    train_loader, test_loader = get_data_loaders(config, encoding_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    patience_counter = 0

    for epoch in range(config.num_epochs):
        model.train()
        for _, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            # Correction: Move data and targets to the device
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            _, mem_rec = model(data)
            # Use the membrane potential of the last layer to calculate the loss
            loss = torch.stack([criterion(mem, targets) for mem in mem_rec['layer4']]).mean()
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        total_correct, total_samples = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                # Correction: Also fix this part in the evaluation loop
                data = data.to(device)
                targets = targets.to(device)

                cur_rec, _ = model(data)
                # Use the accumulated sum of the last layer's current (before spiking) for prediction
                output_activity = cur_rec['layer4'].sum(dim=0)
                _, predicted = output_activity.max(1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        test_acc = 100 * total_correct / total_samples if total_samples > 0 else 0
        print(f"Epoch {epoch + 1}/{config.num_epochs} - Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            print(f"✨ New best accuracy: {best_acc:.2f}%. Saving model to {model_path}")
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def generate_csv_summary(results, filename="fixed_params_layerwise_summary.csv"):
    """Saves the analysis results to a CSV file"""
    summary_data = []
    for mode, mode_data in results.items():
        for i in range(1, 5):
            layer = f'layer{i}'
            currents = mode_data[layer]['currents']
            membranes = mode_data[layer]['membranes']

            summary_data.append({
                'Mode': mode,
                'Layer': i,
                'Current_Mean': np.mean(currents),
                'Current_Std': np.std(currents),
                'Membrane_Mean': np.mean(membranes),
                'Membrane_Std': np.std(membranes),
                'Fixed_Beta': mode_data['beta'],
                'Fixed_Threshold': mode_data['threshold']
            })

    df = pd.DataFrame(summary_data)
    df.to_csv(filename, index=False)
    print(f"\n--- Layer-wise Analysis Summary (Fixed Parameters) ---")
    print(f"Results saved to {filename}")
    print(df.to_string())


def run_fixed_params_layerwise_analysis():
    """Main analysis function"""
    config = Config()

    modes = {
        'Dissipative': {'delta': 10.0, 'alpha': 2.0, 'beta': 0.1, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},
        'Expansive': {'delta': -1.5, 'alpha': 2.0, 'beta': 0.1, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0},
        'Critical': {'delta': 2.0, 'alpha': 2.0, 'beta': 0.1, 'gamma': 0.1, 'omega': 1.0, 'drive': 0.0}
    }

    results = {mode: {} for mode in modes}
    model_dir = "saved_fixed_models"
    os.makedirs(model_dir, exist_ok=True)

    for mode_name, params in modes.items():
        print(f"\n{'=' * 50}\n--- Processing Mode: {mode_name} ---")
        model_path = os.path.join(model_dir, f"fixed_model_delta_{params['delta']:.2f}.pth")

        # 1. Train or load fixed-parameter model
        model = train_or_load_fixed_model(config, params, model_path)

        # 2. Store fixed beta and threshold
        results[mode_name]['beta'] = config.FIXED_BETA
        results[mode_name]['threshold'] = config.FIXED_THRESHOLD

        # 3. Load test data and collect layer-wise data
        _, test_loader = get_data_loaders(config, params)

        layer_data = {f'layer{i}': {'currents': [], 'membranes': []} for i in range(1, 5)}

        model.eval()
        with torch.no_grad():
            for data, _ in tqdm(test_loader, desc=f"Collecting data for {mode_name}"):
                data = data.to(device)
                cur_rec, mem_rec = model(data)

                for i in range(1, 5):
                    layer = f'layer{i}'
                    layer_data[layer]['currents'].append(cur_rec[layer].cpu().numpy().flatten())
                    layer_data[layer]['membranes'].append(mem_rec[layer].cpu().numpy().flatten())

        # Aggregate data
        for i in range(1, 5):
            layer = f'layer{i}'
            results[mode_name][layer] = {
                'currents': np.concatenate(layer_data[layer]['currents']),
                'membranes': np.concatenate(layer_data[layer]['membranes'])
            }

    # 4. Generate CSV report
    generate_csv_summary(results)


if __name__ == "__main__":
    run_fixed_params_layerwise_analysis()
