from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import spikegen
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import csv
import os
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import statistics
import time
import seaborn as sns


    
    
@jit(nopython=True)
def lorenz_transformer_vectorized(x0, y0, z0, sigma=10, beta=2.667, rho=28, tmax=20.0, h=0.01):
    """
    Vectorized and JIT-compiled Lorenz transformer
    """
    # print('param:', sigma, beta, rho, tmax)
    nsteps = round((tmax - 0) / h)
    t = np.linspace(0, tmax, nsteps + 1)

    # Initialize arrays
    x = np.zeros(nsteps + 1)
    y = np.zeros(nsteps + 1)
    z = np.zeros(nsteps + 1)

    # Set initial conditions
    x[0] = x0
    y[0] = y0
    z[0] = z0

    # RK4 integration
    for i in range(nsteps):
        # Current values
        xi, yi, zi = x[i], y[i], z[i]

        # K1
        k1x = h * (sigma * (yi - xi))
        k1y = h * (xi * (rho - zi) - yi)
        k1z = h * (xi * yi - beta * zi)

        # K2
        x2 = xi + k1x / 2
        y2 = yi + k1y / 2
        z2 = zi + k1z / 2
        k2x = h * (sigma * (y2 - x2))
        k2y = h * (x2 * (rho - z2) - y2)
        k2z = h * (x2 * y2 - beta * z2)

        # K3
        x3 = xi + k2x / 2
        y3 = yi + k2y / 2
        z3 = zi + k2z / 2
        k3x = h * (sigma * (y3 - x3))
        k3y = h * (x3 * (rho - z3) - y3)
        k3z = h * (x3 * y3 - beta * z3)

        # K4
        x4 = xi + k3x
        y4 = yi + k3y
        z4 = zi + k3z
        k4x = h * (sigma * (y4 - x4))
        k4y = h * (x4 * (rho - z4) - y4)
        k4z = h * (x4 * y4 - beta * z4)

        # Update
        x[i + 1] = xi + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        y[i + 1] = yi + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        z[i + 1] = zi + (k1z + 2 * k2z + 2 * k3z + k4z) / 6

    return np.column_stack((x, y, z))


def lorenz_encode(data, num_steps, tmax=2, sigma=10, beta=8/3, rho=28):
    """
    Optimized Lorenz encoding
    """
    batch_size = data.shape[0]
    num_features = data.shape[1]
    encoded_data = np.zeros((batch_size, num_steps, num_features * 3))

    # Convert to numpy and normalize
    data_numpy = data.cpu().numpy()
    data_max = np.max(np.abs(data_numpy))
    if data_max > 0:
        data_numpy = data_numpy / data_max

    # Process batches of features
    feature_batch_size = min(50, num_features)  # Process 50 features at a time

    for b in range(batch_size):
        for j in range(0, num_features, feature_batch_size):
            end_idx = min(j + feature_batch_size, num_features)
            current_features = data_numpy[b, j:end_idx]

            # Process batch of features
            for k, value in enumerate(current_features):
                lorenz_output = lorenz_transformer_vectorized(
                    value, value * 0.2, -value,
                    sigma=sigma, beta=beta, rho=rho,
                    tmax=tmax
                )

                # Resample trajectory
                feature_idx = j + k
                for dim in range(3):
                    encoded_data[b, :, feature_idx * 3 + dim] = np.interp(
                        np.linspace(0, 1, num_steps),
                        np.linspace(0, 1, lorenz_output.shape[0]),
                        lorenz_output[:, dim]
                    )

    return torch.from_numpy(encoded_data).float().to(data.device)


def ttfs_encoding(data, num_steps, threshold=0.1, normalize=True):
    """Time-to-first-spike encoding - [time, batch, features]"""
    # Note: This function now expects 2D data: [batch, features]
    batch_size, features = data.shape
    ttfs = torch.ones_like(data) * num_steps
    for step in range(num_steps):
        spike_indices = (data > threshold) & (ttfs == num_steps)
        ttfs[spike_indices] = step
        data = data - threshold
        data.clamp_(min=0)
    if normalize:
        ttfs = ttfs.float() / num_steps

    # [time, batch, features]
    spikes = torch.zeros(num_steps, batch_size, features, device=data.device)

    for t in range(num_steps):
        spikes[t] = (ttfs <= t).float()

    return spikes


def delta_encoding_wrapper(data, num_steps, threshold=0.1):
    """Delta encoding wrapper - [time, batch, features]"""
    # Note: This function now expects 2D data: [batch, features]
    batch_size, features = data.shape
    
    # We need to reshape for spikegen.delta, which expects [batch, channels, height, width]
    # Here, since data is already flat, we need a custom delta encoding logic
    
    delta_spikes = torch.zeros(num_steps, batch_size, features, device=data.device)
    prev_data = data.clone()
    
    for t in range(1, num_steps):
        current_data = data.clone()
        diff = current_data - prev_data
        delta_spikes[t] = (diff.abs() > threshold).float()
        prev_data = current_data
        
    return delta_spikes


def phase_encoding(data, num_steps, phase_max=2 * np.pi):
    """
    Phase encoding - [time, batch, features]
    """
    # Note: This function now expects 2D data: [batch, features]
    device = data.device
    batch_size, features = data.shape

    phase = data * phase_max

    time = torch.arange(num_steps, device=device).float() / num_steps * phase_max

    # [time, batch, features]
    spikes = torch.zeros(num_steps, batch_size, features, device=device)

    for t in range(num_steps):
        spikes[t] = (torch.sin(phase - time[t]) >= 0).float()

    return spikes


def burst_encoding(data, num_steps, beta=0.95, vth=0.5):
    """
    Burst - [time, batch, features]
    """
    # Note: This function now expects 2D data: [batch, features]
    device = data.device
    batch_size, features = data.shape

    spikes = torch.zeros(num_steps, batch_size, features, device=device)
    membrane_potential = torch.zeros(batch_size, features, device=device)
    burst_function = torch.ones(batch_size, features, device=device)

    if torch.max(data) > 0:
        normalized_data = data / torch.max(data)
    else:
        normalized_data = data.clone()

    for t in range(num_steps):
        threshold = burst_function * vth

        membrane_potential = membrane_potential + normalized_data

        spike_out = (membrane_potential >= threshold).float()
        spikes[t] = spike_out

        membrane_potential = membrane_potential - threshold * spike_out

        burst_function = torch.where(
            spike_out == 1,
            beta * burst_function,
            torch.ones_like(burst_function)
        )

    return spikes


######################################

@jit(nopython=True)
def nose_hoover_transformer_vectorized(x0, y0, z0, alpha=1.0, tmax=20.0, h=0.01):
    """
    Nosé-Hoover oscillator transformer
    """
    nsteps = round((tmax - 0) / h)
    t = np.linspace(0, tmax, nsteps + 1)

    x = np.zeros(nsteps + 1, dtype=np.float64)
    y = np.zeros(nsteps + 1, dtype=np.float64)
    z = np.zeros(nsteps + 1, dtype=np.float64)

    x[0] = x0
    y[0] = y0
    z[0] = z0

    for i in range(nsteps):
        xi, yi, zi = x[i], y[i], z[i]

        k1x = h * yi
        k1y = h * (-xi - yi * zi)
        k1z = h * (yi * yi - alpha)

        x2 = xi + k1x / 2
        y2 = yi + k1y / 2
        z2 = zi + k1z / 2
        k2x = h * y2
        k2y = h * (-x2 - y2 * z2)
        k2z = h * (y2 * y2 - alpha)

        x3 = xi + k2x / 2
        y3 = yi + k2y / 2
        z3 = zi + k2z / 2
        k3x = h * y3
        k3y = h * (-x3 - y3 * z3)
        k3z = h * (y3 * y3 - alpha)

        x4 = xi + k3x
        y4 = yi + k3y
        z4 = zi + k3z
        k4x = h * y4
        k4y = h * (-x4 - y4 * z4)
        k4z = h * (y4 * y4 - alpha)

        x[i + 1] = xi + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        y[i + 1] = yi + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        z[i + 1] = zi + (k1z + 2 * k2z + 2 * k3z + k4z) / 6

    return np.column_stack((x, y, z))


@jit(nopython=True)
def sprott_case_c_transformer_vectorized(x0, y0, z0, a=3.0, tmax=20.0, h=0.01):
    """
    Sprott Case C transformer
    """
    nsteps = round((tmax - 0) / h)
    t = np.linspace(0, tmax, nsteps + 1)

    x = np.zeros(nsteps + 1, dtype=np.float64)
    y = np.zeros(nsteps + 1, dtype=np.float64)
    z = np.zeros(nsteps + 1, dtype=np.float64)

    x[0] = x0
    y[0] = y0
    z[0] = z0

    for i in range(nsteps):
        xi, yi, zi = x[i], y[i], z[i]

        dx = yi * zi
        dy = xi - yi
        dz = 1.0 - a * xi * xi

        x[i + 1] = xi + h * dx
        y[i + 1] = yi + h * dy
        z[i + 1] = zi + h * dz

    return np.column_stack((x, y, z))


@jit(nopython=True)
def chua_circuit_transformer_vectorized(x0, y0, z0, alpha=15.6, beta=28.58,
                                        gamma=0.0, tmax=20.0, h=0.01):
    """
    Chua's Circuit transformer
    """
    nsteps = round((tmax - 0) / h)
    t = np.linspace(0, tmax, nsteps + 1)

    x = np.zeros(nsteps + 1, dtype=np.float64)
    y = np.zeros(nsteps + 1, dtype=np.float64)
    z = np.zeros(nsteps + 1, dtype=np.float64)

    x[0] = x0
    y[0] = y0
    z[0] = z0

    m0 = -1.143
    m1 = -0.714

    for i in range(nsteps):
        xi, yi, zi = x[i], y[i], z[i]

        h_x = m1 * xi + 0.5 * (m0 - m1) * (abs(xi + 1) - abs(xi - 1))

        k1x = h * (alpha * (yi - xi - h_x))
        k1y = h * (xi - yi + zi)
        k1z = h * (-beta * yi - gamma * zi)

        x2 = xi + k1x / 2
        y2 = yi + k1y / 2
        z2 = zi + k1z / 2
        h_x2 = m1 * x2 + 0.5 * (m0 - m1) * (abs(x2 + 1) - abs(x2 - 1))

        k2x = h * (alpha * (y2 - x2 - h_x2))
        k2y = h * (x2 - y2 + z2)
        k2z = h * (-beta * y2 - gamma * z2)

        x3 = xi + k2x / 2
        y3 = yi + k2y / 2
        z3 = zi + k2z / 2
        h_x3 = m1 * x3 + 0.5 * (m0 - m1) * (abs(x3 + 1) - abs(x3 - 1))

        k3x = h * (alpha * (y3 - x3 - h_x3))
        k3y = h * (x3 - y3 + z3)
        k3z = h * (-beta * y3 - gamma * z3)

        x4 = xi + k3x
        y4 = yi + k3y
        z4 = zi + k3z
        h_x4 = m1 * x4 + 0.5 * (m0 - m1) * (abs(x4 + 1) - abs(x4 - 1))

        k4x = h * (alpha * (y4 - x4 - h_x4))
        k4y = h * (x4 - y4 + z4)
        k4z = h * (-beta * y4 - gamma * z4)

        x[i + 1] = xi + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        y[i + 1] = yi + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        z[i + 1] = zi + (k1z + 2 * k2z + 2 * k3z + k4z) / 6

    return np.column_stack((x, y, z))


def chen_encode(data, num_steps, tmax=2):
    """Chen attractor encoding"""
    return _attractor_encode(data, num_steps, tmax, chen_transformer_vectorized)


def rossler_encode(data, num_steps, tmax=2):
    """Rossler attractor encoding"""
    return _attractor_encode(data, num_steps, tmax, rossler_transformer_vectorized)


def aizawa_encode(data, num_steps, tmax=2):
    """Aizawa attractor encoding"""
    return _attractor_encode(data, num_steps, tmax, aizawa_transformer_vectorized)


def _attractor_encode(data, num_steps, tmax, attractor_fn):
    """Generic attractor encoding function"""
    batch_size = data.shape[0]
    num_features = data.shape[1]
    encoded_data = np.zeros((batch_size, num_steps, num_features * 3))

    data_numpy = data.cpu().numpy() if data.is_cuda else data.numpy()
    data_max = np.max(np.abs(data_numpy))

    if data_max > 1e-8:
        data_numpy = data_numpy / (data_max + 1e-8)

    feature_batch_size = min(50, num_features)

    for b in range(batch_size):
        for j in range(0, num_features, feature_batch_size):
            end_idx = min(j + feature_batch_size, num_features)
            current_features = data_numpy[b, j:end_idx].flatten()

            for k in range(len(current_features)):
                value = current_features[k].item()

                attractor_output = attractor_fn(
                    float(value),
                    float(value * 0.2),
                    float(-value),
                    tmax=tmax
                )

                feature_idx = j + k
                for dim in range(3):
                    encoded_data[b, :, feature_idx * 3 + dim] = np.interp(
                        np.linspace(0, 1, num_steps),
                        np.linspace(0, 1, len(attractor_output)),
                        attractor_output[:, dim].astype(np.float32)
                    )

    return torch.from_numpy(encoded_data).float().to(data.device)


def encode_dataset(dataset, encoding, num_steps, device, batch_size, tmax, custom_params=None):
    all_encoded = []
    all_targets = []

    # Get the shape of the first data sample to pre-allocate
    sample_data, _ = dataset[0]
    sample_data = sample_data.unsqueeze(0).to(device)

    # Process a sample batch to determine the output feature dimension
    test_encoded = process_batch(sample_data, encoding, num_steps, tmax, custom_params)
    feature_dim = test_encoded.shape[1:]

    # Pre-allocate tensors for all data
    full_encoded = torch.zeros(len(dataset), *feature_dim).to(device)
    full_targets = torch.zeros(len(dataset), dtype=torch.long)

    for start_idx in range(0, len(dataset), batch_size):
        end_idx = min(start_idx + batch_size, len(dataset))
        current_batch_size = end_idx - start_idx

        # Extract batch data and targets
        batch_data_list = []
        batch_targets_list = []
        for idx in range(start_idx, end_idx):
            data, target = dataset[idx]
            batch_data_list.append(data)
            batch_targets_list.append(target)
            
        batch_data = torch.stack(batch_data_list).to(device)
        encoded_batch = process_batch(batch_data, encoding, num_steps, tmax, custom_params)

        full_encoded[start_idx:end_idx] = encoded_batch
        full_targets[start_idx:end_idx] = torch.tensor(batch_targets_list)

    return TensorDataset(full_encoded.cpu(), full_targets.cpu())


def process_batch(data, encoding, num_steps, tmax, custom_params=None):
    batch_size = data.size(0)

    # All encodings except 'default' and 'umap' now receive a 2D tensor of shape [batch, features]
    # For 'default' and 'umap' they still receive the raw data or UMAP data directly.
    if encoding == 'default':
        return data.view(batch_size, -1)
    
    elif encoding in ['rate', 'latency', 'ttfs', 'delta', 'phase', 'burst']:
        # Ensure input data is 2D, which it will be after UMAP, or from the raw data flattening below
        if data.dim() > 2:
            flat_data = data.view(batch_size, -1)
        else:
            flat_data = data
            
        # Call the appropriate encoding function
        if encoding == 'rate':
            return spikegen.rate(flat_data, num_steps=num_steps).permute(1, 0, 2)
        elif encoding == 'latency':
            return spikegen.latency(flat_data, num_steps=num_steps, normalize=True).permute(1, 0, 2)
        elif encoding == 'ttfs':
            return ttfs_encoding(flat_data, num_steps=num_steps).permute(1, 0, 2)
        elif encoding == 'delta':
            # --- 针对 Delta 编码的修正 ---
            # spikegen.delta的输入需要是4D张量 [batch, channels, height, width]
            # UMAP数据是2D，所以需要先重塑为4D
            # 假设channels=1, height=1, width=features
            input_4d = flat_data.view(batch_size, 1, 1, -1)
            
            # spikegen.delta返回一个稀疏张量，形状为 [num_steps, batch_size, 1, 1, features]
            spk_gen_sparse = spikegen.delta(input_4d, threshold=0.1, num_steps=num_steps)
            
            # 在进行permute和view之前，需要将其转换为稠密张量
            spk_gen_dense = spk_gen_sparse.to_dense()

            # 将形状从 [num_steps, batch_size, 1, 1, features] 转换为 [batch_size, num_steps, features]
            return spk_gen_dense.permute(1, 0, 2, 3, 4).view(batch_size, num_steps, -1)
        elif encoding == 'phase':
            return phase_encoding(flat_data, num_steps=num_steps).permute(1, 0, 2)
        elif encoding == 'burst':
            return burst_encoding(flat_data, num_steps=num_steps).permute(1, 0, 2)
            
    elif encoding == 'umap':
        return data.view(data.size(0), -1)

    elif encoding == 'lorenz':
        # For Lorenz encoding, first flatten the input
        flat_data = data.view(batch_size, -1)
        encoded_list = []

        # Process each sample individually to maintain consistent dimensions
        for i in range(batch_size):
            sample = flat_data[i:i + 1]
            encoded = lorenz_encode(sample, num_steps, tmax)
            encoded_list.append(encoded)

        return torch.cat(encoded_list, dim=0)
    
    elif encoding in ['chen', 'rossler', 'aizawa', 'nose_hoover', 'sprott', 'chua']:
         # Attractor encodings take flat data
        if data.dim() > 2:
            flat_data = data.view(batch_size, -1)
        else:
            flat_data = data
            
        encoded_list = []
        for i in range(batch_size):
            sample = flat_data[i:i + 1]
            if encoding == 'chen':
                encoded = chen_encode(sample, num_steps, tmax)
            elif encoding == 'rossler':
                encoded = rossler_encode(sample, num_steps, tmax)
            elif encoding == 'aizawa':
                encoded = aizawa_encode(sample, num_steps, tmax)
            elif encoding == 'nose_hoover':
                encoded = nose_hoover_encode(sample, num_steps, tmax)
            elif encoding == 'sprott':
                encoded = sprott_encode(sample, num_steps, tmax)
            elif encoding == 'chua':
                encoded = chua_encode(sample, num_steps, tmax)
            encoded_list.append(encoded)
        
        return torch.cat(encoded_list, dim=0)

    elif encoding == 'mixed_oscillator' and custom_params is not None:
        flat_data = data.view(batch_size, -1)
        return mixed_oscillator_encode(flat_data, num_steps, tmax, params=custom_params)
    else:
        raise ValueError(f"Unknown encoding technique: {encoding}")

