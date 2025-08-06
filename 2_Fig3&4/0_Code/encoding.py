from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import spikegen
import matplotlib.pyplot as plt
import numpy as np
import umap
from numba import jit
import csv
import os
# from util import train_printer
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import statistics
import time
import seaborn as sns

import numpy as np
from numba import jit
import torch




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
                    sigma=sigma, beta=beta, rho=rho,  # 使用传入参数
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
    """Time-to-first-spike encoding - Ensures output format is [time, batch, features]"""
    shape = data.shape
    batch_size, channels, height, width = shape
    data = data.view(batch_size, -1)
    ttfs = torch.ones_like(data) * num_steps
    for step in range(num_steps):
        spike_indices = (data > threshold) & (ttfs == num_steps)
        ttfs[spike_indices] = step
        data = data - threshold
        data.clamp_(min=0)
    if normalize:
        ttfs = ttfs.float() / num_steps

    features = channels * height * width
    # Create output tensor [time, batch, features]
    spikes = torch.zeros(num_steps, batch_size, features, device=data.device)

    # Generate spikes based on ttfs
    for t in range(num_steps):
        # Set spikes at the corresponding time step
        spikes[t] = (ttfs <= t).float()

    return spikes


def delta_encoding_wrapper(data, num_steps, threshold=0.1):
    """[time, batch, features]"""
    # Flattens the spatial dimensions into features
    batch_size = data.shape[0]
    delta_spikes = spikegen.delta(data, threshold=threshold)
    delta_spikes = delta_spikes.view(batch_size, -1)

    # Creates the output tensor with shape [time, batch, features]
    spikes = torch.zeros(num_steps, batch_size, delta_spikes.shape[1], device=data.device)

    # Uses the same delta encoding for all time steps
    for t in range(num_steps):
        spikes[t] = delta_spikes

    return spikes


def phase_encoding(data, num_steps, phase_max=2 * np.pi):
    """
    Ensures output format is [time, batch, features]
    """
    device = data.device
    batch_size, channels, height, width = data.shape
    features = channels * height * width

    # Flatten data to [batch, features]
    data_flat = data.view(batch_size, features)

    # Calculate phase values
    phase = data_flat * phase_max

    # Create time steps
    time = torch.arange(num_steps, device=device).float() / num_steps * phase_max

    # Create output tensor [time, batch, features]
    spikes = torch.zeros(num_steps, batch_size, features, device=device)

    # Generate spikes
    for t in range(num_steps):
        # sin(phase - time) >= 0 generates spikes
        spikes[t] = (torch.sin(phase - time[t]) >= 0).float()

    return spikes


def burst_encoding(data, num_steps, beta=0.95, vth=0.5):
    """
    Burst encoding - ensures output format is [time, batch, features]
    """
    device = data.device
    batch_size, channels, height, width = data.shape
    features = channels * height * width

    # Flatten spatial dimensions
    data_flat = data.view(batch_size, features)

    # Initialize output tensor and state variables
    spikes = torch.zeros(num_steps, batch_size, features, device=device)
    membrane_potential = torch.zeros(batch_size, features, device=device)
    burst_function = torch.ones(batch_size, features, device=device)

    # Normalize input data
    if torch.max(data_flat) > 0:
        normalized_data = data_flat / torch.max(data_flat)
    else:
        normalized_data = data_flat.clone()

    # Simulate SNN temporal dynamics
    for t in range(num_steps):
        # Calculate effective threshold for the current timestep
        threshold = burst_function * vth

        # Update membrane potential
        membrane_potential = membrane_potential + normalized_data

        # Generate spikes
        spike_out = (membrane_potential >= threshold).float()
        spikes[t] = spike_out

        # Update membrane potential (reset after firing)
        membrane_potential = membrane_potential - threshold * spike_out

        # Update burst function value
        burst_function = torch.where(
            spike_out == 1,
            beta * burst_function,  # Decrease burst function value if a spike is fired
            torch.ones_like(burst_function)  # Reset burst function to 1 if no spike is fired
        )

    return spikes

######################################

@jit(nopython=True)
def nose_hoover_transformer_vectorized(x0, y0, z0, alpha=1.0, tmax=20.0, h=0.01):
    """
    Nosé-Hoover oscillator transformer

    The Nosé-Hoover oscillator is a chaotic system that represents
    a harmonic oscillator coupled to a thermostat.

    Parameters:
        x0, y0, z0: Initial conditions
        alpha: Control parameter (typically 1.0)
        tmax: Maximum integration time
        h: Step size for numerical integration
    """
    nsteps = round((tmax - 0) / h)
    t = np.linspace(0, tmax, nsteps + 1)

    # Initialize arrays with explicit data type
    x = np.zeros(nsteps + 1, dtype=np.float64)
    y = np.zeros(nsteps + 1, dtype=np.float64)
    z = np.zeros(nsteps + 1, dtype=np.float64)

    # Set initial conditions
    x[0] = x0
    y[0] = y0
    z[0] = z0

    # RK4 integration
    for i in range(nsteps):
        xi, yi, zi = x[i], y[i], z[i]

        # K1
        k1x = h * yi
        k1y = h * (-xi - yi * zi)
        k1z = h * (yi * yi - alpha)

        # K2
        x2 = xi + k1x / 2
        y2 = yi + k1y / 2
        z2 = zi + k1z / 2
        k2x = h * y2
        k2y = h * (-x2 - y2 * z2)
        k2z = h * (y2 * y2 - alpha)

        # K3
        x3 = xi + k2x / 2
        y3 = yi + k2y / 2
        z3 = zi + k2z / 2
        k3x = h * y3
        k3y = h * (-x3 - y3 * z3)
        k3z = h * (y3 * y3 - alpha)

        # K4
        x4 = xi + k3x
        y4 = yi + k3y
        z4 = zi + k3z
        k4x = h * y4
        k4y = h * (-x4 - y4 * z4)
        k4z = h * (y4 * y4 - alpha)

        # Update with weighted average
        x[i + 1] = xi + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        y[i + 1] = yi + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        z[i + 1] = zi + (k1z + 2 * k2z + 2 * k3z + k4z) / 6

    return np.column_stack((x, y, z))


@jit(nopython=True)
def sprott_case_c_transformer_vectorized(x0, y0, z0, a=3.0, tmax=20.0, h=0.01):
    """
    Sprott Case C transformer

    One of the simpler chaotic systems from Sprott's collection.
    This is "Case C" which is one of the most commonly used.

    Parameters:
        x0, y0, z0: Initial conditions
        a: Control parameter (typically 3.0)
        tmax: Maximum integration time
        h: Step size for numerical integration
    """
    nsteps = round((tmax - 0) / h)
    t = np.linspace(0, tmax, nsteps + 1)

    # Initialize arrays with explicit data type
    x = np.zeros(nsteps + 1, dtype=np.float64)
    y = np.zeros(nsteps + 1, dtype=np.float64)
    z = np.zeros(nsteps + 1, dtype=np.float64)

    # Set initial conditions
    x[0] = x0
    y[0] = y0
    z[0] = z0


    for i in range(nsteps):
        xi, yi, zi = x[i], y[i], z[i]

        # Compute derivatives
        dx = yi * zi
        dy = xi - yi
        dz = 1.0 - a * xi * xi

        # Update with Euler step
        x[i + 1] = xi + h * dx
        y[i + 1] = yi + h * dy
        z[i + 1] = zi + h * dz

    return np.column_stack((x, y, z))


@jit(nopython=True)
def chua_circuit_transformer_vectorized(x0, y0, z0, alpha=15.6, beta=28.58,
                                        gamma=0.0, tmax=20.0, h=0.01):
    """
    Chua's Circuit transformer

    A chaotic system that models an electronic circuit
    which was one of the first physical systems proven to be chaotic.

    Parameters:
        x0, y0, z0: Initial conditions
        alpha, beta: System parameters
        gamma: Typically 0.0 in the standard model
        tmax: Maximum integration time
        h: Step size for numerical integration
    """
    nsteps = round((tmax - 0) / h)
    t = np.linspace(0, tmax, nsteps + 1)

    # Initialize arrays with explicit data type
    x = np.zeros(nsteps + 1, dtype=np.float64)
    y = np.zeros(nsteps + 1, dtype=np.float64)
    z = np.zeros(nsteps + 1, dtype=np.float64)

    # Set initial conditions
    x[0] = x0
    y[0] = y0
    z[0] = z0

    # Non-linearity parameters for Chua's diode
    m0 = -1.143
    m1 = -0.714

    # RK4 integration
    for i in range(nsteps):
        xi, yi, zi = x[i], y[i], z[i]

        # Chua's diode non-linear function
        h_x = m1 * xi + 0.5 * (m0 - m1) * (abs(xi + 1) - abs(xi - 1))

        # K1
        k1x = h * (alpha * (yi - xi - h_x))
        k1y = h * (xi - yi + zi)
        k1z = h * (-beta * yi - gamma * zi)

        # K2
        x2 = xi + k1x / 2
        y2 = yi + k1y / 2
        z2 = zi + k1z / 2
        h_x2 = m1 * x2 + 0.5 * (m0 - m1) * (abs(x2 + 1) - abs(x2 - 1))

        k2x = h * (alpha * (y2 - x2 - h_x2))
        k2y = h * (x2 - y2 + z2)
        k2z = h * (-beta * y2 - gamma * z2)

        # K3
        x3 = xi + k2x / 2
        y3 = yi + k2y / 2
        z3 = zi + k2z / 2
        h_x3 = m1 * x3 + 0.5 * (m0 - m1) * (abs(x3 + 1) - abs(x3 - 1))

        k3x = h * (alpha * (y3 - x3 - h_x3))
        k3y = h * (x3 - y3 + z3)
        k3z = h * (-beta * y3 - gamma * z3)

        # K4
        x4 = xi + k3x
        y4 = yi + k3y
        z4 = zi + k3z
        h_x4 = m1 * x4 + 0.5 * (m0 - m1) * (abs(x4 + 1) - abs(x4 - 1))

        k4x = h * (alpha * (y4 - x4 - h_x4))
        k4y = h * (x4 - y4 + z4)
        k4z = h * (-beta * y4 - gamma * z4)

        # Update with weighted average
        x[i + 1] = xi + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        y[i + 1] = yi + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        z[i + 1] = zi + (k1z + 2 * k2z + 2 * k3z + k4z) / 6

    return np.column_stack((x, y, z))


# Encoding functions for each attractor
def nose_hoover_encode(data, num_steps, tmax=2):
    """Nosé-Hoover attractor encoding"""
    return _attractor_encode(data, num_steps, tmax, nose_hoover_transformer_vectorized)


def sprott_encode(data, num_steps, tmax=2):
    """Sprott Case C attractor encoding"""
    return _attractor_encode(data, num_steps, tmax, sprott_case_c_transformer_vectorized)


def chua_encode(data, num_steps, tmax=2):
    """Chua's Circuit attractor encoding"""
    return _attractor_encode(data, num_steps, tmax, chua_circuit_transformer_vectorized)


@jit(nopython=True)
def chen_transformer_vectorized(x0, y0, z0, a=60, b=2.667, c=97, tmax=20.0, h=0.01):
    """Chen attractor transformer"""
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

        dx = a * (yi - xi)
        dy = (c - a) * xi - xi * zi + c * yi
        dz = xi * yi - b * zi

        x[i + 1] = xi + h * dx
        y[i + 1] = yi + h * dy
        z[i + 1] = zi + h * dz

    return np.column_stack((x, y, z))


@jit(nopython=True)
def rossler_transformer_vectorized(x0, y0, z0, a=0.2, b=0.2, c=5.7, tmax=20.0, h=0.01):
    """Rossler attractor transformer"""
    nsteps = round((tmax - 0) / h)
    t = np.linspace(0, tmax, nsteps + 1)

    x = np.zeros(nsteps + 1)
    y = np.zeros(nsteps + 1)
    z = np.zeros(nsteps + 1)

    x[0], y[0], z[0] = x0, y0, z0

    for i in range(nsteps):
        xi, yi, zi = x[i], y[i], z[i]

        dx = -yi - zi
        dy = xi + a * yi
        dz = b + zi * (xi - c)

        x[i + 1] = xi + h * dx
        y[i + 1] = yi + h * dy
        z[i + 1] = zi + h * dz

    return np.column_stack((x, y, z))


@jit(nopython=True)
def aizawa_transformer_vectorized(x0, y0, z0, alpha=0.95, beta=0.7, gamma=0.6, delta=3.5, epsilon=0.25, zeta=0.1,
                                  tmax=20.0, h=0.01):
    """Aizawa attractor transformer"""
    nsteps = round((tmax - 0) / h)
    t = np.linspace(0, tmax, nsteps + 1)

    x = np.zeros(nsteps + 1)
    y = np.zeros(nsteps + 1)
    z = np.zeros(nsteps + 1)

    x[0], y[0], z[0] = x0, y0, z0

    for i in range(nsteps):
        xi, yi, zi = x[i], y[i], z[i]

        dx = (zi - beta) * xi - delta * yi
        dy = delta * xi + (zi - beta) * yi
        dz = gamma + alpha * zi - (zi ** 3) / 3 - (xi ** 2 + yi ** 2) * (1 + epsilon * zi) + zeta * zi * xi ** 3

        x[i + 1] = xi + h * dx
        y[i + 1] = yi + h * dy
        z[i + 1] = zi + h * dz

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

    # Move data to CPU and convert to numpy array if it's on GPU
    data_numpy = data.cpu().numpy() if data.is_cuda else data.numpy()
    data_max = np.max(np.abs(data_numpy))

    # Add a small epsilon to prevent division by zero
    if data_max > 1e-8:
        data_numpy = data_numpy / (data_max + 1e-8)

    feature_batch_size = min(50, num_features)

    for b in range(batch_size):
        for j in range(0, num_features, feature_batch_size):
            end_idx = min(j + feature_batch_size, num_features)
            current_features = data_numpy[b, j:end_idx].flatten()

            for k in range(len(current_features)):
                value = current_features[k].item()

                # Generate attractor trajectory
                attractor_output = attractor_fn(
                    float(value),
                    float(value * 0.2),
                    float(-value),
                    tmax=tmax
                )

                # Linear interpolation to specified steps
                feature_idx = j + k
                for dim in range(3):
                    encoded_data[b, :, feature_idx * 3 + dim] = np.interp(
                        np.linspace(0, 1, num_steps),
                        np.linspace(0, 1, len(attractor_output)),
                        attractor_output[:, dim].astype(np.float32)
                    )

    return torch.from_numpy(encoded_data).float().to(data.device)
######################################


def encode_dataset(dataset, encoding, num_steps, device, batch_size, tmax):
    all_encoded = []
    all_targets = []

    sample_data, _ = dataset[0]
    sample_data = sample_data.unsqueeze(0).to(device)

    test_encoded = process_batch(sample_data, encoding, num_steps, tmax)
    feature_dim = test_encoded.shape[1:]

    full_encoded = torch.zeros(len(dataset), *feature_dim).to(device)
    full_targets = torch.zeros(len(dataset), dtype=torch.long)

    for start_idx in range(0, len(dataset), batch_size):
        end_idx = min(start_idx + batch_size, len(dataset))
        
        batch_data = []
        batch_targets = []
        for idx in range(start_idx, end_idx):
            data, target = dataset[idx]
            batch_data.append(data)
            batch_targets.append(target)

        batch_data = torch.stack(batch_data).to(device)
        encoded_batch = process_batch(batch_data, encoding, num_steps, tmax)

        full_encoded[start_idx:end_idx] = encoded_batch
        full_targets[start_idx:end_idx] = torch.tensor(batch_targets)

    return TensorDataset(full_encoded.cpu(), full_targets.cpu())


def process_batch(data, encoding, num_steps, tmax):
    """
    Processes a batch of data and applies the selected encoding scheme.
    """
    batch_size = data.size(0)

    if encoding == 'default':
        return data.view(batch_size, -1)
    elif encoding == 'lorenz':
        # For Lorenz encoding, first flatten the input
        flat_data = data.view(batch_size, -1)
        encoded_list = []

        # Process each sample individually to maintain consistent dimensions
        for i in range(batch_size):
            sample = flat_data[i:i + 1]
            encoded = lorenz_encode(sample, num_steps, tmax)
            encoded_list.append(encoded)

        # Stack all samples along batch dimension
        return torch.cat(encoded_list, dim=0)
    elif encoding in ['rossler', 'aizawa', 'nose_hoover', 'sprott', 'chua']:
        return encode_data(data, encoding, num_steps)
    else:  # All time-based encodings: 'rate', 'latency', 'ttfs', 'delta', 'phase', 'burst'
        # Call encode_data to get the encoded data
        encoded = encode_data(data, encoding, num_steps)

        # Check the dimensions of the encoded data
        if encoded.dim() == 3:  # [time, batch, features]
            # Transpose to [batch, time, features]
            encoded = encoded.permute(1, 0, 2).contiguous()
        elif encoded.dim() == 5:  # [time, batch, c, h, w]
            # Transpose to [batch, time, c, h, w]
            encoded = encoded.permute(1, 0, 2, 3, 4).contiguous()
            # Flatten the spatial dimensions [batch, time, features]
            encoded = encoded.view(batch_size, num_steps, -1)
        return encoded


def encode_data(data, encoding='default', num_steps=5, tmax=2):

    # batch_size, num_steps, num_features
    if encoding == 'default':
        return data.view(data.size(0), -1)
    elif encoding == 'rate':
        return spikegen.rate(data, num_steps=num_steps)
    elif encoding == 'latency':
        return spikegen.latency(data, num_steps=num_steps, normalize=True)
    elif encoding == 'ttfs':
        return ttfs_encoding(data, num_steps=num_steps)
    elif encoding == 'delta':
        return delta_encoding_wrapper(data, num_steps=num_steps)
    elif encoding == 'phase':
        return phase_encoding(data, num_steps=num_steps)
    elif encoding == 'burst':
        return burst_encoding(data, num_steps=num_steps)

    elif encoding == 'umap':
        return data.view(data.size(0), -1)



    #################

    elif encoding == 'lorenz':
        return lorenz_encode(data, num_steps, tmax)

    #################


    elif encoding == 'chen':
        return chen_encode(data, num_steps, tmax)
    elif encoding == 'rossler':
        return rossler_encode(data, num_steps, tmax)
    elif encoding == 'aizawa':
        return aizawa_encode(data, num_steps, tmax)
    elif encoding == 'nose_hoover':
        return nose_hoover_encode(data, num_steps, tmax)
    elif encoding == 'sprott':
        return sprott_encode(data, num_steps, tmax)
    elif encoding == 'chua':
        return chua_encode(data, num_steps, tmax)

    else:
        raise ValueError(f"Unknown encoding technique: {encoding}")
