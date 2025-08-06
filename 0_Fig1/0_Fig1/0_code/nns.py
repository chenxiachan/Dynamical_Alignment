import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F


class ANNNet(nn.Module):
    def __init__(self, config, encoding):
        super().__init__()
        self.encoding = encoding

        if encoding in 'lorenz':
            input_size = config.n_components * 3
        elif encoding == 'umap':
            input_size = config.n_components
        elif encoding == 'default':
            input_size = config.num_inputs
        else:
            input_size = config.num_inputs * config.num_steps

        print(f"Initializing ANN with input size: {input_size}")

        self.fc1 = nn.Linear(input_size, config.num_hidden)
        self.fc2 = nn.Linear(config.num_hidden, config.num_hidden)
        self.fc3 = nn.Linear(config.num_hidden, config.num_hidden)
        self.fc4 = nn.Linear(config.num_hidden, config.num_outputs)


    def forward(self, x):
        if self.encoding == 'lorenz':
            if len(x.shape) > 2:
                x = x[:, -1, :]
                # x = x.mean(dim=1)
        x = x.view(x.size(0), -1)

        # x = F.relu(self.bn1(self.fc1(x)))
        # x = self.dropout(x)
        # x = F.relu(self.bn2(self.fc2(x)))
        # x = self.dropout(x)
        # x = F.relu(self.bn3(self.fc3(x)))
        # x = self.dropout(x)
        # x = self.fc4(x)

        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = F.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc4(x)

        return x

class Net(nn.Module):
    def __init__(self, config, encoding):
        super().__init__()
        self.encoding = encoding
        self.num_steps = config.num_steps

        if encoding in ['lorenz', 'chen', 'rossler', 'aizawa', 'nose_hoover', 'sprott', 'chua','mixed_oscillator']:
            self.input_dim = config.n_components * 3
        elif encoding == 'default':
            self.input_dim = config.num_inputs
        elif encoding == 'umap':
            self.input_dim = config.n_components
        else:
            self.input_dim = config.num_inputs

        self.fc1 = nn.Linear(self.input_dim, config.num_hidden)

        # Hidden layers
        self.fc2 = nn.Linear(config.num_hidden, config.num_hidden)
        self.fc3 = nn.Linear(config.num_hidden, config.num_hidden)
        self.fc4 = nn.Linear(config.num_hidden, config.num_outputs)

        # LIF neurons
        self.lif1 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)
        self.lif2 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)
        self.lif3 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)
        self.lif4 = snn.Leaky(beta=config.beta, learn_beta=True, learn_threshold=True)

    def forward(self, x):
        # Initialize spike and membrane potential records for all layers
        spk1_rec, spk2_rec, spk3_rec, spk4_rec = [], [], [], []
        mem1_rec, mem2_rec, mem3_rec, mem4_rec = [], [], [], []

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        # Temporal processing
        for step in range(self.num_steps):
            if self.encoding in ['default', 'umap']:
                cur1 = self.fc1(x)
            else:  # 'lorenz', 'rate', 'latency', 'ttfs', 'delta', 'phase', 'burst'
                cur1 = self.fc1(x[:, step, :])

            # Process through layers
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            # Record all layer activities
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            spk4_rec.append(spk4)
            mem1_rec.append(mem1)
            mem2_rec.append(mem2)
            mem3_rec.append(mem3)
            mem4_rec.append(mem4)

        # Stack temporal sequences
        spk_rec = {
            'layer1': torch.stack(spk1_rec, dim=0),
            'layer2': torch.stack(spk2_rec, dim=0),
            'layer3': torch.stack(spk3_rec, dim=0),
            'layer4': torch.stack(spk4_rec, dim=0)
        }
        mem_rec = {
            'layer1': torch.stack(mem1_rec, dim=0),
            'layer2': torch.stack(mem2_rec, dim=0),
            'layer3': torch.stack(mem3_rec, dim=0),
            'layer4': torch.stack(mem4_rec, dim=0)
        }

        return spk_rec, mem_rec

