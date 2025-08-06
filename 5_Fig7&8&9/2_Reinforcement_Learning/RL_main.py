import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

try:
    from encoding import mixed_oscillator_encode
    from nns import Net, ANNNet

    SNN_AVAILABLE = True
    ANN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Import error: {e}")
    try:
        from nns import ANNNet

        ANN_AVAILABLE = True
        SNN_AVAILABLE = False
    except ImportError:
        print("Neither ANN nor SNN modules available")
        ANN_AVAILABLE = False
        SNN_AVAILABLE = False

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 1. UnifiedRLConfig ---
class UnifiedRLConfig:
    """Unified Reinforcement Learning configuration to ensure identical base settings for both ANN and SNN."""

    # Environment Parameters
    env_name = "CartPole-v1"
    state_dim = 4  # CartPole state dimension
    action_dim = 2  # CartPole action dimension

    # Network Parameters - Compatible with nns.py
    num_inputs = 4  # Input dimension
    num_hidden = 32  # Hidden layer dimension (unified setting)
    num_outputs = 2  # Output dimension
    n_components = 4  # Number of components for encoding

    # SNN Specific Parameters
    beta = 0.95  # LIF neuron parameter
    num_steps = 5  # Number of time steps

    # Dynamical Encoding Parameters
    tmax_encoding = 8.0
    osc_alpha = 2.0
    osc_beta_osc = 0.1
    osc_gamma = 0.1
    osc_omega = 1.0
    osc_drive = 0.0

    # Reinforcement Learning Parameters
    learning_rate = 0.001
    gamma = 0.99
    num_episodes_train = 800
    max_steps_per_episode = 500

    # Experiment Parameters
    random_state = 42

    def get_unified_config(self):
        """Returns itself for network initialization"""
        return self

# --- 2. ANN Policy Wrapper（ANNNet） ---
class ANNPolicyWrapper(nn.Module):
    def __init__(self, config: UnifiedRLConfig):
        super().__init__()
        self.config = config
        # Use ANNNet, but adapt it for RL output
        self.ann_core = ANNNet(config, encoding='default')

        # Check and adjust output layer dimension
        if self.ann_core.fc4.out_features != config.action_dim:
            print(f"Adjusting ANN output layer from {self.ann_core.fc4.out_features} to {config.action_dim}")
            self.ann_core.fc4 = nn.Linear(config.num_hidden, config.action_dim)

    def forward(self, state_tensor):
        """Forward pass, returns action probabilities"""
        # ANNNet expected input format
        logits = self.ann_core(state_tensor)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs


# --- 3. SNN Policy Wrapper（Net） ---
class SNNPolicyWrapper(nn.Module):
    def __init__(self, config: UnifiedRLConfig, dynamic_params: Dict):
        super().__init__()
        self.config = config
        self.dynamic_params = dynamic_params

        # Use the new Net class
        self.snn_core = Net(config, encoding='mixed_oscillator')

        # Check and adjust output layer dimension
        if self.snn_core.fc_out.out_features != config.action_dim:
            print(f"Adjusting SNN output layer from {self.snn_core.fc_out.out_features} to {config.action_dim}")
            self.snn_core.fc_out = nn.Linear(config.num_hidden, config.action_dim)

    def encode_state(self, state_np):
        """State encoding - unchanged"""
        state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
        osc_params = {
            'alpha': self.config.osc_alpha,
            'beta': self.config.osc_beta_osc,
            'delta': self.dynamic_params['delta'],
            'gamma': self.config.osc_gamma,
            'omega': self.config.osc_omega,
            'drive': self.config.osc_drive
        }

        encoded_state = mixed_oscillator_encode(
            state_tensor,
            num_steps=self.config.num_steps,
            tmax=self.config.tmax_encoding,
            params=osc_params
        )
        return encoded_state

    def forward(self, encoded_state):
        """Forward pass, return action probabilities and spike records"""
        spk_rec, mem_rec = self.snn_core(encoded_state)

        # === Key change: Adapt to the new Net output format ===
        if hasattr(self.config, 'use_temporal_integration') and self.config.use_temporal_integration:
            # If temporal integration is used, the new Net has already summed internally
            # spk_rec['layer4'] now stores the output of each time step
            action_logits = spk_rec['layer4'].sum(dim=0)  # [batch_size, action_dim]
        else:
            # If temporal integration is not used, use the last time step
            action_logits = spk_rec['layer4'][-1]  # [batch_size, action_dim]

        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs, spk_rec


# --- 4. Unified REINFORCE Agent ---
class UnifiedReinforceAgent:
    def __init__(self, config: UnifiedRLConfig, agent_type: str, dynamic_params: Optional[Dict] = None):
        self.config = config
        self.agent_type = agent_type
        self.dynamic_params = dynamic_params or {}

        # Initialize policy network
        if agent_type == "ANN" and ANN_AVAILABLE:
            self.policy_net = ANNPolicyWrapper(config).to(device)
        elif agent_type == "SNN" and SNN_AVAILABLE:
            self.policy_net = SNNPolicyWrapper(config, dynamic_params).to(device)
        else:
            raise ValueError(f"Agent type {agent_type} not available or supported")

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.rewards_buffer = []
        self.log_probs_buffer = []
        self.spikes_this_episode = 0

    def select_action(self, state_np):
        """Selects an action"""
        self.policy_net.train()

        if self.agent_type == "ANN":
            state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
            action_probs = self.policy_net(state_tensor)
            spk_rec = None
        else:  # SNN
            encoded_state = self.policy_net.encode_state(state_np)
            action_probs, spk_rec = self.policy_net(encoded_state)

        m = Categorical(action_probs)
        action = m.sample()
        self.log_probs_buffer.append(m.log_prob(action))

        # Calculate spike count (for SNN only)
        if self.agent_type == "SNN" and spk_rec:
            current_step_spikes = sum(spk_rec[layer].sum().item() for layer in spk_rec)
            self.spikes_this_episode += current_step_spikes

        return action.item()

    def store_reward(self, reward):
        """Stores the reward"""
        self.rewards_buffer.append(reward)

    def update_policy(self):
        """Updates the policy"""
        if not self.log_probs_buffer:
            self.clear_buffers()
            return 0.0, 0

        # Calculate discounted returns
        discounted_returns = []
        cumulative_return = 0
        for r in reversed(self.rewards_buffer):
            cumulative_return = r + self.config.gamma * cumulative_return
            discounted_returns.insert(0, cumulative_return)

        discounted_returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32).to(device)

        # Calculate policy loss
        policy_loss_terms = []
        for log_prob, G_t in zip(self.log_probs_buffer, discounted_returns_tensor):
            policy_loss_terms.append(-log_prob * G_t)

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss = torch.stack(policy_loss_terms).sum()
        total_loss.backward()
        self.optimizer.step()

        loss_value = total_loss.item()
        spikes_for_episode = self.spikes_this_episode
        self.clear_buffers()
        return loss_value, spikes_for_episode

    def clear_buffers(self):
        """Clears the buffers"""
        self.rewards_buffer = []
        self.log_probs_buffer = []
        self.spikes_this_episode = 0


# --- 5. Single Agent Train ---
def train_unified_rl_agent(config: UnifiedRLConfig, agent_type: str, mode_name: str,
                           dynamic_params: Optional[Dict], run_id: int, base_results_dir: str):
    """Trains a single agent"""

    env = gym.make(config.env_name)
    agent = UnifiedReinforceAgent(config, agent_type, dynamic_params)

    all_episode_rewards = []
    all_episode_losses = []
    all_episode_spikes = []

    # Create results directory
    current_run_mode_dir = os.path.join(base_results_dir, f"run_{run_id}", mode_name)
    os.makedirs(current_run_mode_dir, exist_ok=True)

    print(f"\n--- Starting {agent_type} RL training: Run {run_id} - {mode_name} ---")
    if dynamic_params:
        print(f"    Dynamic params: {dynamic_params}")

    training_start_time = time.time()
    solved_at_episode = -1

    for i_episode in range(1, config.num_episodes_train + 1):
        state, _ = env.reset()
        current_episode_reward_sum = 0
        agent.clear_buffers()

        for t_step in range(config.max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward)
            current_episode_reward_sum += reward
            state = next_state

            if done:
                break

        loss_this_episode, spikes_this_episode = agent.update_policy()
        all_episode_rewards.append(current_episode_reward_sum)
        all_episode_losses.append(loss_this_episode)
        all_episode_spikes.append(spikes_this_episode)

        # Print progress
        if i_episode % 50 == 0 or i_episode == config.num_episodes_train:
            window = min(50, len(all_episode_rewards))
            avg_reward = np.mean(all_episode_rewards[-window:])
            avg_loss = np.mean(all_episode_losses[-window:]) if all_episode_losses else 0.0
            avg_spikes = np.mean(all_episode_spikes[-window:]) if all_episode_spikes else 0.0
            print(f"Run {run_id} | {mode_name} | Episode {i_episode}/{config.num_episodes_train} | "
                  f"Avg Reward: {avg_reward:.2f} | Avg Loss: {avg_loss:.3f} | Avg Spikes: {avg_spikes:.0f}")

        # Check if solved
        if solved_at_episode == -1 and len(all_episode_rewards) >= 100:
            if np.mean(all_episode_rewards[-100:]) >= 475.0:
                print(f"Run {run_id} | {mode_name}: Environment solved in {i_episode} episodes!")
                solved_at_episode = i_episode

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print(f"--- Run {run_id} | {mode_name} training completed. Time: {total_training_time:.2f}s ---")

    # Save training log
    results_log_df = pd.DataFrame({
        'episode': range(1, len(all_episode_rewards) + 1),
        'reward': all_episode_rewards,
        'loss': all_episode_losses,
        'spikes_per_episode': all_episode_spikes
    })
    results_log_df.to_csv(
        os.path.join(current_run_mode_dir, f"training_log_{mode_name}_run_{run_id}.csv"),
        index=False
    )

    # Generate learning curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(results_log_df['episode'], results_log_df['reward'].rolling(window=50).mean(),
             color='blue', label='Avg Reward (Rolling 50)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'{mode_name} - Reward Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if agent_type == "SNN":
        plt.subplot(1, 2, 2)
        plt.plot(results_log_df['episode'], results_log_df['spikes_per_episode'].rolling(window=50).mean(),
                 color='red', label='Avg Spikes (Rolling 50)')
        plt.xlabel('Episode')
        plt.ylabel('Average Spikes')
        plt.title(f'{mode_name} - Spikes per Episode')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.suptitle(f'{mode_name} Learning Curves (Run {run_id})')
    plt.tight_layout()
    plt.savefig(os.path.join(current_run_mode_dir, f"learning_curves_{mode_name}_run_{run_id}.png"), dpi=300)
    plt.close()

    # Save model
    model_save_path = os.path.join(current_run_mode_dir, f"{mode_name}_model_run_{run_id}.pth")
    torch.save(agent.policy_net.state_dict(), model_save_path)

    env.close()

    # Collect statistics
    avg_reward_last_100 = np.mean(all_episode_rewards[-100:]) if len(all_episode_rewards) >= 100 else np.mean(
        all_episode_rewards)
    avg_spikes_last_100 = np.mean(all_episode_spikes[-100:]) if len(all_episode_spikes) >= 100 else np.mean(
        all_episode_spikes)
    max_reward_achieved = np.max(all_episode_rewards) if all_episode_rewards else 0

    summary_stats = {
        'run_id': run_id,
        'agent_type': agent_type,
        'mode_name': mode_name,
        'delta_value': dynamic_params.get('delta') if dynamic_params else None,
        'avg_reward_last_100_ep': avg_reward_last_100,
        'max_reward_achieved': max_reward_achieved,
        'avg_spikes_last_100_ep': avg_spikes_last_100,
        'total_training_time_sec': total_training_time,
        'solved_at_episode': solved_at_episode,
        'total_episodes_trained': len(all_episode_rewards)
    }

    return results_log_df, summary_stats

# --- 6. Single Run ---
def run_single_rl_repetition(repetition_idx: int, config: UnifiedRLConfig, base_results_dir: str):
    """Run a single repetition of the experiment to test ANN and SNN agents"""

    print(f"\n{'#' * 15} Starting RL Repetition {repetition_idx} {'#' * 15}")

    repetition_summary_list = []

    # 1. Train ANN baseline
    if ANN_AVAILABLE:
        print(f"\n{'=' * 20} ANN Baseline {'=' * 20}")
        _, ann_summary = train_unified_rl_agent(
            config, "ANN", "ANN_Baseline", None, repetition_idx, base_results_dir
        )
        repetition_summary_list.append(ann_summary)
    else:
        print("ANN experiments skipped - ANNNet not available")

    # 2. Train SNN in various modes
    if SNN_AVAILABLE:
        snn_modes = {
            "SNN_Dissipative": {'delta': 10.0},
            "SNN_Expansive": {'delta': -1.5},
            "SNN_Tough": {'delta': 2.5}
        }

        for mode_name, dynamic_params in snn_modes.items():
            print(f"\n{'=' * 20} {mode_name} (δ={dynamic_params['delta']}) {'=' * 20}")
            _, snn_summary = train_unified_rl_agent(
                config, "SNN", mode_name, dynamic_params, repetition_idx, base_results_dir
            )
            repetition_summary_list.append(snn_summary)
    else:
        print("SNN experiments skipped - modules not available")

    # Save summary for this repetition
    repetition_summary_df = pd.DataFrame(repetition_summary_list)
    summary_csv_path = os.path.join(base_results_dir, f"repetition_{repetition_idx}",
                                    f"unified_rl_summary_repetition_{repetition_idx}.csv")
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    repetition_summary_df.to_csv(summary_csv_path, index=False)

    print(f"\n--- Repetition {repetition_idx} Complete ---")
    print(repetition_summary_df[['agent_type', 'mode_name', 'avg_reward_last_100_ep', 'avg_spikes_last_100_ep']])

    return repetition_summary_df


# --- 7. Main Experiment ---
def run_unified_rl_experiment_suite(num_runs: int = 10):
    """Run the unified reinforcement learning experiment suite."""

    config = UnifiedRLConfig()

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_results_dir = f"unified_rl_results_{timestamp}"
    os.makedirs(overall_results_dir, exist_ok=True)

    print(f"Starting unified RL experiment with {num_runs} repetitions")
    print(f"ANN modules available: {ANN_AVAILABLE}")
    print(f"SNN modules available: {SNN_AVAILABLE}")
    print(f"Results will be saved to: {overall_results_dir}")

    if not ANN_AVAILABLE and not SNN_AVAILABLE:
        print("Neither ANN nor SNN modules are available. Cannot run experiments.")
        return

    all_repetitions_summaries = []
    start_time_total = time.time()

    # Run multiple repetitions
    for i in range(num_runs):
        try:
            repetition_df = run_single_rl_repetition(i, config, overall_results_dir)
            if repetition_df is not None and not repetition_df.empty:
                all_repetitions_summaries.append(repetition_df)
            else:
                print(f"Warning: Repetition {i} returned empty results")
        except Exception as e:
            print(f"Error in repetition {i}: {e}")
            continue

    if not all_repetitions_summaries:
        print("No valid data collected from any repetition")
        return

    # Combine all results
    full_summary_df = pd.concat(all_repetitions_summaries, ignore_index=True)
    full_summary_path = os.path.join(overall_results_dir, "all_unified_rl_repetitions_summary.csv")
    full_summary_df.to_csv(full_summary_path, index=False)

    print(f"\n--- All {num_runs} RL Repetitions Complete ---")
    print(f"Combined summary saved to {full_summary_path}")

    # Compute statistical summary
    print("\n--- Statistical Summary ---")
    if 'agent_type' in full_summary_df.columns and 'avg_reward_last_100_ep' in full_summary_df.columns:
        # Group by agent type and mode
        stats_summary = full_summary_df.groupby(['agent_type', 'mode_name']).agg({
            'avg_reward_last_100_ep': ['mean', 'std', 'count'],
            'max_reward_achieved': ['mean', 'std'],
            'avg_spikes_last_100_ep': ['mean', 'std'],
            'solved_at_episode': lambda x: (x != -1).sum()
        }).round(2)

        print("\nPerformance Statistics by Agent Type and Mode:")
        print(stats_summary)

        # Save statistical results
        stats_path = os.path.join(overall_results_dir, "rl_statistical_summary.csv")
        stats_summary.to_csv(stats_path)
        print(f"Statistical summary saved to {stats_path}")

    # Generate comparison plots
    try:
        plt.figure(figsize=(15, 10))

        # Subplot 1: Average Reward Comparison
        plt.subplot(2, 2, 1)
        if 'agent_type' in full_summary_df.columns:
            agent_types = full_summary_df['agent_type'].unique()
            plot_data = []
            plot_labels = []

            for agent_type in agent_types:
                agent_data = full_summary_df[full_summary_df['agent_type'] == agent_type]
                if agent_type == 'ANN':
                    plot_data.append(agent_data['avg_reward_last_100_ep'].dropna().values)
                    plot_labels.append('ANN')
                else:
                    for mode in agent_data['mode_name'].unique():
                        mode_data = agent_data[agent_data['mode_name'] == mode][
                            'avg_reward_last_100_ep'].dropna().values
                        if len(mode_data) > 0:
                            plot_data.append(mode_data)
                            plot_labels.append(mode.replace('SNN_', ''))

            if plot_data:
                plt.boxplot(plot_data, labels=plot_labels)
                plt.title('Average Reward (Last 100 Episodes)')
                plt.ylabel('Reward')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)

        # Subplot 2: Success Rate Comparison
        plt.subplot(2, 2, 2)
        if 'agent_type' in full_summary_df.columns:
            solve_rates = full_summary_df.groupby(['agent_type', 'mode_name']).apply(
                lambda x: (x['solved_at_episode'] != -1).sum() / len(x) * 100
            ).reset_index(name='solve_rate')

            x_pos = range(len(solve_rates))
            plt.bar(x_pos, solve_rates['solve_rate'])
            plt.xlabel('Configuration')
            plt.ylabel('Success Rate (%)')
            plt.title('Environment Solving Success Rate')
            plt.xticks(x_pos, [f"{row['agent_type']}-{row['mode_name'].replace('SNN_', '').replace('ANN_', '')}"
                               for _, row in solve_rates.iterrows()], rotation=45)
            plt.grid(True, alpha=0.3)

        # Subplot 3: Energy Efficiency Comparison (SNN only)
        plt.subplot(2, 2, 3)
        snn_data = full_summary_df[full_summary_df['agent_type'] == 'SNN']
        if not snn_data.empty:
            spike_data = []
            spike_labels = []
            for mode in snn_data['mode_name'].unique():
                mode_data = snn_data[snn_data['mode_name'] == mode]['avg_spikes_last_100_ep'].dropna().values
                if len(mode_data) > 0:
                    spike_data.append(mode_data)
                    spike_labels.append(mode.replace('SNN_', ''))

            if spike_data:
                plt.boxplot(spike_data, labels=spike_labels)
                plt.title('Average Spikes per Episode (SNN only)')
                plt.ylabel('Spikes')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)

        # Subplot 4: Training Time Comparison
        plt.subplot(2, 2, 4)
        if 'total_training_time_sec' in full_summary_df.columns:
            time_data = []
            time_labels = []
            for agent_type in full_summary_df['agent_type'].unique():
                agent_data = full_summary_df[full_summary_df['agent_type'] == agent_type]
                if agent_type == 'ANN':
                    time_data.append(agent_data['total_training_time_sec'].dropna().values)
                    time_labels.append('ANN')
                else:
                    for mode in agent_data['mode_name'].unique():
                        mode_data = agent_data[agent_data['mode_name'] == mode][
                            'total_training_time_sec'].dropna().values
                        if len(mode_data) > 0:
                            time_data.append(mode_data)
                            time_labels.append(mode.replace('SNN_', ''))

            if time_data:
                plt.boxplot(time_data, labels=time_labels)
                plt.title('Training Time Distribution')
                plt.ylabel('Time (seconds)')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)

        plt.suptitle(f'Unified RL Experiment Results ({num_runs} runs)', fontsize=16)
        plt.tight_layout()

        plot_path = os.path.join(overall_results_dir, "unified_rl_performance_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison plot saved to {plot_path}")
        plt.close()

    except Exception as e:
        print(f"Error generating plots: {e}")

    end_time_total = time.time()
    print(f"\nTotal execution time: {(end_time_total - start_time_total) / 60:.2f} minutes")


if __name__ == "__main__":
    num_repetitions_to_run = 30

    # For quick test
    # num_repetitions_to_run = 3

    run_unified_rl_experiment_suite(num_runs=num_repetitions_to_run)