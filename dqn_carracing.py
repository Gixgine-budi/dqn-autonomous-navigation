"""
Deep Q-Network for CarRacing-v0
Implementasi untuk makalah Aljabar Linear dan Geometri
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import json
from datetime import datetime

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Preprocessing:
    """
    Modul preprocessing untuk mengubah frame game menjadi tensor input.
    
    Operasi:
    1. Grayscale conversion: RGB -> Luminansi
       I_gray = 0.299*R + 0.587*G + 0.114*B
    
    2. Cropping: Menghilangkan dashboard di bagian bawah
       I_crop = I_gray[0:84, 6:90]
    
    3. Normalization: Skala piksel ke [0, 1]
       I_norm = I_crop / 255.0
    """
    
    @staticmethod
    def to_grayscale(frame):
        """Konversi RGB ke grayscale menggunakan formula luminansi."""
        # Formula: 0.299*R + 0.587*G + 0.114*B
        return np.dot(frame[..., :3], [0.299, 0.587, 0.114])
    
    @staticmethod
    def crop_frame(frame):
        """Crop frame untuk menghilangkan dashboard."""
        # Ambil area 84x84 dari tengah atas
        return frame[0:84, 6:90]
    
    @staticmethod
    def normalize(frame):
        """Normalisasi nilai piksel ke range [0, 1]."""
        return frame / 255.0
    
    @staticmethod
    def process(frame):
        """Pipeline lengkap preprocessing."""
        gray = Preprocessing.to_grayscale(frame)
        cropped = Preprocessing.crop_frame(gray)
        normalized = Preprocessing.normalize(cropped)
        return normalized.astype(np.float32)


class FrameStacker:
    """
    Menumpuk 4 frame berurutan untuk memberikan informasi temporal.
    
    State tensor S_t = [I_{t-3}, I_{t-2}, I_{t-1}, I_t] ∈ R^{4x84x84}
    
    Perbedaan antar frame mengandung informasi:
    - Kecepatan mobil
    - Arah gerakan
    - Sudut steering
    """
    
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
    
    def reset(self, initial_frame):
        """Reset stack dengan frame awal yang diulang."""
        processed = Preprocessing.process(initial_frame)
        for _ in range(self.stack_size):
            self.frames.append(processed)
        return self.get_state()
    
    def add_frame(self, frame):
        """Tambah frame baru dan return state terbaru."""
        processed = Preprocessing.process(frame)
        self.frames.append(processed)
        return self.get_state()
    
    def get_state(self):
        """Return stacked frames sebagai tensor."""
        return np.array(self.frames)


class DQN(nn.Module):
    """
    Deep Q-Network dengan arsitektur CNN.
    
    Arsitektur:
    - Input: Tensor (4, 84, 84)
    - Conv1: 16 filters, kernel 8×8, stride 4 → (16, 20, 20)
    - Conv2: 32 filters, kernel 4×4, stride 2 → (32, 9, 9)
    - Flatten: 32×9×9 = 2592
    - FC: 256 neurons
    - Output: 5 nilai Q (satu per aksi)
    
    Total parameter: ~677,429
    """
    
    def __init__(self, input_channels=4, num_actions=5):
        super(DQN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        
        # Calculate size after convolutions
        # Input: 84×84 -> Conv1: 20×20 -> Conv2: 9×9
        self.fc_input_size = 32 * 9 * 9  # 2592
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, num_actions)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inisialisasi bobot menggunakan Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass: S → Q-values
        
        Operasi per layer:
        Z_1 = W_1 * S + b_1, H_1 = ReLU(Z_1)
        Z_2 = W_2 * H_1 + b_2, H_2 = ReLU(Z_2)
        h_flat = flatten(H_2)
        z_3 = W_3 · h_flat + b_3, h_3 = ReLU(z_3)
        q = W_4 · h_3 + b_4
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def count_parameters(self):
        """Hitung total parameter yang dapat dilatih."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ReplayBuffer:
    """
    Experience Replay Buffer untuk menyimpan transisi.
    
    Menyimpan tuple (s, a, r, s', done) untuk training batch.
    Membantu memutus korelasi temporal antara sampel.
    """
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Simpan transisi ke buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Ambil batch random dari buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Agen DQN dengan epsilon-greedy policy dan target network.
    
    Komponen:
    - Policy network: Untuk memilih aksi
    - Target network: Untuk stabilitas training
    - Replay buffer: Untuk experience replay
    - Epsilon scheduler: Untuk eksplorasi
    """
    
    # Discrete action mapping
    ACTIONS = [
        np.array([0.0, 0.0, 0.0]),    # 0: Do nothing
        np.array([-1.0, 0.0, 0.0]),   # 1: Turn left
        np.array([1.0, 0.0, 0.0]),    # 2: Turn right
        np.array([0.0, 1.0, 0.0]),    # 3: Gas
        np.array([0.0, 0.0, 0.8]),    # 4: Brake
    ]
    
    def __init__(self, 
                 learning_rate=0.0001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=100000,
                 batch_size=32,
                 buffer_size=100000,
                 target_update_freq=1000):
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Epsilon scheduling
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        
        # Networks
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer (Adam)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.training_step = 0
        self.losses = []
        
        print(f"DQN initialized with {self.policy_net.count_parameters():,} parameters")
    
    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection.
        
        a* = argmax_a Q(s, a)  dengan probabilitas 1-ε
        a* = random action     dengan probabilitas ε
        """
        if training and random.random() < self.epsilon:
            return random.randrange(len(self.ACTIONS))
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(1).item()
    
    def get_action_array(self, action_idx):
        """Convert action index ke array untuk environment."""
        return self.ACTIONS[action_idx]
    
    def train_step(self):
        """
        Satu langkah training dengan batch dari replay buffer.
        
        Loss function (temporal difference):
        L(θ) = E[(r + γ max_a' Q(s', a'; θ^-) - Q(s, a; θ))²]
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Current Q values: Q(s, a; θ)
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values: r + γ max_a' Q(s', a'; θ^-)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss (MSE)
        loss = F.mse_loss(current_q, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping untuk stabilitas
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def save(self, path):
        """Simpan model checkpoint."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'losses': self.losses,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.losses = checkpoint.get('losses', [])
        print(f"Model loaded from {path}")


def train(num_episodes=200, save_freq=50, render=False):
    """
    Training loop utama.
    
    Args:
        num_episodes: Jumlah episode training
        save_freq: Frekuensi penyimpanan checkpoint
        render: Apakah menampilkan visualisasi
    """
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize environment
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array" if not render else "human")
    
    # Initialize agent
    agent = DQNAgent(
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=50000,  # Faster decay for shorter training
        batch_size=32,
        buffer_size=50000,
        target_update_freq=500
    )
    
    # Initialize frame stacker
    stacker = FrameStacker(stack_size=4)
    
    # Training stats
    episode_rewards = []
    episode_lengths = []
    
    print(f"\n{'='*60}")
    print(f"Starting training for {num_episodes} episodes")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    for episode in range(num_episodes):
        # Reset environment
        obs, info = env.reset(seed=SEED + episode)
        state = stacker.reset(obs)
        
        episode_reward = 0
        episode_length = 0
        episode_loss = []
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select and execute action
            action_idx = agent.select_action(state)
            action = agent.get_action_array(action_idx)
            
            # For discrete environment, use action index directly
            next_obs, reward, done, truncated, info = env.step(action_idx)
            
            # Get next state
            next_state = stacker.add_frame(next_obs)
            
            # Store transition
            agent.buffer.push(state, action_idx, reward, next_state, done or truncated)
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Early stopping untuk episode yang terlalu lama
            if episode_length > 1000:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Logging
        avg_reward = np.mean(episode_rewards[-50:])
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        
        print(f"Episode {episode+1:4d} | "
              f"Reward: {episode_reward:7.1f} | "
              f"Avg(50): {avg_reward:7.1f} | "
              f"Length: {episode_length:4d} | "
              f"Loss: {avg_loss:.4f} | "
              f"Epsilon: {agent.epsilon:.3f}")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(f"checkpoints/dqn_episode_{episode+1}.pt")
            
            # Save training stats
            stats = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'losses': agent.losses,
                'epsilon_history': [agent.epsilon],
            }
            with open('results/training_stats.json', 'w') as f:
                json.dump(stats, f)
    
    env.close()
    
    # Final save
    agent.save("checkpoints/dqn_final.pt")
    
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {elapsed}")
    print(f"Final average reward (last 50): {np.mean(episode_rewards[-50:]):.1f}")
    print(f"{'='*60}")
    
    return episode_rewards, agent.losses


def evaluate(model_path, num_episodes=5, render=True):
    """Evaluasi model yang sudah dilatih."""
    env = gym.make("CarRacing-v3", continuous=False, 
                   render_mode="human" if render else "rgb_array")
    
    agent = DQNAgent()
    agent.load(model_path)
    
    stacker = FrameStacker()
    rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = stacker.reset(obs)
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action_idx = agent.select_action(state, training=False)
            next_obs, reward, done, truncated, _ = env.step(action_idx)
            state = stacker.add_frame(next_obs)
            total_reward += reward
        
        rewards.append(total_reward)
        print(f"Episode {episode+1}: Reward = {total_reward:.1f}")
    
    env.close()
    print(f"\nAverage reward: {np.mean(rewards):.1f}")
    return rewards


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DQN CarRacing Training")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", type=str, help="Evaluate model from path")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render environment")
    
    args = parser.parse_args()
    
    if args.train:
        train(num_episodes=args.episodes, render=args.render)
    elif args.eval:
        evaluate(args.eval, render=args.render)
    else:
        print("Usage: python dqn_carracing.py --train --episodes 200")
        print("       python dqn_carracing.py --eval checkpoints/dqn_final.pt")

