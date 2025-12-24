"""
Deep Q-Network Training Script
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


class DQN(nn.Module):
    """
    Deep Q-Network dengan arsitektur fully connected.
    
    Arsitektur untuk CartPole:
    - Input: State vector (4 dimensi)
    - FC1: 128 neurons + ReLU
    - FC2: 128 neurons + ReLU  
    - Output: 2 nilai Q (left/right)
    
    Operasi aljabar linear:
    h_1 = ReLU(W_1 · x + b_1)
    h_2 = ReLU(W_2 · h_1 + b_2)
    q = W_3 · h_2 + b_3
    """
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inisialisasi bobot menggunakan Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass dengan ReLU activation."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ReplayBuffer:
    """Experience Replay Buffer."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent dengan epsilon-greedy policy."""
    
    def __init__(self,
                 state_dim=4,
                 action_dim=2,
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 batch_size=64,
                 buffer_size=10000,
                 target_update_freq=10):
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Epsilon scheduling
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Adam optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Stats
        self.training_step = 0
        self.losses = []
        self.episode_count = 0
        
        print(f"DQN Agent initialized with {self.policy_net.count_parameters():,} parameters")
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(1).item()
    
    def train_step(self):
        """Single training step."""
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
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss (MSE)
        loss = F.mse_loss(current_q, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_step += 1
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def update_target_network(self):
        """Update target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train(num_episodes=500, max_steps=500):
    """
    Training loop.
    
    CartPole-v1 dianggap "solved" jika average reward >= 475 over 100 episodes.
    """
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize
    env = gym.make("CartPole-v1")
    agent = DQNAgent(
        state_dim=4,
        action_dim=2,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_size=10000,
        target_update_freq=10
    )
    
    # Stats tracking
    episode_rewards = []
    episode_lengths = []
    epsilon_history = []
    
    print(f"\n{'='*60}")
    print(f"Starting DQN Training")
    print(f"Environment: CartPole-v1")
    print(f"Episodes: {num_episodes}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0
        
        for step in range(max_steps):
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.buffer.push(state, action, reward, next_state, done)
            
            # Train
            agent.train_step()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Post-episode updates
        agent.decay_epsilon()
        agent.episode_count += 1
        
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        epsilon_history.append(agent.epsilon)
        
        # Logging
        avg_reward = np.mean(episode_rewards[-100:])
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1:4d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Avg(100): {avg_reward:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {np.mean(agent.losses[-100:]) if agent.losses else 0:.4f}")
        
        # Save checkpoint every 100 episodes
        if (episode + 1) % 100 == 0:
            save_stats(episode_rewards, agent.losses, epsilon_history)
    
    env.close()
    
    # Final save
    save_stats(episode_rewards, agent.losses, epsilon_history)
    
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {elapsed}")
    print(f"Final average reward (last 100): {np.mean(episode_rewards[-100:]):.1f}")
    print(f"Total training steps: {agent.training_step}")
    print(f"{'='*60}")
    
    return episode_rewards, agent.losses, epsilon_history


def save_stats(rewards, losses, epsilons):
    """Save training statistics to JSON."""
    stats = {
        'episode_rewards': rewards,
        'losses': losses[-10000:] if len(losses) > 10000 else losses,  # Limit size
        'epsilon_history': epsilons,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('results/training_stats.json', 'w') as f:
        json.dump(stats, f)
    
    print(f"Stats saved to results/training_stats.json")


if __name__ == "__main__":
    print("="*60)
    print("DQN Training for Paper Demonstration")
    print("="*60)
    
    # Run training
    rewards, losses, epsilons = train(num_episodes=500, max_steps=500)
    
    print("\nTraining complete! Run generate_real_figures.py to create figures.")

