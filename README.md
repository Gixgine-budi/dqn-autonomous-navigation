# Deep Q-Network for Autonomous Navigation

Implementation code for the paper:

**"Tensor Calculus in Autonomous Navigation: A Linear Algebraic Deconstruction of Deep Q-Learning in Simulated Environments"**

## Overview

This project demonstrates how tensors and linear algebra operations are used in deep reinforcement learning for autonomous navigation. The code implements a Deep Q-Network (DQN) agent that learns to navigate through simulated environments.

Two implementations are provided:
1. **CarRacing-v3**: Full CNN-based DQN for visual autonomous driving
2. **CartPole-v1**: Simplified MLP-based DQN for quick demonstrations

## Project Structure

```
dqn-autonomous-navigation/
├── dqn_carracing.py    # Full DQN for CarRacing (visual input)
├── dqn_training.py     # Simplified DQN for CartPole
├── requirements.txt    # Python dependencies
├── results/
│   └── training_stats.json
└── checkpoints/
```

## Files Description

### `dqn_carracing.py`

Complete DQN implementation for the CarRacing-v3 environment featuring:

| Component        | Description                                           |
|------------------|-------------------------------------------------------|
| Preprocessing    | Grayscale conversion, cropping (84×84), normalization |
| Frame Stacking   | 4 consecutive frames for temporal information         |
| CNN Architecture | 2 conv layers + 2 FC layers (677,429 parameters)      |
| Replay Buffer    | 100,000 transitions for experience replay             |
| Target Network   | Stabilizes training with periodic updates             |

**CNN Architecture:**
```
Input (4×84×84) → Conv1(16, 8×8, s=4) → Conv2(32, 4×4, s=2) → FC(256) → Output(5)
```

**Discrete Actions:**

| Index | Action     |
|-------|------------|
| 0     | Do nothing |
| 1     | Turn left  |
| 2     | Turn right |
| 3     | Accelerate |
| 4     | Brake      |

### `dqn_training.py`

Simplified DQN for CartPole-v1. Demonstrates core concepts without visual dependencies:
- Fully connected neural network (17,410 parameters)
- Experience replay and target network
- Epsilon-greedy exploration with decay

## Installation

### Requirements

- Python 3.10 - 3.12 (recommended)
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- Gymnasium >= 1.0.0

### Setup

```bash
git clone https://https://github.com/Gixgine-budi/dqn-autonomous-navigation.git
cd dqn-autonomous-navigation
pip install -r requirements.txt
```

### Note on Python Version

- **Python 3.14**: CarRacing dependencies (pygame, box2d) may not be available
- **Python 3.10-3.12**: Fully compatible with all features

## Usage

### Training CarRacing (Visual)

```bash
# Train with visual rendering
python dqn_carracing.py --train --episodes 100 --render

# Train without rendering (faster)
python dqn_carracing.py --train --episodes 200
```

**Estimated training time:**
- With rendering: ~45 sec/episode
- Without rendering: ~15 sec/episode

### Training CartPole (Quick Demo)

```bash
python dqn_training.py
```

Runs 500 episodes in ~2-3 minutes.

### Evaluate a Trained Model

```bash
python dqn_carracing.py --eval checkpoints/dqn_final.pt --render
```

## Training Results

### CartPole-v1

| Metric             | Value        |
|--------------------|--------------|
| Episodes           | 500          |
| Training Time      | ~2.5 minutes |
| Final Avg Reward   | 96.6         |
| Total Steps        | 33,793       |
| Network Parameters | 17,410       |

### CarRacing-v3

| Metric               | Value       |
|----------------------|-------------|
| Network Parameters   | 677,429     |
| Input Shape          | (4, 84, 84) |
| Output Actions       | 5           |
| Recommended Episodes | 200+        |

## References

1. Mnih, V., et al. "Human-level control through deep reinforcement learning." *Nature* 518.7540 (2015): 529-533.
2. Gymnasium Documentation. "CarRacing Environment". https://gymnasium.farama.org/environments/box2d/car_racing/
3. Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*. MIT Press, 2016.

## Author

Muhammad Iqbal Raihan
