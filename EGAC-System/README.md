# EGAC: Entropy-Guided Actor-Critic with Meta-Learning Dynamics

EGAC (Entropy-Guided Actor-Critic) is a modular reinforcement learning system designed for adaptive, stable, and exploratory learning in continuous and discrete control environments. This repo provides a TensorFlow-based implementation with built-in prioritized experience replay, meta-gradient dynamics, reward normalization, dynamic entropy control, GAE advantage estimation, and Polyak-averaged critic targets.

## Features

- **Entropy-Guided Exploration** with adaptive entropy coefficients
- **GAE (Generalized Advantage Estimation)** for smoother advantage computation
- **Prioritized Experience Replay (PER)** with dynamic priority updates
- **Exponential Moving Average Loss Tracking** for training stability
- **Reward Normalization** and reward-variance tracking
- **Polyak Averaging** for target critic network updates
- **Modular Actor-Critic Architecture** via TensorFlow Agents
- **Evaluation Mode** with deterministic policy and performance metrics

---

## File Structure

```
EGAC-System/
├── egac.py          # Main module containing the EGACAgent class
├── tests/
     |--- System-Evaluation      # Example training script using the module
     |----Compare-Algorithms.py
└── README.md
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/EGAC.git
cd EGAC
```

### 2. Install Dependencies

```bash
pip install tensorflow tf-agents matplotlib gym numpy
```

### 3. Training the Agent

```python
from egac import EGACAgent
from tf_agents.environments import suite_gym

env = suite_gym.load('LunarLander-v2')
agent = EGACAgent(env)

# Train for 20,000 iterations
agent.train(num_iterations=20000)
```

### 4. Evaluating the Agent

```python
agent.evaluate(env, num_episodes=10, render=False)
```

---

## Code Example

```python
from tf_agents.environments import suite_gym
from egac import EGACAgent

env = suite_gym.load('LunarLander-v2')
agent = EGACAgent(env)

# Train
agent.train(num_iterations=5000)

# Evaluate
agent.evaluate(env, num_episodes=5)
```

---

## How It Works

EGAC is built upon classic Actor-Critic principles with multiple innovations:

- **Entropy Adjustment**: If the variance in episodic reward is too low, entropy coefficient is increased to encourage exploration.
- **Loss EMA**: If policy loss converges too fast, learning rates decay automatically.
- **PER**: Experiences with higher advantage values are prioritized.
- **Exploration Bonus**: Introduced based on deviation from average rewards.

---

## License

This project is released under the MIT License.

---

## Citation

If you use EGAC for research, please cite:

```
@misc{egac2025,
  title={EGAC: Entropy-Guided Actor-Critic with Meta-Learning Dynamics},
  author={Your Name},
  year={2025},
  howpublished={GitHub},
  url={https://github.com/yourusername/EGAC}
}
```

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## Contact

Created by [Your Name] — feel free to reach out with questions or feedback!
```

---

Let me know if you'd like this styled for a specific license, PyPI packaging, or to include experiment results in plots or logs.
