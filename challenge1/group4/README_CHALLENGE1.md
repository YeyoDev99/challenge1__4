# Challenge 1: Deep Q-Network (DQN) on ALE/Gravitar-v5

**Group 4**
**Authors**: Yader Ibraldo Quiroga Torres, Rosa Alejandra Lopez Lizarazo, Diego Alejandro Garzon Rodriguez
**Lecturer**: Carlos Andrés Sierra Virgüez
**Environment**: ALE/Gravitar-v5

---

## Overview

This repository contains the implementation of a Deep Q-Network (DQN) agent for the Atari 2600 game Gravitar, as part of Challenge 1 of the Machine Learning course. The agent learns to navigate through solar systems, destroy reactors, and manage fuel in a gravity-driven physics environment.

## Project Files

- `gravitar_dqn.py` - Main DQN training script (train, play, sweep, inspect modes)
- `sweep_configs.json` - 5 optimized DQN hyperparameter configurations
- `README_CHALLENGE1.md` - This file

Generated at runtime:
- `models/` - Trained model archives (`.zip`)
- `logs/gravitar_dqn/` - DQN TensorBoard event files

---

## Installation

```bash
# Create virtual environment
python -m venv venv_gravitar

# Activate virtual environment
# Windows:
venv_gravitar\Scripts\activate
# Linux/Kali:
source venv_gravitar/bin/activate

# Install dependencies
pip install stable-baselines3 gymnasium[atari] ale-py opencv-python torch tensorboard
```

---

## Quick Start

### Quick Test (1 minute)

```bash
python gravitar_dqn.py --mode train --model-path models/test_dqn --timesteps 10000
```

### Train Single Model (300k steps, ~12-18h CPU / ~3-4h GPU)

```bash
python gravitar_dqn.py --mode train --model-path models/gravitar_dqn_g4
```

### Run Full Sweep (5 experiments × 3 seeds, ~36-54h CPU / ~9-12h GPU)

```bash
python gravitar_dqn.py --mode sweep --sweep-file sweep_configs.json --model-path models/gravitar_dqn_best
```

### Watch Trained Agent (requires display)

```bash
python gravitar_dqn.py --mode play --model-path models/gravitar_dqn_best --episodes 5
```

### Inspect Model Hyperparameters

```bash
python gravitar_dqn.py --mode inspect --model-path models/gravitar_dqn_best
```

---

## Monitor Training with TensorBoard

In a separate terminal:

```bash
tensorboard --logdir logs/gravitar_dqn/sweep --port 6006
```

Open http://localhost:6006 in your browser.

**Key metrics**:
- `rollout/ep_rew_mean` - Rolling mean reward (last 100 episodes)
- `training/episode_reward` - Reward per episode
- `train/loss` - TD-error loss
- `train/learning_rate` - Current learning rate

---

## Hyperparameter Experiments

### DQN Configurations

5 configurations in `sweep_configs.json`:

| Experiment | LR | Buffer | Batch | Purpose |
|---|---|---|---|---|
| exp_01_baseline | 1e-4 | 50k | 64 | Standard reference |
| exp_02_large_buffer | 5e-5 | 200k | 32 | Best for Gravitar — Deep exploration |
| exp_03_high_lr | 2e-4 | 50k | 64 | Fast learning (unstable) |
| exp_04_medium_balance | 1e-4 | 100k | 64 | Conservative balance |
| exp_05_small_batch | 1e-4 | 100k | 32 | Rich gradients |

All use 300,000 timesteps per run.

---

## Algorithm Details

**Deep Q-Network (DQN):**
- CNN processes 84×84 grayscale game frames
- Stacks 4 consecutive frames to capture motion
- Learns Q-values (expected future reward per action)
- Uses replay buffer (memory) to break correlation
- Target network for stable learning
- ε-greedy exploration (decays from 1.0 to 0.01)

**Key hyperparameters**:
- **learning_rate**: How fast the agent learns (default 1e-4)
- **buffer_size**: Memory of past experiences (default 50k-200k)
- **batch_size**: Samples per update (32 or 64)
- **gamma**: Future discount (0.99 = future matters almost as much)
- **train_freq**: Update interval in steps (4 = every 4 steps)
- **exploration_fraction**: How long to explore (0.167 = first 50k steps of 300k)
- **exploration_final_eps**: Minimum exploration rate (0.01 = always 1% random)

---

## Results Summary

**Best DQN Configuration** (exp_02_large_buffer):
- **Learning rate**: 5e-5
- **Buffer size**: 200,000
- **Batch size**: 32
- **Smoothed mean reward**: 335.7081
- **Training throughput**: 6 FPS
- **Steps to 200 reward**: ~180,000
- **Training loss**: Volatile (0.002-0.01)

**Key findings**:
- DQN achieved moderate performance on Gravitar
- Suffered from severe computational overhead (6 FPS bottleneck)
- ε-greedy exploration decayed before mastering thrust sequences
- Policy rigidity following epsilon-decay led to plateau at 335.7
- Off-policy replay provided efficient learning from rare successful episodes

---

## Output File Structure

```
models/
├── test_dqn.zip                    # Quick test model
├── gravitar_dqn_g4.zip             # Single training run
└── gravitar_dqn_best.zip           # Best model from sweep

logs/gravitar_dqn/sweep/
├── exp_01_baseline/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
├── exp_02_large_buffer/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
└── ...
```

---

## Customization

Modify hyperparameters in `sweep_configs.json`:

```json
{
  "name": "exp_custom",
  "learning_rate": 1e-4,
  "buffer_size": 100000,
  "batch_size": 32,
  "gamma": 0.99,
  "train_freq": 4,
  "target_update_interval": 1000,
  "exploration_fraction": 0.167,
  "exploration_final_eps": 0.01,
  "timesteps": 300000
}
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `No module named gymnasium` | `pip install gymnasium[atari]` |
| `No module named stable_baselines3` | `pip install stable-baselines3` |
| `ALE not found` | `pip install ale-py` |
| `CUDA out of memory` | Reduce `batch_size` in sweep_configs.json |
| `Model not found` | Run training first with `--mode train` |

---

## Command Reference

```bash
# Training modes
python gravitar_dqn.py --mode train --model-path models/model_name
python gravitar_dqn.py --mode train --experiment exp_01_baseline --model-path models/model_name

# Sweep mode
python gravitar_dqn.py --mode sweep --sweep-file sweep_configs.json --model-path models/best

# Play mode
python gravitar_dqn.py --mode play --model-path models/model_name --episodes 5

# Inspect mode
python gravitar_dqn.py --mode inspect --model-path models/model_name

# Common options
--seed 42                    # Set random seed
--timesteps 300000          # Override timesteps
--tensorboard-log logs/     # Custom log directory
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{challenge1_dqn,
  title={Deep Q-Network Implementation for ALE/Gravitar-v5},
  author={Quiroga Torres, Yader Ibraldo and Lopez Lizarazo, Rosa Alejandra and Garzon Rodriguez, Diego Alejandro},
  journal={Universidad Distrital Francisco José de Caldas},
  year={2026},
  note={Machine Learning Course - Challenge 1}
}
```

---

## License

This project is part of the Machine Learning course at Universidad Distrital Francisco José de Caldas.
