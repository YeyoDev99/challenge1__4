# Challenge 3: Proximal Policy Optimization (PPO) on ALE/Gravitar-v5

**Group 4**
**Authors**: Yader Ibraldo Quiroga Torres, Rosa Alejandra Lopez Lizarazo, Diego Alejandro Garzon Rodriguez
**Lecturer**: Carlos Andrés Sierra Virgüez
**Environment**: ALE/Gravitar-v5

---

## Overview

This repository contains the implementation of a Proximal Policy Optimization (PPO) agent for the Atari 2600 game Gravitar, as part of Challenge 3 of the Machine Learning course. PPO is an on-policy actor-critic method that achieves the reliability of trust-region methods while being easier to implement and tune.

## Project Files

- `gravitar_ppo.py` - Main PPO training script (train, play, sweep, inspect modes)
- `sweep_configs_ppo.json` - 9 optimized PPO hyperparameter configurations
- `CHECKLIST.md` - Training commands, seeds, and comparative summary
- `comparative_analysis_dqn_vs_ppo.md` - Comparative analysis template
- `ieee_paper_dqn_to_ppo_gravitar.md` - IEEE paper template
- `Challenge3.md` - Challenge 3 requirements document
- `README_CHALLENGE3.md` - This file

Generated at runtime:
- `models/` - Trained model archives (`.zip`)
- `logs/gravitar_ppo/` - PPO TensorBoard event files

---

## Installation

```bash
# Create virtual environment (if not already created)
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
python gravitar_ppo.py --mode train --model-path ../../models/test_ppo --timesteps 10000
```

### Train Single Model (5M steps, ~12-18h CPU / ~3-4h GPU)

```bash
python gravitar_ppo.py --mode train --model-path ../../models/gravitar_ppo_g4
```

### Run Full Sweep (9 experiments × 3 seeds, ~36-54h CPU / ~9-12h GPU)

```bash
python gravitar_ppo.py --mode sweep --sweep-file sweep_configs_ppo.json --model-path ../../models/gravitar_ppo_best
```

### Watch Trained Agent (requires display)

```bash
python gravitar_ppo.py --mode play --model-path ../../models/gravitar_ppo_best --episodes 5
```

### Inspect Model Hyperparameters

```bash
python gravitar_ppo.py --mode inspect --model-path ../../models/gravitar_ppo_best
```

---

## Monitor Training with TensorBoard

In a separate terminal:

```bash
tensorboard --logdir logs/gravitar_ppo/sweep --port 6006
```

Open http://localhost:6006 in your browser.

**Key metrics**:
- `rollout/ep_rew_mean` - Rolling mean reward (last 100 episodes)
- `training/episode_reward` - Reward per episode
- `train/policy_loss` - PPO policy loss (clipped surrogate)
- `train/value_loss` - Value function loss
- `train/entropy_loss` - Entropy bonus term
- `train/learning_rate` - Current learning rate
- `train/clip_fraction` - Fraction of clipped updates
- `train/explained_variance` - Critic prediction accuracy

---

## Hyperparameter Experiments

### PPO Configurations

9 configurations in `sweep_configs_ppo.json` (systematic search of LR, horizon, entropy):

| Experiment | LR | n_steps | Entropy | Purpose |
|---|---|---|---|---|
| exp_01_baseline | 2.5e-4 | 1024 | 0.01 | Challenge 3 starter for Gravitar |
| exp_02_high_lr | 5e-4 | 1024 | 0.01 | Test faster learning |
| exp_03_low_lr | 1e-4 | 1024 | 0.01 | Test stable learning |
| exp_04_short_horizon | 2.5e-4 | 512 | 0.01 | More frequent updates |
| exp_05_long_horizon | 2.5e-4 | 2048 | 0.01 | Better credit assignment |
| exp_06_high_entropy | 2.5e-4 | 1024 | 0.02 | More exploration |
| exp_07_low_entropy | 2.5e-4 | 1024 | 0.001 | More exploitation |
| exp_08_high_lr_high_entropy | 5e-4 | 1024 | 0.02 | Aggressive exploration |
| exp_09_long_horizon_high_entropy | 2.5e-4 | 2048 | 0.02 | Recommended for Gravitar |

All use 5,000,000 timesteps per run (Challenge 3 budget).

---

## Algorithm Details

**Proximal Policy Optimization (PPO):**
- Actor-Critic architecture with shared CNN backbone
- Clipped surrogate objective prevents large policy updates
- Generalized Advantage Estimation (GAE) for variance reduction
- Entropy bonus encourages exploration
- On-policy learning (discards data after update)

**Key hyperparameters**:
- **learning_rate**: How fast the agent learns (default 2.5e-4 for Gravitar)
- **n_steps**: Rollout buffer size (1024 steps per environment for Gravitar)
- **batch_size**: Minibatch size for PPO updates (128 for Gravitar)
- **n_epochs**: Number of optimization epochs per update (4 for Gravitar)
- **gamma**: Future discount (0.99)
- **gae_lambda**: GAE parameter for advantage estimation (0.95)
- **clip_range**: PPO clipping parameter (0.2 = prevents large policy updates)
- **ent_coef**: Entropy coefficient for exploration (0.01)
- **vf_coef**: Value function loss coefficient (0.5)
- **max_grad_norm**: Gradient clipping for stability (0.5)

---

## Results Summary

**Best PPO Configuration** (exp_01_baseline):
- **Learning rate**: 2.5e-4
- **Horizon (n_steps)**: 1024
- **Batch size**: 128
- **N epochs**: 4
- **Entropy coefficient**: 0.01
- **Smoothed mean reward**: 374.7062
- **Training throughput**: 171 FPS
- **Steps to 200 reward**: ~165,000
- **Explained variance**: 0.406 (peak: 0.625)
- **Entropy loss**: -1.9571
- **Clip fraction**: 0.336

**Key findings**:
- PPO achieved superior performance over DQN (374.7 vs 335.7)
- Dramatically improved wall-clock efficiency (171 FPS vs 6 FPS, 28.5× faster)
- On-policy learning with GAE better captured Gravitar's continuous thrust requirements
- Entropy bonus maintained exploratory thrust patterns throughout training
- Balanced action distribution with consistent thrust engagement
- Clipped objective prevented policy collapse but led to conservative fuel management

---

## Comparison with DQN (Challenge 1)

| Metric | DQN (Challenge 1) | PPO (Challenge 3) |
|--------|------------------|------------------|
| Max Smoothed Reward | 335.7081 | **374.7062** |
| Training Throughput (FPS) | 6 | **171** |
| Steps to 200 Reward | ~180,000 | **~165,000** |
| Final Entropy Loss | N/A | -1.9571 |
| Explained Variance | N/A | **0.406** |
| Stability (AUC Normalized) | 0.58 | **0.72** |

**Conclusion**: PPO is superior for physics-based Atari games like Gravitar, achieving higher performance with dramatically better computational efficiency.

---

## Output File Structure

```
models/
├── test_ppo.zip                    # Quick test model
├── gravitar_ppo_g4.zip             # Single training run
└── gravitar_ppo_best.zip           # Best model from sweep

logs/gravitar_ppo/sweep/
├── exp_01_baseline/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
├── exp_02_high_lr/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
└── ...
```

---

## Customization

Modify hyperparameters in `sweep_configs_ppo.json`:

```json
{
  "name": "exp_custom",
  "learning_rate": 2.5e-4,
  "n_steps": 1024,
  "batch_size": 128,
  "n_epochs": 4,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "clip_range": 0.2,
  "ent_coef": 0.01,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5,
  "timesteps": 5000000
}
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `No module named gymnasium` | `pip install gymnasium[atari]` |
| `No module named stable_baselines3` | `pip install stable-baselines3` |
| `ALE not found` | `pip install ale-py` |
| `CUDA out of memory` | Reduce `batch_size` in sweep_configs_ppo.json |
| `Model not found` | Run training first with `--mode train` |

---

## Command Reference

```bash
# Training modes
python gravitar_ppo.py --mode train --model-path ../../models/model_name
python gravitar_ppo.py --mode train --experiment exp_01_baseline --model-path ../../models/model_name

# Sweep mode
python gravitar_ppo.py --mode sweep --sweep-file sweep_configs_ppo.json --model-path ../../models/best

# Play mode
python gravitar_ppo.py --mode play --model-path ../../models/model_name --episodes 5

# Inspect mode
python gravitar_ppo.py --mode inspect --model-path ../../models/model_name

# Common options
--seed 42                    # Set random seed
--timesteps 5000000         # Override timesteps
--tensorboard-log logs/     # Custom log directory
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{challenge3_ppo,
  title={Proximal Policy Optimization Implementation for ALE/Gravitar-v5},
  author={Quiroga Torres, Yader Ibraldo and Lopez Lizarazo, Rosa Alejandra and Garzon Rodriguez, Diego Alejandro},
  journal={Universidad Distrital Francisco José de Caldas},
  year={2026},
  note={Machine Learning Course - Challenge 3}
}
```

---

## License

This project is part of the Machine Learning course at Universidad Distrital Francisco José de Caldas.
