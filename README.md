# Gravitar DQN & PPO — Challenge 1 & 3 Machine Learning (Group 4)

**Authors**: Yader Ibraldo Quiroga Torres, Rosa Alejandra Lopez Lizarazo, Diego Alejandro Garzon Rodriguez
**Lecturer**: Carlos Andrés Sierra Virgüez
**Environment**: ALE/Gravitar-v5
**Institution**: Universidad Distrital Francisco José de Caldas
**Date**: May 2026

This repository contains implementations of Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents for the Atari 2600 game Gravitar, as part of Challenges 1 and 3 of the Machine Learning course. The project compares off-policy (DQN) and on-policy (PPO) deep reinforcement learning algorithms on a physics-based environment requiring precise thrust control and gravity compensation.

---

## Quick Links

- **[Challenge 1: DQN](challenge1/group4/README_CHALLENGE1.md)** - Deep Q-Network implementation and results
- **[Challenge 3: PPO](challenge3/group4/README_CHALLENGE3.md)** - Proximal Policy Optimization implementation and results
- **[Challenge 3 Checklist](challenge3/group4/CHECKLIST.md)** - Training commands and verification

---

## Repository Structure

```
challenge1__4/
├── README.md                           # This file - General repository overview
│
├── challenge1/
│   └── group4/
│       ├── README_CHALLENGE1.md       # Challenge 1 (DQN) detailed documentation
│       ├── gravitar_dqn.py            # DQN training script (train, play, sweep, inspect)
│       ├── sweep_configs.json         # DQN hyperparameter configurations (5 experiments)
│       ├── pyproject.toml             # Project metadata and dependencies
│       ├── challenge1_4.pdf           # Challenge 1 requirements document
│       ├── models/                    # Generated at runtime - DQN trained models
│       └── logs/                      # Generated at runtime - DQN TensorBoard logs
│
└── challenge3/
    └── group4/
        ├── README_CHALLENGE3.md       # Challenge 3 (PPO) detailed documentation
        ├── CHECKLIST.md               # Training commands, seeds, comparative summary
        ├── gravitar_ppo.py            # PPO training script (train, play, sweep, inspect)
        ├── sweep_configs_ppo.json     # PPO hyperparameter configurations (9 experiments)
        ├── challenge3_group4_paper.pdf # Challenge 3 paper (PDF)
        ├── models/                    # Generated at runtime - PPO trained models
        └── logs/                      # Generated at runtime - PPO TensorBoard logs
```

---

## Results Summary

| Metric | DQN (Challenge 1) | PPO (Challenge 3) |
|--------|------------------|------------------|
| Max Smoothed Reward | 335.7081 | **374.7062** |
| Training Throughput (FPS) | 6 | **171** |
| Steps to 200 Reward | ~180,000 | **~165,000** |
| Explained Variance | N/A | **0.406** |
| Stability (AUC Normalized) | 0.58 | **0.72** |

**Key Finding**: PPO achieved superior performance with dramatically better computational efficiency (28.5× faster than DQN), making it the preferred algorithm for physics-based Atari games like Gravitar.

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

### Challenge 1 (DQN)

```bash
cd challenge1/group4

# Quick test
python gravitar_dqn.py --mode train --model-path models/test_dqn --timesteps 10000

# Train single model
python gravitar_dqn.py --mode train --model-path models/gravitar_dqn_g4

# Run sweep
python gravitar_dqn.py --mode sweep --sweep-file sweep_configs.json --model-path models/gravitar_dqn_best

cd ../..
```

### Challenge 3 (PPO)

```bash
cd challenge3/group4

# Quick test
python gravitar_ppo.py --mode train --model-path models/test_ppo --timesteps 10000

# Train single model
python gravitar_ppo.py --mode train --model-path models/gravitar_ppo_g4

# Run sweep
python gravitar_ppo.py --mode sweep --sweep-file sweep_configs_ppo.json --model-path models/gravitar_ppo_best

cd ../..
```

For detailed instructions, see the challenge-specific READMEs linked above.

---

## Citation

If you use this code, please cite:

```bibtex
@article{challenge_dqn_ppo,
  title={Empirical Evaluation of On-Policy vs. Off-Policy Deep Reinforcement Learning in Gravity-Driven Physics Environments: A Comparative Study of DQN and PPO on ALE/Gravitar-v5},
  author={Quiroga Torres, Yader Ibraldo and Lopez Lizarazo, Rosa Alejandra and Garzon Rodriguez, Diego Alejandro},
  journal={Universidad Distrital Francisco José de Caldas},
  year={2026},
  note={Machine Learning Course - Challenges 1 \& 3}
}
```

---

## License

This project is part of the Machine Learning course at Universidad Distrital Francisco José de Caldas.  
**Date:** May 2026

