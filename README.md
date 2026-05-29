# Gravitar: DQN, PPO & GAIL — Machine Learning Challenges 1, 3 & 4 (Group 4)

**Authors**: Yader Ibraldo Quiroga Torres, Rosa Alejandra Lopez Lizarazo, Diego Alejandro Garzon Rodriguez
**Lecturer**: Carlos Andrés Sierra Virgüez
**Environment**: ALE/Gravitar-v5
**Institution**: Universidad Distrital Francisco José de Caldas
**Date**: May 2026

This repository contains comprehensive implementations of three deep reinforcement learning and imitation learning algorithms for the Atari 2600 game Gravitar, as part of Challenges 1, 3, and 4 of the Machine Learning course:
- **Challenge 1**: Deep Q-Network (DQN) — off-policy value-based learning
- **Challenge 3**: Proximal Policy Optimization (PPO) — on-policy actor-critic learning
- **Challenge 4**: Generative Adversarial Imitation Learning (GAIL) — adversarial imitation learning

All algorithms are evaluated on physics-based control requiring precise thrust management and gravity compensation.

---

## Quick Links

- **[Challenge 1: DQN](challenge1/group4/README_CHALLENGE1.md)** - Deep Q-Network implementation and results
- **[Challenge 3: PPO](challenge3/group4/README_CHALLENGE3.md)** - Proximal Policy Optimization implementation and results
- **[Challenge 3 Checklist](challenge3/group4/CHECKLIST.md)** - Training commands, seeds, and comparative analysis
- **[Challenge 4: GAIL](challenge4/group4/README_CHALLENGE4.md)** - Generative Adversarial Imitation Learning implementation
- **[Challenge 4 Checklist](challenge4/group4/CHECKLIST.md)** - Training commands and verification

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
│       ├── models/                    # Generated at runtime - DQN trained models
│       └── logs/                      # Generated at runtime - DQN TensorBoard logs
│
├── challenge3/
│   └── group4/
│       ├── README_CHALLENGE3.md       # Challenge 3 (PPO) detailed documentation
│       ├── CHECKLIST.md               # Training commands, seeds, comparative summary
│       ├── gravitar_ppo.py            # PPO training script (train, play, sweep, inspect)
│       ├── sweep_configs_ppo.json     # PPO hyperparameter configurations (9 experiments)
│       ├── models/                    # Generated at runtime - PPO trained models
│       └── logs/                      # Generated at runtime - PPO TensorBoard logs
│
└── challenge4/
    └── group4/
        ├── README_CHALLENGE4.md       # Challenge 4 (GAIL) detailed documentation
        ├── CHECKLIST.md               # Training commands and verification
        ├── gravitar_gail.py           # GAIL training script (collect, bc, gail, sweep)
        ├── sweep_configs_gail.json    # GAIL hyperparameter configurations (9 experiments)
        ├── pyproject.toml             # Project metadata and dependencies
        ├── demos_ppo.npz              # Demonstration dataset from PPO policy
        ├── bc_policy_ppo.pt           # Behavioral Cloning policy (warmstart)
        ├── gail_metrics.npz           # Computed GAIL evaluation metrics
        ├── models/                    # Generated at runtime - GAIL trained models
        │   ├── bc_policy_seed42.pt    # Behavioral Cloning reference policy
        │   ├── gail_policy.pt         # Final trained GAIL policy
        │   ├── gail_discriminator.pt  # Final trained discriminator
        │   └── gail/                  # Sweep results (exp_01-exp_09)
        ├── logs/                      # Generated at runtime - GAIL TensorBoard logs
        └── rl_env/                    # Virtual environment (optional)
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

### Challenge 4 (GAIL)

```bash
cd challenge4/group4

# Collect demonstrations from PPO policy
python gravitar_gail.py --mode collect --expert-policy models/gravitar_ppo_best.zip --num-demos 10

# Train Behavioral Cloning policy (warmstart)
python gravitar_gail.py --mode bc --demos demos_ppo.npz --model-path models/bc_policy_ppo.pt

# Quick test GAIL training
python gravitar_gail.py --mode gail --demo-path demos_ppo.npz --model-path models/test_gail --timesteps 10000

# Train single GAIL model
python gravitar_gail.py --mode gail --demo-path demos_ppo.npz --model-path models/gravitar_gail_g4

# Run sweep
python gravitar_gail.py --mode sweep --sweep-file sweep_configs_gail.json --demo-path demos_ppo.npz --model-path models/gravitar_gail_best

cd ../..
```

For detailed instructions and parameter explanations, see the challenge-specific READMEs linked above.

---

## Citation

If you use this code, please cite:

```bibtex
@article{challenge_dqn_ppo_gail,
  title={Empirical Comparison of Deep Reinforcement Learning and Imitation Learning Algorithms on ALE/Gravitar-v5: DQN, PPO, and GAIL},
  author={Quiroga Torres, Yader Ibraldo and Lopez Lizarazo, Rosa Alejandra and Garzon Rodriguez, Diego Alejandro},
  journal={Universidad Distrital Francisco José de Caldas},
  year={2026},
  note={Machine Learning Course - Challenges 1, 3 \& 4}
}
```

---

## License

This project is part of the Machine Learning course at Universidad Distrital Francisco José de Caldas.  
**Challenges**: 1, 3 & 4  
**Date**: May 2026

