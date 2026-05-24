# Gravitar DQN & PPO — Challenge 1 & 3 Machine Learning (Group 4)

**Authors**: Yader Ibraldo Quiroga Torres, Rosa Alejandra Lopez Lizarazo, Diego Alejandro Garzon Rodriguez
**Lecturer**: Carlos Andrés Sierra Virgüez
**Environment**: ALE/Gravitar-v5

This repository contains implementations of Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents for the Atari 2600 game Gravitar, as part of Challenges 1 and 3 of the Machine Learning course at Universidad Distrital Francisco José de Caldas.

---

## Quick Links

- **[Challenge 1: DQN](README_CHALLENGE1.md)** - Deep Q-Network implementation and results
- **[Challenge 3: PPO](challenge3/group4/README_CHALLENGE3.md)** - Proximal Policy Optimization implementation and results
- **[Execution Guide](EXECUTION_GUIDE.md)** - Step-by-step commands for running experiments
- **[IEEE Paper](article.tex)** - Comparative study of DQN vs PPO on Gravitar

---

## Project Structure

**Root (Challenge 1 - DQN):**
- `gravitar_dqn.py` — Main DQN training script
- `sweep_configs.json` — 5 optimized DQN hyperparameter configurations
- `README_CHALLENGE1.md` — Detailed DQN documentation

**challenge3/group4/ (Challenge 3 - PPO):**
- `gravitar_ppo.py` — Main PPO training script
- `sweep_configs_ppo.json` — 9 optimized PPO hyperparameter configurations
- `CHECKLIST.md` — Training commands, seeds, and comparative summary
- `comparative_analysis_dqn_vs_ppo.md` — Comparative analysis template
- `ieee_paper_dqn_to_ppo_gravitar.md` — IEEE paper template
- `Challenge3.md` — Challenge 3 requirements document
- `README_CHALLENGE3.md` — Detailed PPO documentation

**Shared:**
- `README.md` — This index file
- `README_CHALLENGE1.md` — Challenge 1 documentation
- `README_CHALLENGE3.md` — Challenge 3 documentation
- `EXECUTION_GUIDE.md` - Complete execution guide for both challenges
- `article.tex` — IEEE paper comparing DQN and PPO
- `pyproject.toml` — Project metadata

Generated at runtime:
- `models/` — Trained model archives (`.zip`)
- `logs/gravitar_dqn/` — DQN TensorBoard event files
- `logs/gravitar_ppo/` — PPO TensorBoard event files

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
# Quick test
python gravitar_dqn.py --mode train --model-path models/test_dqn --timesteps 10000

# Train single model
python gravitar_dqn.py --mode train --model-path models/gravitar_dqn_g4

# Run sweep
python gravitar_dqn.py --mode sweep --sweep-file sweep_configs.json --model-path models/gravitar_dqn_best
```

### Challenge 3 (PPO)

```bash
cd challenge3/group4

# Quick test
python gravitar_ppo.py --mode train --model-path ../../models/test_ppo --timesteps 10000

# Train single model
python gravitar_ppo.py --mode train --model-path ../../models/gravitar_ppo_g4

# Run sweep
python gravitar_ppo.py --mode sweep --sweep-file sweep_configs_ppo.json --model-path ../../models/gravitar_ppo_best

cd ../..
```

For detailed instructions, see [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md).

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
**Date:** March 2026

