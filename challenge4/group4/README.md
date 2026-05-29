# Challenge 4: GAIL (Generative Adversarial Imitation Learning) on ALE/Gravitar-v5

**Group 4**
**Authors**: Yader Ibraldo Quiroga Torres, Rosa Alejandra Lopez Lizarazo, Diego Alejandro Garzon Rodriguez
**Lecturer**: Carlos Andrés Sierra Virgüez
**Environment**: ALE/Gravitar-v5

---

## Overview

This repository contains the implementation of Generative Adversarial Imitation Learning (GAIL) for the Atari 2600 game Gravitar, as part of Challenge 4 of the Machine Learning course. GAIL combines ideas from Generative Adversarial Networks (GANs) and Reinforcement Learning to allow an agent to learn from demonstrations without requiring an explicit reward function from the environment.

This challenge completes a three-algorithm comparison series:
- **Challenge 1**: DQN (Deep Q-Network) - value-based RL
- **Challenge 3**: PPO (Proximal Policy Optimization) - on-policy actor-critic RL
- **Challenge 4**: GAIL (Generative Adversarial Imitation Learning) - imitation learning with adversarial training

---

## Project Files

- `gravitar_gail.py` - Main GAIL training script (collect, bc, gail, sweep modes)
- `sweep_configs_gail.json` - 9 optimized GAIL hyperparameter configurations
- `pyproject.toml` - Project metadata and dependencies
- `Challenge4.md` - Challenge 4 requirements document
- `README.md` - This file

Generated at runtime:
- `demos.npz` - Demonstration dataset (state-action pairs)
- `bc_policy.pt` - Behavioral Cloning trained policy
- `models/gail/` - Trained GAIL models (policy and discriminator)
- `logs/gail/` - GAIL TensorBoard event files

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
pip install numpy torch "gymnasium[atari,accept-rom-license]>=0.29.1,<1.1.0" \
    "ale-py==0.10.1" "stable-baselines3>=2.3,<3" \
    "opencv-python>=4.8" tensorboard
```

---

## Quick Start

### Step 1: Collect Demonstrations

First, collect demonstrations from a trained checkpoint (DQN from Challenge 1 or PPO from Challenge 3):

```bash
# Option A: Collect from best DQN checkpoint (Challenge 1)
python gravitar_gail.py --mode collect \
    --checkpoint-path ../../challenge1/group4/models/best.zip \
    --n-steps 30000 \
    --demos-path demos.npz

# Option B: Collect from best PPO checkpoint (Challenge 3)
python gravitar_gail.py --mode collect \
    --checkpoint-path ../../challenge3/group4/models/gravitar_ppo.zip \
    --n-steps 30000 \
    --demos-path demos_ppo.npz
```

### Step 2: Train Behavioral Cloning (BC) Baseline

Train the BC baseline to establish a supervised learning lower bound:

```bash
# If using DQN demonstrations
python gravitar_gail.py --mode bc \
    --demos-path demos.npz \
    --bc-epochs 25 \
    --bc-model-path bc_policy.pt

# If using PPO demonstrations
python gravitar_gail.py --mode bc \
    --demos-path demos_ppo.npz \
    --bc-epochs 25 \
    --bc-model-path bc_policy_ppo.pt
```

### Step 3: Train GAIL

Train GAIL with BC warm-start:

```bash
# If using DQN demonstrations
python gravitar_gail.py --mode gail \
    --demos-path demos.npz \
    --bc-warmstart bc_policy.pt \
    --total-steps 500000 \
    --log-dir logs/gail

# If using PPO demonstrations
python gravitar_gail.py --mode gail \
    --demos-path demos_ppo.npz \
    --bc-warmstart bc_policy_ppo.pt \
    --total-steps 500000 \
    --log-dir logs/gail_ppo
```

### Step 4: Run Full Sweep

Run hyperparameter sweep across 9 configurations × 3 seeds:

```bash
# If using DQN demonstrations
python gravitar_gail.py --mode sweep \
    --sweep-file sweep_configs_gail.json \
    --demos-path demos.npz \
    --seeds 42 43 44

# If using PPO demonstrations
python gravitar_gail.py --mode sweep \
    --sweep-file sweep_configs_gail.json \
    --demos-path demos_ppo.npz \
    --seeds 42 43 44
```

---

## Algorithm Details

### Behavioral Cloning (BC)

BC treats imitation as a supervised learning problem, minimizing the negative log-likelihood between demonstrations and policy outputs:

$$\mathcal{L}_{BC}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log \pi_\theta(a_i | s_i)$$

BC is fast to train but suffers from distributional shift: the policy's errors accumulate at test time.

### Generative Adversarial Imitation Learning (GAIL)

GAIL frames imitation as a two-player game between a policy π_θ (generator) and a discriminator D_φ:

**Discriminator objective** (distinguish expert from policy):
$$\max_{D_\phi} \mathbb{E}_{(s,a) \sim \mathcal{D}}[\log D_\phi(s, a)] + \mathbb{E}_{(s,a) \sim \pi_\theta}[\log(1 - D_\phi(s, a))]$$

**Policy objective** (using adversarial reward):
$$r_{adv}(s, a) = \log D_\phi(s, a)$$

The policy is trained with PPO using the adversarial reward signal instead of the environment's true reward.

---

## Hyperparameter Experiments

### GAIL Configurations

9 configurations in `sweep_configs_gail.json`:

| Experiment | LR Policy | LR Disc | Disc Updates | BC Warm-start | Entropy | Purpose |
|---|---|---|---|---|---|---|
| exp_01_baseline | 2.5e-4 | 3e-4 | 3 | Yes | 0.01 | Baseline with BC warm-start |
| exp_02_high_disc_lr | 2.5e-4 | 5e-4 | 3 | Yes | 0.01 | Faster discriminator adaptation |
| exp_03_low_disc_lr | 2.5e-4 | 1e-4 | 3 | Yes | 0.01 | More stable discriminator |
| exp_04_more_disc_updates | 2.5e-4 | 3e-4 | 5 | Yes | 0.01 | More frequent discriminator updates |
| exp_05_no_bc_warmstart | 2.5e-4 | 3e-4 | 3 | No | 0.01 | Random initialization |
| exp_06_high_entropy | 2.5e-4 | 3e-4 | 3 | Yes | 0.02 | More exploration |
| exp_07_low_entropy | 2.5e-4 | 3e-4 | 3 | Yes | 0.001 | More exploitation |
| exp_08_long_horizon | 2.5e-4 | 3e-4 | 3 | Yes | 0.01 | Better credit assignment |
| exp_09_short_horizon | 2.5e-4 | 3e-4 | 3 | Yes | 0.01 | More frequent updates |

All use 5,000,000 timesteps per run (Challenge 4 budget).

---

## Key Hyperparameters

- **lr_policy**: Policy learning rate (default 2.5e-4)
- **lr_disc**: Discriminator learning rate (default 3e-4)
- **disc_updates_per_rollout**: Number of discriminator updates per PPO rollout (default 3)
- **horizon**: PPO rollout horizon (default 1024)
- **ent_coef**: Entropy coefficient for exploration (default 0.01)
- **bc_warmstart**: Whether to initialize policy with BC weights (default True)

---

## Research Questions

This implementation addresses the following research questions:

1. **Does GAIL outperform pure RL (DQN, PPO) on Gravitar?**
   - Gravitar requires maintaining specific thrust sequences. GAIL may learn control style from demonstrations before discovering reward.

2. **How sensitive is GAIL to demonstration quality and quantity?**
   - Ablation experiments test different demonstration sizes and quality levels.

3. **Does the adversarial reward remain informative throughout training?**
   - Discriminator accuracy is tracked to detect collapse or loss of informativeness.

4. **In what regime is BC alone competitive with full GAIL or PPO?**
   - BC baseline provides a supervised learning lower bound for comparison.

---

## Command Reference

```bash
# Collect demonstrations
python gravitar_gail.py --mode collect \
    --checkpoint-path <path_to_checkpoint> \
    --n-steps 30000 \
    --demos-path demos.npz

# Train BC baseline
python gravitar_gail.py --mode bc \
    --demos-path demos.npz \
    --bc-epochs 25 \
    --bc-model-path bc_policy.pt

# Train single GAIL model
python gravitar_gail.py --mode gail \
    --demos-path demos.npz \
    --bc-warmstart bc_policy.pt \
    --total-steps 5000000 \
    --lr-policy 2.5e-4 \
    --lr-disc 3e-4 \
    --disc-updates 3 \
    --ent-coef 0.01 \
    --log-dir logs/gail

# Run hyperparameter sweep
python gravitar_gail.py --mode sweep \
    --sweep-file sweep_configs_gail.json \
    --demos-path demos.npz \
    --seeds 42 43 44

# Common options
--device cuda  # Use GPU (default: auto-detect)
--seed 42      # Random seed
```

---

## Output File Structure

```
challenge4/group4/
├── demos.npz                          # Demonstration dataset
├── bc_policy.pt                       # BC trained policy
├── models/
│   └── gail/
│       ├── exp_01_baseline/
│       │   ├── policy_seed42.pt
│       │   ├── policy_seed43.pt
│       │   ├── policy_seed44.pt
│       │   ├── discriminator_seed42.pt
│       │   ├── discriminator_seed43.pt
│       │   └── discriminator_seed44.pt
│       └── ...
├── logs/
│   └── gail/
│       └── sweep/
│           ├── exp_01_baseline/
│           │   ├── seed_42/
│           │   ├── seed_43/
│           │   └── seed_44/
│           └── ...
└── sweep_results_gail.json            # Sweep results summary
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `No module named gymnasium` | `pip install gymnasium[atari]` |
| `No module named stable_baselines3` | `pip install stable-baselines3` |
| `ALE not found` | `pip install ale-py` |
| `CUDA out of memory` | Reduce `horizon` or use `--device cpu` |
| `Demonstrations not found` | Run `--mode collect` first |
| `Discriminator collapses (acc ~0.5)` | Reduce `lr_disc` or increase `disc_updates` |

---

## Citation

If you use this code, please cite:

```bibtex
@article{challenge4_gail,
  title={Generative Adversarial Imitation Learning Implementation for ALE/Gravitar-v5},
  author={Quiroga Torres, Yader Ibraldo and Lopez Lizarazo, Rosa Alejandra and Garzon Rodriguez, Diego Alejandro},
  journal={Universidad Distrital Francisco José de Caldas},
  year={2026},
  note={Machine Learning Course - Challenge 4}
}
```

---

## License

This project is part of the Machine Learning course at Universidad Distrital Francisco José de Caldas.
