# Gravitar DQN & PPO — Challenge 1 & 3 Machine Learning (Group 4)

Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents trained on Atari Gravitar using Stable-Baselines3 and Gymnasium.

## Project Structure

**Root (Challenge 1 - DQN):**
- `gravitar_dqn.py` — Main DQN training script (train, play, sweep, inspect modes)
- `sweep_configs.json` — 5 optimized DQN hyperparameter configurations

**challenge3/group4/ (Challenge 3 - PPO):**
- `gravitar_ppo.py` — Main PPO training script (train, play, sweep, inspect modes)
- `sweep_configs_ppo.json` — 9 optimized PPO hyperparameter configurations
- `CHECKLIST.md` — Training commands, seeds, and comparative summary
- `comparative_analysis_dqn_vs_ppo.md` — Comparative analysis template
- `ieee_paper_dqn_to_ppo_gravitar.md` — IEEE paper template
- `Challenge3.md` — Challenge 3 requirements document

**Shared:**
- `README.md` — Complete installation and usage guide (this file)
- `pyproject.toml` — Project metadata

Generated at runtime:
- `models/` — Trained model archives (`.zip`)
- `logs/gravitar_dqn/` — DQN TensorBoard event files
- `logs/gravitar_ppo/` — PPO TensorBoard event files

---

## Installation

### KALI LINUX 

Copy and paste all commands below in order:

```bash
sudo apt-get update && sudo apt-get upgrade -y

sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev build-essential \
    cmake git wget libopenblas-dev liblapack-dev libgomp1 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libatlas-base-dev libjpeg-dev libpng-dev libtiff-dev \
    libfreetype6-dev libharfbuzz0b

python3.11 -m venv venv_gravitar
source venv_gravitar/bin/activate

pip install --upgrade pip setuptools wheel

pip install numpy torch "stable-baselines3[extra]>=2.3,<3" \
    "gymnasium[atari,accept-rom-license]>=0.29.1,<1.1.0" \
    "ale-py==0.10.1" "autorom[accept-rom-license]" \
    "opencv-python-headless>=4.8" tqdm rich tensorboard

python -m autorom --accept-rom-license

python -c "import gymnasium as gym; import ale_py; gym.register_envs(ale_py); env = gym.make('ALE/Gravitar-v5'); print('✓ Gravitar ready'); env.close()"
```

### WINDOWS 11 

Open PowerShell **as Administrator** and copy-paste all commands:

```powershell
python -m venv venv_gravitar

venv_gravitar\Scripts\activate

python -m pip install --upgrade pip setuptools wheel

pip install numpy torch "stable-baselines3[extra]>=2.3,<3" ^
    "gymnasium[atari,accept-rom-license]>=0.29.1,<1.1.0" ^
    "ale-py==0.10.1" "autorom[accept-rom-license]" ^
    "opencv-python>=4.8" tqdm rich tensorboard

python -m autorom --accept-rom-license

python -c "import gymnasium as gym; import ale_py; gym.register_envs(ale_py); env = gym.make('ALE/Gravitar-v5'); print('✓ Gravitar ready'); env.close()"
```

**Note:** Replace `^` with `\` if using Git Bash instead of PowerShell.

---

## Usage

### Activate Environment

```bash
# Linux/Kali
source venv_gravitar/bin/activate

# Windows
venv_gravitar\Scripts\activate
```

### Quick Test (1 minute)

```bash
python gravitar_dqn.py --mode train --model-path models/test --timesteps 10000
```

### Train Single Model (300k steps, ~45 min CPU / ~10 min GPU)

```bash
python gravitar_dqn.py --mode train --model-path models/gravitar_g4
```

### Run Full Sweep (5 experiments × 3 seeds, ~12-18h CPU / ~3-4h GPU)

```bash
python gravitar_dqn.py --mode sweep --sweep-file sweep_configs.json --model-path models/gravitar_best
```

This automatically:
- Trains 5 different hyperparameter configurations
- Runs each configuration 3 times with different random seeds (42, 43, 44)
- Calculates mean ± std deviation for each experiment
- Selects and saves the best overall model
- Saves logs organized by experiment and seed

### Watch Trained Agent (requires display)

```bash
python gravitar_dqn.py --mode play --model-path models/gravitar_best --episodes 5
```

### Inspect Model Hyperparameters

```bash
python gravitar_dqn.py --mode inspect --model-path models/gravitar_best
```

---

## Challenge 3: PPO (Proximal Policy Optimization)

### Quick Test PPO (1 minute)

```bash
cd challenge3/group4
python gravitar_ppo.py --mode train --model-path ../../models/test_ppo --timesteps 10000
```

### Train Single PPO Model (5M steps, ~12-18h CPU / ~3-4h GPU)

```bash
cd challenge3/group4
python gravitar_ppo.py --mode train --model-path ../../models/gravitar_ppo_g4
```

### Run Full PPO Sweep (9 experiments × 3 seeds, ~36-54h CPU / ~9-12h GPU)

```bash
cd challenge3/group4
python gravitar_ppo.py --mode sweep --sweep-file sweep_configs_ppo.json --model-path ../../models/gravitar_ppo_best
```

### Watch Trained PPO Agent (requires display)

```bash
cd challenge3/group4
python gravitar_ppo.py --mode play --model-path ../../models/gravitar_ppo_best --episodes 5
```

### Inspect PPO Model Hyperparameters

```bash
cd challenge3/group4
python gravitar_ppo.py --mode inspect --model-path ../../models/gravitar_ppo_best
```

### Monitor PPO Training with TensorBoard

In a separate terminal:

```bash
tensorboard --logdir logs/gravitar_ppo/sweep --port 6006
```

Key PPO metrics:
- `rollout/ep_rew_mean` — Rolling average reward (higher is better)
- `train/policy_loss` — PPO clipped surrogate loss (lower is better)
- `train/value_loss` — Value function loss (lower is better)
- `train/entropy_loss` — Entropy bonus (encourages exploration)

---

## Monitor Training with TensorBoard

In a separate terminal:

```bash
tensorboard --logdir logs/gravitar_dqn/sweep --port 6006
```

Then open in browser: `http://localhost:6006`

Key metrics:
- `rollout/ep_rew_mean` — Rolling average reward (higher is better)
- `training/epsilon` — Exploration decay (1.0 → 0.01)
- `train/loss` — TD error (lower is better)

---

## Hyperparameter Experiments

### DQN (Challenge 1)

5 configurations in `sweep_configs.json`:

| Experiment | LR | Buffer | Batch | Purpose |
|---|---|---|---|---|
| exp_01_baseline | 1e-4 | 50k | 64 | Standard reference |
| exp_02_large_buffer | 5e-5 | 200k | 32 | Best for Gravitar — Deep exploration |
| exp_03_high_lr | 2e-4 | 50k | 64 | Fast learning (unstable) |
| exp_04_medium_balance | 1e-4 | 100k | 64 | Conservative balance |
| exp_05_small_batch | 1e-4 | 100k | 32 | Rich gradients |

All use 300,000 timesteps per run.

### PPO (Challenge 3)

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

**Deep Q-Network (DQN):**
- CNN processes 84×84 grayscale game frames
- Stacks 4 consecutive frames to capture motion
- Learns Q-values (expected future reward per action)
- Uses replay buffer (memory) to break correlation
- ε-greedy exploration: 90% best action, 10% random (decays over time)
- Target network sync every 1000 steps (stabilizes learning)

**Proximal Policy Optimization (PPO):**
- Actor-Critic architecture with CNN base for Atari images
- Actor (policy network): outputs action probabilities
- Critic (value network): estimates state value V(s)
- Clipped surrogate objective prevents large policy updates
- Generalized Advantage Estimation (GAE) for advantage computation
- Entropy bonus encourages exploration
- On-policy learning: collects trajectories, then optimizes multiple epochs
- Gradient clipping for stability

**Preprocessing (automatic):**
- RGB → Grayscale
- Resize to 84×84
- Frame skip every 4 steps
- Frame stacking (4 frames)
- Terminal-on-life-loss

---

## Output Files

After DQN training:

```
models/
  └── gravitar_best.zip          # Best trained DQN model

logs/gravitar_dqn/
  └── sweep/
      ├── exp_01_baseline/
      │   ├── seed_42/
      │   ├── seed_43/
      │   └── seed_44/
      ├── exp_02_large_buffer/...
      ├── exp_03_high_lr/...
      ├── exp_04_medium_balance/...
      └── exp_05_small_batch/...
```

After PPO training:

```
models/
  └── gravitar_ppo_best.zip     # Best trained PPO model

logs/gravitar_ppo/
  └── sweep/
      ├── exp_01_baseline/
      │   ├── seed_42/
      │   ├── seed_43/
      │   └── seed_44/
      ├── exp_02_high_entropy/...
      ├── exp_03_tight_clip/...
      ├── exp_04_high_gae/...
      └── exp_05_more_epochs/...
```

---

## Customization

### DQN

Modify hyperparameters in `sweep_configs.json`:

```json
{
  "name": "exp_custom",
  "learning_rate": 1e-4,
  "buffer_size": 100000,
  "learning_starts": 12000,
  "batch_size": 64,
  "gamma": 0.99,
  "train_freq": 4,
  "target_update_interval": 1200,
  "exploration_fraction": 0.15,
  "exploration_final_eps": 0.01,
  "timesteps": 300000
}
```

### PPO

Modify hyperparameters in `sweep_configs_ppo.json`:

```json
{
  "name": "exp_custom",
  "learning_rate": 2.5e-4,
  "n_steps": 1024,
  "batch_size": 128,
  "n_epochs": 6,
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
| `Could not load any Atari ROM` | Run `python -m autorom --accept-rom-license` again |
| `ImportError: libGL.so.1` (Kali) | `sudo apt-get install libgl1` |
| `ImportError: libglib2.0-0` (Kali) | `sudo apt-get install libglib2.0-0` |
| Training very slow | Check GPU: `python -c "import torch; print(torch.cuda.is_available())"` |

---

## Key Parameters

### DQN
- **learning_rate**: How fast the agent learns (default 1e-4)
- **buffer_size**: Memory of past experiences (default 50k-200k)
- **batch_size**: Samples per update (32 or 64)
- **gamma**: Future discount (0.99 = future matters almost as much)
- **train_freq**: Update interval in steps (4 = every 4 steps)
- **exploration_fraction**: How long to explore (0.15 = first 45k steps of 300k)
- **exploration_final_eps**: Minimum exploration rate (0.01 = always 1% random)

### PPO
- **learning_rate**: How fast the agent learns (default 2.5e-4 for Gravitar)
- **n_steps**: Rollout buffer size (1024 steps per environment for Gravitar)
- **batch_size**: Minibatch size for PPO updates (128 for Gravitar)
- **n_epochs**: Number of optimization epochs per update (6 for Gravitar)
- **gamma**: Future discount (0.99)
- **gae_lambda**: GAE parameter for advantage estimation (0.95)
- **clip_range**: PPO clipping parameter (0.2 = prevents large policy updates)
- **ent_coef**: Entropy coefficient for exploration (0.01)
- **vf_coef**: Value function loss coefficient (0.5)
- **max_grad_norm**: Gradient clipping for stability (0.5)

---

## Author & License

**Challenge:** Machine Learning — Atari DQN & PPO  
**Group:** Grupo 4  
**Professor:** Prof. Carlos Andrés Sierra (cavirguezs@udistrital.edu.co)  
**License:** GNU/GPL 3.0  
**Date:** March 2026

