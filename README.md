# Gravitar DQN — Challenge 1 Machine Learning (Group 4)

Deep Q-Network agent trained on Atari Gravitar using Stable-Baselines3 and Gymnasium.

## Project Files

- `gravitar_dqn.py` — Main training script (train, play, sweep, inspect modes)
- `sweep_configs.json` — 5 optimized hyperparameter configurations
- `README.md` — Complete installation and usage guide (this file)
- `pyproject.toml` — Project metadata

Generated at runtime:
- `models/` — Trained model archives (`.zip`)
- `logs/gravitar_dqn/` — TensorBoard event files

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

### Monitor Training with TensorBoard

In a separate terminal:

```bash
tensorboard --logdir logs/gravitar_dqn
```

Then open in browser: `http://localhost:6006`

Key metrics:
- `rollout/ep_rew_mean` — Rolling average reward (higher is better)
- `training/epsilon` — Exploration decay (1.0 → 0.01)
- `train/loss` — TD error (lower is better)

---

## Hyperparameter Experiments

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
- ε-greedy exploration: 90% best action, 10% random (decays over time)
- Target network sync every 1000 steps (stabilizes learning)

**Preprocessing (automatic):**
- RGB → Grayscale
- Resize to 84×84
- Frame skip every 4 steps
- Frame stacking (4 frames)
- Terminal-on-life-loss

---

## Output Files

After training:

```
models/
  └── gravitar_best.zip          # Best trained model
  
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

---

## Customization

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

- **learning_rate**: How fast the agent learns (default 1e-4)
- **buffer_size**: Memory of past experiences (default 50k-200k)
- **batch_size**: Samples per update (32 or 64)
- **gamma**: Future discount (0.99 = future matters almost as much)
- **train_freq**: Update interval in steps (4 = every 4 steps)
- **exploration_fraction**: How long to explore (0.15 = first 45k steps of 300k)
- **exploration_final_eps**: Minimum exploration rate (0.01 = always 1% random)

---

## Author & License

**Challenge:** Machine Learning — Atari DQN  
**Group:** Grupo 4  
**Professor:** Prof. Carlos Andrés Sierra (cavirguezs@udistrital.edu.co)  
**License:** GNU/GPL 3.0  
**Date:** March 2026

