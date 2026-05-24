# Challenge 3 Checklist - Group 4 (Gravitar)

## Best PPO Run Training Commands

### Single Run (Best Configuration)
```bash
# Activate environment first
# Linux/Kali: source venv_gravitar/bin/activate
# Windows: venv_gravitar\Scripts\activate

# Navigate to challenge3/group4 directory
cd challenge3/group4

# Train with best configuration (to be determined after sweep)
python gravitar_ppo.py --mode train --experiment [EXPERIMENT_NAME] --model-path ../../models/gravitar_ppo_best --seed 42 --timesteps 5000000
```

### Full Sweep (All Configurations)
```bash
cd challenge3/group4
python gravitar_ppo.py --mode sweep --sweep-file sweep_configs_ppo.json --model-path ../../models/gravitar_ppo_best --seed 42
```

### Watch Trained Agent
```bash
cd challenge3/group4
python gravitar_ppo.py --mode play --model-path ../../models/gravitar_ppo_best --episodes 5
```

### Inspect Model Hyperparameters
```bash
cd challenge3/group4
python gravitar_ppo.py --mode inspect --model-path ../../models/gravitar_ppo_best
```

---

## Seed Values

All experiments use the following random seeds for reproducibility:
- **Seed 42**: Primary seed
- **Seed 43**: Secondary seed
- **Seed 44**: Tertiary seed

Total: 3 seeds per configuration (as required by Challenge 3)

---

## Pointers to Logs and Figures

### PPO Logs Location
```
logs/gravitar_ppo/sweep/
├── exp_01_baseline/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
├── exp_02_high_lr/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
├── exp_03_low_lr/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
├── exp_04_short_horizon/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
├── exp_05_long_horizon/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
├── exp_06_high_entropy/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
├── exp_07_low_entropy/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
├── exp_08_high_lr_high_entropy/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
└── exp_09_long_horizon_high_entropy/
    ├── seed_42/
    ├── seed_43/
    └── seed_44/
```

### DQN Logs Location (for comparison)
```
logs/gravitar_dqn/sweep/
├── exp_01_baseline/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
├── exp_02_large_buffer/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
├── exp_03_high_lr/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
├── exp_04_medium_balance/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
└── exp_05_small_batch/
    ├── seed_42/
    ├── seed_43/
    └── seed_44/
```

### TensorBoard Commands
```bash
# View PPO logs
tensorboard --logdir logs/gravitar_ppo/sweep --port 6006

# View DQN logs
tensorboard --logdir logs/gravitar_dqn/sweep --port 6007
```

### Model Locations
- **Best PPO model**: `models/gravitar_ppo_best.zip`
- **Best DQN model**: `models/gravitar_dqn_best.zip`

### Analysis Documents
- **Comparative analysis template**: `comparative_analysis_dqn_vs_ppo.md`
- **IEEE paper template**: `ieee_paper_dqn_to_ppo_gravitar.md`

---

## 200-Word Comparative Summary: DQN vs PPO on Gravitar

This study compares Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) on ALE/Gravitar-v5 under identical computational budgets. PPO achieved superior final performance (smoothed mean reward: 374.7 vs 335.7), demonstrating better asymptotic performance and dramatically improved wall-clock efficiency (171 FPS vs 6 FPS, 28.5× faster). The performance difference can be attributed to PPO's on-policy learning with Generalized Advantage Estimation (GAE) and entropy bonus, which better captured Gravitar's continuous thrust requirements and maintained exploratory thrust patterns throughout training. DQN's off-policy replay provided efficient learning from rare successful episodes but suffered from severe computational overhead and policy rigidity following epsilon-decay. Action distribution analysis revealed PPO maintained balanced action usage with consistent thrust engagement, while DQN exhibited skewed distributions favoring horizontal rotation. These results suggest that on-policy methods are better suited for physics-based Atari games requiring sustained action sequences, with PPO's 28.5× throughput advantage making it the only viable algorithm for rigorous hyperparameter sweeps under laboratory time constraints.

---

## Hyperparameter Configurations

### PPO Best Configuration (exp_01_baseline)
```json
{
  "learning_rate": 2.5e-4,
  "n_steps": 1024,
  "batch_size": 128,
  "n_epochs": 4,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "clip_range": 0.2,
  "ent_coef": 0.01,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5
}
```

**Results**:
- Smoothed mean reward: 374.7062
- Training throughput: 171 FPS
- Steps to 200 reward: ~165,000
- Explained variance: 0.406 (peak: 0.625)
- Entropy loss: -1.9571
- Clip fraction: 0.336

### DQN Best Configuration (from Challenge 1)
```json
{
  "learning_rate": 1e-4,
  "buffer_size": 100000,
  "batch_size": 32,
  "gamma": 0.99,
  "train_freq": 4,
  "target_update_interval": 1000,
  "exploration_fraction": 0.167,
  "exploration_final_eps": 0.01
}
```

**Results**:
- Smoothed mean reward: 335.7081
- Training throughput: 6 FPS
- Steps to 200 reward: ~180,000
- Training loss: volatile (0.002-0.01)

---

## Implementation Verification

### PPO Required Components (All Implemented ✓)
- ✓ Clipped surrogate objective (clip_range parameter)
- ✓ Generalized Advantage Estimation (gae_lambda parameter)
- ✓ Entropy bonus (ent_coef parameter)
- ✓ Actor-Critic architecture (CnnPolicy)
- ✓ Mini-batch updates (batch_size parameter)
- ✓ Multiple epochs per rollout (n_epochs parameter)

### Preprocessing (Identical to Challenge 1 ✓)
- ✓ Grayscale conversion
- ✓ Resize to 84×84
- ✓ Frame skip (4 steps)
- ✓ Frame stacking (4 frames)
- ✓ Terminal-on-life-loss

### Experimental Protocol (Challenge 3 Requirements ✓)
- ✓ Fixed computational budget: 5,000,000 steps
- ✓ 3 random seeds per configuration (42, 43, 44)
- ✓ Identical preprocessing to Challenge 1
- ✓ Systematic hyperparameter search
- ✓ Logged metrics (TensorBoard)
- ✓ Reproducible runs

---

## Notes

- All experiments use Stable-Baselines3 library
- Environment: ALE/Gravitar-v5
- Preprocessing implemented via AtariWrapper from Stable-Baselines3
- Logs automatically include rollout/ep_rew_mean, train/loss, and custom metrics
- Comparative analysis and IEEE paper templates provided for post-experiment analysis
