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

*To be filled after completing experiments*

[This section should contain a concise 200-word summary comparing DQN and PPO performance on ALE/Gravitar-v5. The summary should address:]

- Which algorithm achieved better final performance
- Sample efficiency differences
- Training stability comparison
- Key algorithmic differences that explain the results
- Implications for physics-based Atari games

*Template for summary:*

This study compares Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) on ALE/Gravitar-v5 under identical 5M-step budgets. [PPO/DQN] achieved superior final performance (mean score: [value] vs [value]), demonstrating [better/worse] sample efficiency (reaching target in [value] vs [value] steps) and [more/less] training stability (CV: [value] vs [value]). The performance difference can be attributed to [key algorithmic factor]: PPO's on-policy learning with entropy bonus better captured Gravitar's continuous thrust requirements, while DQN's off-policy replay provided efficient learning from rare successful episodes. Action distribution analysis revealed [key finding about thrust usage]. These results suggest that [on-policy/off-policy] methods are better suited for physics-based Atari games requiring sustained action sequences, though both algorithms face challenges with Gravitar's precise control demands. The entropy bonus in PPO proved particularly valuable for maintaining exploratory thrust patterns, while DQN's ε-greedy exploration sometimes disrupted coherent control sequences.

---

## Hyperparameter Configurations

### PPO Best Configuration (to be determined after sweep)
```json
{
  "learning_rate": [value],
  "n_steps": [value],
  "batch_size": [value],
  "n_epochs": [value],
  "gamma": 0.99,
  "gae_lambda": [value],
  "clip_range": [value],
  "ent_coef": [value],
  "vf_coef": 0.5,
  "max_grad_norm": 0.5
}
```

### DQN Best Configuration (from Challenge 1)
```json
{
  "learning_rate": [value],
  "buffer_size": [value],
  "batch_size": [value],
  "gamma": 0.99,
  "train_freq": [value],
  "target_update_interval": [value],
  "exploration_fraction": [value],
  "exploration_final_eps": [value]
}
```

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
