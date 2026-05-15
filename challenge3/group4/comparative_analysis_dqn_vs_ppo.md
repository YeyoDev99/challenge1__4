# Comparative Analysis: DQN vs PPO on ALE/Gravitar-v5

**Challenge 3 - Machine Learning Course**
**Group 4**
**Environment: ALE/Gravitar-v5**

---

## Executive Summary

*Brief summary of which algorithm performs better and why (to be filled after experiments)*

---

## 1. Experimental Setup

### 1.1 Computational Budget
- **Total environment steps**: 5,000,000 per configuration
- **Number of seeds**: 3 per configuration (42, 43, 44)
- **Evaluation protocol**: Deterministic evaluation at fixed intervals

### 1.2 DQN Configuration (Challenge 1)
- **Best configuration**: [To be filled after sweep]
- **Learning rate**: [value]
- **Buffer size**: [value]
- **Batch size**: [value]
- **Gamma**: [value]
- **Exploration schedule**: [value]

### 1.3 PPO Configuration (Challenge 3)
- **Best configuration**: [To be filled after sweep]
- **Learning rate**: [value]
- **Horizon (n_steps)**: [value]
- **Batch size**: [value]
- **N epochs**: [value]
- **Gamma**: [value]
- **GAE lambda**: [value]
- **Clip range**: [value]
- **Entropy coefficient**: [value]

### 1.4 Preprocessing (Identical for Both)
- Grayscale conversion
- Resize to 84×84
- Frame skip: 4
- Frame stacking: 4 frames
- Terminal-on-life-loss

---

## 2. Sample Efficiency Analysis

### 2.1 Learning Curves

**Metric**: Episode return vs. environment steps

| Algorithm | Steps to reach 50% of max performance | Steps to reach 75% of max performance | Steps to reach 90% of max performance |
|-----------|--------------------------------------|--------------------------------------|--------------------------------------|
| DQN       | [value]                              | [value]                              | [value]                              |
| PPO       | [value]                              | [value]                              | [value]                              |

### 2.2 Sample Efficiency Comparison

**Analysis**:
- Which algorithm reaches target scores faster?
- How does the learning curve shape differ?
- Early learning performance comparison (first 500k steps)

**Key Findings**:
- [To be filled with data]

---

## 3. Training Stability Analysis

### 3.1 Variance Across Seeds

| Algorithm | Mean Final Score (± std) | Coefficient of Variation | Min Score | Max Score |
|-----------|--------------------------|-------------------------|-----------|-----------|
| DQN       | [value] ± [value]        | [value]                 | [value]   | [value]   |
| PPO       | [value] ± [value]        | [value]                 | [value]   | [value]   |

### 3.2 Training Stability Metrics

**Area Under Learning Curve (AUC)**:
- DQN: [value]
- PPO: [value]

**Variance during training**:
- DQN: [analysis of loss curves, reward variance]
- PPO: [analysis of policy loss, value loss, entropy]

### 3.3 Stability Comparison

**Analysis**:
- Which algorithm shows more consistent performance across seeds?
- How do loss curves compare (DQN TD-error vs PPO policy/value/entropy losses)?
- Presence of catastrophic forgetting or performance collapse?

**Key Findings**:
- [To be filled with data]

---

## 4. Final Performance Comparison

### 4.1 Asymptotic Performance

| Algorithm | Mean Final Score | Median Final Score | Best Single Run | Worst Single Run |
|-----------|------------------|-------------------|-----------------|------------------|
| DQN       | [value]          | [value]           | [value]         | [value]          |
| PPO       | [value]          | [value]           | [value]         | [value]          |

### 4.2 Statistical Significance

**Hypothesis Test**: Mean score comparison
- Test: [t-test / Mann-Whitney U]
- p-value: [value]
- Conclusion: [significant / not significant]

### 4.3 Performance Summary

**Analysis**:
- Which algorithm achieves higher final scores?
- Is the difference statistically significant?
- How does performance scale with computational budget?

**Key Findings**:
- [To be filled with data]

---

## 5. Qualitative Analysis: Gravitar Physics and Control

### 5.1 Gravitar-Specific Challenges

Gravitar presents unique challenges:
- **Gravity mechanics**: Continuous thrust required to counteract gravity
- **Precise control**: Small adjustments needed for navigation
- **Multi-stage gameplay**: Different zones with different physics
- **Fuel management**: Limited resources require efficient action sequences

### 5.2 Algorithm-Specific Behavior

#### DQN Characteristics
- **Exploration**: ε-greedy with decay schedule
- **Action selection**: Q-value maximization
- **Memory**: Off-policy replay buffer
- **Update pattern**: Periodic target network sync

**Expected behavior for Gravitar**:
- [Analysis based on results]
- How does ε-greedy exploration affect thrust control?
- Does replay buffer help learn gravity mechanics?
- Target network sync frequency impact on stability

#### PPO Characteristics
- **Exploration**: Entropy bonus encourages stochastic policy
- **Action selection**: Stochastic policy with clipped updates
- **Memory**: On-policy rollouts
- **Update pattern**: Multiple epochs per rollout

**Expected behavior for Gravitar**:
- [Analysis based on results]
- How does entropy bonus affect thrust variability?
- Does on-policy learning help with precise control?
- Clipped ratio impact on policy stability for continuous thrust

### 5.3 Physics Handling Comparison

**Thrust Action Distribution**:
- DQN thrust frequency: [percentage of thrust actions]
- PPO thrust frequency: [percentage of thrust actions]
- Comparison: [which uses thrust more appropriately]

**Gravity Compensation**:
- DQN: [analysis of how well it compensates gravity]
- PPO: [analysis of how well it compensates gravity]

**Precision Control**:
- DQN: [analysis of action granularity]
- PPO: [analysis of action granularity]

**Key Findings**:
- [To be filled with data]

---

## 6. Action Distribution Analysis

### 6.1 Overall Action Distribution

**DQN Action Frequencies** (across all episodes):
| Action | Frequency | Description |
|--------|-----------|-------------|
| NOOP   | [value]%  | No operation |
| FIRE   | [value]%  | Fire/thrust |
| UP     | [value]%  | Thrust up |
| RIGHT  | [value]%  | Rotate right |
| LEFT   | [value]%  | Rotate left |
| DOWN   | [value]%  | (if applicable) |

**PPO Action Frequencies** (across all episodes):
| Action | Frequency | Description |
|--------|-----------|-------------|
| NOOP   | [value]%  | No operation |
| FIRE   | [value]%  | Fire/thrust |
| UP     | [value]%  | Thrust up |
| RIGHT  | [value]%  | Rotate right |
| LEFT   | [value]%  | Rotate left |
| DOWN   | [value]%  | (if applicable) |

### 6.2 Temporal Action Patterns

**Action sequence analysis**:
- DQN average action streak length: [value]
- PPO average action streak length: [value]
- Thrust burst patterns: [comparison]

### 6.3 Action Distribution Insights

**Analysis**:
- Which algorithm shows more diverse action usage?
- How does action distribution relate to performance?
- Are there systematic differences in how each algorithm uses thrust?

**Key Findings**:
- [To be filled with data]

---

## 7. Hyperparameter Sensitivity

### 7.1 DQN Hyperparameter Impact

| Hyperparameter | Tested Values | Best Value | Impact on Performance |
|----------------|---------------|------------|----------------------|
| Learning rate  | [values]      | [value]    | [analysis]           |
| Buffer size    | [values]      | [value]    | [analysis]           |
| Batch size     | [values]      | [value]    | [analysis]           |

### 7.2 PPO Hyperparameter Impact

| Hyperparameter | Tested Values | Best Value | Impact on Performance |
|----------------|---------------|------------|----------------------|
| Learning rate  | [values]      | [value]    | [analysis]           |
| n_steps        | [values]      | [value]    | [analysis]           |
| ent_coef       | [values]      | [value]    | [analysis]           |

### 7.3 Sensitivity Comparison

**Analysis**:
- Which algorithm is more sensitive to hyperparameter choices?
- Which hyperparameters have the largest impact for each algorithm?
- Practical implications for tuning

**Key Findings**:
- [To be filled with data]

---

## 8. Computational Efficiency

### 8.1 Wall-Clock Time

| Algorithm | Time per 1M steps (CPU) | Time per 1M steps (GPU) | Total training time (5M steps) |
|-----------|------------------------|------------------------|--------------------------------|
| DQN       | [value]                | [value]                | [value]                        |
| PPO       | [value]                | [value]                | [value]                        |

### 8.2 Memory Usage

| Algorithm | Peak memory usage | Buffer/rollout size |
|-----------|-------------------|---------------------|
| DQN       | [value]           | [value]             |
| PPO       | [value]           | [value]             |

### 8.3 Computational Efficiency Comparison

**Analysis**:
- Which algorithm is more computationally efficient?
- Trade-off between sample efficiency and wall-clock time
- Memory requirements comparison

**Key Findings**:
- [To be filled with data]

---

## 9. Algorithmic Discussion

### 9.1 Why DQN Performs as It Does

**Strengths for Gravitar**:
- Off-policy learning with replay buffer
- Stable target network
- ε-greedy exploration

**Weaknesses for Gravitar**:
- [Analysis based on results]

**Failure Modes**:
- [Observed issues]

### 9.2 Why PPO Performs as It Does

**Strengths for Gravitar**:
- On-policy learning with fresh trajectories
- Entropy bonus for exploration
- Clipped updates for stability
- Actor-critic architecture

**Weaknesses for Gravitar**:
- [Analysis based on results]

**Failure Modes**:
- [Observed issues]

### 9.3 Theoretical vs. Empirical

**Comparison with theoretical expectations**:
- How do results align with PPO theory (sample efficiency, stability)?
- Does on-policy vs off-policy matter for Gravitar?
- Role of entropy bonus in continuous control tasks

**Key Insights**:
- [To be filled with data]

---

## 10. Conclusions

### 10.1 Summary of Findings

**Sample Efficiency**:
- [Which algorithm is more sample efficient?]
- [Quantitative comparison]

**Stability**:
- [Which algorithm is more stable?]
- [Quantitative comparison]

**Final Performance**:
- [Which algorithm achieves higher scores?]
- [Statistical significance]

### 10.2 Algorithm Recommendation for Gravitar

**Recommended algorithm**: [DQN / PPO]

**Rationale**:
- [Key reasons based on analysis]

**Hyperparameter recommendations**:
- [Best configuration for future work]

### 10.3 Broader Implications

**General insights for continuous control in Atari**:
- [What does this comparison teach us about RL for physics-based games?]

**Limitations**:
- [Study limitations]

**Future work**:
- [Suggested improvements or extensions]

---

## 11. Appendix: Training Commands

### 11.1 DQN Training Commands

```bash
# Best DQN configuration
python gravitar_dqn.py --mode train --experiment [exp_name] --model-path models/gravitar_dqn_best --seed 42

# DQN sweep
python gravitar_dqn.py --mode sweep --sweep-file sweep_configs.json --model-path models/gravitar_dqn_best
```

### 11.2 PPO Training Commands

```bash
# Best PPO configuration
python gravitar_ppo.py --mode train --experiment [exp_name] --model-path models/gravitar_ppo_best --seed 42

# PPO sweep
python gravitar_ppo.py --mode sweep --sweep-file sweep_configs_ppo.json --model-path models/gravitar_ppo_best
```

### 11.3 TensorBoard Commands

```bash
# View DQN logs
tensorboard --logdir logs/gravitar_dqn/sweep

# View PPO logs
tensorboard --logdir logs/gravitar_ppo/sweep
```

---

## 12. Appendix: Data Files

### 12.1 Log Locations

**DQN logs**: `logs/gravitar_dqn/sweep/`
**PPO logs**: `logs/gravitar_ppo/sweep/`

### 12.2 Model Locations

**Best DQN model**: `models/gravitar_dqn_best.zip`
**Best PPO model**: `models/gravitar_ppo_best.zip`

---

*This template should be filled with actual experimental data after completing the DQN and PPO training sweeps.*
