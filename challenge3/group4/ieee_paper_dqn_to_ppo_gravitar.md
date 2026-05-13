# From Off-Policy to On-Policy: A Comparative Study of DQN and PPO on Atari Gravitar

**[Author Names]**
**Group 4**
**Universidad Distrital Francisco José de Caldas**
**Machine Learning Course**
**Prof. Carlos Andrés Sierra**

---

## Abstract

Deep Reinforcement Learning has achieved remarkable success in Atari games through value-based methods like Deep Q-Networks (DQN) and policy-based methods like Proximal Policy Optimization (PPO). This paper presents a comprehensive comparative study of both algorithms on ALE/Gravitar-v5, an Atari game characterized by complex physics, gravity mechanics, and precise control requirements. We implement and tune both algorithms under identical computational budgets (5 million environment steps) and preprocessing pipelines, enabling a fair algorithmic comparison. Our results show that [PPO/DQN] achieves [higher/lower] final performance with [better/worse] sample efficiency and [more/less] training stability. We analyze these differences through the lens of algorithmic properties: DQN's off-policy learning with experience replay versus PPO's on-policy rollouts with clipped policy updates. Our findings suggest that [on-policy/off-policy] methods are better suited for physics-based Atari games requiring continuous thrust and precise control, providing insights for algorithm selection in similar domains.

**Index Terms**—Reinforcement Learning, Deep Q-Network, Proximal Policy Optimization, Atari, Gravitar, On-Policy, Off-Policy

---

## I. Introduction

Atari games have served as a benchmark for Deep Reinforcement Learning (DRL) since the seminal work by Mnih et al. [1], demonstrating that deep neural networks can learn to play games from raw pixels. The Arcade Learning Environment (ALE) provides a standardized platform for comparing algorithms across diverse game mechanics, from simple reflex-based games to complex physics-based challenges.

Gravitar, one of the more challenging Atari games, presents unique difficulties that distinguish it from other ALE titles. The game features continuous gravity mechanics, requiring the agent to maintain thrust to counteract gravitational pull while navigating through underground caverns. Precise control is essential—small timing errors can lead to catastrophic failure. This physics-based nature makes Gravitar an ideal testbed for comparing different RL paradigms.

Deep Q-Networks (DQN) [1] revolutionized Atari RL through value-based off-policy learning, using experience replay to break correlation and a target network for stability. In contrast, Proximal Policy Optimization (PPO) [2] represents the policy gradient paradigm, using on-policy rollouts with a clipped surrogate objective to ensure stable policy improvements.

The central research question this study addresses is: **Under a fixed computational budget on a physics-based Atari game, does PPO converge faster, achieve higher performance, or exhibit different failure modes compared to DQN?** This question is motivated by the theoretical differences between the algorithms: DQN's off-policy nature allows efficient reuse of past experiences, while PPO's on-policy approach may better capture the temporal dependencies inherent in physics-based control.

Our contributions are:
1. A systematic implementation and hyperparameter tuning of both DQN and PPO on ALE/Gravitar-v5 under identical experimental conditions
2. A comprehensive empirical comparison focusing on sample efficiency, training stability, and final performance
3. An analysis of how algorithmic properties (off-policy vs on-policy, exploration strategies) interact with Gravitar's physics-based mechanics
4. Insights on the suitability of on-policy versus off-policy methods for physics-based Atari games

The remainder of this paper is organized as follows: Section II reviews related work, Section III describes our methodology, Section IV presents experimental results, Section V discusses the findings, and Section VI concludes.

---

## II. Related Work

### A. Deep Q-Networks and Value-Based Methods

Deep Q-Networks (DQN) [1] introduced the combination of deep convolutional neural networks with Q-learning, enabling end-to-end learning from raw pixels. Key innovations include experience replay to break temporal correlation and a target network to stabilize training. Subsequent improvements like Double DQN [3], Dueling DQN [4], and prioritized experience replay [5] further enhanced performance.

DQN's off-policy nature allows it to learn from past experiences efficiently, making it sample-efficient in many Atari domains. However, the ε-greedy exploration strategy can be suboptimal for tasks requiring sustained action sequences, as random actions disrupt coherent behavior.

### B. Policy Gradient Methods and PPO

Policy gradient methods [6] directly optimize the policy by gradient ascent on expected return. Vanilla policy gradients suffer from high variance, leading to the development of trust region methods like Trust Region Policy Optimization (TRPO) [7] and its approximation, Proximal Policy Optimization (PPO) [2].

PPO introduces a clipped surrogate objective that limits policy updates to a trust region, preventing catastrophic performance drops. The algorithm also incorporates Generalized Advantage Estimation (GAE) [8] for variance reduction and an entropy bonus to encourage exploration. PPO has become a default algorithm for many RL tasks due to its simplicity and robustness.

### C. On-Policy vs Off-Policy for Continuous Control

The distinction between on-policy and off-policy learning has been studied extensively in continuous control domains [9]. Off-policy methods like DQN can reuse data efficiently but may struggle with temporal credit assignment in tasks requiring sustained action sequences. On-policy methods like PPO learn from fresh trajectories, potentially better capturing temporal dependencies but at higher sample cost.

Recent work has compared these paradigms in various domains [10], but systematic comparisons on physics-based Atari games remain limited. Our study contributes to this gap by focusing on Gravitar, where continuous thrust and gravity create unique challenges.

---

## III. Methodology

### A. Environment: ALE/Gravitar-v5

Gravitar is an Atari game released in 1982 featuring the following mechanics:
- **Gravity**: Continuous downward force requiring sustained thrust
- **Fuel management**: Limited fuel resources requiring efficient thrust usage
- **Multi-level structure**: Multiple planets with different layouts and challenges
- **Precise control**: Small timing errors lead to mission failure
- **Enemy interactions**: Hostile terrain and enemies requiring avoidance strategies

The action space consists of 18 discrete actions: NOOP, FIRE, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT, UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE, UPRIGHTFIRE, UPLEFTFIRE, DOWNRIGHTFIRE, DOWNLEFTFIRE. The observation space consists of RGB frames (210×160 pixels) which we preprocess as described below.

### B. Preprocessing Pipeline

Both algorithms use identical preprocessing to ensure fair comparison:
1. **Grayscale conversion**: RGB frames converted to single-channel grayscale
2. **Resize**: Frames resized to 84×84 pixels
3. **Frame skipping**: Each action repeated for 4 frames, reducing computational load
4. **Frame stacking**: 4 consecutive frames stacked as input, providing motion information
5. **Terminal-on-life-loss**: Each life treated as a separate episode

This preprocessing follows standard Atari RL protocols [1] and is implemented using Stable-Baselines3's Atari wrappers.

### C. DQN Implementation

Our DQN implementation uses the Stable-Baselines3 library [11] with the following architecture:
- **Convolutional base**: Three convolutional layers (32, 64, 64 filters) with ReLU activations
- **Fully connected layers**: 512-unit hidden layer
- **Output layer**: Q-values for 18 actions

**Key hyperparameters** (after tuning):
- Learning rate: [value]
- Replay buffer size: [value]
- Batch size: [value]
- Target network update interval: [value]
- Exploration schedule: ε decays from [value] to [value] over [value] steps
- Discount factor γ: 0.99

DQN uses ε-greedy exploration, where the agent selects a random action with probability ε and the greedy action otherwise. ε decays linearly during training, balancing exploration and exploitation.

### D. PPO Implementation

Our PPO implementation also uses Stable-Baselines3 with an Actor-Critic architecture:
- **Shared convolutional base**: Same CNN structure as DQN
- **Actor head**: Policy network outputting action logits
- **Critic head**: Value network estimating state value V(s)

**Key hyperparameters** (after tuning):
- Learning rate: [value]
- Rollout horizon (n_steps): [value]
- Batch size: [value]
- Number of optimization epochs per rollout (n_epochs): [value]
- Discount factor γ: 0.99
- GAE parameter λ: [value]
- Clipping parameter ε: [value]
- Entropy coefficient: [value]
- Value function coefficient: [value]

PPO uses the clipped surrogate objective:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio and $\hat{A}_t$ is the GAE advantage estimate.

The total loss combines policy, value, and entropy terms:

$$L(\theta) = -L^{CLIP}(\theta) + c_1 L^{VF}(\theta) - c_2 L^{ENT}(\theta)$$

### E. Hyperparameter Search

We conducted systematic hyperparameter searches for both algorithms:

**DQN search space**:
- Learning rate: $\{1\times10^{-4}, 2\times10^{-4}, 5\times10^{-4}\}$
- Buffer size: $\{50k, 100k, 200k\}$
- Batch size: $\{32, 64\}$

**PPO search space** (following Challenge 3 guidelines):
- Learning rate: $\{1\times10^{-4}, 2.5\times10^{-4}, 5\times10^{-4}\}$
- Horizon (n_steps): $\{512, 1024, 2048\}$
- Entropy coefficient: $\{0.001, 0.01, 0.02\}$

Each configuration was evaluated with 3 random seeds (42, 43, 44) to capture variance. The best configuration for each algorithm was selected based on mean final performance across seeds.

---

## IV. Experimental Results

### A. Experimental Setup

**Computational budget**: 5 million environment steps per configuration
**Evaluation protocol**: Deterministic evaluation every [value] steps
**Metrics tracked**:
- Episode return
- Training loss (TD-error for DQN, policy/value/entropy losses for PPO)
- Action frequencies
- TensorBoard logs for visualization

**Hardware**: [Specify hardware used]

### B. Sample Efficiency

**Figure 1**: Learning curves (episode return vs. environment steps) for DQN and PPO

```
[Insert learning curve plot showing both algorithms with shaded std bands]
```

**Table I**: Sample efficiency metrics

| Metric | DQN | PPO |
|--------|-----|-----|
| Steps to 50% max performance | [value] | [value] |
| Steps to 75% max performance | [value] | [value] |
| Steps to 90% max performance | [value] | [value] |
| Initial learning rate (first 100k steps) | [value] | [value] |

**Analysis**: [Describe which algorithm learns faster, early learning differences]

### C. Training Stability

**Figure 2**: Loss curves for DQN (TD-error) and PPO (policy, value, entropy losses)

```
[Insert loss curves showing training stability]
```

**Table II**: Stability metrics across 3 seeds

| Metric | DQN | PPO |
|--------|-----|-----|
| Mean final score (± std) | [value] ± [value] | [value] ± [value] |
| Coefficient of variation | [value] | [value] |
| Min score across seeds | [value] | [value] |
| Max score across seeds | [value] | [value] |
| AUC (normalized) | [value] | [value] |

**Analysis**: [Describe which algorithm is more stable, variance patterns]

### D. Final Performance

**Figure 3**: Final performance comparison (box plots across seeds)

```
[Insert box plot showing final score distribution for both algorithms]
```

**Table III**: Final performance comparison

| Metric | DQN | PPO | p-value |
|--------|-----|-----|---------|
| Mean final score | [value] | [value] | [value] |
| Median final score | [value] | [value] | - |
| Best single run | [value] | [value] | - |
| Worst single run | [value] | [value] | - |

**Statistical test**: [t-test / Mann-Whitney U], p = [value]

**Analysis**: [Describe which algorithm performs better, statistical significance]

### E. Action Distribution Analysis

**Figure 4**: Action frequency distributions for DQN and PPO

```
[Insert bar chart showing action frequencies for both algorithms]
```

**Table IV**: Thrust action usage

| Metric | DQN | PPO |
|--------|-----|-----|
| Thrust action frequency | [value]% | [value]% |
| Average thrust streak length | [value] | [value] |
| Thrust burst pattern | [description] | [description] |

**Analysis**: [Describe how each algorithm uses thrust, implications for gravity compensation]

---

## V. Discussion

### A. Why [PPO/DQN] Performed Better

**[If PPO won]**:
PPO's superior performance can be attributed to several algorithmic properties:

1. **On-policy learning**: PPO learns from fresh trajectories, which may better capture the temporal dependencies in Gravitar's physics. Continuous thrust requires coherent action sequences over multiple timesteps—on-policy learning preserves these temporal correlations better than off-policy replay.

2. **Entropy bonus**: The entropy coefficient encourages stochastic exploration, preventing premature convergence to suboptimal deterministic policies. In Gravitar, where precise timing matters, maintaining action variability helps discover effective thrust patterns.

3. **Clipped updates**: The clipping mechanism prevents catastrophic policy drops, allowing more aggressive exploration without risk of performance collapse. This is particularly valuable in Gravitar where bad episodes can be highly punishing.

4. **Actor-Critic architecture**: The separate value head provides dense learning signals even when rewards are sparse, helping the agent learn value estimates for states that may not yield immediate rewards.

**[If DQN won]**:
DQN's superior performance can be attributed to:

1. **Experience replay**: The replay buffer allows efficient reuse of past experiences, making DQN more sample-efficient. In Gravitar, where fuel is limited, learning from past successful thrust patterns is crucial.

2. **Target network stability**: The periodic target network sync provides stable learning targets, preventing the oscillation that can occur in pure on-policy learning.

3. **ε-greedy exploration**: While simple, ε-greedy provides consistent exploration that may be sufficient for Gravitar's action space, where the optimal policy is relatively deterministic once learned.

### B. Algorithmic Properties and Gravitar's Physics

**Gravity compensation**:
- [DQN/PPO] showed [better/worse] ability to maintain sustained thrust
- The [off-policy/on-policy] nature [helped/hindered] learning continuous thrust patterns

**Precise control**:
- [DQN/PPO] exhibited [more/less] precise action timing
- The [exploration strategy] affected [positively/negatively] the development of precise control

**Fuel efficiency**:
- [DQN/PPO] achieved [higher/lower] fuel efficiency
- The [algorithmic property] influenced fuel management

### C. On-Policy vs Off-Policy for Physics-Based Games

Our results suggest that [on-policy/off-policy] methods are better suited for physics-based Atari games like Gravitar. This aligns with theoretical expectations:

**Arguments for on-policy (PPO)**:
- Physics-based control requires coherent action sequences over time
- On-policy learning preserves temporal correlations better
- Entropy bonus maintains exploration needed for discovering precise control patterns
- Clipped updates enable safe exploration in punishing environments

**Arguments for off-policy (DQN)**:
- Experience replay enables efficient learning from rare successful episodes
- Target network provides stability in complex dynamics
- Sample efficiency may be more important when fuel is limited

**Our findings**: [Summarize which paradigm worked better and why]

### D. Failure Modes

**DQN failure modes**:
- [Observed issues: e.g., getting stuck in local optima, insufficient exploration]

**PPO failure modes**:
- [Observed issues: e.g., policy collapse, insufficient sample efficiency]

### E. Limitations

1. **Single environment**: Results may not generalize to other physics-based Atari games
2. **Computational budget**: 5M steps may be insufficient for asymptotic performance
3. **Hyperparameter sensitivity**: Results may depend on specific hyperparameter choices
4. **Implementation details**: Different implementations may yield different results

---

## VI. Conclusion

We presented a comprehensive comparison of DQN and PPO on ALE/Gravitar-v5, a physics-based Atari game requiring continuous thrust and precise control. Under identical computational budgets and preprocessing, [PPO/DQN] achieved [better/worse] performance with [more/less] sample efficiency and [more/less] training stability.

Our key findings are:
1. [Algorithm] achieved higher final performance ([value] vs [value] mean score)
2. [Algorithm] demonstrated better sample efficiency ([value] vs [value] steps to target)
3. [Algorithm] showed greater stability (CV: [value] vs [value])
4. The [on-policy/off-policy] paradigm is [more/less] suitable for physics-based control

These results suggest that for physics-based Atari games requiring continuous thrust and precise timing, [on-policy/off-policy] methods offer advantages due to [reason]. The entropy bonus in PPO [helped/hindered] exploration of thrust patterns, while experience replay in DQN [helped/hindered] learning from rare successful episodes.

**Practical implications**: When applying RL to physics-based games, practitioners should consider [recommendation based on results]. The choice between on-policy and off-policy methods should be guided by the specific requirements of the task: [guidance].

**Future work** should extend this comparison to other physics-based Atari games (e.g., Solaris, Venture) and investigate hybrid approaches that combine the strengths of both paradigms. Additionally, more sophisticated exploration strategies tailored to physics-based control could further improve performance.

---

## References

[1] V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, vol. 518, no. 7540, pp. 529–533, 2015.

[2] J. Schulman et al., "Proximal policy optimization algorithms," arXiv preprint arXiv:1707.06347, 2017.

[3] H. Van Hasselt, A. Guez, and D. Silver, "Deep reinforcement learning with double q-learning," in AAAI, 2016.

[4] Z. Wang et al., "Dueling network architectures for deep reinforcement learning," in ICML, 2016.

[5] T. Schaul et al., "Prioritized experience replay," arXiv preprint arXiv:1511.05952, 2015.

[6] R. S. Sutton, D. McAllester, S. Singh, and Y. Mansour, "Policy gradient methods for reinforcement learning with function approximation," in NIPS, 2000.

[7] J. Schulman et al., "Trust region policy optimization," in ICML, 2015.

[8] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, "High-dimensional continuous control using generalized advantage estimation," arXiv preprint arXiv:1506.02438, 2015.

[9] M. Fujimoto, J. Hoof, and D. Meger, "Addressing function approximation error in actor-critic methods," in ICML, 2018.

[10] A. Hill et al., "Stable-baselines3: Reliable reinforcement learning implementations," arXiv preprint arXiv:2005.05519, 2020.

[11] A. Raffin et al., "Stable-baselines3: Reliable reinforcement learning implementations," Journal of Machine Learning Research, vol. 22, no. 268, pp. 1–8, 2021.

---

## Appendix

### A. Hyperparameter Configurations

**Best DQN configuration**:
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

**Best PPO configuration**:
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
  "vf_coef": 0.5
}
```

### B. Training Commands

```bash
# DQN training
python gravitar_dqn.py --mode train --experiment [exp_name] --model-path models/gravitar_dqn_best --seed 42

# PPO training
python gravitar_ppo.py --mode train --experiment [exp_name] --model-path models/gravitar_ppo_best --seed 42
```

### C. Reproducibility

All experiments use random seeds 42, 43, 44. Code and logs are available at: [repository URL]
