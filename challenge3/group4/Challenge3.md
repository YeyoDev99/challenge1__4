

## Machine Learning — Challenge 3
Proximal Policy Optimization for Atari (PPO)
Algorithmic Comparison with Challenge 1
Prof.  Carlos Andrés Sierra, M.Sc.
## Full-time Adjunct Professor
## Computer Engineering Program
School of Engineering
Universidad Distrital Francisco José de Caldas
## Overview
This document describes Challenge 3 for the Machine Learning course. It is a direct
extension of Challenge 1, where each group implemented and tuned a Deep Q-Network
(DQN) agent on a specific Atari ALE environment.  In Challenge 3, each group works
on the same environment as in Challenge 1 but now implements a Proximal Policy
Optimization (PPO) agent, conducts a comparable set of experiments, and produces a
side-by-side empirical comparison between the two algorithms.
The central research question this challenge asks students to answer is:
“Under a fixed computational budget and on the same environment, does PPO
converge faster, reach higher performance, or exhibit different failure modes com-
pared to the DQN agent designed in Challenge 1? Why?”
Challenge objective
Groups will:
- Implement a PPO agent with clipped surrogate objective, Generalised Advantage Es-
timation (GAE), and shared or separate actor/critic network for the ALE environment
assigned in Challenge 1.
- Conduct a hyperparameter search for the PPO agent analogous to the search performed
in Challenge 1, following the same experimental protocol (fixed seeds, repeated runs,
logged metrics).
Carlos Andrés Sierra, Computer Engineer, M.Sc. in Computer Engineering, Full-time Adjunct Professor
at Universidad Distrital Francisco José de Caldas.
Any comment or concern about this document can be sent to: cavirguezs@udistrital.edu.co.

## ML CHALLENGE 3 — PPO VS DQN2
- Directly compare the best PPO run and the best DQN run from Challenge 1 using
equivalent metrics: sample efficiency (reward vs. environment steps), final asymptotic
performance, and training stability (variance across seeds).
- Produce a scientific report in IEEE format that extends the Challenge 1 paper with
PPO results and a rigorous algorithmic comparison.
Environments - same assignment as Challenge 1
Each group works on the same ALE game assigned in Challenge 1. The table below
is reproduced for reference. Use the same ALE version identifier as before so that DQN and
PPO results are directly comparable.
- Group 1 — ALE/MontezumaRevenge-v5: very sparse rewards; long-term plan-
ning and room memory required.
- Group 2 — ALE/Pitfall-v5:  sparse/negative rewards; delicate action sequences
needed to score.
- Group 3 — ALE/PrivateEye-v5:  extremely sparse rewards;  complex multi-
objective structure.
- Group 4 — ALE/Gravitar-v5: challenging physics; precise control under gravity.
- Group 5 — ALE/Solaris-v5: multi-stage dynamics; varied objectives across levels.
- Group 6 — ALE/Venture-v5: sparse positive signals; noisy action space early.
- Group 7 — ALE/MsPacman-v5: dense moving objects; partial observability; reac-
tive planning.
- Group 8 — ALE/Phoenix-v5: fast action dynamics; non-trivial reward structure.
Preprocessing must be identical to Challenge 1: grayscale, resize to 84 ×
84, frame-stack 4, frame-skip 4.  Keeping preprocessing constant ensures any observed
performance difference is attributable to the algorithm, not to environment configuration.
PPO algorithm - required elements
At minimum, the PPO implementation must include:
- On-policy rollout collection for T environment steps (horizon).
- Advantage estimation using Generalised Advantage Estimation (GAE) with parameters
γ and λ.
- Clipped surrogate objective:  L
## CLIP
## =E
h
min
## 
r
t
## (θ)
## ˆ
## A
t
, clip(r
t
## (θ), 1−ε, 1+ε)
## ˆ
## A
t
## i
where r
t
(θ) = π
θ
## (a
t
## |s
t
## )/π
θ
old
## (a
t
## |s
t
## ).
- Value-function loss: L
## VF
## =E
## 
## (V
θ
## (s
t
## )− R
t
## )
## 2
## 
## .

## ML CHALLENGE 3 — PPO VS DQN3
- Entropy bonus: L
## ENT
=E[H(π
θ
## (·|s
t
))] to encourage exploration.
- Combined loss: L =−L
## CLIP
+ c
## 1
## L
## VF
− c
## 2
## L
## ENT
## .
- Multiple mini-batch epochs per horizon (typically K = 4–10).
- Optional (but encouraged): gradient norm clipping; separate actor and critic networks
or a shared convolutional backbone with dual heads.
Groups may use the reference PPO implementation discussed in course lectures as a
starting point but must adapt and instrument it for the Atari observation space (convolu-
tional feature extractor, discrete action head).
Suggested hyperparameter search space for PPO
- Learning rate: 1× 10
## −4
## , 2.5× 10
## −4
## , 5× 10
## −4
## .
- Horizon (T steps per rollout): 512; 1 024; 2 048.
- Number of epochs per horizon (K): 4; 6; 10.
- Mini-batch size: 64; 128; 256.
- Discount factor γ: 0.99; 0.995.
- GAE λ: 0.90; 0.95; 0.97.
- Clip epsilon ε: 0.1; 0.2.
- Entropy coefficient c
## 2
## : 0.001; 0.01; 0.02.
- Value-loss coefficient c
## 1
## : 0.5; 1.0.
- Gradient norm clip: 0.5; 1.0; none.
Apply the same exploration strategy chosen in Challenge 1 (or justify a different one)
and ensure each condition is repeated for at least three independent random seeds.
Comparison methodology
The following protocol must be followed so that the DQN vs. PPO comparison is fair:
- Budget parity: define a fixed number of environment steps (e.g., 5 000 000) and
train both agents to that exact budget. Report learning curves sampled at the same
step intervals.
- Identical preprocessing: same grayscale/resize/stack/skip as Challenge 1.
- Identical evaluation protocol: at fixed intervals, pause training and run 10 de-
terministic evaluation episodes (epsilon = 0 for DQN; greedy policy for PPO). Report
mean and standard deviation of episode return.

## ML CHALLENGE 3 — PPO VS DQN4
- Metrics to report:
- Learning curve: episode return vs. environment steps.
- Sample efficiency: steps needed to reach a target score threshold (choose a mean-
ingful threshold, e.g. 50 or 100 points above baseline).
- Final performance: mean ± std over 3 seeds at the end of training.
- Training stability: area under the learning curve (AUC) normalised by total
steps.
- Wall-clock time (if compute conditions are comparable).
- Analysis: discuss which algorithm converges faster, which is more stable, and which
achieves higher asymptotic performance.  Explain these differences in terms of the
algorithmic properties of DQN (off-policy, replay buffer, epsilon-greedy) vs. PPO (on-
policy, rollouts, clipped ratio, entropy).
## Deliverables
- Repository folder: Add a challenge3/group<k>/ folder to the same GitHub repos-
itory used for Challenge 1. Include all PPO source code, a README.md with reproducible
run instructions, and logging artifacts.
- Extended IEEE paper: An 8-page IEEE conference paper extending the Chal-
lenge 1 paper.  The paper must include all DQN results from Challenge 1 and
the new PPO results, a comparison section, and updated conclusions.  Submit as
challenge3_group<k>_paper.pdf.
- Checklist: A short CHECKLIST.md inside the repository folder containing the exact
training commands for the best PPO run, seed values, pointers to logs/figures, and a
200-word comparative summary of DQN vs PPO.
Evaluation criteria
- Implementation correctness (25%):  PPO algorithm is correctly implemented
(clipped objective, GAE, entropy bonus, mini-batch updates).
- Experimental rigour (30%): hyperparameter search is systematic, variance is re-
ported, runs are reproducible.
- Comparison quality (30%): fair protocol, meaningful metrics, and insightful anal-
ysis that links algorithmic properties to observed behaviour.
- Presentation and writing (15%): quality of the IEEE extension (clarity, figures,
tables, citations, and updated conclusions).

## ML CHALLENGE 3 — PPO VS DQN5
Per-game starter guides
The following skeletons share a common PPO loop structure. Each snippet highlights
the environment setup and per-game-specific considerations students should investigate.
The shared preprocessing wrapper shown below applies to all groups:
Listing 1: Shared preprocessing wrapper (all groups)
1  import  gymnasium  as gym
2  import  numpy as np
3  from  gymnasium.wrappers  import (
4       GrayscaleObservation ,
5       ResizeObservation ,
6       FrameStackObservation ,
7       AtariPreprocessing ,
## 8  )
## 9
10  def  make_env(env_id: str , seed: int = 0):
11       """Build a pre -processed \texttt{ALE} environment  compatible  with \
texttt{PPO}."""
12       env = gym.make(env_id , render_mode=None)
13       env = AtariPreprocessing(
14            env ,
15            noop_max =30,
16            frame_skip =4,
17            screen_size =84,
18            grayscale_obs=True ,
19            scale_obs=True ,         # pixel  values  in [0, 1]
20            grayscale_newaxis=True ,
## 21       )
22       env = FrameStackObservation(env , 4)    # stack 4 frames
23       env.reset(seed=seed)
24       return  env
Listing 2: Shared convolutional actor-critic backbone (all groups)
1  import  torch
2  import  torch.nn as nn
## 3
4  class  AtariActorCritic(nn.Module):
5       """Shared  CNN  backbone  with  separate  actor  and  critic  heads."""
## 6
7       def  __init__(self , n_actions: int):
8            super().__init__ ()
9            # Input: (batch , 4, 84, 84) - 4 stacked  greyscale  frames
10            self.cnn = nn.Sequential(
11                 nn.Conv2d(4, 32,  kernel_size =8, stride =4), nn.ReLU(),
12                 nn.Conv2d (32, 64,  kernel_size =4, stride =2), nn.ReLU(),
13                 nn.Conv2d (64, 64,  kernel_size =3, stride =1), nn.ReLU(),
14                 nn.Flatten (),
## 15            )
16            cnn_out = 64 * 7 * 7   # 3136
## 17
18            self.actor = nn.Sequential(
19                 nn.Linear(cnn_out , 512), nn.ReLU(),
20                 nn.Linear (512,  n_actions),

## ML CHALLENGE 3 — PPO VS DQN6
## 21            )
22            self.critic = nn.Sequential(
23                 nn.Linear(cnn_out , 512), nn.ReLU(),
24                 nn.Linear (512, 1),
## 25            )
## 26
27       def  forward(self , x):
28            # x: (batch , 4, 84, 84), dtype  float32
29            feats = self.cnn(x)
30            return  self.actor(feats), self.critic(feats).squeeze (-1)
Listing 3: Minimal PPO training loop (all groups - adapt as needed)
1  import  torch
2  import  torch.optim as  optim
3  from  torch.distributions  import  Categorical
## 4
5  def  train_ppo(env_id , total_steps =5_000_000 , horizon =1024 ,
6                   n_epochs=4,  batch_size =128, lr=2.5e-4,
7                   gamma =0.99 ,  gae_lambda =0.95 ,
8                   clip_eps =0.2,  ent_coef =0.01 ,  vf_coef =0.5,
9                   max_grad_norm =0.5,  seed =42):
## 10
11       env = make_env(env_id , seed=seed)
12       n_actions = env.action_space.n
13       model = AtariActorCritic(n_actions).to("cuda" if  torch.cuda.
is_available () else "cpu")
14       optimizer = optim.Adam(model.parameters (), lr=lr)
## 15
16       obs , _ = env.reset()
17       episode_return , all_returns = 0.0, []
## 18
19       for  global_step  in  range(0, total_steps , horizon):
20            # --- rollout  collection  ---
21            obs_buf , act_buf , logp_buf , rew_buf , done_buf , val_buf = [], [],
## [], [], [], []
## 22
23            for _ in  range(horizon):
24                 obs_t = torch.tensor(obs , dtype=torch.float32).unsqueeze (0)
25                 with  torch.no_grad ():
26                     logits , value = model(obs_t)
27                 dist = Categorical(logits=logits)
28                 action = dist.sample ()
## 29
30                 obs_buf.append(obs_t.squeeze (0))
31                 act_buf.append(action)
32                 logp_buf.append(dist.log_prob(action))
33                 val_buf.append(value.squeeze ())
## 34
35                 obs , reward , terminated , truncated , _ = env.step(action.item
## ())
36                 done = terminated  or  truncated
37                 rew_buf.append(reward)
38                 done_buf.append(done)
39                 episode_return  +=  reward
## 40

## ML CHALLENGE 3 — PPO VS DQN7
41                 if done:
42                     all_returns.append(episode_return)
43                      episode_return = 0.0
44                     obs , _ = env.reset()
## 45
46            # --- compute  GAE  advantages  ---
47            with  torch.no_grad ():
48                 obs_t = torch.tensor(obs , dtype=torch.float32).unsqueeze (0)
49                 _, next_val = model(obs_t)
50            advantages , returns = compute_gae(
51                 rew_buf , val_buf , done_buf , next_val.item(), gamma ,
gae_lambda
## 52            )
## 53
54            # --- policy  updates (K epochs) ---
55            obs_t    = torch.stack(obs_buf)
56            act_t    = torch.stack(act_buf)
57            logp_t   = torch.stack(logp_buf).detach ()
58            adv_t    = (advantages  - advantages.mean()) / (advantages.std() +
## 1e-8)
59            ret_t    = returns
## 60
61            idx = torch.randperm(horizon)
62            for _ in  range(n_epochs):
63                 for  start in  range(0, horizon , batch_size):
64                     mb = idx[start:start + batch_size]
65                     logits , val_new = model(obs_t[mb])
66                     dist_new = Categorical(logits=logits)
67                     logp_new = dist_new.log_prob(act_t[mb])
68                     entropy   = dist_new.entropy ().mean()
69                     ratio     = (logp_new  - logp_t[mb]).exp()
## 70
71                     surr1 = ratio * adv_t[mb]
72                     surr2 = ratio.clamp(1 - clip_eps , 1 + clip_eps) * adv_t[
mb]
73                     loss_pi   = -torch.min(surr1 , surr2).mean()
74                     loss_vf   = (( val_new  - ret_t[mb]) ** 2).mean()
75                     loss      = loss_pi + vf_coef * loss_vf  - ent_coef *
entropy
## 76
77                     optimizer.zero_grad ()
78                     loss.backward ()
79                     nn.utils.clip_grad_norm_(model.parameters (),
max_grad_norm)
80                     optimizer.step()
## 81
82            if len(all_returns) % 10 == 0:
83                 print(f"step={ global_step}   mean_ret ={np.mean(all_returns
## [ -100:]) :.1f}")
## 84
85       env.close()
86       return  model , all_returns

## ML CHALLENGE 3 — PPO VS DQN8
Group 1 — ALE/MontezumaRevenge-v5
Key challenge: Reward is extremely sparse.  A standard PPO agent will rarely
receive a non-zero reward signal. Students should investigate: (a) using a large entropy
coefficient to promote exploration, (b) normalising rewards and advantages carefully, and
(c) studying whether PPO finds any reward compared to DQN.
Listing 4: Group 1 starter — Montezuma’s Revenge
1  # Suggested  starting  call  for  Group 1
2  # Increase  entropy  to  drive  exploration  in  sparse  reward  setting
3  model , returns = train_ppo(
4       env_id       = "ALE/MontezumaRevenge -v5",
5       total_steps = 5_000_000 ,
6       horizon      = 2048,
7       n_epochs     = 4,
8       batch_size   = 64,
9       lr            = 2.5e-4,
10       gamma         = 0.99,
11       gae_lambda   = 0.95,
12       clip_eps     = 0.1,
13       ent_coef     = 0.02,    # higher  entropy  for  sparse  exploration
14       seed          = 42,
## 15  )
16  # NOTE: Track  how  often  the  agent  enters  Room 1.
17  # Compare  with \texttt{DQN}’s first -room  entry  statistics  from  Challenge
## 1.
Group 2 — ALE/Pitfall-v5
Key challenge: Many actions cause immediate negative rewards. PPO’s on-policy
nature means it will unlearn risky behaviours quickly, but it may also avoid entire regions
of the level. Students should investigate whether PPO discovers any positive reward, and
if not, document why and compare failure mode with DQN’s from Challenge 1.
Listing 5: Group 2 starter — Pitfall!
1  model , returns = train_ppo(
2       env_id       = "ALE/Pitfall -v5",
3       total_steps = 5_000_000 ,
4       horizon      = 1024,
5       n_epochs     = 6,
6       batch_size   = 128,
7       lr            = 1e-4,
8       gamma         = 0.995 ,    # longer  horizon  helps in  sparse  settings
9       gae_lambda   = 0.97,
10       clip_eps     = 0.2,
11       ent_coef     = 0.01,
12       seed          = 42,
## 13  )
14  # NOTE: Log  the  fraction  of  episodes  with  positive  return.
15  # Track  the  average  score  per  episode  vs. env  steps.

## ML CHALLENGE 3 — PPO VS DQN9
Group 3 — ALE/PrivateEye-v5
Key challenge: One of the hardest Atari games. Rewards are so sparse that most
agents score 0 throughout training. Students should document the training collapse, com-
pare reward histograms from PPO and DQN, and reason about whether on-policy vs. off-
policy matters here.
Listing 6: Group 3 starter — Private Eye
1  model , returns = train_ppo(
2       env_id       = "ALE/PrivateEye -v5",
3       total_steps = 5_000_000 ,
4       horizon      = 2048,
5       n_epochs     = 4,
6       batch_size   = 64,
7       lr            = 2.5e-4,
8       gamma         = 0.99,
9       gae_lambda   = 0.95,
10       clip_eps     = 0.1,
11       ent_coef     = 0.02,
12       seed          = 42,
## 13  )
14  # NOTE: Record a histogram  of per -episode  rewards  across  all  runs.
15  # Measure  and  report  the  proportion  of non -zero -reward  episodes.
Group 4 — ALE/Gravitar-v5
Key challenge: Continuous thrust and gravity require precise, sustained control.
PPO’s entropy bonus can maintain exploratory behaviour while the clipped ratio prevents
policy collapse after a bad episode. Students should sweep entropy coefficient and horizon.
Listing 7: Group 4 starter — Gravitar
1  model , returns = train_ppo(
2       env_id       = "ALE/Gravitar -v5",
3       total_steps = 5_000_000 ,
4       horizon      = 1024,
5       n_epochs     = 6,
6       batch_size   = 128,
7       lr            = 2.5e-4,
8       gamma         = 0.99,
9       gae_lambda   = 0.95,
10       clip_eps     = 0.2,
11       ent_coef     = 0.01,
12       seed          = 42,
## 13  )
14  # NOTE: Track  the  distribution  of  thrust  actions  to see if the  agent
15  # learns  to use  the  engine  at all. Compare  action  histograms  DQN vs. PPO.
Group 5 — ALE/Solaris-v5
Key challenge: Multi-stage game with varying objectives. A longer horizon helps
PPO capture the multi-step structures needed to navigate between stages. Students should

## ML CHALLENGE 3 — PPO VS DQN10
compare whether DQN’s experience replay or PPO’s longer trajectory windows work better
here.
Listing 8: Group 5 starter — Solaris
1  model , returns = train_ppo(
2       env_id       = "ALE/Solaris -v5",
3       total_steps = 5_000_000 ,
4       horizon      = 2048,
5       n_epochs     = 4,
6       batch_size   = 128,
7       lr            = 2.5e-4,
8       gamma         = 0.995 ,    # long -horizon  discounting
9       gae_lambda   = 0.97,
10       clip_eps     = 0.2,
11       ent_coef     = 0.01,
12       seed          = 42,
## 13  )
14  # NOTE: Log  whether  the  agent  starts  the  second in -game  stage.
15  # Maximum  achieved in -game  stage  per run is a useful  comparison  metric.
Group 6 — ALE/Venture-v5
Key challenge: Enemies and sparse item rewards. PPO tends to be conservative
once a small positive signal is found; this may prevent further exploration. Students should
investigate whether a higher entropy coefficient or a smaller clip epsilon helps PPO explore
deeper into dungeons.
Listing 9: Group 6 starter — Venture
1  model , returns = train_ppo(
2       env_id       = "ALE/Venture -v5",
3       total_steps = 5_000_000 ,
4       horizon      = 1024,
5       n_epochs     = 6,
6       batch_size   = 64,
7       lr            = 1e-4,
8       gamma         = 0.99,
9       gae_lambda   = 0.95,
10       clip_eps     = 0.1,    # more  conservative  updates
11       ent_coef     = 0.02,
12       seed          = 42,
## 13  )
14  # NOTE: Track  how  many  dungeons  the  agent  enters  per run.
15  # Compare  with \texttt{DQN}’s exploration  depth  from  Challenge  1.
Group 7 — ALE/MsPacman-v5
Key challenge:  Denser rewards than other games in this list; this is the most
tractable environment for PPO. Students should expect PPO to show clear learning curves.
Focus on sample efficiency comparison: does PPO reach a score of 1000 faster or slower than
## DQN?

## ML CHALLENGE 3 — PPO VS DQN11
Listing 10: Group 7 starter — Ms. Pac-Man
1  model , returns = train_ppo(
2       env_id       = "ALE/MsPacman -v5",
3       total_steps = 5_000_000 ,
4       horizon      = 1024,
5       n_epochs     = 4,
6       batch_size   = 128,
7       lr            = 2.5e-4,
8       gamma         = 0.99,
9       gae_lambda   = 0.95,
10       clip_eps     = 0.2,
11       ent_coef     = 0.01,
12       seed          = 42,
## 13  )
14  # NOTE: Sample -efficiency  comparison  with \texttt{DQN} is the  main  result
here.
15  # Plot  reward -vs -steps  for  both  algorithms  with  shaded  std  bands.
Group 8 — ALE/Phoenix-v5
Key challenge: Fast-moving objects require reactive policies. PPO with short hori-
zons and many epochs should be contrasted with a longer horizon that allows credit assign-
ment across multi-wave sequences. Students should study the effect of horizon length on
PPO performance specifically.
Listing 11: Group 8 starter — Phoenix
1  model , returns = train_ppo(
2       env_id       = "ALE/Phoenix -v5",
3       total_steps = 5_000_000 ,
4       horizon      = 512,     # start  short  for  reactive  tasks
5       n_epochs     = 10,
6       batch_size   = 128,
7       lr            = 2.5e-4,
8       gamma         = 0.99,
9       gae_lambda   = 0.95,
10       clip_eps     = 0.2,
11       ent_coef     = 0.01,
12       seed          = 42,
## 13  )
14  # NOTE: Ablate  horizon  lengths: 512 vs. 1024 vs. 2048.
15  # Report  the  effect  on  final  score  and  training  stability.
Notes on scope and computational budget
PPO is typically more sample-efficient than vanilla DQN on Atari but may require more
wall-clock time per environment step because it processes full rollouts before updating. Plan
experiments accordingly. If compute is limited:
- Reduce total_steps to 2 000 000 and document this constraint clearly.
- Prioritise a 3-seed comparison between the best DQN and best PPO configuration over
a wide hyperparameter sweep.

## ML CHALLENGE 3 — PPO VS DQN12
- Report trends rather than absolute best performance when budget is constrained.
Grading checklist
The CHECKLIST.md file inside the repository must contain:
- Exact command to reproduce the best PPO run.
- Seeds used for PPO repeated experiments.
- Pointers to logs and figures for both DQN and PPO.
- A paragraph (max 200 words) summarising the key algorithmic difference observed
empirically between DQN and PPO on the assigned environment.
References and further reading
Standard starting references for this challenge include the original PPO paper by Schul-
man et al. (2017), the GAE paper by Schulman et al. (2016), and the DQN paper by Mnih
et al. (2015). Students should also consult current survey articles on deep reinforcement
learning and Atari benchmarking. All citations must follow IEEE style.
Challenge 3 is designed to develop critical algorithmic thinking. The goal is not to make
PPO beat DQN at all costs, but to understand when, why, and by how much one algorithm
outperforms the other, and to express that understanding clearly in a scientific paper.