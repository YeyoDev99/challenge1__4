from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import ale_py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

gym.register_envs(ale_py)  # register ALE environments in the gymnasium namespace

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# CHANGE THIS to try a different Atari game
ENV_ID = "ALE/Gravitar-v5"

# Number of consecutive frames stacked together as a single observation.
# Stacking gives the agent a sense of motion (e.g. ball direction/speed).
N_STACK = 4


# Environment Builders

def make_env(env_id: str, seed: int = 0):
    """Create a single preprocessed Atari environment."""
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = AtariWrapper(env)
        env.seed(seed)
        return env
    return _init


def build_training_environment(seed: int) -> VecFrameStack:
    """Create a vectorised, preprocessed Atari environment for training.

    Applies the standard Atari preprocessing pipeline automatically via
    make_atari_env + VecFrameStack:
      - Grayscale conversion
      - Frame resize to 84 * 84
      - Frame skipping (repeat each action 4 steps)
      - Terminal-on-life-loss (treat each life as a separate episode)
      - Frame stacking (last N_STACK frames as one observation)

    Args:
        seed: Random seed for reproducibility.

    Returns:
        A VecFrameStack-wrapped vectorised environment ready for PPO.
    """
    env = make_atari_env(
        env_id=ENV_ID,
        n_envs=1,
        seed=seed,
        wrapper_kwargs={"clip_reward": False},
    )
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


# Actor-Critic Network (reused from Challenge 3)

class AtariActorCritic(nn.Module):
    """CNN-based Actor-Critic network for Atari environments.

    The network processes stacked grayscale frames (4 × 84 × 84) through
    three convolutional layers, then splits into two heads:
      - Actor: outputs action logits (policy)
      - Critic: outputs state value V(s)
    """

    def __init__(self, n_actions: int):
        super().__init__()
        # Shared CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # Compute CNN output size: 64 * 7 * 7 = 3136
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84)
            cnn_out = self.cnn(dummy).shape[1]

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(cnn_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(cnn_out, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, obs):
        """Forward pass: returns (logits, value)."""
        feats = self.cnn(obs)
        logits = self.actor(feats)
        value = self.critic(feats)
        return logits, value

    def get_action(self, obs, deterministic=False):
        """Sample action from policy."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), value


# GAIL Discriminator Network

class GAILDiscriminator(nn.Module):
    """
    Takes a stacked-frame observation (and optionally a one-hot encoded
    action) and outputs P(expert | s, a) in (0, 1).

    Using observation only (obs-only variant) is simpler and often
    sufficient for image-based environments.
    """
    def __init__(self, n_actions: int, use_action: bool = False):
        super().__init__()
        self.use_action = use_action
        # Shared CNN - same architecture as the policy backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out = 64 * 7 * 7  # 3136
        fc_in = cnn_out + n_actions if use_action else cnn_out

        self.fc = nn.Sequential(
            nn.Linear(fc_in, 512), nn.Tanh(),
            nn.Linear(512, 1), nn.Sigmoid(),
        )

    def forward(self, obs, actions_onehot=None):
        feats = self.cnn(obs)
        if self.use_action and actions_onehot is not None:
            feats = torch.cat([feats, actions_onehot], dim=-1)
        return self.fc(feats).squeeze(-1)


# GAE Computation

def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation (GAE)."""
    advantages = []
    gae = 0
    values = values + [next_value]
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


# Demonstration Collector

def collect_demonstrations(env_id: str, checkpoint_path: str,
                          n_steps: int = 30_000, seed: int = 0,
                          device: str = "cpu") -> dict:
    """
    Roll out a saved policy and record (obs, action) pairs.

    Returns a dict with keys 'observations' and 'actions',
    each a numpy array of shape (n_steps, ...).
    """
    print(f"Collecting {n_steps} demonstration steps from {checkpoint_path}...")
    
    env = make_env(env_id, seed=seed)()
    n_actions = env.action_space.n

    model = AtariActorCritic(n_actions).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    obs_buf, act_buf = [], []
    obs, _ = env.reset()
    
    for i in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(obs_t)
            action = logits.argmax(dim=-1).item()  # greedy / deterministic

        obs_buf.append(obs)
        act_buf.append(action)

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        
        if (i + 1) % 5000 == 0:
            print(f"  Collected {i + 1}/{n_steps} steps")

    env.close()
    demos = {
        "observations": np.array(obs_buf, dtype=np.float32),
        "actions": np.array(act_buf, dtype=np.int64),
    }
    print(f"Collected {n_steps} demo steps")
    return demos


# Behavioral Cloning (BC) Baseline

def train_bc(env_id: str, demos_path: str = "demos.npz",
             n_epochs: int = 25, batch_size: int = 256,
             lr: float = 1e-4, device: str = "cpu",
             model_path: str = "bc_policy.pt") -> AtariActorCritic:
    """
    Supervised imitation: minimise cross-entropy between
    demonstrations and policy logits.
    """
    print(f"Training BC baseline from {demos_path}...")
    
    data = np.load(demos_path)
    obs_t = torch.tensor(data["observations"], dtype=torch.float32)
    act_t = torch.tensor(data["actions"], dtype=torch.long)
    dataset = TensorDataset(obs_t, act_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    env = make_env(env_id)()
    n_actions = env.action_space.n
    env.close()

    model = AtariActorCritic(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        total_loss = 0.0
        for obs_b, act_b in loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            logits, _ = model(obs_b)
            loss = criterion(logits, act_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"  BC epoch {epoch+1}/{n_epochs} loss={avg:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"BC policy saved to {model_path}")
    return model


# GAIL Training Loop

def train_gail(env_id, demos_path="demos.npz",
               total_steps=5_000_000, horizon=1024,
               n_ppo_epochs=4, batch_size=128,
               lr_policy=2.5e-4, lr_disc=3e-4,
               disc_updates_per_rollout=3,
               gamma=0.99, gae_lambda=0.95,
               clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
               max_grad_norm=0.5, seed=42,
               device="cuda" if torch.cuda.is_available() else "cpu",
               bc_warmstart_path=None,
               log_dir="logs/gail"):
    """
    Train GAIL agent with PPO as the inner RL algorithm.
    """
    print(f"Training GAIL on {env_id} for {total_steps} steps...")
    print(f"Device: {device}")
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Load demonstrations
    data = np.load(demos_path)
    demo_obs = torch.tensor(data["observations"], dtype=torch.float32)
    demo_act = torch.tensor(data["actions"], dtype=torch.long)
    n_demos = len(demo_obs)
    print(f"Loaded {n_demos} demonstration steps")

    env = build_training_environment(seed=seed)
    n_actions = env.action_space.n

    # Initialize policy
    policy = AtariActorCritic(n_actions).to(device)
    
    # Optional BC warm-start
    if bc_warmstart_path and os.path.exists(bc_warmstart_path):
        print(f"Loading BC warm-start from {bc_warmstart_path}")
        policy.load_state_dict(torch.load(bc_warmstart_path, map_location=device))

    # Initialize discriminator
    disc = GAILDiscriminator(n_actions, use_action=False).to(device)

    opt_policy = optim.Adam(policy.parameters(), lr=lr_policy)
    opt_disc = optim.Adam(disc.parameters(), lr=lr_disc)
    bce = torch.nn.BCELoss()

    obs, _ = env.reset()
    ep_return = 0.0
    all_returns = []
    disc_acc_history = []

    for global_step in range(0, total_steps, horizon):

        # ---- rollout collection ----
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf = [], [], []

        for _ in range(horizon):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, value = policy(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()

            obs_buf.append(obs_t.squeeze(0))
            act_buf.append(action)
            logp_buf.append(dist.log_prob(action))
            val_buf.append(value.squeeze())

            obs, env_reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            done_buf.append(done)
            ep_return += env_reward

            if done:
                all_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()

        # ---- adversarial reward (replace env reward) ----
        obs_stack = torch.stack(obs_buf).to(device)
        with torch.no_grad():
            d_scores = disc(obs_stack)  # P(expert | s)
            # reward: log D(s,a) -- agent wants to look like the expert
            adv_rewards = torch.log(d_scores + 1e-8).cpu()
        rew_buf = adv_rewards.tolist()

        # ---- GAE advantages ----
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            _, nv = policy(obs_t)
            advantages, returns = compute_gae(
                rew_buf, val_buf, done_buf, nv.item(), gamma, gae_lambda
            )

        # ---- discriminator update ----
        act_one_hot = F.one_hot(
            torch.stack(act_buf), n_actions).float().to(device)

        for _ in range(disc_updates_per_rollout):
            # sample expert mini-batch
            idx_e = torch.randint(0, n_demos, (batch_size,))
            e_obs = demo_obs[idx_e].to(device)
            # agent mini-batch
            idx_a = torch.randint(0, horizon, (batch_size,))
            a_obs = obs_stack[idx_a]

            d_expert = disc(e_obs)
            d_agent = disc(a_obs)

            loss_disc = bce(d_expert, torch.ones_like(d_expert)) + \
                        bce(d_agent, torch.zeros_like(d_agent))
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

        # Compute discriminator accuracy
        with torch.no_grad():
            d_acc = ((d_expert > 0.5).float().mean() +
                     (d_agent < 0.5).float().mean()) / 2
            disc_acc_history.append(d_acc.item())

        # ---- PPO update ----
        act_t = torch.stack(act_buf).to(device)
        logp_t = torch.stack(logp_buf).detach().to(device)
        adv_t = (torch.tensor(advantages) - torch.tensor(advantages).mean()) / (torch.tensor(advantages).std() + 1e-8)
        ret_t = torch.tensor(returns).to(device)

        idx = torch.randperm(horizon)
        for _ in range(n_ppo_epochs):
            for start in range(0, horizon, batch_size):
                mb = idx[start:start + batch_size]
                lg, vn = policy(obs_stack[mb])
                dn = Categorical(logits=lg)
                lp_new = dn.log_prob(act_t[mb])
                ent = dn.entropy().mean()
                ratio = (lp_new - logp_t[mb]).exp()

                s1 = ratio * adv_t[mb]
                s2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t[mb]
                l_pi = -torch.min(s1, s2).mean()
                l_vf = ((vn - ret_t[mb]) ** 2).mean()
                loss = l_pi + vf_coef * l_vf - ent_coef * ent

                opt_policy.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                opt_policy.step()

        if len(all_returns) % 10 == 0 and all_returns:
            mean_ret = np.mean(all_returns[-100:])
            print(f"step={global_step:7d} ret={mean_ret:7.1f} "
                  f"disc_loss={loss_disc.item():.3f} "
                  f"disc_acc={d_acc.item():.2f}")

    env.close()
    return policy, disc, all_returns, disc_acc_history


# Sweep Function

def run_sweep(sweep_file, demos_path, seeds=[42, 43, 44], device="cpu"):
    """Run hyperparameter sweep for GAIL."""
    with open(sweep_file, 'r') as f:
        configs = json.load(f)
    
    print(f"Running sweep with {len(configs)} configurations")
    print(f"Seeds: {seeds}")
    print(f"Device: {device}")
    
    results = {}
    
    for config in configs:
        exp_name = config["name"]
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"Description: {config.get('description', 'N/A')}")
        print(f"{'='*60}")
        
        exp_results = []
        
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            
            # Create log directory for this experiment and seed
            log_dir = f"logs/gail/sweep/{exp_name}/seed_{seed}"
            
            # Determine BC warm-start path
            bc_warmstart_path = None
            if config.get("bc_warmstart", False):
                bc_warmstart_path = f"models/bc_policy_seed{seed}.pt"
                if not os.path.exists(bc_warmstart_path):
                    print(f"  Training BC for seed {seed}...")
                    train_bc(
                        env_id=ENV_ID,
                        demos_path=demos_path,
                        n_epochs=25,
                        device=device,
                        model_path=bc_warmstart_path,
                    )
            
            # Train GAIL
            try:
                policy, disc, returns, disc_acc = train_gail(
                    env_id=ENV_ID,
                    demos_path=demos_path,
                    total_steps=config["timesteps"],
                    horizon=config["horizon"],
                    lr_policy=config["lr_policy"],
                    lr_disc=config["lr_disc"],
                    disc_updates_per_rollout=config["disc_updates_per_rollout"],
                    ent_coef=config["ent_coef"],
                    bc_warmstart_path=bc_warmstart_path,
                    seed=seed,
                    device=device,
                    log_dir=log_dir,
                )
                
                # Save models
                model_dir = f"models/gail/{exp_name}"
                os.makedirs(model_dir, exist_ok=True)
                torch.save(policy.state_dict(), f"{model_dir}/policy_seed{seed}.pt")
                torch.save(disc.state_dict(), f"{model_dir}/discriminator_seed{seed}.pt")
                
                # Calculate metrics
                final_return = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
                exp_results.append({
                    "seed": seed,
                    "final_return": final_return,
                    "disc_acc_final": disc_acc[-1] if disc_acc else 0.0,
                })
                
                print(f"  Seed {seed} complete. Final return: {final_return:.1f}")
                
            except Exception as e:
                print(f"  Seed {seed} failed with error: {e}")
                exp_results.append({
                    "seed": seed,
                    "final_return": 0.0,
                    "disc_acc_final": 0.0,
                    "error": str(e),
                })
        
        # Calculate experiment statistics
        returns = [r["final_return"] for r in exp_results]
        results[exp_name] = {
            "config": config,
            "results": exp_results,
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
        }
        
        print(f"\nExperiment {exp_name} summary:")
        print(f"  Mean return: {results[exp_name]['mean_return']:.1f} ± {results[exp_name]['std_return']:.1f}")
    
    # Save sweep results
    with open("sweep_results_gail.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find best experiment
    best_exp = max(results.items(), key=lambda x: x[1]["mean_return"])
    print(f"\n{'='*60}")
    print(f"Best experiment: {best_exp[0]}")
    print(f"Mean return: {best_exp[1]['mean_return']:.1f} ± {best_exp[1]['std_return']:.1f}")
    print(f"{'='*60}")
    
    return results


# Main CLI

def main():
    parser = argparse.ArgumentParser(description="GAIL Training for ALE/Gravitar-v5")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["collect", "bc", "gail", "sweep"],
                        help="Mode: collect demonstrations, train BC, train GAIL, or run sweep")
    
    # Demonstration collection
    parser.add_argument("--checkpoint-path", type=str,
                        help="Path to trained checkpoint for demo collection")
    parser.add_argument("--n-steps", type=int, default=30_000,
                        help="Number of demonstration steps to collect")
    parser.add_argument("--demos-path", type=str, default="demos.npz",
                        help="Path to save/load demonstrations")
    
    # BC training
    parser.add_argument("--bc-epochs", type=int, default=25,
                        help="Number of BC training epochs")
    parser.add_argument("--bc-lr", type=float, default=1e-4,
                        help="BC learning rate")
    parser.add_argument("--bc-model-path", type=str, default="bc_policy.pt",
                        help="Path to save BC model")
    
    # GAIL training
    parser.add_argument("--total-steps", type=int, default=5_000_000,
                        help="Total training steps for GAIL")
    parser.add_argument("--horizon", type=int, default=1024,
                        help="PPO rollout horizon")
    parser.add_argument("--lr-policy", type=float, default=2.5e-4,
                        help="Policy learning rate")
    parser.add_argument("--lr-disc", type=float, default=3e-4,
                        help="Discriminator learning rate")
    parser.add_argument("--disc-updates", type=int, default=3,
                        help="Discriminator updates per rollout")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--bc-warmstart", type=str, default=None,
                        help="Path to BC model for warm-start")
    parser.add_argument("--log-dir", type=str, default="logs/gail",
                        help="TensorBoard log directory")
    
    # Sweep mode
    parser.add_argument("--sweep-file", type=str, default="sweep_configs_gail.json",
                        help="Path to sweep configuration JSON file")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 43, 44],
                        help="Seeds for sweep experiments")
    
    # Common
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    
    args = parser.parse_args()
    
    if args.mode == "collect":
        if not args.checkpoint_path:
            raise ValueError("--checkpoint-path required for collect mode")
        demos = collect_demonstrations(
            env_id=ENV_ID,
            checkpoint_path=args.checkpoint_path,
            n_steps=args.n_steps,
            seed=args.seed,
            device=args.device,
        )
        np.savez_compressed(args.demos_path, **demos)
        print(f"Demonstrations saved to {args.demos_path}")
    
    elif args.mode == "bc":
        model = train_bc(
            env_id=ENV_ID,
            demos_path=args.demos_path,
            n_epochs=args.bc_epochs,
            lr=args.bc_lr,
            device=args.device,
            model_path=args.bc_model_path,
        )
        print(f"BC training complete. Model saved to {args.bc_model_path}")
    
    elif args.mode == "gail":
        if not os.path.exists(args.demos_path):
            raise ValueError(f"Demonstrations not found at {args.demos_path}")
        
        policy, disc, returns, disc_acc = train_gail(
            env_id=ENV_ID,
            demos_path=args.demos_path,
            total_steps=args.total_steps,
            horizon=args.horizon,
            lr_policy=args.lr_policy,
            lr_disc=args.lr_disc,
            disc_updates_per_rollout=args.disc_updates,
            ent_coef=args.ent_coef,
            bc_warmstart_path=args.bc_warmstart,
            seed=args.seed,
            device=args.device,
            log_dir=args.log_dir,
        )
        
        # Save final models
        os.makedirs("models", exist_ok=True)
        torch.save(policy.state_dict(), "models/gail_policy.pt")
        torch.save(disc.state_dict(), "models/gail_discriminator.pt")
        print("GAIL training complete. Models saved to models/")
        
        # Save training metrics
        np.savez("gail_metrics.npz", returns=returns, disc_acc=disc_acc)
        print("Training metrics saved to gail_metrics.npz")
    
    elif args.mode == "sweep":
        if not os.path.exists(args.demos_path):
            raise ValueError(f"Demonstrations not found at {args.demos_path}")
        
        results = run_sweep(
            sweep_file=args.sweep_file,
            demos_path=args.demos_path,
            seeds=args.seeds,
            device=args.device,
        )
        print("Sweep complete. Results saved to sweep_results_gail.json")


if __name__ == "__main__":
    main()
