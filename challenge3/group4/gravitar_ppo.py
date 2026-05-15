from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)  # register ALE environments in the gymnasium namespace

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# CHANGE THIS to try a different Atari game
ENV_ID = "ALE/Gravitar-v5"

# Number of consecutive frames stacked together as a single observation.
# Stacking gives the agent a sense of motion (e.g. ball direction/speed).
N_STACK = 4


# Logging into Tensor Board

class TensorBoardCallback(BaseCallback):
    """Custom callback that logs per-episode metrics to TensorBoard.

    Attaches to the same SummaryWriter that SB3 creates internally, so our
    custom scalars land in the exact same event file as the built-in
    rollout/ and train/ metrics.

    Scalars added by this callback:
      - training/episode_reward : total reward accumulated in each episode

    SB3 built-in scalars (also visible in the same run):
      - rollout/ep_rew_mean : rolling mean reward over the last 100 episodes
      - train/policy_loss    : PPO policy loss (clipped surrogate)
      - train/value_loss     : Value function loss
      - train/entropy_loss   : Entropy bonus term
      - train/learning_rate  : current learning rate
    """

    def __init__(self) -> None:
        super().__init__()
        self._writer: SummaryWriter | None = None
        self._episode_reward = 0.0

    def _on_training_start(self) -> None:
        # Reuse SB3's own TensorBoard writer so every scalar ends up in the
        # same event file. SB3 stores it inside TensorBoardOutputFormat.writer.
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        for fmt in self.model._logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                self._writer = fmt.writer
                return
        # Fallback: SB3 was not given a tensorboard_log dir
        self._writer = None

    def _on_step(self) -> bool:
        if self._writer is None:
            return True

        self._episode_reward += float(self.locals["rewards"][0])

        if self.locals["dones"][0]:
            self._writer.add_scalar("training/episode_reward",
                                    self._episode_reward,
                                    self.num_timesteps)
            self._episode_reward = 0.0

        return True  # returning False would abort training


# Environment Builders

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
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


def build_playing_environment() -> VecFrameStack:
    """Create a human-rendered Atari environment for watching the agent play.

    Differences from the training environment:
      - render_mode="human" opens a visible game window.
      - terminal_on_life_loss=True: on each life loss the wrapper sends the
        FIRE action automatically, so the ball/game restarts without the agent
        getting stuck waiting for input.
      - clip_reward=False: show the real score instead of the clipped {-1,0,+1}.

    Returns:
        A VecFrameStack-wrapped vectorised environment with a human-visible window.
    """
    def _make_single_env() -> AtariWrapper:
        base_env = gym.make(ENV_ID, render_mode="human")
        return AtariWrapper(base_env, terminal_on_life_loss=True, clip_reward=False)

    env = DummyVecEnv([_make_single_env])
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


# Core Logic

def train_agent(
    model_path: str,
    timesteps: int,
    seed: int,
    tensorboard_log: str,
    hparams: dict | None = None,
) -> float:
    """Train a PPO agent and save the model.

    PPO uses an Actor-Critic architecture with:
      - Convolutional neural network base for processing Atari images
      - Policy head (actor) that outputs action probabilities
      - Value head (critic) that estimates state value
      - Clipped surrogate objective for stable policy updates
      - Generalized Advantage Estimation (GAE) for advantage computation
      - Entropy bonus to encourage exploration

    When called without `hparams` the function uses the built-in defaults.
    The sweep runner passes its own `hparams` dict so the same training logic
    is reused across all experiments.

    Returns:
        Mean episode reward over the last episodes stored in SB3's episode
        info buffer — used by run_sweep to rank experiments.

    Hyperparameter notes (tuned for 8 GB RAM, 300k-step budget):

    PPO-specific hyperparameters:
      learning_rate      3e-4    — standard PPO learning rate; higher than DQN
                                    due to on-policy nature
      n_steps           2048     — number of steps to run for each environment
                                    per update (PPO rollout buffer size)
      batch_size         64      — minibatch size for PPO updates
      n_epochs           10      — number of epochs when optimizing the surrogate
                                    (how many times to reuse the same data)
      gamma              0.99    — discount factor for future rewards
      gae_lambda         0.95    — GAE parameter for advantage estimation
                                    (higher = more variance reduction)
      clip_range         0.2     — PPO clipping parameter (prevents large policy updates)
      ent_coef          0.01     — entropy coefficient for exploration bonus
                                    (higher = more stochastic policy)
      vf_coef           0.5      — value function loss coefficient
                                    (balances policy vs value learning)
      max_grad_norm      0.5     — gradient clipping for stability

    Args:
        model_path:      Path (without .zip) where the trained model is saved.
        timesteps:       Total environment steps to train for.
        seed:            Random seed for reproducibility.
        tensorboard_log: Directory where TensorBoard event files are written.
        hparams:         Optional hyperparameter dict; uses built-in defaults
                         when None. Must include all keys listed above plus
                         'env_id', 'timesteps', 'seed'.
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # Use provided hparams or fall back to built-in defaults.
    if hparams is None:
        hparams = dict(
            env_id=ENV_ID,
            learning_rate=2.5e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=6,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            timesteps=timesteps,
            seed=seed,
        )

    # Write hparams to TensorBoard → visible in the HPARAMS tab.
    _tb_writer = SummaryWriter(log_dir=tensorboard_log)
    _tb_writer.add_hparams(hparams, metric_dict={"hparam/episode_reward": 0})
    _tb_writer.close()

    env = build_training_environment(seed=seed)

    model = PPO(
        policy="CnnPolicy",  # CNN-based Actor-Critic for Atari
        env=env,
        learning_rate=hparams["learning_rate"],
        n_steps=hparams["n_steps"],
        batch_size=hparams["batch_size"],
        n_epochs=hparams["n_epochs"],
        gamma=hparams["gamma"],
        gae_lambda=hparams["gae_lambda"],
        clip_range=hparams["clip_range"],
        ent_coef=hparams["ent_coef"],
        vf_coef=hparams["vf_coef"],
        max_grad_norm=hparams["max_grad_norm"],
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=seed,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=TensorBoardCallback(),
        progress_bar=True,
    )
    model.save(model_path)
    env.close()
    print(f"Model saved -> {model_path}.zip")

    # Return mean episode reward over the last recorded episodes.
    # SB3 maintains ep_info_buffer (deque of {r, l, t} dicts) during training.
    if model.ep_info_buffer:
        return float(np.mean([ep["r"] for ep in model.ep_info_buffer]))
    return 0.0


def play_agent(model_path: str, episodes: int) -> None:
    """Load a trained model and watch it play in a visible game window.

    Each full game (all lives exhausted) counts as one episode. The agent
    automatically fires at the start of each life thanks to the FireResetEnv
    wrapper inside build_playing_environment().

    Args:
        model_path: Path to the saved model (with or without .zip extension).
        episodes:   Number of full games to play before exiting.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(
            f"Model not found: {model_path}.zip\n"
            "Run with --mode train first to create a model."
        )

    env = build_playing_environment()
    model = PPO.load(model_path, env=env)

    completed = 0
    obs = env.reset()
    episode_reward = 0.0

    while completed < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        episode_reward += float(rewards[0])

        if dones[0]:
            # 'lives' > 0 means a mid-game life loss; the env auto-resets and fires.
            # 'lives' == 0 means the full game is over — count it as one episode.
            if infos[0].get("lives", 0) == 0:
                completed += 1
                print(f"Episode {completed}/{episodes}  reward: {episode_reward:.2f}")
                episode_reward = 0.0

    env.close()


# Sweep

def run_sweep(
    sweep_path: str,
    default_timesteps: int,
    seed: int,
    base_log_dir: str,
    best_model_path: str,
    num_seeds: int = 1,
) -> None:
    """Run all experiments defined in a JSON config file with multiple seeds.

    For each experiment, runs training ``num_seeds`` times with different random
    seeds (base_seed, base_seed+1, base_seed+2, etc.) to capture variance in
    training outcomes.

    Each experiment uses the ``timesteps`` value from its JSON entry. If the
    entry omits ``timesteps``, the value from ``--timesteps`` (default 300k)
    is used as a fallback.

    TensorBoard logs for every run are written to
    ``<base_log_dir>/sweep/<experiment_name>/<seed_suffix>/`` so all runs
    (across all seeds and experiments) are visible together by pointing
    TensorBoard at ``<base_log_dir>/sweep``.

    After all experiments finish, statistics (mean and std of final reward)
    are computed across all seeds, and the best model overall is kept at
    ``best_model_path``. All intermediate models are deleted to save disk space.

    Args:
        sweep_path:        Path to the JSON file containing experiment configs.
        default_timesteps: Fallback timestep budget for experiments that do not
                           define ``timesteps`` in their JSON entry.
        seed:              Base random seed; actual seeds will be seed, seed+1, etc.
        base_log_dir:      Root TensorBoard log directory.
        best_model_path:   Where to save the winning model (without .zip).
        num_seeds:         Number of distinct random seeds to run per experiment (default 3).
    """
    with open(sweep_path) as f:
        configs = json.load(f)

    tmp_model_dir = Path("models") / "_sweep_tmp"
    tmp_model_dir.mkdir(parents=True, exist_ok=True)

    # Store results: (exp_name, seed, score)
    all_results: list[tuple[str, int, float]] = []
    total_experiments = len(configs)

    for idx, cfg in enumerate(configs, start=1):
        name = cfg.get("name", f"exp_{idx:02d}")
        note = cfg.get("note", "")
        exp_timesteps = cfg.get("timesteps", default_timesteps)

        print(f"\n{'='*70}")
        print(f"Experiment {idx}/{total_experiments}: {name}")
        print(f"  Timesteps: {exp_timesteps:,} | Seeds: {num_seeds} (base_seed={seed})")
        if note:
            print(f"  {note}")
        print(f"{'='*70}")

        # Build the shared hparams dict for this experiment
        hparams_template = {
            "env_id": ENV_ID,
            "learning_rate": cfg["learning_rate"],
            "n_steps": cfg["n_steps"],
            "batch_size": cfg["batch_size"],
            "n_epochs": cfg["n_epochs"],
            "gamma": cfg["gamma"],
            "gae_lambda": cfg["gae_lambda"],
            "clip_range": cfg["clip_range"],
            "ent_coef": cfg["ent_coef"],
            "vf_coef": cfg["vf_coef"],
            "max_grad_norm": cfg["max_grad_norm"],
            "timesteps": exp_timesteps,
        }

        # Run training with multiple seeds
        seed_scores = []
        for seed_offset in range(num_seeds):
            current_seed = seed + seed_offset
            hparams = {**hparams_template, "seed": current_seed}

            model_path = str(tmp_model_dir / f"{name}_seed{current_seed}")
            log_dir = f"{base_log_dir}/sweep/{name}/seed_{current_seed}"

            print(f"  [{seed_offset + 1}/{num_seeds}] Running with seed={current_seed}...")

            score = train_agent(
                model_path=model_path,
                timesteps=exp_timesteps,
                seed=current_seed,
                tensorboard_log=log_dir,
                hparams=hparams,
            )
            seed_scores.append(score)
            all_results.append((name, current_seed, score))
            print(f"    → final mean reward: {score:.2f}")

        # Summary for this experiment across all seeds
        mean_score = float(np.mean(seed_scores))
        std_score = float(np.std(seed_scores))
        print(f"\n  Summary for {name}:")
        print(f"    Mean reward: {mean_score:.2f} ± {std_score:.2f}")
        print(f"    All scores: {[f'{s:.2f}' for s in seed_scores]}")

    # Overall summary: aggregate results by experiment name
    print(f"\n{'='*70}")
    print("SWEEP COMPLETE — Aggregate results by experiment (mean ± std):")
    print(f"{'='*70}")

    experiment_stats: dict[str, tuple[float, float]] = {}
    for exp_name in [cfg.get("name", f"exp_{i:02d}") for i, cfg in enumerate(configs, 1)]:
        scores_for_exp = [score for name, _, score in all_results if name == exp_name]
        if scores_for_exp:
            mean_val = float(np.mean(scores_for_exp))
            std_val = float(np.std(scores_for_exp))
            experiment_stats[exp_name] = (mean_val, std_val)

    # Sort by mean reward (descending)
    sorted_stats = sorted(experiment_stats.items(), key=lambda x: x[1][0], reverse=True)
    for rank, (exp_name, (mean_reward, std_reward)) in enumerate(sorted_stats, start=1):
        marker = "  ← BEST" if rank == 1 else ""
        print(f"  {rank}. {exp_name:<40s}  {mean_reward:7.2f} ± {std_reward:6.2f}{marker}")

    # Find the single best run (across all seeds)
    if all_results:
        best_result = max(all_results, key=lambda x: x[2])  # max by score
        best_exp_name, best_seed, best_score = best_result
    else:
        best_exp_name, best_seed, best_score = "unknown", -1, 0.0

    print(f"{'='*70}\n")

    # Save best model
    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
    best_model_src = tmp_model_dir / f"{best_exp_name}_seed{best_seed}.zip"
    if best_model_src.exists():
        shutil.copy(str(best_model_src), f"{best_model_path}.zip")
    shutil.rmtree(tmp_model_dir)

    print(f"✓ Best model saved: {best_model_path}.zip")
    print(f"  Experiment: {best_exp_name}, Seed: {best_seed}, Score: {best_score:.2f}")
    print(f"✓ TensorBoard logs (all experiments, all seeds): {base_log_dir}/sweep/")
    print(f"\n  View results:\n    tensorboard --logdir {base_log_dir}/sweep")


def inspect_model(model_path: str) -> None:
    """Load a saved model and print its hyperparameters.

    SB3 serialises all constructor arguments inside the .zip file, so this
    works for any model saved by this script.

    Args:
        model_path: Path to the saved model (with or without .zip extension).

    Example:
        python gravitar_ppo.py --mode inspect --model-path models/gravitar_ppo
    """
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")

    model = PPO.load(model_path)

    # Parameters SB3 saves inside the zip
    params = {
        "policy": model.policy_class.__name__,
        "learning_rate": model.learning_rate,
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "clip_range": model.clip_range,
        "ent_coef": model.ent_coef,
        "vf_coef": model.vf_coef,
        "max_grad_norm": model.max_grad_norm,
        "num_timesteps_trained": model.num_timesteps,
    }

    print(f"\n── Saved model: {model_path}.zip")
    for key, value in params.items():
        print(f"  {key:30s}: {value}")
    print("─" * 55 + "\n")


# CLI

def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train or watch a PPO agent on an Atari game.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["train", "play", "inspect", "sweep"], required=True,
        help="'train' single run, 'play' watch agent, 'inspect' print params, "
             "'sweep' run all experiments from --sweep-file.",
    )
    parser.add_argument(
        "--sweep-file", default="sweep_configs_ppo.json",
        help="Path to JSON file with experiment configs (used by --mode sweep and --experiment).",
    )
    parser.add_argument(
        "--experiment", default=None,
        help="Name of a single experiment in --sweep-file to run with --mode train. "
             "Uses built-in defaults when omitted.",
    )
    parser.add_argument(
        "--model-path", default="models/gravitar_ppo",
        help="Path to save (train) or load (play) the model (without .zip).",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training steps. Overrides the value in the JSON config when set. "
             "Defaults to the JSON value, or 5M if neither is specified.",
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of full games to play in play mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tensorboard-log", default="logs/gravitar_ppo",
        help="Directory for TensorBoard logs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        hparams = None
        timesteps = args.timesteps or 5_000_000

        if args.experiment:
            # Load a named experiment from the JSON so the same configs work
            # for both single runs (--mode train) and full sweeps (--mode sweep).
            with open(args.sweep_file) as f:
                configs = {c["name"]: c for c in json.load(f)}
            if args.experiment not in configs:
                raise ValueError(
                    f"Experiment '{args.experiment}' not found in {args.sweep_file}.\n"
                    f"Available: {', '.join(configs)}"
                )
            cfg = configs[args.experiment]
            timesteps = args.timesteps or cfg.get("timesteps", 5_000_000)
            hparams = {
                "env_id": ENV_ID,
                "learning_rate": cfg["learning_rate"],
                "n_steps": cfg["n_steps"],
                "batch_size": cfg["batch_size"],
                "n_epochs": cfg["n_epochs"],
                "gamma": cfg["gamma"],
                "gae_lambda": cfg["gae_lambda"],
                "clip_range": cfg["clip_range"],
                "ent_coef": cfg["ent_coef"],
                "vf_coef": cfg["vf_coef"],
                "max_grad_norm": cfg["max_grad_norm"],
                "timesteps": timesteps,
                "seed": args.seed,
            }
            print(f"Loaded experiment '{args.experiment}' from {args.sweep_file}")

        train_agent(
            model_path=args.model_path,
            timesteps=timesteps,
            seed=args.seed,
            tensorboard_log=args.tensorboard_log,
            hparams=hparams,
        )

    elif args.mode == "play":
        play_agent(model_path=args.model_path, episodes=args.episodes)

    elif args.mode == "sweep":
        run_sweep(
            sweep_path=args.sweep_file,
            default_timesteps=args.timesteps or 5_000_000,
            seed=args.seed,
            base_log_dir=args.tensorboard_log,
            best_model_path=args.model_path,
        )

    else:
        inspect_model(model_path=args.model_path)


if __name__ == "__main__":
    main()
