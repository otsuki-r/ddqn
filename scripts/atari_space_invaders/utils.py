import argparse
import logging
import pathlib

import ale_py  # noqa: F401
import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
from ddqn.action_selection.epsilon_greedy import make_epsilon_greedy
from ddqn.adaptors import make_torch_from_int
from ddqn.configure import (
    DEVICE,
    EnvConfig,
    RunConfig,
    TrainingConfig,
    get_render_mode,
)
from ddqn.early_stop import RunningReward
from ddqn.utils import run
from ddqn.ddqn import DoubleDQN


def pad_batch(x: torch.tensor, **kwargs) -> torch.Tensor:
    return torch.tensor(x, dtype=kwargs["dtype"], device=kwargs["device"])


def build_env_config(cli_args: argparse.Namespace) -> EnvConfig:
    env = gym.make(
        "ALE/SpaceInvaders-v5",
        render_mode=get_render_mode(cli_args),
        obs_type="grayscale",
        # render_mode="human",
    )
    state_space_size = 716
    return EnvConfig(
        env=env,
        action_space_size=6,
        state_space_size=512,
        state_space_adaptor=pad_batch,
        # state_space_adaptor=make_torch_from_int(state_space_size),
        state_space_dtype=torch.float32,
    )


def build_training_config(
    ddqn: DoubleDQN, cli_args: argparse.Namespace, env_config: EnvConfig
) -> TrainingConfig:
    return TrainingConfig(
        epsilon_start=cli_args.epsilon_start,
        epsilon_end=cli_args.epsilon_end,
        epsilon_decay=cli_args.epsilon_decay,
        epsilon_exploration_steps=cli_args.epsilon_exploration_steps,
        num_episodes=cli_args.num_episodes,
        batch_size=cli_args.batch_size,
        gamma=cli_args.gamma,
        optimiser=optim.AdamW(
            ddqn.pnet.parameters(), lr=cli_args.alpha, amsgrad=True
        ),
        tau=cli_args.tau,
        action_selector_fn=make_epsilon_greedy(
            ddqn.pnet,
            env_config.env,
            smoothed=True,
            dtype=env_config.action_space_dtype,
            device=DEVICE,
        ),
        # loss_fn=nn.MSELoss(),
        loss_fn=nn.SmoothL1Loss(),
        early_return_fn=RunningReward(
            eval_frequency=50,
            short_sample=1_000,
            long_sample=10_000,
            tolerance=0.001,
            min_mean=0.875,
        ),
        outdir=pathlib.Path(cli_args.outdir),
    )


def build_run_config(cli_args: argparse.Namespace) -> RunConfig:
    outdir = pathlib.Path(cli_args.outdir)
    spath = pathlib.Path(cli_args.save_path) if cli_args.save_path else None
    return RunConfig(weights_dir=outdir, save_path=spath, num_iterations=1)


def custom_eval(
    ddqn: DoubleDQN, env_config: EnvConfig, run_config: RunConfig
) -> None:
    rewards: list[int] = []
    durations: list[int] = []
    for _ in range(1_000):
        res = run(ddqn, env_config=env_config, run_config=run_config)
        logging.info(res)
        rewards.append(res.total_reward)
        durations.append(res.epsiode_duration)
    logging.info("Evaluation (mean of 1,000 runs):")
    logging.info(f"Rewards: {sum(rewards) / len(rewards):.3f}")
    logging.info(f"Epsidoe duration: {sum(durations) / len(durations):.3f}")
    return
    return (sum(rewards) / len(rewards),)
