import argparse
import logging
import pathlib

import icu_sepsis  # noqa: F401
import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
from ddqn.action_selection import make_epsilon_greedy
from ddqn.adaptors import make_torch_from_int
from ddqn.structures import DoubleDQN
from ddqn.configure import DEVICE, EnvConfig, RunConfig, TrainingConfig
from ddqn.utils import run


def build_env_config(cli_args: argparse.Namespace) -> EnvConfig:
    env = gym.make(
        # "Sepsis/ICU-Sepsis-v2", inadmissible_action_strategy="terminate"
        "Sepsis/ICU-Sepsis-v2"
    )
    state_space_size = 716
    return EnvConfig(
        env=env,
        action_space_size=25,
        state_space_size=state_space_size,
        state_space_adaptor=make_torch_from_int(state_space_size),
        state_space_dtype=torch.float32,
    )


def build_training_config(
    ddqn: DoubleDQN, cli_args: argparse.Namespace, env_config: EnvConfig
) -> TrainingConfig:
    return TrainingConfig(
        epsilon_start=cli_args.epsilon_start,
        epsilon_end=cli_args.epsilon_end,
        epsilon_decay=cli_args.epsilon_decay,
        num_episodes=cli_args.num_episodes,
        batch_size=cli_args.batch_size,
        gamma=cli_args.gamma,
        optimiser=optim.AdamW(
            ddqn.pnet.parameters(), lr=cli_args.alpha, amsgrad=True
        ),
        tau=cli_args.tau,
        action_selector_fn=make_epsilon_greedy(
            env_config.env, dtype=env_config.action_space_dtype, device=DEVICE
        ),
        loss_fn=nn.SmoothL1Loss(),  # Generalisation of robust Huber loss
        early_return_fn=None,
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
