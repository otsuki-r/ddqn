import argparse
from dataclasses import dataclass
from typing import Callable, Iterator

import gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.envs.registration import Env
from .action_selection import make_action_selector


parser = argparse.ArgumentParser("bloom_cart_pole")
parser.add_argument(
    "--epsilon-start",
    help="Start value for parameter used in epsilon-greedy action selection.",
    type=float,
    default=0.9,
)
parser.add_argument(
    "--epsilon-end",
    help="End value for parameter used in epsilon-greedy action selection.",
    type=float,
    default=0.05,
)
parser.add_argument(
    "--epsilon-decay",
    help="Decay rate for parameter used in epsilon-greedy action selection.",
    type=float,
    default=0.001,
)
parser.add_argument(
    "--num-episodes",
    help="Number of episodes to train for.",
    type=int,
    default=50,
)
parser.add_argument(
    "--batch-size",
    help="Batch size to be used in training.",
    type=int,
    default=128,
)
parser.add_argument(
    "--discount",
    help="Discounting power of past actions in computing reward",
    type=float,
    default=0.95,
)
parser.add_argument(
    "--learning-rate",
    help="Learing rate LR to be used in training.",
    type=float,
    default=1e-4,
)
parser.add_argument(
    "--update-rate",
    help="Update rate \tau of target network to be used in training.",
    type=float,
    default=0.005,
)
parser.add_argument(
    "--outdir",
    help="Directory to save results to. Will be created if it does not exist.",
    type=str,
    default="./out",
)
parser.add_argument(
    "--use-bloom",
    help="Whether to use a Bloom filter for action selection during training.",
    type=bool,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--eval",
    help="Whether to run a model stored in --outdir",
    type=bool,
    action=argparse.BooleanOptionalAction,
)


@dataclass
class TrainingConfig:
    action_selector: Callable[[torch.Tensor, nn.Module], torch.Tensor]
    num_episodes: int
    batch_size: int
    discount: float
    optimiser: optim.Optimizer
    update_rate: float
    loss_fn: torch.nn.modules.loss._Loss
    device: torch.device


@dataclass
class EnvConfig:
    env: Env
    action_space_size: int
    state_space_size: int


def process_cli_args() -> argparse.Namespace:
    cli_args = parser.parse_args()

    if cli_args.epsilon_start < 0.0 or cli_args.epsilon_start > 1.0:
        raise ValueError("Start value of epsilon must lie in range [0, 1]")

    if cli_args.epsilon_end > cli_args.epsilon_start:
        raise ValueError("End value of epsilon must be less than start value")

    if cli_args.epsilon_end < 0.0 or cli_args.epsilon_end > 1.0:
        raise ValueError("End value of epsilon must lie in range [0, 1]")

    if cli_args.epsilon_decay < 0.0:
        raise ValueError("Decay rate of epsilon must lie in range [0, 1]")

    if cli_args.num_episodes < 0:
        raise ValueError("Num. training episodes must be a positive integer")

    if cli_args.batch_size < 0:
        raise ValueError("Batch size must be a positive integer")

    if cli_args.discount < 0.0 or cli_args.discount > 1.0:
        raise ValueError("Discounting power must lie in range [0, 1]")

    if cli_args.learning_rate < 0.0:
        raise ValueError("Learning rate must be positive")

    if cli_args.update_rate < 0.0 or cli_args.update_rate > 1.0:
        raise ValueError("Update rate must lie in range [0, 1]")

    return cli_args


def get_env_config(cli_args: argparse.Namespace) -> EnvConfig:
    env = gymnasium.make(
        "CartPole-v1", render_mode="human" if cli_args.eval else None
    )
    initial_state, _ = env.reset(seed=42)
    return EnvConfig(
        env=env,
        action_space_size=int(env.action_space.n),  # type: ignore
        state_space_size=len(initial_state),
    )


def build_training_config(
    cli_args: argparse.Namespace,
    env_config: EnvConfig,
    parameters: Iterator[nn.Parameter],
) -> TrainingConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return TrainingConfig(
        action_selector=make_action_selector(
            eps_start=cli_args.epsilon_start,
            eps_end=cli_args.epsilon_end,
            eps_decay=cli_args.epsilon_decay,
            use_bloom=cli_args.use_bloom,
            device=device,
            env=env_config.env,
        ),
        num_episodes=cli_args.num_episodes,
        batch_size=cli_args.batch_size,
        discount=cli_args.discount,
        optimiser=optim.AdamW(
            parameters, lr=cli_args.learning_rate, amsgrad=True
        ),
        update_rate=cli_args.update_rate,
        loss_fn=nn.SmoothL1Loss(),
        device=device,
    )
