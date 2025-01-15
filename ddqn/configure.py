import argparse
from dataclasses import dataclass

import torch
import torch.optim as optim
from gymnasium.envs.registration import Env
from .action_selection import (
    ActionSelector,
    make_epsilon_greedy,
    make_epsilon_greedy_with_bloom_filter,
)
from .early_stop import EarlyReturnFn


parser = argparse.ArgumentParser("bloom_cart_pole")

parser.add_argument(
    "--memory-size",
    help="Max size of the replay memory buffer.",
    type=int,
    default=10_000,
)
parser.add_argument(
    "--warmup-episodes-upper",
    help="Number of episodes to append to warmup memory for",
    type=int,
    default=60,
)
parser.add_argument(
    "--main-episodes-lower",
    help="Number of episodes to wait before starting to append to main memory",
    type=int,
    default=200,
)
parser.add_argument(
    "--epsilon-start",
    help="Start value for ε used in epsilon-greedy action selection.",
    type=float,
    default=0.90,
)
parser.add_argument(
    "--epsilon-end",
    help="End value for ε used in epsilon-greedy action selection.",
    type=float,
    default=0.05,
)
parser.add_argument(
    "--epsilon-decay",
    help="Exponential decay rate of ε used in epsilon-greedy action selection.",
    type=float,
    default=1e-4,
)
parser.add_argument(
    "--num-episodes",
    help="Number of episodes to train for.",
    type=int,
    default=600,
)
parser.add_argument(
    "--batch-size",
    help="Batch size to be used in training.",
    type=int,
    default=128,
)
parser.add_argument(
    "--gamma",
    help="Discounting power γ of past actions in computing reward",
    type=float,
    default=0.95,
)
parser.add_argument(
    "--alpha",
    help="Learing rate α of the optimiser to be used in training.",
    type=float,
    default=1e-3,
)
parser.add_argument(
    "--tau",
    help="Update rate τ of target network (soft update).",
    type=float,
    default=1e-2,
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
    action_selector: ActionSelector
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    num_episodes: int
    batch_size: int
    gamma: float
    optimiser: optim.Optimizer
    tau: float
    loss_fn: torch.nn.modules.loss._Loss
    device: torch.device
    early_return_fn: EarlyReturnFn | None


@dataclass
class EnvConfig:
    env: Env
    action_space_size: int
    state_space_size: int


def process_cli_args() -> argparse.Namespace:
    """
    Parse and validate arguments from the CLI

    Returns
    -------
    argparse.Namespace
        Parsed CLI args.
    """

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

    if cli_args.gamma < 0.0 or cli_args.gamma > 1.0:
        raise ValueError("Discounting power must lie in range [0, 1]")

    if cli_args.alpha < 0.0:
        raise ValueError("Learning rate must be positive")

    if cli_args.tau < 0.0 or cli_args.tau > 1.0:
        raise ValueError("Update rate must lie in range [0, 1]")

    return cli_args


def build_training_config(
    cli_args: argparse.Namespace,
    env_config: EnvConfig,
    *,
    optimiser: optim.Optimizer,
    loss_fn: torch.nn.modules.loss._Loss,
    early_return_fn: EarlyReturnFn | None = None,
) -> TrainingConfig:
    """
    Collate CLI args and environment configurations into a training
    configuration.

    Parameters
    ----------
    cli_args : argparse.Namespace
        Args parsed from the command line.
    env_config : EnvConfig
        Configuration of the environment to train in.
    optimiser : optim.Optimizer
        Optimizer to use.
    loss : torch.nn.modules.loss._Loss,
        Loss function to use,
    early_return_fn : EarlyReturnFn | None, optional
        Condition to return early from training.

    Returns
    -------
    TrainingConfig
        Traning configuration constructed from inputs.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cli_args.use_bloom:
        action_selector = make_epsilon_greedy_with_bloom_filter(
            env=env_config.env, device=device
        )
    else:
        action_selector = make_epsilon_greedy(env=env_config.env, device=device)

    return TrainingConfig(
        action_selector=action_selector,
        epsilon_start=cli_args.epsilon_start,
        epsilon_end=cli_args.epsilon_end,
        epsilon_decay=cli_args.epsilon_decay,
        num_episodes=cli_args.num_episodes,
        batch_size=cli_args.batch_size,
        gamma=cli_args.gamma,
        optimiser=optimiser,
        tau=cli_args.tau,
        loss_fn=loss_fn,
        device=device,
        early_return_fn=early_return_fn,
    )
