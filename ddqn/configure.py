import argparse
import pathlib
from dataclasses import dataclass

import torch
import torch.optim as optim
from gymnasium.envs.registration import Env
from .action_selection import ActionSelector
from .early_stop import EarlyReturnFn


ddqn_parser = argparse.ArgumentParser("ddqn")

ddqn_parser.add_argument(
    "--memory-size",
    help="Max size of the replay memory buffer.",
    type=int,
    default=10_000,
)
ddqn_parser.add_argument(
    "--warmup-episodes-upper",
    help="Number of episodes to append to warmup memory for",
    type=int,
    default=60,
)
ddqn_parser.add_argument(
    "--main-episodes-lower",
    help="Number of episodes to wait before starting to append to main memory",
    type=int,
    default=200,
)
ddqn_parser.add_argument(
    "--epsilon-start",
    help="Start value for ε used in epsilon-greedy action selection.",
    type=float,
    default=0.90,
)
ddqn_parser.add_argument(
    "--epsilon-end",
    help="End value for ε used in epsilon-greedy action selection.",
    type=float,
    default=0.05,
)
ddqn_parser.add_argument(
    "--epsilon-decay",
    help="Exponential decay rate of ε used in epsilon-greedy action selection.",
    type=float,
    default=1e-4,
)
ddqn_parser.add_argument(
    "--num-episodes",
    help="Number of episodes to train for.",
    type=int,
    default=600,
)
ddqn_parser.add_argument(
    "--batch-size",
    help="Batch size to be used in training.",
    type=int,
    default=128,
)
ddqn_parser.add_argument(
    "--gamma",
    help="Discounting power γ of past actions in computing reward",
    type=float,
    default=0.95,
)
ddqn_parser.add_argument(
    "--alpha",
    help="Learing rate α of the optimiser to be used in training.",
    type=float,
    default=1e-3,
)
ddqn_parser.add_argument(
    "--tau",
    help="Update rate τ of target network (soft update).",
    type=float,
    default=1e-2,
)
ddqn_parser.add_argument(
    "--outdir",
    help="Directory to save results to. Will be created if it does not exist.",
    type=str,
    default="./out",
)
ddqn_parser.add_argument(
    "--eval",
    help="Runs the model specified in --outdir and renders the run",
    type=bool,
    action=argparse.BooleanOptionalAction,
)
ddqn_parser.add_argument(
    "--save-path",
    help="If supplied with the --eval flag set, will save a GIF of a run to the supplied path",  # type:ignore
    type=str,
    default=None,
)


@dataclass
class TrainingConfig:
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    num_episodes: int
    batch_size: int
    gamma: float
    optimiser: optim.Optimizer
    tau: float
    action_selector_fn: ActionSelector
    loss_fn: torch.nn.modules.loss._Loss
    device: torch.device
    early_return_fn: EarlyReturnFn | None
    outdir: pathlib.Path


@dataclass
class RunConfig:
    weights_dir: pathlib.Path
    save_path: pathlib.Path | None = None


@dataclass
class EnvConfig:
    env: Env
    action_space_size: int
    state_space_size: int


def process_cli_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parse and validate arguments from the CLI

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser to process.

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


def get_render_mode(cli_args: argparse.Namespace) -> str | None:
    """
    Computes the render mode from the parsed CLI args.

    Parameters
    ----------
    cli_args : argparse.Namespace
        Arguments parsed from the CLI.

    Return
    ------
    str | None
        The appropriate render mode parsed from the cli config.
    """

    render_mode: str | None
    if cli_args.eval:
        if cli_args.save_path:
            render_mode = "rgb_array"
        else:
            render_mode = "human"
    else:
        # For training
        render_mode = None

    return render_mode
