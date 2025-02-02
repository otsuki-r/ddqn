import argparse
import pathlib
from typing import Callable
from dataclasses import dataclass

import torch
import torch.optim as optim
from gymnasium.envs.registration import Env
from .action_selection import ActionSelector
from .early_stop import EarlyStop, NullEarlyStop
from .adaptors import torch_from_array

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    epsilon_exploration_steps: int = 0
    num_episodes: int
    batch_size: int
    gamma: float
    optimiser: optim.Optimizer
    tau: float
    action_selector_fn: ActionSelector
    loss_fn: torch.nn.modules.loss._Loss
    outdir: pathlib.Path
    early_return_fn: EarlyStop = NullEarlyStop


@dataclass
class RunConfig:
    weights_dir: pathlib.Path
    save_path: pathlib.Path | None = None
    num_iterations: int | None = None


@dataclass
class EnvConfig:
    env: Env
    action_space_size: int
    state_space_size: int
    state_space_adaptor: Callable[[...], torch.Tensor] = torch_from_array
    state_space_dtype: torch.dtype = torch.float32
    action_space_dtype: torch.dtype = torch.int64
    reward_dtype: torch.dtype = torch.float32


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

    if cli_args.epsilon_exploration_steps < 0:
        raise ValueError("Number of exploration steps must be poitive")

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
