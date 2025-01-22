import argparse
import functools
import hashlib
import logging
import pathlib
import random
from typing import Generic, Generator, TypeVar
from collections.abc import Hashable

from ddqn.configure import DEVICE, EnvConfig
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from ddqn.structures import DoubleDQN
from ddqn.early_stop import winsorised_durations_early_return
from ddqn.action_selection import ActionSelector, make_epsilon_greedy
from ddqn.configure import (
    RunConfig,
    TrainingConfig,
    ddqn_parser,
    get_render_mode,
)

T = TypeVar("T", bound=Hashable)

ddqn_parser.add_argument(
    "--use-bloom",
    help="Whether to use a Bloom filter for action selection during training.",
    type=bool,
    action=argparse.BooleanOptionalAction,
)


class CenteredCartPole(gym.Env):
    """
    Modified CartPole env with a penalty to encourage it to stay in
    the middle of the viewport.
    """

    def __init__(self, render_mode: str | None = None) -> None:
        self.env = gym.make("CartPole-v1", render_mode=render_mode)

        def penalise_non_centered(fn):
            @functools.wraps(fn)
            def inner(*args, **kwargs):
                next_state, reward, *g = fn(*args, **kwargs)

                # Penalise if cart position is too far from center
                # n.b. -2.4 <= cart_position <= 2.4
                reward -= min(next_state[0] ** 2 * 10, 1)

                return next_state, reward, *g

            return inner

        setattr(self.env, "step", penalise_non_centered(self.env.step))


class BloomFilter(Generic[T]):
    """
    A probabilistic data structure that can:
    1. definitively determine that an element *has not* been seen
       before;
    2. probabilistically determine that an element *has* been seen
       before.

    However, it cannot count how many times the element has been seen
    before.

    It has a very light memory footprint, and the accuracy of the set
    membership is determined primarily by the number of hashes that it
    computes internally. It thus trades of space for computation and
    is useful in cases where the cardinality of the set can be
    extremely large, and where the accuracy of set membership is not
    critical.
    """

    def __init__(self, capacity: int, num_hashes: int = 5) -> None:
        """
        Initialise the Bloom filter with linear size `capacity` and
        number of hashes `num_hashes`.

        Parameters
        ----------
        capacity : int
            Linear size of the filter.
        num_hashes : int, optional
            Number of hashes to compute. More hashes equate to fewer
            false positives in identifying set membership, at the
            expense of more compute power.
        """

        self.field = [False] * capacity
        self.capacity = capacity
        self.num_hashes = num_hashes

    def _hash_multiple(self, obj: T) -> Generator[int, None, None]:
        """
        Helper function to yield hash results of the element `obj`.

        Parameters
        ----------
        obj : T
            Object whose hashes are to be computed.

        Returns
        -------
        int
            Successive hashes of object.
        """

        for i in range(self.num_hashes):
            hasher = hashlib.sha256()
            hasher.update(str((i, obj)).encode())
            yield int(hasher.hexdigest(), 16) % self.capacity

    def insert(self, obj: T) -> None:
        """
        Insert operation for the Bloom filter.

        Parameters
        ----------
        obj : T
            Object to insert.
        """

        for b in self._hash_multiple(obj):
            self.field[b] = True

    def contains(self, obj: T) -> bool:
        """
        Checks set membership of `obj`.

        Parameters
        ----------
        obj : T
            Object to check set membership of.

        Returns
        -------
        bool
            Set membership of `obj`.
        """

        return all(self.field[b] for b in self._hash_multiple(obj))


def make_epsilon_greedy_with_bloom_filter(
    env_config: EnvConfig,
) -> ActionSelector:
    """
    Helper function to create an ε-greedy action selection function
    that also uses a Bloom filter to help exploration.

    Parameters
    ----------
    env_config : EnvConfig
        Environment config with an action space that is to be sampled.
    """

    bloom_filter = BloomFilter[tuple[torch.Tensor, int]](10_000)

    def inner(
        this_epsilon: float,
        state: torch.Tensor,
        pnet: nn.Module,
    ) -> torch.Tensor:
        """
        Decision function for choosing which action to take.
        This function is largely the same as an ε-greedy algorithm,
        but it also maintains a Bloom filter to remember what actions
        have been taken in what state.

        If a random choice is chosen and the random action chosen
        from the current state (i.e. the combination $(s_t, a_t)$)
        has been taken before, then this duplication will be picked up
        by the Bloom filter, and it will reroll the random choice.

        Note that we permit this reroll to land on the same action
        choice again as rerunning the same action from the same state
        can serve to reinforce the decision. However, if the reroll
        differs from the initial random choice, then we will have
        encouraged the network to explore.

        Parameters
        ----------
        this_epsilon : float
            Value of decaying ε parameter to evaluation the action
            selection at.
        state : torch.Tensor
            The current state of the agent.
        pnet : nn.Module
            The policy net of the double DQN  being trained.

        Returns
        -------
        torch.Tensor
            Choice of action taken, sampled from the environment's
            action space according to the ε-greedy algorithm.

        Notes
        -----
        Since the state dtype if float32, we round each value in the
        state to 2 d.p. to group similar states into discrete buckets
        for the purposes of the Bloom filter.
        """

        if random.random() > this_epsilon:
            with torch.no_grad():
                return pnet(state).max(1).indices.view(1, 1)

        next_choice = env_config.env.action_space.sample()
        v1 = (state.round(decimals=2), next_choice)
        if bloom_filter.contains(v1):
            # Resample
            next_choice = env_config.env.action_space.sample()
            v2 = (state.round(decimals=2), next_choice)
            bloom_filter.insert(v2)
        else:
            bloom_filter.insert(v1)

        return torch.tensor(
            [[next_choice]], dtype=env_config.state_space_dtype, device=DEVICE
        )

    return inner


def build_env_config(cli_args: argparse.Namespace) -> EnvConfig:
    """
    Helper function to construct environment configurations from CLI args

    Paramaters
    ----------
    cli_args: argparse.Namespace
        Arguments parsed from the CLI

    Return
    ------
    EnvConfig
        Environment configuration built form CLI args.
    """

    render_mode = get_render_mode(cli_args)
    cart_pole = CenteredCartPole(render_mode=render_mode)
    initial_state, _ = cart_pole.env.reset(seed=42)

    return EnvConfig(
        env=cart_pole.env,
        action_space_size=int(cart_pole.env.action_space.n),  # type: ignore
        state_space_size=len(initial_state),
    )


def build_training_config(
    ddqn: DoubleDQN,
    cli_args: argparse.Namespace,
    env_config: EnvConfig,
) -> TrainingConfig:
    """
    Collate CLI args and environment configurations into a training
    configuration.

    Parameters
    ----------
    ddqn : DoubleDQN
        Network to train.
    cli_args : argparse.Namespace
        Args parsed from the command line.
    env_config : EnvConfig
        Configuration of the environment to train in.

    Returns
    -------
    TrainingConfig
        Traning configuration constructed from inputs.
    """

    if cli_args.use_bloom:
        action_selector = make_epsilon_greedy_with_bloom_filter(env=env_config)
    else:
        action_selector = make_epsilon_greedy(env=env_config)

    tc = TrainingConfig(
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
        action_selector_fn=action_selector,
        loss_fn=nn.SmoothL1Loss(),  # Generalisation of robust Huber loss
        early_return_fn=winsorised_durations_early_return(
            last_n=20, clip_lower=5, clip_upper=0, score_threshold=470.0
        ),
        outdir=pathlib.Path(cli_args.outdir),
    )

    logging.info("Training with config: %s", tc)

    return tc


def build_run_config(cli_args: argparse.Namespace) -> RunConfig:
    """
    Helper function to build run configurations

    Parameters
    ----------
    cli_args : argparse.Namespace
        Arguments parsed from the command line.

    Returns
    -------
    RunConfig
        Configuration required for running the model.
    """

    outdir = pathlib.Path(cli_args.outdir)
    spath = pathlib.Path(cli_args.save_path) if cli_args.save_path else None
    rc = RunConfig(weights_dir=outdir, save_path=spath)

    logging.info("Running with config: %s", rc)

    return rc
