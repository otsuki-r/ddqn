import math
import random
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env

from . import ActionSelector
from ..parser import ddqn_parser


# Register args
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
    "--epsilon-exploration-steps",
    help="Number of pure exploration steps",
    type=int,
    default=0,
)


class EpsilonSchedule(Protocol):
    def __call__(self, global_step_number: int) -> float: ...


class ExponentialDecay:
    def __init__(
        self,
        start: float,
        end: float,
        decay: float,
        exploration_steps: int = 0,
    ) -> None:
        self.epsilon_start = start
        self.epsilon_end = end
        self.epsilon_decay = decay
        self.epsilon_delta = self.epsilon_start - self.epsilon_end
        self.exploration_steps = exploration_steps

    def __call__(self, global_step_number: int) -> float:
        r"""
        Epsilon schedule with a built in exploration phase

        $$
        \epsilon(t) = H(\epsilon_{\text{exp}} - t)
            + H(t - \epsilon_{\text{exp}}) \times (
                \epsilon_f + (\epsilon_{\text{i}} - \epsilon_{\text{f}}) \times e^{- t * \epsilon_{\text{decay}}}
            )
        $$

        where
        * $\epsilon_{\text{exp}}$ is a number of exploration steps
            for which $\epsilon$ is fixed to 1 (100% exploration)
        * $\espilon_{\text{i}}$ is the initial $\epsilon$
        * $\espilon_{\text{f}}$ is the target final $\epsilon$
        * $\espilon_{\text{decay}}$ is the decay rate of $\epsilon$
        """

        return (
            np.heaviside(self.exploration_steps - global_step_number, 0.5)
            + np.heaviside(global_step_number - self.exploration_steps, 0.5)
            * (
                self.epsilon_end
                + self.epsilon_delta
                * math.exp(-global_step_number * self.epsilon_decay)
            )
        ).item()


def make_epsilon_greedy(
    pnet: nn.Module,
    env: Env,
    *,
    smoothed: bool = False,
    dtype: torch.dtype,
    device: torch.device,
    regularisation: float = 1e-6,
) -> ActionSelector:
    """
    Helper function to create an ε-greedy action selection function.

    Parameters
    ----------
    env : Env
        Environment to train in.
    dtype : torch.dtype
        Dtype of tensor to create
    device : torch.device
        Device to put tensor on.
    smoothed : bool, optional
        Whether to choose the action based on max of state values,
        or over a distribution of probabilities (when the random
        action is not chosen).)
    regularisation : float, optional
        Regularisation to prevent zero probabilities in the softmax.
    """

    def inner(this_epsilon: float, state: torch.Tensor) -> torch.Tensor:
        """
        Decision function for choosing which action to take.

        Parameters
        ----------
        this_epsilon : float
            Value of decaying ε parameter to evaluation the action
            selection at.
        state : torch.Tensor
            The current state of the agent.

        Returns
        -------
        torch.Tensor
            Choice of action taken, sampled from the environment's
            action space according to the ε-greedy algorithm.
        """

        if random.random() > this_epsilon:
            with torch.no_grad():
                return pnet(state).argmax().unsqueeze(0)

        next_choice = env.action_space.sample()

        return torch.tensor([next_choice], dtype=dtype, device=device)

    def inner_smoothed(
        this_epsilon: float, state: torch.Tensor
    ) -> torch.Tensor:
        """
        Decision function for choosing which action to take.

        Parameters
        ----------
        this_epsilon : float
            Value of decaying ε parameter to evaluation the action
            selection at.
        state : torch.Tensor
            The current state of the agent.

        Returns
        -------
        torch.Tensor
            Choice of action taken, sampled from the environment's
            action space according to the ε-greedy algorithm.

        Notes
        -----
        In this smoothed variant, the state-aciton values are taken as
        relative weights in a probability distribution. This is to
        smooth out sharp changes to the action selection under small
        changes to the Q-values.

        In the unsmoothed versions, if the state-action values for
        3 possible actions are [0.2, 0.6, 0.7] before an update and
        [0.3, 0.7, 0.6] after an update, there will be a hard change
        in action seletion from action 3 to action 2. The smooth
        version translates smooth changes in the Q-value to smooth
        changes in the action selection.
        """

        if random.random() > this_epsilon:
            with torch.no_grad():
                return torch.clamp(pnet(state), regularisation).multinomial(
                    num_samples=1
                )

        next_choice = env.action_space.sample()

        return torch.tensor([next_choice], dtype=dtype, device=device)

    # Separate out the function defs entirely so we don't need to
    # do an extra `if` state inside the bodies to handle the
    # `smoothed` parameter.
    if smoothed:
        return inner_smoothed
    else:
        return inner
