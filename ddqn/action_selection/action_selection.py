import random
from typing import Callable

import torch
import torch.nn as nn
from ..configure import DEVICE, EnvConfig

ActionSelector = Callable[[float, torch.Tensor, nn.Module], torch.Tensor]


def make_epsilon_greedy(env_config: EnvConfig) -> ActionSelector:
    """
    Helper function to create an ε-greedy action selection function.

    Parameters
    ----------
    env_config : EnvConfig
        Environment config with an action space that is to be sampled.
    """

    def inner(
        this_epsilon: float,
        state: torch.Tensor,
        pnet: nn.Module,
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
        pnet : nn.Module
            The policy net of the double DQN  being trained.

        Returns
        -------
        torch.Tensor
            Choice of action taken, sampled from the environment's
            action space according to the ε-greedy algorithm.
        """

        if random.random() > this_epsilon:
            with torch.no_grad():
                return pnet(state).argmax().unsqueeze(0)

        next_choice = env_config.env.action_space.sample()

        return torch.tensor(
            [next_choice], dtype=env_config.state_space_dtype, device=DEVICE
        )

    return inner
