import random
from typing import Callable

import torch
import torch.nn as nn
from gymnasium import Env

ActionSelector = Callable[[float, torch.Tensor, nn.Module], torch.Tensor]


def make_epsilon_greedy(
    *,
    env: Env,
    device: torch.device,
) -> ActionSelector:
    """
    Helper function to create an ε-greedy action selection function.

    Parameters
    ----------
    env : Env
        Environment with an action space that is to be sampled from.
    device : torch.device
        Device on which to place torch tensors.
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
                return pnet(state).max(1).indices.view(1, 1)

        next_choice = env.action_space.sample()

        return torch.tensor([[next_choice]], device=device)

    return inner
