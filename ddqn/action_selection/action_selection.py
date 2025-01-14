import random
from typing import Callable

import torch
import torch.nn as nn
from gymnasium import Env
from ..structures import BloomFilter

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


def make_epsilon_greedy_with_bloom_filter(
    *,
    device: torch.device,
    env: Env,
) -> ActionSelector:
    """
    Helper function to create an ε-greedy action selection function
    that also uses a Bloom filter to help exploration.

    Parameters
    ----------
    env : Env
        Environment with an action space that is to be sampled from.
    device : torch.device
        Device on which to place torch tensors.
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

        next_choice = env.action_space.sample()
        v1 = (state.round(decimals=2), next_choice)
        if bloom_filter.contains(v1):
            # Resample
            next_choice = env.action_space.sample()
            v2 = (state.round(decimals=2), next_choice)
            bloom_filter.insert(v2)
        else:
            bloom_filter.insert(v1)

        return torch.tensor([[next_choice]], device=device)

    return inner
