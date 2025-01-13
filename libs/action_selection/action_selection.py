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
    def inner(
        this_epsilon: float,
        state: torch.Tensor,
        pnet: nn.Module,
    ) -> torch.Tensor:
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
    bloom_filter = BloomFilter[tuple[torch.Tensor, int]](10_000)

    def inner(
        this_epsilon: float,
        state: torch.Tensor,
        pnet: nn.Module,
    ) -> torch.Tensor:
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
