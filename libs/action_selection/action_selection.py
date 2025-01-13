import random
from typing import Callable

import torch
import torch.nn as nn
from gymnasium import Env
from ..bloom_filter import BloomFilter


def make_action_selector(
    *,
    use_bloom: bool = False,
    device: torch.device,
    env: Env,
) -> Callable[[float, torch.Tensor, nn.Module], torch.Tensor]:
    bloom_filter: BloomFilter[tuple[torch.Tensor, int]] | None
    if use_bloom:
        bloom_filter = BloomFilter(10_000)
    else:
        bloom_filter = None

    def inner(
        this_epsilon: float,
        state: torch.Tensor,
        pnet: nn.Module,
    ) -> torch.Tensor:
        if random.random() > this_epsilon:
            with torch.no_grad():
                return pnet(state).max(1).indices.view(1, 1)

        next_choice = env.action_space.sample()
        if bloom_filter:
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
