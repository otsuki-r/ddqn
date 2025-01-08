import math
import random
from typing import Callable

import torch
import torch.nn as nn
from gymnasium import Env


def make_action_selector(
    *,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    use_bloom: bool = False,
    device: torch.device,
    env: Env,
) -> Callable[[int, torch.Tensor, nn.Module], torch.Tensor]:
    def inner(
        step_number: int,
        state: torch.Tensor,
        pnet: nn.Module,
    ) -> torch.Tensor:
        this_eps_threshold = eps_end + (eps_start - eps_end) * math.exp(
            -step_number * eps_decay
        )
        if random.random() > this_eps_threshold:
            with torch.no_grad():
                return pnet(state).max(1).indices.view(1, 1)

        return torch.tensor([[env.action_space.sample()]], device=device)

    return inner
