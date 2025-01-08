import random
from dataclasses import dataclass
from collections import deque
from typing import TypeVar

import torch

T = TypeVar("T")


@dataclass
class StateChange:
    state: torch.Tensor  # s_t
    action: torch.Tensor  # a_t
    reward: torch.Tensor  # r_{t+1}
    next_state: torch.Tensor | None  # s_{t+1}


@dataclass
class StateChanges:
    states: list[torch.Tensor]
    actions: list[torch.Tensor]
    rewards: list[torch.Tensor]
    next_states: list[torch.Tensor | None]

    def __init__(self, state_changes: list[StateChange]) -> None:
        self.states = [s.state for s in state_changes]
        self.actions = [s.action for s in state_changes]
        self.rewards = [s.reward for s in state_changes]
        self.next_states = [s.next_state for s in state_changes]


class ReplayMemory(deque[T]):
    def __init__(self, capacity: int) -> None:
        super().__init__([], maxlen=capacity)

    def sample(self, num_samples: int) -> list[T]:
        return random.sample(self, num_samples)
