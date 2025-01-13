import random
from dataclasses import dataclass
from collections import deque
from typing import Generic, TypeVar

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


class ReplayMemory(Generic[T]):
    def __init__(self, capacity: int) -> None:
        self.warmup_memory = deque[T]([], maxlen=capacity)
        self.main_memory = deque[T]([], maxlen=capacity)

    def sample(self, batch_size: int, frac: float) -> list[T] | None:
        num_warmup_samples = int(batch_size * frac)
        num_main_samples = batch_size - num_warmup_samples

        if len(self.main_memory) < 500:
            return None
        if (
            len(self.warmup_memory) < num_warmup_samples
            or len(self.main_memory) < num_main_samples
        ):
            # Pass until we have enough experience to bootstrap from
            return None

        return random.sample(
            self.warmup_memory, num_warmup_samples
        ) + random.sample(self.main_memory, num_main_samples)

    def append_warmup(self, sample: T) -> None:
        self.warmup_memory.append(sample)

    def append_main(self, sample: T) -> None:
        self.main_memory.append(sample)
