from collections import deque
from .state_change import StateChange

import numpy as np
import torch


class ExperienceReplay:
    def __init__(self, capacity: int, min_batch_multiple: int = 4) -> None:
        self.min_batch_multiple = min_batch_multiple
        self.buffer = deque[StateChange](maxlen=capacity)
        self.capacity = capacity

    def sample(self, batch_size: int) -> list[StateChange] | None:
        if len(self.buffer) < self.min_batch_multiple * batch_size:
            return None

        indices = torch.tensor(
            np.random.choice(
                min(len(self.buffer), self.capacity), batch_size, replace=False
            )
        )
        return [self.buffer[i] for i in indices]

    def append(self, sample: StateChange) -> None:
        self.buffer.append(sample)
