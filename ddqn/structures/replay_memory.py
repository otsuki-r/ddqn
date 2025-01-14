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
    """
    Replay memory to store the target networks' experiences in and
    with which to train the policy network with. We maintain two
    working memories; a warmup memory and a main memory.

    The former will primarliy hold 'bad' experiences early on
    when the target network primarily selects the next action at
    random. The latter will primarily hold 'good' experiences from
    later on in the training.

    When sampling from memory, we always inject a small amount of
    'bad' experiences from the warmup memory to try and discourage
    catastrophic forgetting wherein the network can forget how to
    handle a 'bad' state as it had seen 'good' states for so long in
    the training.
    """

    def __init__(
        self,
        capacity: int,
        warmup_episodes_upper: int,
        main_episodes_lower: int,
    ) -> None:
        """
        Initialise replay memory.

        Parameters
        ----------
        capacity : int
            Maximum size of both the warmup memory and main memories.
        warmup_episodes_upper : int
            Number of episodes after which appending to the warmup
            memory will *stop*.
        main_episodes_lower : int
            Number of episodes after which appending to the main
            memory will *start*.
        """

        self.warmup_episodes_upper = warmup_episodes_upper
        self.main_episodes_lower = main_episodes_lower

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
