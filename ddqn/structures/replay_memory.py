import random
from dataclasses import dataclass
from collections import deque
from typing import Generic, Protocol, TypeVar

import torch

T = TypeVar("T")


@dataclass
class StateChange:
    state: torch.Tensor  # s_t
    action: torch.Tensor  # a_t
    reward: torch.Tensor  # r_{t+1}
    next_state: torch.Tensor | None  # s_{t+1}
    td_error: torch.Tensor | None


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


class ReplayBuffer(Protocol, Generic[T]):
    """
    Protocol for replay buffers.
    """

    def sample(self, n_samples: int) -> list[T] | None: ...

    def append(self, sample: T) -> None: ...


class DoubleReplayMemory(Generic[T]):
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
        warmup_episodes: int,
        frac_warmup: float = 0.05,
    ) -> None:
        """
        Initialise replay memory.

        Parameters
        ----------
        capacity : int
            Maximum size of both the warmup memory and main memories.
        warmup_episodes : int
            Number of episodes after which appending to the warmup
            memory will *stop*.
        frac_warmup : float, optional
            Fraction of samples that should be taken from the warmup
            memory instead of the main memory.
        """

        self.warmup_episodes = warmup_episodes

        self.warmup_memory = deque[T]([], maxlen=capacity)
        self.main_memory = deque[T]([], maxlen=capacity)
        self.frac_warmup = frac_warmup

    def sample(self, batch_size: int) -> list[T] | None:
        """
        Sample from the main memory. If `self.frac_warmup` is not
        zero, then a proportionate number of samples will be drawn
        from the warmup memory.

        Parameters
        ----------
        batch_size : int
            Total number of objects to sample from memory.

        Returns
        -------
        list[T] | None
            `None` if there are insufficient elements in the two
            memories to sample from. Otherwise, `list[T]` of samples.
        """

        num_warmup_samples = int(batch_size * self.frac_warmup)
        num_main_samples = batch_size - num_warmup_samples

        if len(self.warmup_memory) < num_warmup_samples or len(
            self.main_memory
        ) < max(num_main_samples, 4 * batch_size):
            # Pass until we have enough experience to bootstrap from
            return None

        return random.sample(
            self.warmup_memory, num_warmup_samples
        ) + random.sample(self.main_memory, num_main_samples)

    def append(self, sample: T) -> None:
        if len(self.warmup_memory) < self.warmup_episodes:
            self.warmup_memory.append(sample)
        else:
            self.main_memory.append(sample)
