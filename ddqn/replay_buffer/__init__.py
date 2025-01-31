from typing import Generic, TypeVar, Protocol


from .experience_replay import ExperienceReplay
from .double_experience_replay import DoubleExperienceReplay
from .state_change import StateChange, StateChanges

T = TypeVar("T")


class ReplayBuffer(Protocol, Generic[T]):
    """
    Protocol for replay buffers.
    """

    def sample(self, n_samples: int) -> list[T] | None: ...

    def append(self, sample: T) -> None: ...


__all__ = [
    "DoubleExperienceReplay",
    "ExperienceReplay",
    "ReplayBuffer",
    "StateChange",
    "StateChanges",
]
