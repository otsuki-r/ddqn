import enum
import logging
from collections import deque
from typing import Generic, Protocol, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


class WatcherTarget(enum.StrEnum):
    REWARD = enum.auto()
    LOSS = enum.auto()
    EPISODE_DURATIONS = enum.auto()
    EPSILON = enum.auto()


class WatcherType(enum.StrEnum):
    EPISODES = enum.auto()
    GLOBAL_STEP = enum.auto()


class Watcher(Protocol, Generic[T]):
    def append(self, val: T, *, counter: int) -> None: ...


class ValueWatcher(Generic[T]):
    def __init__(
        self,
        target: WatcherTarget,
        watcher_type: str,
        freq: int = 50,
    ) -> None:
        self.target = target
        self.watcher_type = WatcherType(watcher_type)
        self.freq = freq
        self.counter_str = {
            WatcherType.EPISODES: "Episode",
            WatcherType.GLOBAL_STEP: "Step",
        }[self.watcher_type]

    def append(self, val: T, *, counter: int) -> None:
        if val:
            val = f"{val:.8f}"

        if counter % self.freq != 0:
            return

        logger.info(f"{self.counter_str} {counter} {self.target}: {val}")
        return


class MovingAverageWatcher(Generic[T]):
    def __init__(
        self,
        capacity: int,
        target: WatcherTarget,
        watcher_type: str,
        freq: int = 50,
    ) -> None:
        self.capacity = capacity
        self.target = target
        self.watcher_type = WatcherType(watcher_type)
        self.freq = freq
        self.buffer: deque[T] = deque([], maxlen=capacity)
        self.counter_str = {
            WatcherType.EPISODES: "Episode",
            WatcherType.GLOBAL_STEP: "Step",
        }[self.watcher_type]

    def append(self, val: T, *, counter: int) -> None:
        self.buffer.append(val)

        if counter % self.freq != 0:
            return

        mean = sum(self.buffer) / len(self.buffer)
        logger.info(
            f"{self.counter_str} {counter} {self.target} MA@{self.capacity}: {mean:.3f}"
        )
        return
