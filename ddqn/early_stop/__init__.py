from typing import Generic, Protocol, TypeVar

T = TypeVar("T")


class EarlyStop(Protocol, Generic[T]):
    """Protocol for evaluating an early return from training or not"""

    def update(self, episode_reward: T) -> None:
        """
        Update the internal state of this early condition.

        Paramaters
        ----------
        episode_reward : T
            Reward of the last episode.
        """

    def evaluate(self, episode_num: int) -> bool:
        """
        Evaluates whether this episode should terminate the training
        early or not.

        Paramaters
        ----------
        episode_num : int
            The episode number being evaluated.

        Returns
        --------
        bool
            Whether the early return condition has been met or not.
        """


class NullEarlyStop(Generic[T]):
    def update(self, episode_reward: T) -> None:
        return

    def evaluate(self, episode_num: int) -> bool:
        return False


__all__ = ["EarlyStop", "NullEarlyStop"]
