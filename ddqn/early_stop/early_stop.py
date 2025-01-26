import logging
from collections import deque
from typing import Generic, Protocol, TypeVar

logger = logging.getLogger(__name__)
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


class WinsorisedRewards(Generic[T]):
    def __init__(
        self,
        *,
        num_samples: int,
        eval_frequency: int,
        clip_lower: int,
        clip_upper: int,
        score_threshold: float,
    ) -> None:
        """
        Early stopping condition based on the asymmetric winsorised
        mean of the last `num_samples`.

        Parameters
        ----------
        num_samples : int
            Number of latest episodes to consider.
        eval_frequency : int
            Frequency (in episodes) to evaluate the early return
            condition.
        clip_lower : int
            Number of smallest values to ignore.
        clip_upper : int
            Number of largest values to ignore.
        score_threshold : float
            Minimum average score to decide whether to return early on.
        """

        self.num_samples = num_samples
        self.eval_frequency = eval_frequency
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        self.score_threshold = score_threshold
        self.register = deque([], maxlen=num_samples)

    @classmethod
    def winsorised_mean(
        cls, vals: list[float], *, clip_lower: int = 0, clip_upper: int = 0
    ) -> float:
        """
        Asymmetric winsorised mean that computes the mean after
        discarding the `clip_lower` smallest values and `clip_upper`
        highest values.

        Parameters
        ----------
        vals : list[float]
            List of values to compute the mean of.
        clip_lower : int, optional
            Number of smallest values to remove before coputing mean.
        clip_upper : int, optional
            Number of largest values to remove before computing mean.
        """

        if len(vals) <= clip_lower + clip_upper:
            raise ValueError(
                "Too few values to winsorize (clip_lower, clip_upper)=(%s, %s))",
                clip_lower,
                clip_upper,
            )

        return sum(list(sorted(vals))[clip_lower : len(vals) - clip_upper]) / (
            len(vals) - clip_lower - clip_upper
        )

    def update(self, episode_reward: T) -> None:
        self.register.append(episode_reward)

    def evaluate(self, episode_num: int) -> bool:
        if episode_num % self.eval_frequency != 0:
            return False

        if len(self.register) < self.num_samples:
            return False

        wmean = self.winsorised_mean(
            self.register,
            clip_lower=self.clip_lower,
            clip_upper=self.clip_upper,
        )
        logger.debug("Winsorised mean reward: %s", wmean)

        return wmean > self.score_threshold


class RunningReward(Generic[T]):
    def __init__(
        self,
        *,
        eval_frequency: int,
        short_sample: int,
        long_sample: int,
        tolerance: float,
        min_mean: float | None = None,
    ) -> None:
        """
        Early return condition based on two means--a short range
        mean (taken over `short_sample` episodes) and a long range
        mean (taken over `long_sample` episodes).

        If the two match to within `tolerance`, then the training
        is assumed to have converged.
        """

        self.eval_frequency = eval_frequency
        self.short_sample = short_sample
        self.long_sample = long_sample
        self.tolerance = tolerance
        self.short_register = deque([], maxlen=short_sample)
        self.long_register = deque([], maxlen=long_sample)
        self.min_mean = min_mean

    def update(self, episode_reward: T) -> None:
        self.short_register.append(episode_reward)
        self.long_register.append(episode_reward)

    def evaluate(self, episode_num: int) -> bool:
        if episode_num % self.eval_frequency != 0:
            return False

        if (
            len(self.short_register) != self.short_sample
            or len(self.long_register) != self.long_sample
        ):
            return False

        short_mean = sum(self.short_register) / self.short_sample
        long_mean = sum(self.long_register) / self.long_sample
        deviation = abs(short_mean - long_mean) / max(long_mean, short_mean)

        logger.debug(
            "(short_mean, long_mean)="
            f"({short_mean.item():.3f}, {long_mean.item():.3f}), "
            f"deviation={deviation.item():.3f}",
        )

        if self.min_mean is not None and long_mean < self.min_mean:
            return False

        return deviation < self.tolerance
