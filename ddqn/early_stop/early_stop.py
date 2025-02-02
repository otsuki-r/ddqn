import logging
from collections import deque
from typing import Generic, TypeVar

from ..parser import ddqn_parser

logger = logging.getLogger(__name__)
T = TypeVar("T")

ddqn_parser.add_argument(
    "--early-stop-frequency",
    help="Frequency for evaluating early stop (in episodes)",
    type=int,
    default=50,
)


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
        self.short_total = 0.0
        self.long_total = 0.0
        self.min_mean = min_mean

    def update(self, episode_reward: T) -> None:
        if len(self.short_register) == self.short_sample:
            self.short_total -= self.short_register[0]

        if len(self.long_register) == self.long_sample:
            self.long_total -= self.long_register[0]

        self.short_register.append(episode_reward)
        self.long_register.append(episode_reward)
        self.short_total += episode_reward
        self.long_total += episode_reward

    def evaluate(self, episode_num: int) -> bool:
        if episode_num % self.eval_frequency != 0:
            return False

        if (
            len(self.short_register) != self.short_sample
            or len(self.long_register) != self.long_sample
        ):
            return False

        short_mean = self.short_total / self.short_sample
        long_mean = self.long_total / self.long_sample
        deviation = abs(short_mean - long_mean) / max(long_mean, short_mean)

        logger.debug(
            "(short_mean, long_mean)="
            f"({short_mean.item():.3f}, {long_mean.item():.3f}), "
            f"deviation={deviation.item():.3f}",
        )

        if self.min_mean is not None and long_mean < self.min_mean:
            return False

        return deviation < self.tolerance
