import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from mypy_extensions import NamedArg

    EarlyReturnFn = Callable[
        [
            NamedArg(list[int], "episode_durations"),
            NamedArg(list[float], "rewards"),
            NamedArg(list[float], "losses"),
        ],
        bool,
    ]
else:
    EarlyReturnFn = Any

logger = logging.getLogger(__name__)


def _winsorised_mean(
    vals: list[float], *, clip_lower: int = 0, clip_upper: int = 0
) -> float:
    """
    Asymmetric winsorised mean that computes the mean after discarding
    the `clip_lower` smallest values and `clip_upper` highest values.

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
            "Too few values to winsorize (clip_lower, clip_upper) = (%s, %s))",
            clip_lower,
            clip_upper,
        )

    return sum(list(sorted(vals))[clip_lower:-clip_upper]) / (
        len(vals) - clip_lower - clip_upper
    )


def winsorised_durations_early_return(
    last_n: int, score_threshold: float
) -> EarlyReturnFn:
    """
    Makes an early stopping condition based on winsorised means of
    the cumulative rewards of the last few episodes.

    Parameters
    ----------
    last_n : int
        Number of latest episodes to consider.
    score_threshold : float
        Minimum average score to decide whether to return early on.

    Returns
    --------
    EarlyReturnFn
        Function that computes an whether to return early from
        training based on the outputs of the last few episodes.
    """

    def inner(
        *,
        episode_durations: list[int],
        rewards: list[float],
        losses: list[float],
    ) -> bool:
        """
        epsiode_durations : list[int]
            Durations of each episode.
        rewards : list[float]
            Cumulative rewards of each epsiode.
        losses : list[float]
            Final loss of each episode.
        """

        try:
            wmean = _winsorised_mean(rewards[-last_n:])
        except ValueError:
            return False

        if wmean > score_threshold:
            logger.info("Early finish")
            return True
        return False

    return inner
