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

    return sum(list(sorted(vals))[clip_lower : len(vals) - clip_upper]) / (
        len(vals) - clip_lower - clip_upper
    )


def winsorised_durations_early_return(
    *, last_n: int, clip_lower: int, clip_upper: int, score_threshold: float
) -> EarlyReturnFn:
    """
    Makes an early stopping condition based on winsorised means of
    the cumulative rewards of the last few episodes.

    Parameters
    ----------
    last_n : int
        Number of latest episodes to consider.
    clip_lower : int
        Number of smallest values to ignore.
    clip_upper : int
        Number of largest values to ignore.
    score_threshold : float
        Minimum average score to decide whether to return early on.

    Returns
    --------
    EarlyReturnFn
        Function that computes whether to return early from training
        based on the outputs of the last few episodes.
    """

    def inner(
        *,
        episode_durations: list[int],
        rewards: list[float],
        losses: list[float],
    ) -> bool:
        """
        Inner decision fucntion. Computes the winsorised mean
        according to the parameters in the constructor, and compares
        it to the supplied thresold.

        Parameters
        ----------
        epsiode_durations : list[int]
            Durations of each episode.
        rewards : list[float]
            Cumulative rewards of each epsiode.
        losses : list[float]
            Final loss of each episode.

        Returns
        -------
        bool
            Whether we should stop training early or not.
        """

        try:
            wmean = _winsorised_mean(
                rewards[-last_n:], clip_lower=clip_lower, clip_upper=clip_upper
            )
        except ValueError:
            return False

        if wmean > score_threshold:
            logger.info("Stopping training early.")
            return True
        return False

    return inner
