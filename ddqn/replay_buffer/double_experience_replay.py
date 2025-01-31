import random
from collections import deque
from typing import Generic, TypeVar

from ..parser import ddqn_parser

T = TypeVar("T")

ddqn_parser.add_argument(
    "--warmup-episodes",
    help="Number of episodes to append to the warmup buffer for",
    type=int,
    default=200,
)


class DoubleExperienceReplay(Generic[T]):
    """
    A replay buffer to store the target networks' experiences in and
    with which to train the policy network with. We maintain two
    working memories; a warmup buffer and a main buffer.

    The former will primarliy hold 'bad' experiences early on
    when the target network primarily selects the next action at
    random. The latter will primarily hold 'good' experiences from
    later on in the training.

    When sampling from memory, we always inject a small amount of
    'bad' experiences from the warmup buffer to try and discourage
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
        Initialise the replay buffer.

        Parameters
        ----------
        capacity : int
            Maximum size of both the warmup buffer and main buffer.
        warmup_episodes : int
            Number of episodes after which appending to the warmup
            buffer will *stop*.
        frac_warmup : float, optional
            Fraction of samples that should be taken from the warmup
            buffer instead of the main buffer.
        """

        self.warmup_episodes = warmup_episodes

        self.warmup_buffer = deque[T]([], maxlen=capacity)
        self.main_buffer = deque[T]([], maxlen=capacity)
        self.frac_warmup = frac_warmup

    def sample(self, batch_size: int) -> list[T] | None:
        """
        Sample from the main buffer. If `self.frac_warmup` is not
        zero, then a proportionate number of samples will be drawn
        from the warmup buffer.

        Parameters
        ----------
        batch_size : int
            Total number of objects to sample from buffer.

        Returns
        -------
        list[T] | None
            `None` if there are insufficient elements in the two
            memories to sample from. Otherwise, `list[T]` of samples.
        """

        num_warmup_samples = int(batch_size * self.frac_warmup)
        num_main_samples = batch_size - num_warmup_samples

        if len(self.warmup_buffer) < num_warmup_samples or len(
            self.main_buffer
        ) < max(num_main_samples, 4 * batch_size):
            # Pass until we have enough experience to bootstrap from
            return None

        return random.sample(
            self.warmup_buffer, num_warmup_samples
        ) + random.sample(self.main_buffer, num_main_samples)

    def append(self, sample: T) -> None:
        if len(self.warmup_buffer) < self.warmup_episodes:
            self.warmup_buffer.append(sample)
        else:
            self.main_buffer.append(sample)
