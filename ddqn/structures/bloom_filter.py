import hashlib
from typing import Generic, Generator, TypeVar
from collections.abc import Hashable

T = TypeVar("T", bound=Hashable)


class BloomFilter(Generic[T]):
    """
    A probabilistic data structure that can:
    1. definitively determine that an element *has not* been seen
       before;
    2. probabilistically determine that an element *has* been seen
       before.

    However, it cannot count how many times the element has been seen
    before.

    It has a very light memory footprint, and the accuracy of the set
    membership is determined primarily by the number of hashes that it
    computes internally. It thus trades of space for computation and
    is useful in cases where the cardinality of the set can be
    extremely large, and where the accuracy of set membership is not
    critical.
    """

    def __init__(self, capacity: int, num_hashes: int = 5) -> None:
        """
        Initialise the Bloom filter with linear size `capacity` and
        number of hashes `num_hashes`.

        Parameters
        ----------
        capacity : int
            Linear size of the filter.
        num_hashes : int, optional
            Number of hashes to compute. More hashes equate to fewer
            false positives in identifying set membership, at the
            expense of more compute power.
        """

        self.field = [False] * capacity
        self.capacity = capacity
        self.num_hashes = num_hashes

    def _hash_multiple(self, obj: T) -> Generator[int, None, None]:
        """
        Helper function to yield hash results of the element `obj`.

        Parameters
        ----------
        obj : T
            Object whose hashes are to be computed.

        Returns
        -------
        int
            Successive hashes of object.
        """

        for i in range(self.num_hashes):
            hasher = hashlib.sha256()
            hasher.update(str((i, obj)).encode())
            yield int(hasher.hexdigest(), 16) % self.capacity

    def insert(self, obj: T) -> None:
        """
        Insert operation for the Bloom filter.

        Parameters
        ----------
        obj : T
            Object to insert.
        """

        for b in self._hash_multiple(obj):
            self.field[b] = True

    def contains(self, obj: T) -> bool:
        """
        Checks set membership of `obj`.

        Parameters
        ----------
        obj : T
            Object to check set membership of.

        Returns
        -------
        bool
            Set membership of `obj`.
        """

        return all(self.field[b] for b in self._hash_multiple(obj))
