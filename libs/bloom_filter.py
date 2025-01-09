import hashlib
from typing import Generic, Generator, TypeVar
from collections.abc import Hashable

T = TypeVar("T", bound=Hashable)


class BloomFilter(Generic[T]):
    def __init__(self, capacity: int, num_hashes: int = 5) -> None:
        self.field = [False] * capacity
        self.capacity = capacity
        self.num_hashes = num_hashes

    def hash_multiple(self, obj: T) -> Generator[int, None, None]:
        for i in range(self.num_hashes):
            hasher = hashlib.sha256()
            hasher.update(str((i, obj)).encode())
            yield int(hasher.hexdigest(), 16) % self.capacity

    def insert(self, obj: T) -> None:
        for b in self.hash_multiple(obj):
            self.field[b] = True

    def contains(self, obj: T) -> bool:
        return all(self.field[b] for b in self.hash_multiple(obj))
