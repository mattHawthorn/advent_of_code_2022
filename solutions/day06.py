from collections import defaultdict, deque
from itertools import accumulate
from typing import IO, Deque, Dict, Generic, Hashable, Iterable, TypeVar

H = TypeVar("H", bound=Hashable)


class RollingWindow(Generic[H]):
    def __init__(self, size: int):
        self.size = size
        self.q: Deque[H] = deque()
        self.nunique = 0
        self.counts: Dict[H, int] = defaultdict(int)

    def add(self, item: H) -> "RollingWindow":
        if len(self.q) == self.size:
            self.pop()
        self.q.append(item)
        c = self.counts[item]
        self.counts[item] = c + 1
        if c == 0:
            self.nunique += 1
        return self

    def pop(self) -> H:
        item = self.q.popleft()
        c = self.counts[item]
        self.counts[item] = c - 1
        if c == 1:
            self.nunique -= 1
        return item


def first_unique_substring_index(message: Iterable[H], size: int) -> int:
    window: RollingWindow[H] = RollingWindow(size)
    windows = accumulate(message, RollingWindow.add, initial=window)
    return next(ix for ix, w in enumerate(windows) if w.nunique == size)


def run(input_: IO[str], n: int = 14) -> int:
    chars = iter(lambda: input_.read(1), "")
    return first_unique_substring_index(chars, n)


def test():
    for s, n in [
        ("bvwbjplbgvbhsrlpgdmjqwftvncz", 5),
        ("nppdvjthqldpwncqszvftbrmjlhg", 6),
        ("nznrnfrfntjfmvfwmzdfjlvtqnbhcprsg", 10),
        ("zcfzfwzzqfrljwzlrfnpqdbhtmscgvjw", 11),
    ]:
        assert first_unique_substring_index(s, 4) == n
