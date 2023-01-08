from collections import deque
from functools import reduce
from itertools import chain, repeat, takewhile
from typing import IO, Deque, List, Tuple

from .util import compose, fst

Cycle = Deque[Tuple[int, int]]


def parse(input_: IO[str], key: int = 1) -> Cycle:
    encode = key.__mul__
    return deque(enumerate(map(encode, map(compose(str.strip, int), input_))))


def rotate(cycle: Cycle, offset: int) -> Cycle:
    n = len(cycle)
    offset_ = offset % n
    if offset_ > n // 2:
        offset_ = offset_ - n
    cycle.rotate(offset_)
    return cycle


def mix(cycle: Cycle, idx: int) -> Cycle:
    offset = sum(1 for _ in takewhile(compose(fst, idx.__ne__), cycle))
    cycle = rotate(cycle, -offset)
    idx_, value = cycle.popleft()
    cycle = rotate(cycle, -value)
    cycle.append((idx, value))
    return cycle


def mix_all(cycle: Cycle, rounds: int) -> Cycle:
    return reduce(mix, chain.from_iterable(repeat(range(len(cycle)), rounds)), cycle)


def decode(cycle: Cycle, ixs: List[int]) -> int:
    i = next(i for i, (ix, v) in enumerate(cycle) if v == 0)
    n = len(cycle)
    return sum(cycle[(i + ix) % n][1] for ix in ixs)


def run(input_: IO[str], part_2: bool = True) -> int:
    cycle = parse(input_, key=811589153 if part_2 else 1)
    final_cycle = mix_all(cycle, 10 if part_2 else 1)
    return decode(final_cycle, [1000, 2000, 3000])


test_input = "1 2 -3 3 -2 0 4".replace(" ", "\n")


def test():
    import io

    result = run(io.StringIO(test_input), part_2=False)
    expected = 3
    assert result == expected, (expected, result)

    result = run(io.StringIO(test_input), part_2=True)
    expected = 1623178306
    assert result == expected, (expected, result)
