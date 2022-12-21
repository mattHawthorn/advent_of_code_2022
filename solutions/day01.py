import heapq
from typing import IO, Callable, Iterator, List

from .util import parse_blocks


def parse_ints(s: str):
    return list(map(int, s.strip().split("\n")))


def parse_int_blocks(input_: IO[str]) -> Iterator[List[int]]:
    return parse_blocks(input_, parse_ints)


def run(input_: IO[str], n: int = 1) -> int:
    sum_: Callable[[List[int]], int] = sum
    blocks = map(sum_, parse_blocks(input_, parse_ints))
    return sum(heapq.nlargest(n, blocks))


def test():
    import io

    #         1(3) 2(7)  3(8) 4(5)
    input_ = "1 2  3 4   8    2 3".replace(" ", "\n")
    f = io.StringIO
    assert run(f(input_)) == 8
    assert dict(enumerate(parse_int_blocks(f(input_)), 1)) == {
        1: [1, 2],
        2: [3, 4],
        3: [8],
        4: [2, 3],
    }
