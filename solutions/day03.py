from functools import reduce
from itertools import chain, starmap
from typing import IO, Iterable, Set, Tuple

from .util import chunked, first

PRIORITY = dict(
    chain(
        zip(map(chr, range(ord("a"), ord("z") + 1)), range(1, 27)),
        zip(map(chr, range(ord("A"), ord("Z") + 1)), range(27, 53)),
    )
)


def parse_line(line: str) -> Tuple[str, str]:
    line = line.rstrip()
    size = len(line) // 2
    return line[:size], line[size:]


def common_items(first: str, *others: str) -> Set[str]:
    def inner(coll1: Set[str], coll2: str):
        return {c for c in coll2 if c in coll1}

    return reduce(inner, others, set(first))


def run(input_: IO[str], part_2: bool = True) -> int:
    groups: Iterable[Iterable[str]]
    if part_2:
        groups = chunked(3, map(str.rstrip, input_))
    else:
        groups = map(parse_line, input_)

    common_sets = starmap(common_items, groups)
    common: Iterable[str] = map(first, common_sets)
    priorities = map(PRIORITY.__getitem__, common)
    return sum(priorities)


def test():
    ...
