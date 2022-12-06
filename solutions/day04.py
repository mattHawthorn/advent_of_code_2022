from itertools import starmap
from typing import IO, NamedTuple, Tuple


class Range(NamedTuple):
    lo: int
    hi: int


def parse_range(s: str) -> Range:
    a, b = s.split("-", 1)
    return Range(int(a), int(b))


def parse_line(line: str) -> Tuple[Range, Range]:
    first, second = map(parse_range, line.strip().split(",", 1))
    return first, second


def contained_in(range1: Range, range2: Range):
    return range1.lo >= range2.lo and range1.hi <= range2.hi


def containment(range1: Range, range2: Range) -> bool:
    return contained_in(range1, range2) or contained_in(range2, range1)


def intersection(range1: Range, range2: Range) -> bool:
    return (
        containment(range1, range2)
        or range1.lo <= range2.lo <= range1.hi
        or range1.lo <= range2.hi <= range1.hi
    )


def run(input_: IO[str], part_2: bool = True) -> int:
    ranges = map(parse_line, input_)
    predicate = intersection if part_2 else containment
    return sum(starmap(predicate, ranges))


def test():
    r1 = Range(1, 4)
    r2 = Range(2, 3)
    r3 = Range(2, 3)
    r4 = Range(4, 7)
    for range1, range2, predicate in [
        (r1, r2, containment),
        (r2, r1, containment),
        (r1, r2, intersection),
        (r2, r3, containment),
        (r2, r3, intersection),
        (r1, r4, intersection),
        (r4, r1, intersection),
    ]:
        assert predicate(range1, range2)
