from enum import IntEnum
from functools import cmp_to_key, reduce
from itertools import chain, starmap
from typing import IO, Iterable, Iterator, List, Optional, Tuple, Union, cast

from .tailrec import tail_recursive
from .util import Predicate, chunked, invert, is_not_null, print_, set_verbose, take_until

Packet = Union[List, int]

START_LIST, END_LIST, COMMA = "[", "]", ","

is_not_digit: Predicate[str] = invert(str.isdigit)


# Why use `eval` or `json.loads`? This is more fun!


@tail_recursive
def _read_list(chars: Iterator[str], values: List) -> List:
    next_, last_char = _read_value(chars)
    if next_ is not None:
        values.append(next_)
    if last_char == END_LIST or last_char is None:
        return values
    else:
        return _read_list(chars, values)


def _read_int(chars: Iterator[str], prefix: Iterable[str]) -> Tuple[Optional[int], Optional[str]]:
    chars_ = list(take_until(is_not_digit, chars))
    if not chars_:
        return None, None
    else:
        if is_not_digit(chars_[-1]):
            digits, last = chars_[:-1], chars_[-1]
        else:
            digits, last = chars_, None
        n = "".join(chain(prefix, digits))
        return int(n) if n else None, last


def _read_value(chars: Iterator[str]) -> Tuple[Optional[Packet], Optional[str]]:
    first = next(chars, None)
    if first is None:
        return None, None
    elif first in (COMMA, END_LIST):
        return None, first
    elif first == START_LIST:
        list_ = _read_list(chars, [])
        return list_, next(chars, None)
    elif first.isdigit():
        return _read_int(chars, [first])
    else:
        raise ValueError(first)


def read_value(s: str) -> Optional[Packet]:
    packet = _read_value(iter(s.strip()))[0]
    return packet


class Ternary(IntEnum):
    yes = 1
    maybe = 0
    no = -1


def coalesce_comparisons(c1: Ternary, c2: Ternary) -> Ternary:
    return c2 if c1 == Ternary.maybe else c1


def compare_packets(left: Packet, right: Packet) -> Ternary:
    if isinstance(left, int) and isinstance(right, int):
        if left < right:
            return Ternary.yes
        elif left == right:
            return Ternary.maybe
        else:
            return Ternary.no
    elif isinstance(left, int):
        return compare_packets([left], right)
    elif isinstance(right, int):
        return compare_packets(left, [right])
    else:
        assert isinstance(left, list) and isinstance(right, list)
        comparisons = starmap(compare_packets, zip(left, right))
        comp = reduce(
            coalesce_comparisons, take_until(Ternary.maybe.__ne__, comparisons), Ternary.maybe
        )
        len_comp = compare_packets(len(left), len(right))
        return coalesce_comparisons(comp, len_comp)


def run(input_: IO[str], part_2: bool = True, verbose: bool = False) -> int:
    set_verbose(verbose)
    packets = cast(Iterator[Packet], filter(is_not_null, map(read_value, input_)))
    if part_2:
        packet1: Packet = [[2]]
        packet2: Packet = [[6]]
        sorted_packets = sorted(
            chain([packet1, packet2], packets), key=cmp_to_key(compare_packets), reverse=True
        )
        ix1 = sorted_packets.index(packet1) + 1
        ix2 = sorted_packets.index(packet2) + 1
        print_(ix1, ix2)
        return ix1 * ix2
    else:
        packet_pairs = cast(Iterator[Tuple[Packet, Packet]], map(tuple, chunked(2, packets)))
        correct_order = starmap(compare_packets, packet_pairs)
        correct_indices = [i for i, c in enumerate(correct_order, 1) if c == Ternary.yes]
        print_(correct_indices)
        return sum(correct_indices)


test_input = """
[1,1,3,1,1]
[1,1,5,1,1]

[[1],[2,3,4]]
[[1],4]

[9]
[[8,7,6]]

[[4,4],4,4]
[[4,4],4,4,4]

[7,7,7,7]
[7,7,7]

[]
[3]

[[[]]]
[[]]

[1,[2,[3,[4,[5,6,7]]]],8,9]
[1,[2,[3,[4,[5,6,0]]]],8,9]""".strip()


def test():
    import io

    for list_ in [], [[]], [1, 2, 345], [1, [], 2], [1, [2, 3, []], 4], [1, 2, [3, [4], 5]]:
        s = repr(list_).replace(" ", "")
        l_ = read_value(s)
        assert l_ == list_, (list_, l_)

    result = run(io.StringIO(test_input), part_2=False, verbose=True)
    expected = 13
    assert result == expected, (result, expected)

    result = run(io.StringIO(test_input), part_2=True, verbose=True)
    expected = 140
    assert result == expected, (result, expected)
