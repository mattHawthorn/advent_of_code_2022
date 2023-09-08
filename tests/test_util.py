import operator

import pytest

from solutions import util


def is_inc(a, b):
    return b - a == 1


def longest_common_prefix(s1, s2):
    return "".join(a for a, b in zip(s1, s2) if a == b)


def share_prefix(s1, s2):
    return bool(s1) and bool(s2) and s1[0] == s2[0]


def same_parity(a, b):
    return (b - a) % 2 == 0


@pytest.mark.parametrize(
    "op, agg, values, init, expected",
    [
        (same_parity, operator.add, [1, 2, 3, 1, 5], None, [1, 2, 4, 5]),
        (operator.eq, operator.add, [0, 0, 0, 1, 1, 2, 4], None, [0, 8]),
        (
            share_prefix,
            longest_common_prefix,
            ["bar", "ba", "barn", "baz", "q", "quux"],
            "foo",
            ["foo", "ba", "q"],
        ),
    ],
)
def test_reduce_while(op, agg, values, init, expected):
    actual = list(util.reduce_while(op, agg, values, init))
    assert expected == actual, (expected, actual)
