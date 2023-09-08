import math
from inspect import stack
from typing import Tuple

import pytest

from solutions.tailrec import tail_recursive


@tail_recursive
def factorial(n: int, acc: int = 1, stack_height: int = 0) -> Tuple[int, int]:
    height = max(stack_height, len(stack()))
    if n <= 1:
        return acc, height
    else:
        return factorial(n - 1, acc * n, height)


def recursive_function_overwrites_default(x: int, y: str, default: int = 0) -> int:
    length = len(y) // 2
    if not (x > 30 or len(y) > 20):
        default = 1  # overwrite; ensure it's reset correctly on loop
        print("recurse")
        return recursive_function_overwrites_default(y=y + str(x), x=x + length + default)
    else:
        # using it here ensures failure if it wasn't set correctly on loop
        return x + length + default


recursive_function_overwrites_default_tailrec = tail_recursive(
    recursive_function_overwrites_default
)


@tail_recursive
def recursive_generator_tailrec(x: int):
    if x >= 0:
        if x % 2 == 0:
            yield x, len(stack())
            yield from recursive_generator_tailrec(x - 2)
        else:
            yield from recursive_generator_tailrec(x - 1)


@pytest.mark.parametrize(
    "n",
    range(20),
)
def test_fact(n: int):
    expected = math.factorial(n)
    stack_height = len(stack())
    actual, max_stack_height = factorial(n)
    assert actual == expected
    assert max_stack_height <= stack_height + 2


@pytest.mark.parametrize("x, y", [(1, "foo"), (2, "asdf"), (3, "QWERTY")])
def test_recursive_function_overwrites_default(x: int, y: str):
    assert (
        recursive_function_overwrites_default is not recursive_function_overwrites_default_tailrec
    )
    expected = recursive_function_overwrites_default(x, y)
    actual = recursive_function_overwrites_default_tailrec(x, y)
    assert actual == expected, (expected, actual)


@pytest.mark.parametrize("x", range(10))
def test_recursive_generator_function(x: int):
    expected = list(range(0, x + 1, 2))[::-1]
    stack_height = len(stack())
    actual_ = list(recursive_generator_tailrec(x))
    actual = [n for n, height in actual_]
    max_stack_height = max(height for _, height in actual_)
    assert actual == expected, (expected, actual)
    assert max_stack_height <= stack_height + 2
