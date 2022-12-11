import math
from inspect import stack
from typing import Tuple

import pytest

from solutions.tailrec import tailrec


@tailrec
def factorial(n: int, acc: int = 1, stack_height: int = 0) -> Tuple[int, int]:
    height = max(stack_height, len(stack()))
    if n <= 1:
        return acc, height
    else:
        return factorial(n - 1, acc * n, height)


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
