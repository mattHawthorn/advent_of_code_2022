from functools import partial, reduce
from typing import Any, Callable, Iterable, Iterator, List, TypeVar

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


# Functional


def call(x: T, f: Callable[[T], U]) -> U:
    return f(x)


def compose(f: Callable[[T], Any], *fs: Callable[[Any], V]) -> Callable[[T], V]:
    """Compose multiple functions, chaining output to input from left to right"""
    return partial(reduce, call, (f, *fs))  # type: ignore


# I/O


def parse_blocks(input_: Iterable[str], parse: Callable[[str], T]) -> Iterator[List[T]]:
    block = []
    for line in map(str.rstrip, input_):
        if line:
            block.append(parse(line))
        elif block:
            yield block
            block = []
    if block:
        yield block
