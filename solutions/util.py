from functools import partial, reduce
from itertools import islice, takewhile
from operator import is_not
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple, TypeVar

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


# Functional


def call(x: T, f: Callable[[T], U]) -> U:
    return f(x)


def compose(f: Callable[[T], Any], *fs: Callable[[Any], V]) -> Callable[[T], V]:
    """Compose multiple functions, chaining output to input from left to right"""
    return partial(reduce, call, (f, *fs))  # type: ignore


def swap(t: Tuple[T, U]) -> Tuple[U, T]:
    a, b = t
    return b, a


# Iterators


def chunked(n: int, it: Iterable[T]) -> Iterator[List[T]]:
    return iter(lambda: list(islice(it, n)), [])


def first(it: Iterable[T]) -> T:
    return next(iter(it))


def nonnull_head(it: Iterable[Optional[T]]) -> List[T]:
    return list(takewhile(partial(is_not, None), it))  # type: ignore


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
