import sys
from dataclasses import dataclass
from functools import partial, reduce
from itertools import islice, takewhile
from operator import is_not
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

VERBOSE = False

K = TypeVar("K", bound=Hashable)
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


# Functional


def identity(x: T) -> T:
    return x


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


# Data Structures


@dataclass
class Tree(Generic[K, T]):
    id: K
    data: T
    children: Dict[K, "Tree[K, T]"]
    parent: Optional["Tree[K, T]"] = None

    def __iter__(self) -> Iterator["Tree[K, T]"]:
        return iter(self.children.values())


TreePath = Tuple[K, ...]


def dfs(
    tree: Tree[K, T], path: TreePath[K] = ()
) -> Iterator[Tuple[TreePath[K], Tree[K, T]]]:
    path = path or (tree.id,)
    yield path, tree
    if isinstance(tree, Tree):
        for key, sub_tree in tree.children.items():
            yield from dfs(sub_tree, (*path, key))


def tree_acc(
    tree: Tree[K, T],
    f: Callable[[T], U],
    acc: Callable[[U, U], U],
) -> Tree[K, U]:
    children = {k: tree_acc(t, f, acc) for k, t in tree.children.items()}
    data = reduce(acc, (t.data for t in children.values()), f(tree.data))
    return Tree(tree.id, data, children)


# I/O


def set_verbose(value: bool):
    global VERBOSE
    VERBOSE = value


def print_(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs, file=sys.stderr)


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
