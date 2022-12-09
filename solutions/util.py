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
    Type,
    TypeVar,
    Union,
    overload,
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
class Leaf(Generic[K, T]):
    id: K
    data: T
    parent: Optional["Tree[K, T]"] = None


@dataclass
class Tree(Generic[K, T]):
    id: K
    data: T
    children: Dict[K, Union["Tree[K, T]", Leaf[K, T]]]
    parent: Optional["Tree[K, T]"] = None

    def __iter__(self) -> Iterator[Union[Leaf[K, T], "Tree[K, T]"]]:
        return iter(self.children.values())


AnyTree = Union[Tree[K, T], Leaf[K, T]]
TreePath = Tuple[K, ...]


def dfs(
    tree: AnyTree[K, T], path: TreePath[K] = ()
) -> Iterator[Tuple[TreePath[K], AnyTree[K, T]]]:
    path = path or (tree.id,)
    yield path, tree
    if isinstance(tree, Tree):
        for key, sub_tree in tree.children.items():
            yield from dfs(sub_tree, (*path, key))


@overload
def tree_acc(
    tree: Tree[K, T],
    f: Callable[[T], U],
    acc: Callable[[Iterable[U]], U],
    Tree: Type[Tree] = Tree,
    Leaf: Type[Leaf] = Leaf,
) -> Tree[K, U]:
    ...


@overload
def tree_acc(
    tree: Leaf[K, T],
    f: Callable[[T], U],
    acc: Callable[[Iterable[U]], U],
    Tree: Type[Tree] = Tree,
    Leaf: Type[Leaf] = Leaf,
) -> Leaf[K, U]:
    ...


def tree_acc(
    tree: AnyTree[K, T],
    f: Callable[[T], U],
    acc: Callable[[Iterable[U]], U],
    Tree: Type[Tree] = Tree,
    Leaf: Type[Leaf] = Leaf,
):
    if isinstance(tree, Tree):
        children = {
            k: tree_acc(t, f, acc, Tree, Leaf) for k, t in tree.children.items()
        }
        data = acc(t.data for t in children.values())
        return Tree(tree.id, data, children)
    else:
        return Leaf(tree.id, f(tree.data))


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
