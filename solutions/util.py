import sys
from dataclasses import dataclass
from functools import partial, reduce
from heapq import heappop, heappush
from itertools import chain, islice, product, repeat, takewhile
from operator import add, is_not, itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from .tailrec import tailrec

VERBOSE = False

K = TypeVar("K", bound=Hashable)
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


# Math


@tailrec
def gcd(m, n):
    # Euclid
    larger, smaller = (m, n) if m > n else (n, m)
    if smaller == 0:
        return larger
    _, rem = divmod(larger, smaller)
    return gcd(smaller, rem)


def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


class Inf(int):
    def __eq__(self, other):
        return False

    __lt__ = __le__ = __eq__

    def __ge__(self, other):
        return True

    def __gt__(self, other):
        return not isinstance(other, Inf)


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


def zip_with(f: Callable[[T], U], it: Iterable[T]) -> Iterator[Tuple[T, U]]:
    for i in it:
        yield i, f(i)


def chunked(n: int, it: Iterable[T]) -> Iterator[List[T]]:
    return iter(lambda: list(islice(it, n)), [])


def first(it: Iterable[T]) -> T:
    return next(iter(it))


def nonnull_head(it: Iterable[Optional[T]]) -> List[T]:
    return list(takewhile(partial(is_not, None), it))  # type: ignore


# Data Structures

Grid = List[List[T]]
GridCoordinates = Tuple[int, int]


@dataclass
class Tree(Generic[K, T]):
    id: K
    data: T
    children: Dict[K, "Tree[K, T]"]
    parent: Optional["Tree[K, T]"] = None

    def __contains__(self, item: K):
        return item in self.children

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


# node -> node -> weight
WeightedDiGraph = MutableMapping[K, MutableMapping[K, int]]
Edge = Tuple[K, K]
WeightedEdge = Tuple[Edge[K], int]


def edges(graph: WeightedDiGraph[K]) -> Iterator[WeightedEdge[K]]:
    return (((e, f), w) for e, neighbors in graph.items() for f, w in neighbors.items())


def n_nodes(graph: WeightedDiGraph) -> int:
    return len(set(chain(graph, chain.from_iterable(graph.values()))))


def n_edges(graph: WeightedDiGraph) -> int:
    return sum(map(len, graph.values()))


def edges_to_graph_with_weight(
    weight_fn: Callable[[Edge[K]], Optional[int]], candidate_edges: Iterable[Edge[K]]
) -> WeightedDiGraph[K]:
    maybe_weighted_edges = zip_with(weight_fn, candidate_edges)
    weighted_edges = ((e, w) for e, w in maybe_weighted_edges if w is not None)
    return weighted_edges_to_graph(weighted_edges)


def weighted_edges_to_graph(edges: Iterable[WeightedEdge[K]]) -> WeightedDiGraph[K]:
    graph: WeightedDiGraph[K] = {}
    return reduce(add_weighted_edge, edges, graph)


def add_weighted_edge(
    graph: WeightedDiGraph[K], edge: WeightedEdge[K]
) -> WeightedDiGraph[K]:
    (head, tail), weight = edge
    neighbors = graph.get(head)
    if neighbors is None:
        new_neighbors: MutableMapping[K, int] = {}
        graph[head] = neighbors = new_neighbors
    neighbors[tail] = weight
    return graph


def grid_to_graph(
    grid: Grid[T],
    weight_fn: Callable[[Edge[GridCoordinates]], Optional[int]],
) -> WeightedDiGraph[GridCoordinates]:
    nrows = len(grid)
    ncols = len(grid[0])
    assert all(map(ncols.__eq__, map(len, grid)))

    def neighbors(
        nrows: int, ncols: int, coord: GridCoordinates
    ) -> Iterator[Tuple[GridCoordinates, GridCoordinates]]:
        row, col = coord
        if row > 0:
            yield coord, (row - 1, col)
        if row < nrows - 1:
            yield coord, (row + 1, col)
        if col > 0:
            yield coord, (row, col - 1)
        if col < ncols - 1:
            yield coord, (row, col + 1)

    coords = product(range(nrows), range(ncols))
    candidate_edges = chain.from_iterable(map(partial(neighbors, nrows, ncols), coords))
    return edges_to_graph_with_weight(weight_fn, candidate_edges)


def djikstra(
    graph: WeightedDiGraph[K], start: K, end: K
) -> Tuple[Optional[List[K]], int]:
    unvisited: Set[K] = set(graph)
    distances: Dict[K, int] = dict(zip(graph, repeat(Inf())))
    distances[start] = 0
    min_dist_heap: List[Tuple[int, K]] = []
    predecessors: Dict[K, K] = {}

    def update_dists(node_dists: Iterable[Tuple[K, int]], node: K):
        for n, dist in node_dists:
            if dist < distances[n]:
                distances[n] = dist
                predecessors[n] = node
                heappush(min_dist_heap, (dist, n))

    def explore(node: K) -> Optional[K]:
        dist = distances[node]
        nbrs = [(n, dist) for n, dist in graph[node].items() if n in unvisited]
        nbr_dists = map(partial(add, dist), map(itemgetter(1), nbrs))
        new_dists = zip(map(itemgetter(0), nbrs), nbr_dists)
        update_dists(new_dists, node)
        unvisited.remove(node)
        if min_dist_heap:
            return heappop(min_dist_heap)[1]
        else:
            return None

    node: Optional[K] = start
    while node != end:
        if node is None:
            return None, Inf()
        node = explore(node)

    path = []
    node = end
    while node != start:
        path.append(node)
        node = predecessors[node]
    path.append(node)

    return list(reversed(path)), distances[end]


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
