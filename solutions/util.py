import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import partial, reduce
from heapq import heappop, heappush
from itertools import accumulate, chain, filterfalse, islice, product, repeat
from operator import add, and_, is_, is_not, not_, sub
from typing import (
    AbstractSet,
    Callable,
    Collection,
    Deque,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    overload,
)

from .tailrec import tail_recursive

VERBOSE = False

K = TypeVar("K", bound=Hashable)
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


# Math


def sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


@tail_recursive
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

    def __add__(self, other):
        return INF

    def __radd__(self, other):
        return INF

    def __repr__(self):
        return "Inf()"

    def __str__(self):
        return "+inf"


INF = Inf()


# Functional

Predicate = Callable[[T], bool]

is_not_null: Predicate = partial(is_not, None)
is_null: Predicate = partial(is_, None)


def invert(f: Predicate[T]) -> Predicate[T]:
    return compose(f, not_)


def any_(*fs: Predicate[T]) -> Predicate[T]:
    return lambda t: any(map(call, repeat(t), fs))


def all_(*fs: Predicate[T]) -> Predicate[T]:
    return lambda t: all(map(call, repeat(t), fs))


def identity(x: T) -> T:
    return x


def call(x: T, f: Callable[[T], U]) -> U:
    return f(x)


def compose(f: Callable[[T], U], g: Callable[[U], V]) -> Callable[[T], V]:
    """Compose 2 functions, chaining output to input from left to right"""
    return lambda *x: g(f(*x))


def swap(t: Tuple[T, U]) -> Tuple[U, T]:
    a, b = t
    return b, a


# Iterators


def zip_with(f: Callable[[T], U], it: Iterable[T]) -> Iterator[Tuple[T, U]]:
    for i in it:
        yield i, f(i)


def iterate(f: Callable[[T], T], initial: T) -> Iterator[T]:
    value = initial
    while True:
        yield value
        value = f(value)


def chunked(n: int, it: Iterable[T]) -> Iterator[List[T]]:
    it_ = iter(it)
    return iter(lambda: list(islice(it_, n)), [])


def first(it: Iterable[T]) -> T:
    return next(iter(it))


def nonnull_head(it: Iterable[Optional[T]]) -> Iterator[T]:
    for i in it:
        if i is None:
            break
        yield i


def take_until(f: Predicate, it: Iterable[T]) -> Iterator[T]:
    """like `takewhile` but yields the final value which fails the predicate"""
    for i in it:
        yield i
        if f(i):
            break


def interleave(*its: Iterable[T]) -> Iterator[T]:
    return chain.from_iterable(zip(*its))


def window(size: int, it: Iterable[T]) -> Iterator[Deque[T]]:
    it = iter(it)
    win = deque(islice(it, size))
    if len(win) == size:
        yield win
        for i in it:
            win.popleft()
            win.append(i)
            yield win


def reduce_while(
    condition: Callable[[T, T], bool],
    agg: Callable[[T, T], T],
    it: Iterable[T],
    accumulator: Optional[T] = None,
) -> Iterable[T]:
    """Reduce an iterator in groups connected by a pairwise predicate"""
    it = iter(it)
    next_ = next(it, None)
    if accumulator is None:
        if next_ is not None:
            yield from reduce_while(condition, agg, it, next_)
    else:
        if next_ is None:
            yield accumulator
        else:
            if condition(accumulator, next_):
                accumulator = agg(accumulator, next_)
            else:
                yield accumulator
                accumulator = next_
            yield from reduce_while(condition, agg, it, accumulator)


# Miscellaneous


def non_overlapping(sets: Iterable[AbstractSet]) -> bool:
    intersections = accumulate(sets, and_)
    next(intersections)
    return not any(map(bool, intersections))


# Data Structures

Grid = List[List[T]]
GridCoordinates = Tuple[int, int]
GridCoordinates3D = Tuple[int, int, int]
Vector = Tuple[int, int]
Vector3D = Tuple[int, int, int]
Sprite = Sequence[GridCoordinates]


@overload
def translate(step: Vector, coords: GridCoordinates) -> GridCoordinates:
    ...


@overload
def translate(step: Vector3D, coords: GridCoordinates3D) -> GridCoordinates3D:
    ...


def translate(step, coords):
    return tuple(map(add, step, coords))


@overload
def translate_inv(step: Vector, coords: GridCoordinates) -> GridCoordinates:
    ...


@overload
def translate_inv(step: Vector3D, coords: GridCoordinates3D) -> GridCoordinates3D:
    ...


def translate_inv(step, coords):
    return tuple(map(sub, coords, step))


def translate_all(step: Vector, obj: Sprite) -> Sprite:
    return list(map(partial(translate, step), obj))


@dataclass
class SparseGrid(Generic[T]):
    grid: Dict[GridCoordinates, T]
    x_min: Optional[int] = None
    x_max: Optional[int] = None
    y_min: Optional[int] = None
    y_max: Optional[int] = None

    def __contains__(self, coords: GridCoordinates):
        return coords in self.grid

    @property
    def n_rows(self) -> int:
        return 0 if self.y_min is None or self.y_max is None else (self.y_max + 1 - self.y_min)

    @property
    def n_cols(self) -> int:
        return 0 if self.x_min is None or self.x_max is None else (self.x_max + 1 - self.x_min)

    def get(self, coord: GridCoordinates) -> Optional[T]:
        return self.grid.get(coord, None)

    def set(self, value: T, coord: GridCoordinates) -> "SparseGrid[T]":
        x, y = coord
        self.x_min = _min(self.x_min, x)
        self.x_max = _max(self.x_max, x)
        self.y_min = _min(self.y_min, y)
        self.y_max = _max(self.y_max, y)
        self.grid[x, y] = value
        return self

    def set_all(self, value: T, coords: Iterable[GridCoordinates]) -> "SparseGrid[T]":
        return reduce(lambda grid, coord: grid.set(value, coord), coords, self)


def _min(old: Optional[int], new: int) -> int:
    return new if old is None else min(old, new)


def _max(old: Optional[int], new: int) -> int:
    return new if old is None else max(old, new)


class HeapItem(NamedTuple, Generic[K, T]):
    key: K
    value: T

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value


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


def dfs(tree: Tree[K, T], path: TreePath[K] = ()) -> Iterator[Tuple[TreePath[K], Tree[K, T]]]:
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


def all_edges(graph: WeightedDiGraph[K]) -> Iterator[WeightedEdge[K]]:
    return (((e, f), w) for e, neighbors in graph.items() for f, w in neighbors.items())


def all_nodes(graph: WeightedDiGraph) -> Iterator[K]:
    return chain(graph, chain.from_iterable(graph.values()))


def n_nodes(graph: WeightedDiGraph) -> int:
    return len(set(all_nodes(graph)))


def n_edges(graph: WeightedDiGraph) -> int:
    return sum(map(len, graph.values()))


def reverse_graph(graph: WeightedDiGraph[K]) -> WeightedDiGraph[K]:
    return weighted_edges_to_graph(((f, e), w) for (e, f), w in all_edges(graph))


def edges_to_graph_with_weight(
    weight_fn: Callable[[Edge[K]], Optional[int]], candidate_edges: Iterable[Edge[K]]
) -> WeightedDiGraph[K]:
    maybe_weighted_edges = zip_with(weight_fn, candidate_edges)
    weighted_edges = ((e, w) for e, w in maybe_weighted_edges if w is not None)
    return weighted_edges_to_graph(weighted_edges)


def weighted_edges_to_graph(edges: Iterable[WeightedEdge[K]]) -> WeightedDiGraph[K]:
    graph: WeightedDiGraph[K] = {}
    return reduce(add_weighted_edge, edges, graph)


def add_weighted_edge(graph: WeightedDiGraph[K], edge: WeightedEdge[K]) -> WeightedDiGraph[K]:
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


def induced_subgraph(
    g: WeightedDiGraph[K], nodes_from: Collection[K], nodes_to: Optional[Collection[K]] = None
):
    nodes_to_ = nodes_from if nodes_to is None else nodes_to
    return {
        n: {m: w for m, w in neighbors.items() if m in nodes_to_}
        for n, neighbors in g.items()
        if n in nodes_from
    }


def dfs_graph(graph: WeightedDiGraph[K], node: K, visited: Optional[Set[K]] = None) -> Iterator[K]:
    yield node
    visited_ = set() if visited is None else visited
    visited_.add(node)
    for next_node in filterfalse(visited_.__contains__, graph.get(node, ())):
        yield from dfs_graph(graph, next_node, visited_)


def connected_components(graph: WeightedDiGraph[K]) -> Iterator[Set[K]]:
    all_nodes = set(graph)
    while all_nodes:
        component = set(dfs_graph(graph, next(iter(all_nodes))))
        all_nodes.difference_update(component)
        yield component


def is_complete_graph(g: WeightedDiGraph[K]) -> bool:
    n_nodes = len(set(all_nodes(g)))
    return len(g) == n_nodes and all(len(nbrs) == n_nodes - 1 for nbrs in g.values())


def djikstra(graph: WeightedDiGraph[K], start: K, end: K) -> Tuple[List[K], int]:
    """Return shortest path (if any) from node `start` to node `end`, and the total weight
    of the path"""
    return DjikstraState(graph, start, [end]).shortest_path(end)


def djikstra_all(
    graph: WeightedDiGraph[K], start: K, ends: Collection[K] = frozenset()
) -> Iterator[Tuple[List[K], int]]:
    """Return all shortest paths from node `start` to each node in `ends` (or the whole graph
    if this is empty)"""
    state = DjikstraState(graph, start, ends, False)
    for end in ends or graph:
        yield state.shortest_path(end)


def djikstra_any(graph: WeightedDiGraph[K], start: K, ends: Collection[K]) -> Tuple[List[K], int]:
    """Return all shortest paths from node `start` to each node in `ends` (or the whole graph
    if this is empty)"""
    state = DjikstraState(graph, start, ends, True)
    end = next(iter(state.visited_ends or ends))
    return state.shortest_path(end)


class DjikstraState(Generic[K]):
    def __init__(
        self,
        graph: WeightedDiGraph[K],
        start: K,
        ends: Collection[K] = frozenset(),
        any_: bool = False,
    ):
        self.graph = graph
        self.start = start
        self.ends = set(ends)
        self.any_ = any_
        self.visited_ends: Set[K] = set()
        self.visited: Set[K] = set()
        self.distances: Dict[K, int] = defaultdict(Inf)
        self.min_dist_heap: List[HeapItem[K, int]] = []
        self.predecessors: Dict[K, K] = {}
        self.distances[start] = 0
        self.accumulate_shortest_paths()

    def update_dists(self, node_dists: Iterable[Tuple[K, int]], node: K):
        for n, dist in node_dists:
            if dist < self.distances[n]:
                self.distances[n] = dist
                self.predecessors[n] = node
                heappush(self.min_dist_heap, HeapItem(n, dist))

    def next_candidate_node(self, node: K) -> Optional[K]:
        dist = self.distances[node]
        nbrs = self.graph.get(node, {}).items()
        new_dists = ((n, dist + d) for n, d in nbrs if n not in self.visited)
        self.update_dists(new_dists, node)
        self.visited.add(node)
        return heappop(self.min_dist_heap).key if self.min_dist_heap else None

    def distance(self, end: K) -> int:
        return self.distances[end]

    def shortest_path(self, end: K) -> Tuple[List[K], int]:
        if end not in self.distances:
            return [], INF
        else:
            predecessor = self.predecessors.get
            reverse_path = list(nonnull_head(iterate(predecessor, end)))  # type: ignore
            return list(reversed(reverse_path)), self.distances[end]

    def accumulate_shortest_paths(self):
        for node in iterate(self.next_candidate_node, self.start):
            if node is None:
                return
            elif node in self.ends:
                self.visited_ends.add(node)
                if self.any_ or len(self.visited_ends) == len(self.ends):
                    return


def floyd_warshall(graph: WeightedDiGraph[K]) -> WeightedDiGraph[K]:
    def dist(distances: WeightedDiGraph[K], node1: K, node2: K):
        return 0 if node1 == node2 else distances.get(node1, {}).get(node2, INF)

    def update_dist(distances: WeightedDiGraph[K], nodes: Tuple[K, K, K]) -> WeightedDiGraph[K]:
        node1, node2, node3 = nodes
        candidate_dist = dist(distances, node2, node1) + dist(distances, node1, node3)
        current_dist = dist(distances, node2, node3)
        if node2 != node3 and candidate_dist < current_dist and candidate_dist < INF:
            if node2 in distances:
                neighbors = distances[node2]
            else:
                neighbors = distances[node2] = {}
            neighbors[node3] = candidate_dist
        return distances

    distances: WeightedDiGraph[K] = {node: dict(nbrs) for node, nbrs in graph.items()}
    nodes: Set[K] = set(all_nodes(graph))
    result = reduce(update_dist, product(nodes, nodes, nodes), distances)
    return result


# Hard problems


def branch_and_bound(
    initial: Iterable[T],
    heuristic_solution: T,
    candidate_fn: Callable[[T], Iterable[T]],
    stop_fn: Callable[[T], bool],
    objective_fn: Callable[[T], int],
    lower_bound_fn: Callable[[T], int],
) -> Optional[T]:
    return _branch_and_bound(
        candidate_fn,
        stop_fn,
        objective_fn,
        lower_bound_fn,
        deque(initial),
        heuristic_solution,
        objective_fn(heuristic_solution),
    )


@tail_recursive
def _branch_and_bound(
    candidate_fn: Callable[[T], Iterable[T]],
    stop_fn: Callable[[T], bool],
    objective_fn: Callable[[T], int],
    lower_bound_fn: Callable[[T], int],
    queue: Deque[T],
    best_solution: T,
    objective_upper_bound: int,
) -> Optional[T]:
    if queue:
        candidate = queue.popleft()
        if stop_fn(candidate):
            objective = objective_fn(candidate)
            if objective < objective_upper_bound:
                print_(f"NEW BEST OBJECTIVE: {objective}")
                best_solution, objective_upper_bound = candidate, objective
            return _branch_and_bound(
                candidate_fn,
                stop_fn,
                objective_fn,
                lower_bound_fn,
                queue,
                best_solution,
                objective_upper_bound,
            )
        else:
            is_viable = compose(lower_bound_fn, objective_upper_bound.__gt__)
            viable_candidates = filter(is_viable, candidate_fn(candidate))
            queue.extend(viable_candidates)
            return _branch_and_bound(
                candidate_fn,
                stop_fn,
                objective_fn,
                lower_bound_fn,
                queue,
                best_solution,
                objective_upper_bound,
            )
    else:
        return best_solution


# I/O


def set_verbose(value: bool):
    global VERBOSE
    VERBOSE = value


def print_(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs, file=sys.stderr)


def parse_blocks(input_: Iterable[str], parse: Callable[[str], T]) -> Iterator[T]:
    block = []
    for line in map(str.rstrip, input_):
        if line:
            block.append(line)
        elif block:
            yield parse("\n".join(block))
            block = []
    if block:
        yield parse("\n".join(block))
