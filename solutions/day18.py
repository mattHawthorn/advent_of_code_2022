from functools import partial
from itertools import chain, filterfalse, product, repeat
from operator import itemgetter
from typing import IO, Iterator, List, Set, Tuple

from .util import (
    Edge,
    GridCoordinates,
    GridCoordinates3D,
    djikstra_any,
    translate,
    translate_inv,
    weighted_edges_to_graph,
)

D = 3
S = 2 * D


def parse_coords(line: str) -> GridCoordinates3D:
    x, y, z = line.split(",", 2)
    return int(x), int(y), int(z)


def parse_all_coords(input_: IO[str]) -> Iterator[GridCoordinates3D]:
    return map(parse_coords, filter(None, map(str.strip, input_)))


def pack(ix: int, coords: GridCoordinates, x: int) -> GridCoordinates3D:
    a, b = coords
    return (x, a, b) if ix == 0 else (a, x, b) if ix == 1 else (a, b, x)


def grid_neighbors(coords: GridCoordinates3D) -> List[GridCoordinates3D]:
    return [
        f(v, coords)
        for v, f in product([(0, 0, 1), (0, 1, 0), (1, 0, 0)], [translate_inv, translate])
    ]


def grid_edges_between(
    shape1: Set[GridCoordinates3D], shape2: Set[GridCoordinates3D]
) -> Iterator[Edge[GridCoordinates3D]]:
    def inner(shape: Set[GridCoordinates3D], coords: GridCoordinates3D):
        nbrs = filter(shape.__contains__, grid_neighbors(coords))
        return zip(repeat(coords), nbrs)

    return chain.from_iterable(map(partial(inner, shape2), shape1))


def surface(shape: Set[GridCoordinates3D]) -> Iterator[GridCoordinates3D]:
    return filterfalse(shape.__contains__, chain.from_iterable(map(grid_neighbors, shape)))


def ranges(shape: Set[GridCoordinates3D]) -> Tuple[range, range, range]:
    xs = set(map(itemgetter(0), shape))
    ys = set(map(itemgetter(1), shape))
    zs = set(map(itemgetter(2), shape))
    xrange = range(min(xs), max(xs) + 1)
    yrange = range(min(ys), max(ys) + 1)
    zrange = range(min(zs), max(zs) + 1)
    return xrange, yrange, zrange


def volume(shape: Set[GridCoordinates3D]) -> Iterator[GridCoordinates3D]:
    xrange, yrange, zrange = ranges(shape)
    return product(xrange, yrange, zrange)


def envelope(shape: Set[GridCoordinates3D]) -> Iterator[GridCoordinates3D]:
    xrange, yrange, zrange = ranges(shape)
    return chain.from_iterable(
        chain(
            map(partial(pack, ix), product(xr, yr), repeat(zr.start - 1)),
            map(partial(pack, ix), product(xr, yr), repeat(zr.stop)),
        )
        for ix, xr, yr, zr in [
            (2, xrange, yrange, zrange),
            (1, xrange, zrange, yrange),
            (0, yrange, zrange, xrange),
        ]
    )


def surface_points_reachable_from_envelope(
    shape: Set[GridCoordinates3D],
) -> Iterator[GridCoordinates3D]:
    # boundary points of the shape
    surface_points = set(surface(shape))
    # points inside the cuboid containing the shape and its surface points, but excluding the shape
    vol = set(volume(surface_points)).difference(shape)
    # points just outside the above volume
    env = set(envelope(surface_points))
    # grid edges inside the volume with neither head nor tail inside the shape
    edges = chain(grid_edges_between(vol, vol), grid_edges_between(vol, env))
    graph = weighted_edges_to_graph(zip(edges, repeat(1)))
    # all surface points with a grid path to some point in the envelope
    return (point for point in surface_points if djikstra_any(graph, point, env))


def run(input_: IO[str], part_2: bool = True) -> int:
    points = set(parse_all_coords(input_))

    if part_2:
        surface_points = set(surface_points_reachable_from_envelope(points))
    else:
        surface_points = set(surface(points))

    surface_edges = grid_edges_between(points, surface_points)
    return sum(1 for _ in surface_edges)


test_input = """
2,2,2
1,2,2
3,2,2
2,1,2
2,3,2
2,2,1
2,2,3
2,2,4
2,2,6
1,2,5
3,2,5
2,1,5
2,3,5""".strip()


def test():
    import io

    actual_area = run(io.StringIO(test_input), part_2=False)
    expected_area = 64
    assert actual_area == expected_area, (actual_area, expected_area)

    actual_area = run(io.StringIO(test_input), part_2=True)
    expected_area = 58
    assert actual_area == expected_area, (actual_area, expected_area)
