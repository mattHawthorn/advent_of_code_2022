from functools import partial
from operator import itemgetter
from typing import IO, Optional, Tuple

from .util import Edge, Grid, GridCoordinates, djikstra, grid_to_graph

START, END, START_CHAR, END_CHAR = "S", "E", "a", "z"

_ZERO = ord("a")


def char_to_height(char: str) -> int:
    if char == START:
        return char_to_height(START_CHAR)
    elif char == END:
        return char_to_height(END_CHAR)
    else:
        return ord(char) - _ZERO


def parse_terrain(
    input_: IO[str],
) -> Tuple[Grid[int], GridCoordinates, GridCoordinates]:
    lines = map(str.rstrip, input_)
    char_grid: Grid[str] = list(map(list, lines))  # type: ignore
    start_, end_ = [
        ((i, j), c)
        for i, row in enumerate(char_grid)
        for j, c in enumerate(row)
        if c in (START, END)
    ]
    if start_[1] == START:
        start, end = start_[0], end_[0]
    else:
        start, end = end_[0], start_[0]

    return [list(map(char_to_height, row)) for row in char_grid], start, end


def edge_weight(grid: Grid[int], edge: Edge[GridCoordinates]) -> Optional[int]:
    (row1, col1), (row2, col2) = edge
    height1 = grid[row1][col1]
    height2 = grid[row2][col2]
    diff = height2 - height1
    return 1 if diff <= 1 else None


def run(input_: IO[str], part_2: bool = False):
    grid, start, end = parse_terrain(input_)
    height: int
    if part_2:
        starts = [
            (i, j)
            for i, row in enumerate(grid)
            for j, height in enumerate(row)
            if height == 0
        ]
    else:
        starts = [start]

    graph = grid_to_graph(grid, partial(edge_weight, grid))
    path_dists = map(partial(djikstra, graph, end=end), starts)
    path, dist = min(path_dists, key=itemgetter(1))
    return dist


test_input = """
Sabqponm
abcryxxl
accszExk
acctuvwj
abdefghi""".strip()


def test():
    import io

    input_ = io.StringIO(test_input)
    grid, start, end = parse_terrain(input_)
    assert start == (0, 0)
    assert end == (2, 5)
    graph = grid_to_graph(grid, partial(edge_weight, grid))
    path, dist = djikstra(graph, start, end)
    expected_dist = 31
    assert dist == expected_dist, (dist, expected_dist)
    assert len(path) == expected_dist + 1
