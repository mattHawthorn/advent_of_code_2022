from functools import partial, reduce
from itertools import chain, islice, product, repeat
from operator import itemgetter
from typing import IO, Callable, Iterable, Iterator, List, Optional, Tuple, cast

from .tailrec import tailrec
from .util import (
    GridCoordinates,
    Predicate,
    SparseGrid,
    T,
    chunked,
    compose,
    is_null,
    iterate,
    nonnull_head,
    print_,
    set_verbose,
    swap,
)

CaveGridPath = List[GridCoordinates]

AIR, ROCK, SAND = ".", "#", "o"
DOWN: GridCoordinates = (0, 1)
LEFT: GridCoordinates = (-1, 1)
RIGHT: GridCoordinates = (1, 1)


# Parsing


def parse_coord(s: str) -> GridCoordinates:
    x, y = s.split(",")
    return int(x), int(y)


def parse_path(s: str) -> CaveGridPath:
    coords = s.strip().split(" -> ")
    return list(map(parse_coord, coords))


def segment_coords(start: GridCoordinates, end: GridCoordinates) -> Iterator[GridCoordinates]:
    startx, starty = start
    endx, endy = end
    if starty == endy:
        step = 1 if startx <= endx else -1
        return zip(range(startx, endx, step), repeat(starty))
    elif startx == endx:
        step = 1 if starty <= endy else -1
        return zip(repeat(startx), range(starty, endy, step))
    else:
        raise ValueError(f"Invalid segment: {start} -> {end}")


def path_coords(path: CaveGridPath) -> Iterator[GridCoordinates]:
    return chain(chain.from_iterable(map(segment_coords, path, path[1:])), path[-1:])


def set_value(value: T, grid: SparseGrid[T], coord: GridCoordinates) -> SparseGrid[T]:
    return grid.set(value, coord)


def fill_path(value: T, grid: SparseGrid[T], path: CaveGridPath) -> SparseGrid[T]:
    coords = path_coords(path)
    fill_coord_ = cast(
        Callable[[SparseGrid[T], GridCoordinates], SparseGrid[T]], partial(set_value, value)
    )
    return reduce(fill_coord_, coords, grid)


def fill_paths(value: T, grid: SparseGrid[T], paths: Iterable[CaveGridPath]) -> SparseGrid[T]:
    fill_path_ = partial(fill_path, value)
    return reduce(fill_path_, paths, grid)


def parse_grid(input_: IO[str], value: str = ROCK) -> SparseGrid[str]:
    paths = map(parse_path, input_)
    grid = SparseGrid[str]({}, y_min=0)
    return fill_paths(value, grid, paths)


# Output


def format_grid(grid: SparseGrid) -> str:
    ymin = grid.y_min or 0
    ymax = grid.y_max or 0
    xmin = grid.x_min or 0
    xmax = grid.x_max or 0
    swap_: Callable[[Tuple[int, int]], Tuple[int, int]] = swap
    coords = map(swap_, product(range(ymin, ymax + 1), range(xmin, xmax + 1)))
    values = (v or AIR for v in map(grid.get, coords))
    rows = chunked(xmax + 1 - xmin, values)
    return "\n".join(map("".join, rows))


# Simulation


def next_coord_and_value(
    grid: SparseGrid[T], current: GridCoordinates, step: GridCoordinates
) -> Tuple[GridCoordinates, Optional[T]]:
    x, y = current
    xstep, ystep = step
    ynext, xnext = y + ystep, x + xstep
    next_ = xnext, ynext
    return next_, grid.get(next_)


def sand_grain_next(
    grid: SparseGrid[T],
    current_coord: GridCoordinates,
) -> Optional[GridCoordinates]:
    next_ = partial(next_coord_and_value, grid, current_coord)
    candidates = map(next_, (DOWN, LEFT, RIGHT))
    acceptable_coords = map(itemgetter(0), filter(compose(itemgetter(1), is_null), candidates))
    return next(acceptable_coords, None)


def sand_grain_path(grid: SparseGrid[T], start: GridCoordinates) -> Iterator[GridCoordinates]:
    next_ = partial(sand_grain_next, grid)
    return nonnull_head(iterate(next_, start))


@tailrec
def simulate(
    grid: SparseGrid[T],
    start: GridCoordinates,
    sand_value: T,
    max_steps: int,
    stopping_condition: Predicate[List[GridCoordinates]],
    include_last: bool = False,
    _n_sand_grains: int = 0,
) -> Tuple[SparseGrid[T], int]:
    path = list(islice(sand_grain_path(grid, start), max_steps))
    if stopping_condition(path):
        if include_last and path:
            grid.set(sand_value, path[-1])
            extra = 1
        else:
            extra = 0
        return grid, _n_sand_grains + extra
    else:
        grid.set(sand_value, path[-1])
        return simulate(
            grid, start, sand_value, max_steps, stopping_condition, include_last, _n_sand_grains + 1
        )


def stop_if_path_len_eq(len_: int, path: List[GridCoordinates]) -> bool:
    return len(path) == len_


def run(
    input_: IO[str],
    part_2: bool = True,
    entrypoint: GridCoordinates = (500, 0),
    verbose: bool = False,
):
    set_verbose(verbose)
    grid = parse_grid(input_)
    print_(grid.n_rows, "rows,", grid.n_cols, "cols")
    if part_2:
        max_steps = grid.n_rows + 1
        stop = partial(stop_if_path_len_eq, 1)
    else:
        max_steps = grid.n_rows
        stop = partial(stop_if_path_len_eq, max_steps)

    final_grid, n_grains = simulate(
        grid,
        sand_value=SAND,
        start=entrypoint,
        max_steps=max_steps,
        stopping_condition=stop,
        # really should be True for part 2, but the posted solution has an off-by-1 error
        include_last=False,
    )
    if verbose:
        print_(format_grid(grid))
    return n_grains + part_2


test_input = """498,4 -> 498,6 -> 496,6
503,4 -> 502,4 -> 502,9 -> 494,9"""

test_grid = """
..........
..........
..........
..........
....#...##
....#...#.
..###...#.
........#.
........#.
#########.""".strip()


def test():
    import io
    from copy import deepcopy

    path = parse_path(test_input.splitlines()[0])
    actual = list(path_coords(path))
    expected = [(498, 4), (498, 5), (498, 6), (497, 6), (496, 6)]
    assert actual == expected, (actual, expected)

    expected_grid = SparseGrid(
        {
            (j, i): v
            for i, line in enumerate(test_grid.splitlines(keepends=False))
            for j, v in enumerate(line, 494)
            if v != AIR
        },
        x_min=494,
        x_max=503,
        y_min=0,
        y_max=9,
    )
    actual_grid = parse_grid(io.StringIO(test_input))
    extra_actual = set(actual_grid.grid).difference(expected_grid.grid)
    extra_expected = set(expected_grid.grid).difference(actual_grid.grid)
    assert actual_grid == expected_grid, (extra_expected, extra_actual)

    final_grid, n_grains = simulate(
        deepcopy(actual_grid),
        sand_value=SAND,
        start=(500, 0),
        max_steps=actual_grid.n_rows,
        stopping_condition=partial(stop_if_path_len_eq, actual_grid.n_rows),
        include_last=False,
    )
    print(format_grid(final_grid))
    n_grains_expected = 24
    assert n_grains == n_grains_expected, (n_grains, n_grains_expected)

    final_grid, n_grains = simulate(
        actual_grid,
        sand_value=SAND,
        start=(500, 0),
        max_steps=actual_grid.n_rows + 1,
        stopping_condition=partial(stop_if_path_len_eq, 1),
        include_last=True,
    )
    print(format_grid(final_grid))
    n_grains_expected = 93
    assert n_grains == n_grains_expected, (n_grains, n_grains_expected)
