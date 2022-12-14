from dataclasses import dataclass
from functools import partial, reduce
from itertools import chain, repeat
from operator import itemgetter
from typing import IO, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Tuple, cast

from .tailrec import tailrec
from .util import GridCoordinates, Predicate, T, compose, iterate, nonnull_head, print_, set_verbose

CaveGridPath = List[GridCoordinates]

AIR, ROCK, SAND, NOTHING = ".", "#", "o", ""
DOWN: GridCoordinates = (0, 1)
LEFT: GridCoordinates = (-1, 1)
RIGHT: GridCoordinates = (1, 1)
ABYSS: GridCoordinates = (-1, -1)


@dataclass
class CaveGrid(Generic[T]):
    grid: Dict[GridCoordinates, T]
    empty_value: T
    n_rows: int = 0

    @property
    def n_cols(self) -> int:
        lo = min(map(itemgetter(0), self.grid.keys()))
        hi = max(map(itemgetter(0), self.grid.keys()))
        return hi + 1 - lo

    def get(self, coord: GridCoordinates) -> T:
        return self.grid.get(coord, self.empty_value)

    def insert(self, value: T, coord: GridCoordinates) -> "CaveGrid[T]":
        x, y = coord
        if y >= self.n_rows:
            self.n_rows = y + 1
        self.grid[x, y] = value
        return self


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


def insert(value: T, grid: CaveGrid[T], coord: GridCoordinates) -> CaveGrid[T]:
    return grid.insert(value, coord)


def fill_path(value: T, grid: CaveGrid[T], path: CaveGridPath) -> CaveGrid[T]:
    coords = path_coords(path)
    fill_coord_ = cast(
        Callable[[CaveGrid[T], GridCoordinates], CaveGrid[T]], partial(insert, value)
    )
    return reduce(fill_coord_, coords, grid)


def fill_paths(value: T, grid: CaveGrid[T], paths: Iterable[CaveGridPath]) -> CaveGrid[T]:
    fill_path_ = partial(fill_path, value)
    return reduce(fill_path_, paths, grid)


def parse_grid(input_: IO[str], empty_value: str = AIR, value: str = ROCK) -> CaveGrid[str]:
    paths = map(parse_path, input_)
    grid = CaveGrid[str]({}, empty_value, 0)
    return fill_paths(value, grid, paths)


def next_coord_and_value(
    grid: CaveGrid[T], current: GridCoordinates, step: GridCoordinates
) -> Tuple[GridCoordinates, T]:
    x, y = current
    xstep, ystep = step
    ynext, xnext = y + ystep, x + xstep
    next_ = xnext, ynext
    return next_, grid.get(next_)


def sand_grain_next(
    grid: CaveGrid[T], can_move: Predicate[T], current: Tuple[GridCoordinates, T]
) -> Optional[Tuple[GridCoordinates, T]]:
    current_coord, _ = current
    next_ = partial(next_coord_and_value, grid, current_coord)
    candidates = map(next_, (DOWN, LEFT, RIGHT))
    acceptable_candidates = filter(compose(itemgetter(1), can_move), candidates)
    return next(acceptable_candidates, None)


def sand_grain_path(
    grid: CaveGrid[T], can_move: Predicate[T], start: GridCoordinates
) -> Iterator[GridCoordinates]:
    next_ = partial(sand_grain_next, grid, can_move)
    coord_values = nonnull_head(iterate(next_, (start, grid.empty_value)))
    return (c for c, v in coord_values)


def sand_grain_resting_place(
    grid: CaveGrid[T], can_move: Predicate[T], max_y: int, start: GridCoordinates
) -> GridCoordinates:
    coord = start
    for coord in sand_grain_path(grid, can_move, start):
        print(coord)
        if coord[1] >= max_y:
            return coord
    else:
        return coord


@tailrec
def simulate(
    grid: CaveGrid[T],
    can_move: Predicate[T],
    stop: Predicate[GridCoordinates],
    max_y: int,
    sand_value: T,
    start: GridCoordinates,
    n_sand_grains: int = 0,
) -> Tuple[CaveGrid[T], int]:
    coord = sand_grain_resting_place(grid, can_move, max_y, start)
    if stop(coord):
        return grid, n_sand_grains
    else:
        grid.insert(sand_value, coord)
        return simulate(grid, can_move, stop, sand_value, start, n_sand_grains + 1)


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
        max_y = grid.n_rows + 1

        def stop(coord: GridCoordinates) -> bool:
            return coord == entrypoint

    else:
        max_y = grid.n_rows - 1

        def stop(coord: GridCoordinates) -> bool:
            x, y = coord
            return y >= max_y

    final_grid, n_grains = simulate(
        grid, can_move=AIR.__eq__, sand_value=SAND, max_y=max_y, stop=stop, start=entrypoint
    )
    return n_grains


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

    path = parse_path(test_input.splitlines()[0])
    actual = list(path_coords(path))
    expected = [(498, 4), (498, 5), (498, 6), (497, 6), (496, 6)]
    assert actual == expected, (actual, expected)

    expected_grid = CaveGrid(
        {
            (j, i): v
            for i, line in enumerate(test_grid.splitlines(keepends=False))
            for j, v in enumerate(line, 494)
            if v != AIR
        },
        AIR,
        10,
    )
    actual_grid = parse_grid(io.StringIO(test_input))
    extra_actual = set(actual_grid.grid).difference(expected_grid.grid)
    extra_expected = set(expected_grid.grid).difference(actual_grid.grid)
    assert actual_grid == expected_grid, (extra_expected, extra_actual)

    final_grid, n_grains = simulate(
        actual_grid,
        can_move=AIR.__eq__,
        stop=lambda t: t[1] >= actual_grid.n_rows,
        max_y=actual_grid.n_rows - 1,
        sand_value=SAND,
        start=(500, 0),
    )
    n_grains_expected = 24
    assert n_grains == n_grains_expected, (n_grains, n_grains_expected)
