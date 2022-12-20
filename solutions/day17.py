from functools import partial, reduce
from itertools import cycle, islice, repeat
from typing import IO, Dict, Iterable, List, Tuple, cast

from .util import GridCoordinates, SparseGrid, Sprite, Vector, interleave, translate_all

ROCKS: List[Sprite] = [
    [(0, 0), (1, 0), (2, 0), (3, 0)],  # -
    [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],  # +
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],  # L (flipped horizontal)
    [(0, 0), (0, 1), (0, 2), (0, 3)],  # |
    [(0, 0), (1, 0), (0, 1), (1, 1)],  # ■
]
FLOOR = "-"
ROCK = "#"
AIR_DIRECTIONS: Dict[str, Vector] = {"<": (-1, 0), ">": (1, 0)}
DOWN = (0, -1)
WIDTH = 7
X_OFFSET = 2
Y_OFFSET = 4


def parse_directions(input_: IO[str]) -> List[Vector]:
    return list(map(AIR_DIRECTIONS.__getitem__, input_.read().strip()))


def out_of_bounds(grid: SparseGrid, coords: GridCoordinates) -> bool:
    x, y = coords
    x_min = x if grid.x_min is None else grid.x_min
    x_max = x if grid.x_max is None else grid.x_max
    return x < x_min or x > x_max


def intersects(grid: SparseGrid, obj: Sprite) -> bool:
    return any(coords in grid for coords in obj) or any(map(partial(out_of_bounds, grid), obj))


def move(grid: SparseGrid, direction: Vector, rock: Sprite) -> Tuple[Sprite, bool]:
    new_rock = translate_all(direction, rock)
    return (rock, False) if intersects(grid, new_rock) else (new_rock, True)


def drop_rock(directions: Iterable[Vector], grid: SparseGrid, rock: Sprite) -> SparseGrid:
    directions = iter(directions)
    left_initial = (grid.x_min or 0) + X_OFFSET
    bottom_initial = (grid.y_max or 0) + Y_OFFSET
    rock_start = translate_all((left_initial, bottom_initial), rock)
    while True:
        d = next(directions)
        rock_start, moved = move(grid, d, rock_start)
        if d == DOWN and not moved:
            grid.set_all(ROCK, rock_start)
            break
    return grid


def drop_rocks(
    air_directions: Iterable[Vector], grid: SparseGrid, rocks: Iterable[Sprite]
) -> SparseGrid:
    directions = interleave(air_directions, repeat(DOWN))
    return reduce(partial(drop_rock, directions), rocks, grid)


def simulate(air_directions: List[Vector], n_rocks: int) -> SparseGrid:
    grid = SparseGrid(
        {(x, 0): FLOOR for x in range(1, WIDTH + 1)},  # floor
        x_min=1,
        x_max=WIDTH,
        y_min=0,
        y_max=0,
    )
    return drop_rocks(
        cycle(air_directions),
        grid,
        islice(cycle(ROCKS), n_rocks),
    )


def run(input_: IO[str], part_2: bool = True) -> int:
    n_rocks = 1_000_000_000_000 if part_2 else 2022
    air_directions = parse_directions(input_)
    grid = simulate(air_directions, n_rocks)
    return cast(int, grid.y_max)


test_input = ">>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>"


def test():
    import io

    air_directions = parse_directions(io.StringIO(test_input))
    for n, expected_height in (2022, 3068), (1_000_000_000_000, 1514285714288):
        grid = simulate(air_directions, n)
        actual_height = grid.y_max
        assert expected_height == expected_height, (actual_height, expected_height)
