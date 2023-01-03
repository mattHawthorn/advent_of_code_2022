from functools import partial
from itertools import chain, cycle, filterfalse, islice, repeat, takewhile
from operator import itemgetter
from typing import IO, Dict, Iterable, List, Set, Tuple, cast

from .util import (
    GridCoordinates,
    SparseGrid,
    Sprite,
    Vector,
    compose,
    interleave,
    iterate,
    print_,
    set_verbose,
    translate_all,
)

ROCKS: List[Sprite] = [
    [(0, 0), (1, 0), (2, 0), (3, 0)],  # -
    [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],  # +
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],  # L (flipped horizontal)
    [(0, 0), (0, 1), (0, 2), (0, 3)],  # |
    [(0, 0), (1, 0), (0, 1), (1, 1)],  # ■
]
FLOOR = "-"
ROCK = "■"
AIR_DIRECTIONS: Dict[str, Vector] = {"<": (-1, 0), ">": (1, 0)}
DOWN = (0, -1)
WIDTH = 7
X_OFFSET = 2
Y_OFFSET = 4

State = bytes


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


def start_position(
    grid: SparseGrid, rock: Sprite, offset: GridCoordinates = (X_OFFSET, Y_OFFSET)
) -> Sprite:
    x_offset, y_offset = offset
    left_initial = (grid.x_min or 0) + x_offset
    bottom_initial = (grid.y_max or 0) + y_offset
    return translate_all((left_initial, bottom_initial), rock)


def drop_rock(
    cache: Dict[State, int],
    states: List[State],
    directions: Iterable[Tuple[int, Vector]],
    grid: SparseGrid,
    rock: Tuple[int, Sprite],
) -> Tuple[SparseGrid, bool]:
    rock_ix, rock_template = rock
    rock_ = start_position(grid, rock_template)
    for dir_ix, dir_ in directions:
        rock_, moved = move(grid, dir_, rock_)
        if dir_ == DOWN and not moved:
            grid.set_all(ROCK, rock_)
            break
    else:
        raise StopIteration()

    key = state_key(dir_ix, rock_ix, grid)
    cache_hit = key in cache
    states.append(key)
    global cache_hits
    if not cache_hit:
        cache[key] = cast(int, grid.y_max)

    return grid, cache_hit


def state_key(direction_ix: int, rock_ix: int, grid: SparseGrid) -> State:
    return b"".join(
        [direction_ix.to_bytes(2, "big"), rock_ix.to_bytes(2, "big"), grid_state_key(grid)]
    )


def grid_state_key(grid: SparseGrid) -> bytes:
    # connected empty space from top of grid reachable by each type of rock
    def neighbors(grid: SparseGrid, rock: Sprite, coords: GridCoordinates):
        x, y = coords
        candidates = [(x - 1, y), (x + 1, y), (x, y - 1)]
        return [
            (x, y)
            for x, y in candidates
            if x >= 1 and x <= WIDTH and not intersects(grid, translate_all((x, y), rock))
        ]

    def step(
        grid: SparseGrid,
        rock: Sprite,
        coords_frontier: Tuple[Set[GridCoordinates], Set[GridCoordinates]],
    ):
        coords, frontier = coords_frontier
        neighbors_ = partial(neighbors, grid, rock)
        new_frontier = set(
            filterfalse(coords.__contains__, chain.from_iterable(map(neighbors_, frontier)))
        )
        coords.update(new_frontier)
        return coords, new_frontier

    def coords_for_rock(grid: SparseGrid, rock: Sprite):
        top = cast(int, grid.y_max) + 1
        init_coords = frontier = {
            (x, top)
            for x in range(1, WIDTH + 1)
            if not intersects(grid, translate_all((x, top), rock))
        }
        assert init_coords
        has_frontier = compose(itemgetter(1), bool)
        step_ = partial(step, grid, rock)
        steps = takewhile(has_frontier, iterate(step_, (init_coords, frontier)))
        coords, _ = list(steps)[-1]
        return sorted((x, top - y) for x, y in coords)

    # insert impossible position as separation bytes between each set of rock positions
    sep = (WIDTH + 1, 255)
    available_rock_positions = interleave(map(partial(coords_for_rock, grid), ROCKS), repeat([sep]))
    return bytes(chain.from_iterable(chain.from_iterable(available_rock_positions)))


def final_height(air_directions: List[Vector], n_rocks: int) -> int:
    grid = SparseGrid(
        {(x, 0): FLOOR for x in range(1, WIDTH + 1)},  # floor
        x_min=1,
        x_max=WIDTH,
        y_min=0,
        y_max=0,
    )
    cache: Dict[State, int] = {}
    states: List[State] = []
    rocks = islice(cycle(enumerate(ROCKS)), n_rocks)
    directions = cycle(
        enumerate(list(interleave(air_directions, repeat(DOWN, len(air_directions)))))
    )
    for rock in rocks:
        grid, cache_hit = drop_rock(cache, states, directions, grid, rock)
        if cache_hit:
            return compute_height(cache, states, grid, n_rocks)
    else:
        return cast(int, grid.y_max)


def compute_height(
    cache: Dict[State, int], states: List[State], grid: SparseGrid, n_rocks: int
) -> int:
    # state key at the end of this iteration
    current_step = len(states)
    current_state = states[-1]
    # iteration after which we saw this same state key
    prior_step = states.index(current_state) + 1
    cycle_length = current_step - prior_step
    # gain in height between the two iterations
    current_height = cast(int, grid.y_max)
    initial_height = cache[current_state]
    cycle_height_gain = current_height - initial_height
    # number of cycles to run - total rocks minus the initial steps to start the cycle,
    # divided by the cycle length
    n_cycles, remainder = divmod(n_rocks - prior_step, cycle_length)
    final_state = states[prior_step - 1 + remainder]
    remaining_height = cache[final_state] - initial_height
    final_height = initial_height + cycle_height_gain * n_cycles + remaining_height
    print_(
        f"Cache hit at {len(states)} iterations: \n"
        f"Cycle begins after step {prior_step} at height {initial_height} "
        f"and ends after step {current_step} at height {current_height}. \n"
        f"Run a total of {n_cycles} cycles of "
        f"{cycle_length} iterations each ({n_cycles * cycle_length} iterations), "
        f"gaining {cycle_height_gain} height each,\n"
        f"and finally add {remaining_height} height from {remainder} iterations, "
        "resulting in a height of \n"
        f"{initial_height} + {cycle_height_gain} * {n_cycles} + "
        f"{remaining_height} = {final_height}"
    )
    return final_height


def run(input_: IO[str], part_2: bool = True, verbose: bool = False) -> int:
    set_verbose(verbose)
    n_rocks = 1_000_000_000_000 if part_2 else 2022
    air_directions = parse_directions(input_)
    return final_height(air_directions, n_rocks)


test_input = ">>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>"


def test():
    import io

    set_verbose(True)
    air_directions = parse_directions(io.StringIO(test_input))
    for n, expected_height in (2022, 3068), (1_000_000_000_000, 1514285714288):
        actual_height = final_height(air_directions, n)
        print(actual_height)
        assert actual_height == expected_height, (actual_height, expected_height)
        print_("Passed!")
