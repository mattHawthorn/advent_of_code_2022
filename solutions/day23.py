from collections import Counter, deque
from functools import partial, reduce
from itertools import accumulate, count, islice, repeat, takewhile
from operator import itemgetter
from typing import IO, Dict, Iterator, List, Optional, Tuple

from .util import GridCoordinates, SparseGrid, T, Vector, compose, translate

ELF, EMPTY = "#", "."
DIRECTIONS: List[Tuple[Vector, List[Vector]]] = [
    ((0, -1), [(-1, -1), (0, -1), (1, -1)]),
    ((0, 1), [(-1, 1), (0, 1), (1, 1)]),
    ((-1, 0), [(-1, 1), (-1, 0), (-1, -1)]),
    ((1, 0), [(1, 1), (1, 0), (1, -1)]),
]


def parse_row(row: str) -> Iterator[int]:
    return (ix for ix, c in enumerate(row) if c == ELF)


def insert_row(grid: SparseGrid[str], row_y: Tuple[Iterator[int], int]) -> SparseGrid[str]:
    xs, y = row_y
    return grid.set_all(ELF, zip(xs, repeat(y)))


def parse(input_: IO[str]) -> SparseGrid[str]:
    grid = SparseGrid[str]({})
    rows = map(parse_row, map(str.strip, input_))
    return reduce(insert_row, zip(rows, count()), grid)


def can_move(grid: SparseGrid, elf: GridCoordinates, checks: List[Vector]):
    coords = map(partial(translate, elf), checks)
    return not any(map(grid.__contains__, coords))


def move_proposal(
    grid: SparseGrid, moves: List[Tuple[Vector, List[Vector]]], elf: GridCoordinates
) -> Optional[Tuple[GridCoordinates, GridCoordinates]]:
    vecs = [vec for vec, checks in moves if can_move(grid, elf, checks)]
    if not vecs or len(vecs) == len(moves):
        return None
    else:
        return elf, translate(vecs[0], elf)


def move(grid: SparseGrid, from_to: Tuple[GridCoordinates, GridCoordinates]):
    from_, to = from_to
    return grid.set(grid.grid.pop(from_), to)


def step(grid: SparseGrid, moves: Tuple[Vector, List[Vector]]) -> Tuple[SparseGrid, bool]:
    elves = grid.grid
    propose = partial(move_proposal, grid, moves)
    maybe_moves: Dict[GridCoordinates, GridCoordinates] = dict(filter(None, map(propose, elves)))
    counts = Counter(maybe_moves.values())
    final_moves = [(from_, to) for from_, to in maybe_moves.items() if counts[to] <= 1]
    if not final_moves:
        return grid, False
    else:
        return reduce(move, final_moves, grid), True


def rotate(xs: List[T]) -> Iterator[List[T]]:
    q = deque(xs)
    while True:
        yield list(q)
        q.append(q.popleft())


def simulate(grid: SparseGrid, steps: int) -> SparseGrid:
    directions = islice(rotate(DIRECTIONS), steps)
    step_ = compose(step, itemgetter(0))  # type: ignore
    return reduce(step_, directions, grid)  # type: ignore


def simulate_until_stationary(grid: SparseGrid) -> int:
    directions = rotate(DIRECTIONS)

    def step_(grid_, directions):
        return step(grid_[0], directions)

    nonstationary = itemgetter(1)
    steps = takewhile(nonstationary, accumulate(directions, step_, initial=(grid, True)))
    return sum(1 for _ in steps)


def score(grid: SparseGrid) -> int:
    x_min = min(map(itemgetter(0), grid.grid))
    x_max = max(map(itemgetter(0), grid.grid))
    y_min = min(map(itemgetter(1), grid.grid))
    y_max = max(map(itemgetter(1), grid.grid))
    return (x_max - x_min + 1) * (y_max - y_min + 1) - len(grid.grid)


def run(input_: IO[str], part_2: bool = True):
    grid = parse(input_)
    if part_2:
        return simulate_until_stationary(grid)
    else:
        final_grid = simulate(grid, 10)
        return score(final_grid)


test_input = """
.....
..##.
..#..
.....
..##.
.....""".strip()

test_output = """
..#..
....#
#....
....#
.....
..#..""".strip()


def test():
    import io

    grid = parse(io.StringIO(test_input))
    expected = parse(io.StringIO(test_output))
    result = simulate(grid, 3)
    assert result.grid == expected.grid, (
        set(result.grid).difference(expected.grid),
        set(expected.grid).difference(result.grid),
    )

    area = run(io.StringIO(test_input), part_2=False)
    expected_area = 25
    assert area == expected_area, (expected_area, area)

    steps = run(io.StringIO(test_input), part_2=True)
    expected_steps = 4
    assert steps == expected_steps, (expected_steps, steps)
