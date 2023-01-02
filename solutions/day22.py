import re
from functools import reduce, singledispatch
from itertools import chain, count, islice, takewhile
from typing import IO, Dict, Iterator, List, Tuple, Union

from .util import GridCoordinates, SparseGrid, iterate, translate

SPACE, WALL, NOTHING = ".", "#", " "
L, R = "L", "R"

Turn = str
Distance = int
Direction = int
Instruction = Union[Distance, Turn]
State = Tuple[GridCoordinates, Direction]

DIRECTION_VECTORS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class Board:
    def __init__(self, grid: SparseGrid):
        def update(mapping: Dict[int, Tuple[int, int]], xy: Tuple[int, int]):
            x, y = xy
            if y in mapping:
                lo, hi = mapping[y]
                mapping[y] = min(lo, x), max(hi, x)
            else:
                mapping[y] = x, x
            return mapping

        self.grid = grid
        self.row_ranges: Dict[int, Tuple[int, int]] = reduce(update, grid.grid, {})
        self.col_ranges: Dict[int, Tuple[int, int]] = reduce(
            update, ((y, x) for x, y in grid.grid), {}
        )


def parse_row(s: str, y: int) -> Iterator[Tuple[str, GridCoordinates]]:
    return ((c, (x, y)) for x, c in enumerate(s, 1) if c != NOTHING)


def parse_grid(input_: IO[str]) -> SparseGrid[str]:
    lines = takewhile(bool, map(str.rstrip, input_))
    rows = map(parse_row, lines, count(1))
    values = chain.from_iterable(rows)
    grid = SparseGrid[str]({})

    def set_(grid: SparseGrid[str], value_coord: Tuple[str, GridCoordinates]):
        return grid.set(*value_coord)

    return reduce(set_, values, grid)


def parse_instruction(s: str) -> Instruction:
    return int(s) if s.isdigit() else s


def parse_instructions(s: str) -> List[Instruction]:
    return list(map(parse_instruction, re.findall(rf"\d+|{L}|{R}", s.strip())))


def parse(input_: IO[str]) -> Tuple[Board, List[Instruction]]:
    grid = parse_grid(input_)
    instructions = parse_instructions(input_.read())
    return Board(grid), instructions


@singledispatch
def move(instruction: Instruction, state: State, board: Board) -> State:
    raise TypeError(type(instruction))


@move.register(Turn)
def turn(turn: Turn, state: State, board: Board) -> State:
    coords, dir = state
    rot = 1 if turn == "R" else -1
    return coords, (dir + rot) % 4


@move.register(Distance)
def travel(distance: int, state: State, board: Board) -> State:
    coords, dir = state
    vec = DIRECTION_VECTORS[dir]
    ix = 0 if vec[0] != 0 else 1
    index = [board.row_ranges, board.col_ranges][ix]
    min_, max_ = index[coords[1 - ix]]

    def step(coords: GridCoordinates) -> GridCoordinates:
        new = list(coords)
        if coords[ix] == min_ and vec[ix] < 0:
            new[ix] = max_
            return new[0], new[1]
        elif coords[ix] == max_ and vec[ix] > 0:
            new[ix] = min_
            return new[0], new[1]
        else:
            return translate(vec, coords)

    def not_collision(coords: GridCoordinates) -> bool:
        return board.grid.get(coords) != WALL

    intermediate = takewhile(not_collision, islice(iterate(step, coords), distance + 1))
    new_coords = list(intermediate)[-1]
    return new_coords, dir


def apply_instructions(board: Board, state: State, instructions: List[Instruction]) -> State:
    def move_(state: State, instruction: Instruction) -> State:
        return move(instruction, state, board)

    return reduce(move_, instructions, state)


def decode(state: State) -> int:
    (x, y), dir = state
    return 1000 * y + 4 * x + dir


def run(input_: IO[str]):
    board, instructions = parse(input_)

    row_1_y = board.grid.y_min
    assert row_1_y is not None
    row_1_xmin, row_1_xmax = board.row_ranges[row_1_y]
    row_1_x = min(
        x for x in range(row_1_xmin, row_1_xmax + 1) if board.grid.grid[x, row_1_y] != WALL
    )
    initial_coords = (row_1_x, row_1_y)
    initial_state = (initial_coords, 0)

    final_state = apply_instructions(board, initial_state, instructions)
    return decode(final_state)


test_input = """
        ...#
        .#..
        #...
        ....
...#.......#
........#...
..#....#....
..........#.
        ...#....
        .....#..
        .#......
        ......#.

10R5L5R10L4R5L5"""[
    1:
]


def test():
    import io

    result = run(io.StringIO(test_input))
    expected = 6032
    assert result == expected, (expected, result)
