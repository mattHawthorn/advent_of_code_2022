import re
from functools import partial, reduce, singledispatch
from itertools import chain, count, islice, product, repeat, starmap, takewhile
from operator import mul, sub
from typing import IO, Callable, Dict, Iterable, Iterator, List, Set, Tuple, Union, cast

from .util import (
    GridCoordinates,
    GridCoordinates3D,
    SparseGrid,
    connected_components,
    dfs_graph,
    gcd,
    identity,
    iterate,
    print_,
    set_verbose,
    translate,
    translate_inv,
    weighted_edges_to_graph,
)

SPACE, WALL, NOTHING = ".", "#", " "
L, R = "L", "R"

Turn = str
Distance = int
Direction = int
Instruction = Union[Distance, Turn]
State = Tuple[GridCoordinates, Direction]

E, S, W, N = 0, 1, 2, 3
DIRECTION_VECTORS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
# when matching e.g. the S face of one square with the W face of another (or N with E, etc),
# reverse the order of the mapped coordinate range in the coordinate system order
REVERSE_DIR_TRANSITIONS = {(S, W), (W, S), (N, E), (E, N), (E, E), (W, W), (N, N), (S, S)}


class Board:
    grid: SparseGrid
    row_ranges: Dict[int, Tuple[int, int]]
    col_ranges: Dict[int, Tuple[int, int]]
    transitions: Dict[State, State]

    def __init__(self, grid: SparseGrid, cube_topology: bool = False):
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
        self.transitions = self.cube_topology() if cube_topology else self.flat_topology()

    def flat_topology(self) -> Dict[State, State]:
        return dict(
            chain(
                chain.from_iterable(
                    [(((x_min, y), W), ((x_max, y), W)), (((x_max, y), E), ((x_min, y), E))]
                    for y, (x_min, x_max) in self.row_ranges.items()
                ),
                chain.from_iterable(
                    [(((x, y_min), N), ((x, y_max), N)), (((x, y_max), S), ((x, y_min), S))]
                    for x, (y_min, y_max) in self.col_ranges.items()
                ),
            )
        )

    def cube_topology(self) -> Dict[State, State]:
        xs, ys = (
            chain.from_iterable((lo, hi + 1) for lo, hi in ranges.values())
            for ranges in (self.row_ranges, self.col_ranges)
        )
        # for mypy
        assert self.grid.x_min is not None and self.grid.x_max is not None
        assert self.grid.y_min is not None and self.grid.y_max is not None
        resolution = mesh_resolution(xs, ys)
        x_offset = self.grid.x_min
        y_offset = self.grid.y_min
        x_mesh_size = (self.grid.x_max + 1 - x_offset) // resolution
        y_mesh_size = (self.grid.y_max + 1 - y_offset) // resolution

        def has_square(mesh_x, mesh_y):
            x_lo = x_offset + resolution * mesh_x
            y_lo = y_offset + resolution * mesh_y
            return (x_lo, y_lo) in self.grid

        mesh_square_coords = {
            (mesh_x, mesh_y)
            for mesh_x, mesh_y in product(range(x_mesh_size), range(y_mesh_size))
            if has_square(mesh_x, mesh_y)
        }
        assert len(mesh_square_coords) == 6

        def face_edge_coords(
            mesh_x: int, mesh_y: int, dir1: Direction, reverse: bool
        ) -> Iterator[GridCoordinates]:
            maybe_reverse = cast(
                Callable[[Iterable[int]], Iterable[int]], reversed if reverse else identity
            )
            x_range: Iterable[int]
            y_range: Iterable[int]
            if dir1 in (E, W):
                y_min = y_offset + resolution * mesh_y
                y_range = maybe_reverse(range(y_min, y_min + resolution))
                right_side = dir1 == E
                x = x_offset + resolution * (mesh_x + right_side) - right_side
                x_range = repeat(x, resolution)
            else:
                x_min = x_offset + resolution * mesh_x
                x_range = maybe_reverse(range(x_min, x_min + resolution))
                bottom_side = dir1 == S
                y = y_offset + resolution * (mesh_y + bottom_side) - bottom_side
                y_range = repeat(y, resolution)

            return zip(x_range, y_range)

        abstract_transitions = mesh_transitions(mesh_square_coords)
        return dict(
            chain.from_iterable(
                zip(
                    zip(face_edge_coords(x1, y1, dir1, reverse=False), repeat(dir1)),
                    zip(
                        face_edge_coords(
                            x2, y2, dir2, reverse=(dir1, dir2) in REVERSE_DIR_TRANSITIONS
                        ),
                        repeat((dir2 + 2) % 4),
                    ),
                )
                for ((x1, y1), dir1), ((x2, y2), dir2) in abstract_transitions
            )
        )


def mesh_resolution(xs: Iterable[int], ys: Iterable[int]) -> int:
    xs_ = sorted(set(xs))
    ys_ = sorted(set(ys))
    x_diff = min(map(sub, xs_[1:], xs_))
    y_diff = min(map(sub, xs_[1:], ys_))
    assert x_diff % y_diff == 0 or y_diff % x_diff == 0
    return gcd(x_diff, y_diff)


def mesh_transitions(mesh_coords: Set[GridCoordinates]) -> Iterator[Tuple[State, State]]:
    # given a square in the cube mesh and a direction facing (which determines a face of the square)
    # which square do we transition to, and to which side of that square?
    neighbors = {
        (coords, dir_): translate(vec, coords)
        for coords in mesh_coords
        for dir_, vec in enumerate(DIRECTION_VECTORS)
    }
    edges = ((from_, to) for (from_, _), to in neighbors.items() if to in mesh_coords)
    mesh_graph = weighted_edges_to_graph((e, 1) for e in edges)
    assert len(list(connected_components(mesh_graph))) == 1

    # map each mesh square to a cube face and each cube face to its (E, S, W, N)-ordered neighbors
    def update_mappings(
        mappings: Tuple[
            Dict[GridCoordinates, GridCoordinates3D],
            Dict[GridCoordinates3D, List[GridCoordinates3D]],
        ],
        node: GridCoordinates,
    ):
        mesh_cube_mapping, face_order_mapping = mappings
        nbrs = mesh_graph[node]
        unmapped_neighbors = [n for n in nbrs if n not in mesh_cube_mapping]
        if unmapped_neighbors:
            prev = next(n for n in nbrs if n in mesh_cube_mapping)
            prev_cube_face = mesh_cube_mapping[prev]
            current_cube_face = mesh_cube_mapping[node]
            mesh_cube_mapping.update(
                (n, cube_face_hop(prev, prev_cube_face, node, current_cube_face, n))
                for n in unmapped_neighbors
            )

        nbr = next(iter(nbrs))
        cube_face = mesh_cube_mapping[node]
        nbr_cube_face = mesh_cube_mapping[nbr]
        dir_ = DIRECTION_VECTORS.index(translate_inv(node, nbr))
        direction_ordered_faces = [
            rotate_around(cube_face, nbr_cube_face, (d - dir_) % 4) for d in range(4)
        ]
        face_order_mapping[cube_face] = direction_ordered_faces

        return mesh_cube_mapping, face_order_mapping

    leaf_node, next_node = next(
        (node, next(iter(nbrs))) for node, nbrs in mesh_graph.items() if len(nbrs) == 1
    )
    mesh_cube_mapping: Dict[GridCoordinates, GridCoordinates3D] = {
        leaf_node: (1, 0, 0),
        next_node: (0, 1, 0),
    }
    face_order_mapping: Dict[GridCoordinates3D, List[GridCoordinates3D]] = {}
    mesh_cube_mapping, face_order_mapping = reduce(
        update_mappings,
        dfs_graph(mesh_graph, leaf_node),
        (mesh_cube_mapping, face_order_mapping),
    )
    cube_mesh_mapping = {face: square for square, face in mesh_cube_mapping.items()}
    assert len(mesh_cube_mapping) == len(face_order_mapping) == len(mesh_coords)
    assert set(mesh_cube_mapping) == mesh_coords
    assert set(cube_mesh_mapping) == set(face_order_mapping)

    def state_transitions(face: GridCoordinates3D, ordered_neighbors: List[GridCoordinates3D]):
        mesh_square = cube_mesh_mapping[face]
        return (
            ((mesh_square, dir_), (cube_mesh_mapping[nbr], face_order_mapping[nbr].index(face)))
            for dir_, nbr in enumerate(ordered_neighbors)
            if cube_mesh_mapping[nbr] not in mesh_graph[mesh_square]
        )

    return chain.from_iterable(starmap(state_transitions, face_order_mapping.items()))


def cube_face_hop(
    prior_mesh_coords: GridCoordinates,
    prior_cube_face: GridCoordinates3D,
    mesh_coords: GridCoordinates,
    cube_face: GridCoordinates3D,
    next_mesh_coords: GridCoordinates,
) -> GridCoordinates3D:
    hop_vec1 = translate_inv(prior_mesh_coords, mesh_coords)
    hop_vec2 = translate_inv(mesh_coords, next_mesh_coords)
    direction_1, direction_2 = DIRECTION_VECTORS.index(hop_vec1), DIRECTION_VECTORS.index(hop_vec2)
    turn = (direction_2 - direction_1) % 4
    rotation_axis = cross_product(prior_cube_face, cube_face)
    cube_hop_vec1 = translate_inv(prior_cube_face, cube_face)
    cube_hop_vec2 = rotate_around(cube_face, rotate_around(rotation_axis, cube_hop_vec1, 1), turn)
    next_cube_face = translate(cube_hop_vec2, cube_face)
    return next_cube_face


def matmul(mat: Iterable[GridCoordinates3D], vec: GridCoordinates3D) -> GridCoordinates3D:
    return cast(GridCoordinates3D, tuple(sum(map(mul, row, vec)) for row in mat))


def rotation_matrix(axis: GridCoordinates3D) -> List[GridCoordinates3D]:
    ixs = [i for i, j in enumerate(axis) if j == 0]
    ix, sign = next((i, j) for i, j in enumerate(axis) if j != 0)
    x_ix, y_ix = ixs if ixs[1] - ixs[0] == 1 else reversed(ixs)
    matrix = [[0, 0, 0] for _ in range(3)]
    matrix[ix][ix] = 1
    matrix[x_ix][y_ix] = -sign
    matrix[y_ix][x_ix] = sign
    return [cast(GridCoordinates3D, tuple(row)) for row in matrix]


def rotate_around(axis: GridCoordinates3D, vec: GridCoordinates3D, turns: int) -> GridCoordinates3D:
    matrix = rotation_matrix(axis)
    vecs = islice(iterate(partial(matmul, matrix), vec), turns % 4 + 1)
    return list(vecs)[-1]


def cross_product(vec1: GridCoordinates3D, vec2: GridCoordinates3D) -> GridCoordinates3D:
    x1, y1, z1 = vec1
    x2, y2, z2 = vec2
    return (y1 * z2 - z1 * y2), (z1 * x2 - x1 * z2), (x1 * y2 - y1 * x2)


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


def parse(input_: IO[str], cube_topology: bool) -> Tuple[Board, List[Instruction]]:
    grid = parse_grid(input_)
    instructions = parse_instructions(input_.read())
    return Board(grid, cube_topology), instructions


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
    def step(state: State) -> State:
        to_state = board.transitions.get(state)
        if to_state is None:
            coords, dir = state
            return translate(DIRECTION_VECTORS[dir], coords), dir
        else:
            return to_state

    def not_collision(state: State) -> bool:
        coords, _ = state
        return board.grid.get(coords) != WALL

    intermediate = takewhile(not_collision, islice(iterate(step, state), distance + 1))
    new_state = list(intermediate)[-1]
    return new_state


def apply_instructions(board: Board, state: State, instructions: List[Instruction]) -> State:
    def move_(state: State, instruction: Instruction) -> State:
        return move(instruction, state, board)

    return reduce(move_, instructions, state)


def decode(state: State) -> int:
    (x, y), dir = state
    return 1000 * y + 4 * x + dir


def start_coords(board: Board) -> GridCoordinates:
    y = board.grid.y_min
    assert y is not None
    row_1_xmin, row_1_xmax = board.row_ranges[y]
    x = min(x for x in range(row_1_xmin, row_1_xmax + 1) if board.grid.grid[x, y] != WALL)
    return (x, y)


def run(input_: IO[str], part_2: bool = True, verbose: bool = False):
    set_verbose(verbose)
    board, instructions = parse(input_, cube_topology=part_2)
    initial_state = (start_coords(board), E)
    final_state = apply_instructions(board, initial_state, instructions)
    print_("Initial state:", initial_state)
    print_("Final state:", final_state)
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

10R5L5R10L4R5L5""".lstrip(
    "\n"
)


def test():
    import io

    result = run(io.StringIO(test_input), part_2=False)
    expected = 6032
    assert result == expected, (expected, result)

    result = run(io.StringIO(test_input), part_2=True)
    expected = 5031
    assert result == expected, (expected, result)

    for axis, start, end in [
        ((0, 0, 1), (1, 1, 1), (-1, 1, 1)),
        ((0, 0, -1), (1, 0, 1), (0, -1, 1)),
        ((0, 1, 0), (1, -1, -1), (-1, -1, -1)),
        ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
        ((1, 0, 0), (-1, 0, 1), (-1, -1, 0)),
        ((-1, 0, 0), (1, 1, 0), (1, 0, -1)),
    ]:
        result_vec = rotate_around(axis, start, 1)
        assert result_vec == end, (axis, start, end, result_vec)
