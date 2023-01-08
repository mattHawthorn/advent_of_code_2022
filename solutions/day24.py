import os
import time
from functools import partial
from itertools import chain, islice, repeat
from typing import IO, Iterable, List, NamedTuple, Optional, Set, Tuple

from .util import (
    GridCoordinates,
    T,
    compose,
    djikstra_any,
    fst,
    iterate,
    lcm,
    manhattan_distance,
    translate,
)

L, R, U, D = "<", ">", "^", "v"
LV, RV, UV, DV = (-1, 0), (1, 0), (0, 1), (0, -1)
DIRECTIONS = [LV, RV, UV, DV]
WALL, AIR = "#", "."

TimeStep = int
State = Tuple[GridCoordinates, TimeStep]


class Valley(NamedTuple):
    start: GridCoordinates
    end: GridCoordinates
    width: int
    height: int
    ls: Set[GridCoordinates]
    rs: Set[GridCoordinates]
    us: Set[GridCoordinates]
    ds: Set[GridCoordinates]


def parse_boundary_line(line: str) -> Tuple[int, int]:
    width = len(line) - 2
    start_x = line.index(AIR) - 1
    assert line.count(WALL) == width + 1
    return start_x, width


def parse(input_: IO[str]) -> Valley:
    lines = list(map(str.rstrip, input_))
    internal_lines = lines[1:-1]
    start_x, width = parse_boundary_line(lines[0])
    end_x, width_ = parse_boundary_line(lines[-1])
    assert width_ == width
    height = len(lines) - 2

    def extract_coords(c: str) -> Set[GridCoordinates]:
        return {
            (x, y)
            for y, line in enumerate(internal_lines)
            for x, c_ in enumerate(line[1:])
            if c_ == c
        }

    ls, rs, us, ds = map(extract_coords, [L, R, U, D])
    return Valley(
        (start_x, -1),
        (end_x, height),
        width=width,
        height=height,
        ls=ls,
        rs=rs,
        us=us,
        ds=ds,
    )


def time_step(valley: Valley):
    w, h = valley.width, valley.height
    return Valley(
        valley.start,
        valley.end,
        valley.width,
        valley.height,
        ls=set(((x - 1) % w, y) for x, y in valley.ls),
        rs=set(((x + 1) % w, y) for x, y in valley.rs),
        us=set((x, (y - 1) % h) for x, y in valley.us),
        ds=set((x, (y + 1) % h) for x, y in valley.ds),
    )


def next_states(time_steps: List[Valley], state: State) -> Iterable[State]:
    coords, t = state
    valley = time_steps[t]
    start, end, width, height = valley.start, valley.end, valley.width, valley.height

    def is_valid(xy: GridCoordinates) -> bool:
        x, y = xy
        return (
            xy == start
            or xy == end
            or (
                (0 <= x < width)
                and (0 <= y < height)
                and not any(
                    xy in obstructions
                    for obstructions in (valley.ls, valley.rs, valley.us, valley.ds)
                )
            )
        )

    t_ = (t + 1) % len(time_steps)
    valid_coords = filter(is_valid, chain((coords,), map(partial(translate, coords), DIRECTIONS)))
    return zip(valid_coords, repeat(t_))


def with_const_weight(it: Iterable[T]) -> Iterable[Tuple[T, int]]:
    return zip(it, repeat(1))


def animate(time_steps: List[Valley], path: List[State], delay: Optional[float]):
    D, U, L, R, EMPTY = "˯", "˰", "˱", "˲", " "
    os.system("clear")
    for (coords, t) in path:
        print(coords, t)
        valley = time_steps[(t - 1) % len(time_steps)]
        print(
            "".join(
                "■" if coords == (x, -1) else EMPTY if x == valley.start[0] else WALL
                for x in range(-1, valley.width + 1)
            )
        )
        for y in range(valley.height):
            print(WALL, end="")
            print(
                "".join(
                    "■"
                    if xy == coords
                    else U
                    if xy in valley.us
                    else D
                    if xy in valley.ds
                    else L
                    if xy in valley.ls
                    else R
                    if xy in valley.rs
                    else EMPTY
                    for xy in zip(range(valley.width), repeat(y))
                ),
                end="",
            )
            print("#")
        print(
            "".join(
                "■" if coords == (x, valley.height) else EMPTY if x == valley.end[0] else WALL
                for x in range(-1, valley.width + 1)
            )
        )
        if delay is None:
            input()  # press enter to continue
        else:
            time.sleep(delay)
        os.system("clear")


def run(
    input_: IO[str],
    part_2: bool = True,
    verbose: bool = False,
    delay: Optional[float] = 0.05,
) -> int:
    valley = parse(input_)
    n_time_steps = lcm(valley.width, valley.height)
    time_steps = list(islice(iterate(time_step, valley), n_time_steps))

    next_states_ = compose(partial(next_states, time_steps), with_const_weight)  # type: ignore
    is_end = compose(fst, valley.end.__eq__)
    is_start = compose(fst, valley.start.__eq__)
    dist_to_end = compose(fst, partial(manhattan_distance, valley.end))
    initial_state = (valley.start, 1)

    def shortest_path(state, is_terminal):
        path_n_steps = djikstra_any(next_states_, state, is_terminal, dist_to_end)
        assert path_n_steps is not None
        return path_n_steps[0]

    path = shortest_path(initial_state, is_end)
    if part_2:
        path_to_start = shortest_path(path[-1], is_start)
        path_to_end = shortest_path(path_to_start[-1], is_end)
        full_path = [*path, *path_to_start[1:], *path_to_end[1:]]
    else:
        full_path = path

    if verbose:
        animate(time_steps, full_path, delay)

    return len(full_path) - 1


test_input = """
#.######
#>>.<^<#
#.<..<<#
#>v.><>#
#<^v^^>#
######.#""".strip()


def test():
    import io

    actual_time = run(io.StringIO(test_input), part_2=False)
    expected_time = 18
    assert actual_time == expected_time, (expected_time, actual_time)

    actual_time = run(io.StringIO(test_input), part_2=True, verbose=True)
    expected_time = 54
    assert actual_time == expected_time, (expected_time, actual_time)
