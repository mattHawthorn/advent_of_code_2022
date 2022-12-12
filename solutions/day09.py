from itertools import accumulate
from operator import itemgetter
from typing import IO, Dict, Iterable, Iterator, List, NewType, Tuple, cast

from .util import print_, set_verbose

Direction = NewType("Direction", str)
Distance = NewType("Distance", int)
Coord = NewType("Coord", int)
Instruction = Tuple[Direction, Distance]
Position = Tuple[Coord, Coord]
Vector = Tuple[Distance, Distance]
State = List[Position]

R, L, U, D = Direction("R"), Direction("L"), Direction("U"), Direction("D")
STEP_VECTORS = cast(
    Dict[Direction, Vector],
    {R: (1, 0), L: (-1, 0), U: (0, 1), D: (0, -1)},
)


def parse_instruction(line: str) -> Instruction:
    direction, dist = line.rstrip().split(maxsplit=1)
    return Direction(direction), Distance(int(dist))


def move(pos: Position, vec: Vector) -> Position:
    return Coord(pos[0] + vec[0]), Coord(pos[1] + vec[1])


def sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def step_dist(distance: Distance) -> Distance:
    return Distance(min(abs(distance), 1) * sign(distance))


def catch_up(head: Position, tail: Position) -> Position:
    hx, hy = head
    tx, ty = tail
    vec = vx, vy = Distance(hx - tx), Distance(hy - ty)
    if max(map(abs, vec)) == 1:  # type: ignore
        return tail
    else:
        step = step_dist(vx), step_dist(vy)
        return move(tail, step)


def catch_up_all(head: Position, tail: State) -> State:
    return list(accumulate(tail, catch_up, initial=head))


def transition(initial: State, instruction: Instruction) -> Iterator[State]:
    direction, distance = instruction
    step = STEP_VECTORS[direction]
    state = initial
    for _ in range(distance):
        head, tail = state[0], state[1:]
        head = move(head, step)
        state = catch_up_all(head, tail)
        yield state


def all_transitions(
    initial: State, instructions: Iterable[Instruction]
) -> Iterator[State]:
    print_(initial)
    yield initial
    for instruction in instructions:
        state = initial
        for state in transition(initial, instruction):
            print_(state)
            yield state
        initial = state


def run(input_: IO[str], n: int = 10, verbose: bool = False):
    set_verbose(verbose)
    instructions = list(map(parse_instruction, input_))
    initial = [(Coord(0), Coord(0))] * n
    states = all_transitions(initial, instructions)
    return len(set(map(itemgetter(-1), states)))


test_input = """R 4
U 4
L 3
D 1
R 4
D 1
L 5
R 2"""


def test():
    import io

    actual = run(io.StringIO(test_input), n=2, verbose=True)
    expected = 13
    assert actual == expected, (actual, expected)
