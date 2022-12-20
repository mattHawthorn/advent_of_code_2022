from itertools import accumulate
from operator import itemgetter
from typing import IO, Dict, Iterable, Iterator, Tuple

from .util import GridCoordinates, Sprite, Vector, print_, set_verbose, sign, translate

Direction = str
Distance = int
Instruction = Tuple[Direction, Distance]

R, L, U, D = Direction("R"), Direction("L"), Direction("U"), Direction("D")
STEP_VECTORS: Dict[Direction, Vector] = {R: (1, 0), L: (-1, 0), U: (0, 1), D: (0, -1)}


def parse_instruction(line: str) -> Instruction:
    direction, dist = line.rstrip().split(maxsplit=1)
    return Direction(direction), Distance(int(dist))


def step_dist(distance: Distance) -> Distance:
    return Distance(min(abs(distance), 1) * sign(distance))


def catch_up(head: GridCoordinates, tail: GridCoordinates) -> GridCoordinates:
    hx, hy = head
    tx, ty = tail
    vec = vx, vy = Distance(hx - tx), Distance(hy - ty)
    if max(map(abs, vec)) == 1:  # type: ignore
        return tail
    else:
        step = step_dist(vx), step_dist(vy)
        return translate(step, tail)


def catch_up_all(head: GridCoordinates, tail: Sprite) -> Sprite:
    return list(accumulate(tail, catch_up, initial=head))


def transition(initial: Sprite, instruction: Instruction) -> Iterator[Sprite]:
    direction, distance = instruction
    step = STEP_VECTORS[direction]
    state = initial
    for _ in range(distance):
        head, tail = state[0], state[1:]
        head = translate(step, head)
        state = catch_up_all(head, tail)
        yield state


def all_transitions(initial: Sprite, instructions: Iterable[Instruction]) -> Iterator[Sprite]:
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
    initial = [(0, 0)] * n
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
