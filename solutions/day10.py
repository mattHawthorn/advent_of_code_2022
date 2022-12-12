from itertools import accumulate, chain, cycle, islice, starmap
from operator import mul
from typing import IO, Iterable, Iterator, List, Optional, Tuple

from .util import chunked

NOOP = "noop"
ADDX = "addx"
OFF, ON = PIXEL_STATES = (" ", "#")

Op = str
State = int
Instruction = Tuple[Op, Optional[State]]


def parse_instruction(line: str) -> Instruction:
    tokens = line.rstrip().split(" ", maxsplit=1)
    op = tokens[0]
    arg = None if len(tokens) == 1 else int(tokens[1])
    return op, arg


def run_instruction(prior_state: List[State], instruction: Instruction) -> List[State]:
    """[value *during* each cycle], state at the end of all cycles"""
    op, arg = instruction
    state = prior_state[-1]
    if op == ADDX:
        assert arg is not None
        return [state, state + arg]
    elif op == NOOP:
        return [state]
    else:
        raise ValueError(f"Unknown op: {op}")


def simulate(instructions: Iterable[Instruction], initial: int = 1) -> Iterator[State]:
    results = accumulate(instructions, run_instruction, initial=[initial])
    return chain.from_iterable(results)


def pixel_activation(crt_position: int, state: State) -> bool:
    return abs(state - crt_position) <= 1


def terminal_output(
    states: Iterable[State],
    screen_width: int = 40,
    pixel_values: Tuple[str, str] = PIXEL_STATES,
) -> str:
    pixel_position = cycle(range(screen_width))
    pixel_activations = starmap(pixel_activation, zip(pixel_position, states))
    pixel_values_ = map(pixel_values.__getitem__, pixel_activations)
    lines = map("".join, chunked(screen_width, pixel_values_))
    return "\n".join(lines)


def run(
    input_: IO[str],
    start: int = 20,
    stop=220,
    step: int = 40,
    part_2: bool = True,
    on: str = ON,
    off: str = OFF,
) -> str:
    instructions = map(parse_instruction, input_)
    states = simulate(instructions)
    if part_2:
        return terminal_output(states, pixel_values=(off, on))
    else:
        indexed_states = enumerate(states, 1)
        interesting_states = list(islice(indexed_states, start - 1, stop, step))
        signal_strengths = starmap(mul, interesting_states)
        return str(sum(signal_strengths))


test_input = """addx 15|addx -11|addx 6|addx -3|addx 5|addx -1|addx -8|addx 13|addx 4|noop
addx -1|addx 5|addx -1|addx 5|addx -1|addx 5|addx -1|addx 5|addx -1|addx -35|addx 1|addx 24
addx -19|addx 1|addx 16|addx -11|noop|noop|addx 21|addx -15|noop|noop|addx -3|addx 9|addx 1
addx -3|addx 8|addx 1|addx 5|noop|noop|noop|noop|noop|addx -36|noop|addx 1|addx 7|noop|noop
noop|addx 2|addx 6|noop|noop|noop|noop|noop|addx 1|noop|noop|addx 7|addx 1|noop|addx -13
addx 13|addx 7|noop|addx 1|addx -33|noop|noop|noop|addx 2|noop|noop|noop|addx 8|noop|addx -1
addx 2|addx 1|noop|addx 17|addx -9|addx 1|addx 1|addx -3|addx 11|noop|noop|addx 1|noop
addx 1|noop|noop|addx -13|addx -19|addx 1|addx 3|addx 26|addx -30|addx 12|addx -1|addx 3
addx 1|noop|noop|noop|addx -9|addx 18|addx 1|addx 2|noop|noop|addx 9|noop|noop|noop|addx -1
addx 2|addx -37|addx 1|addx 3|noop|addx 15|addx -21|addx 22|addx -6|addx 1|noop|addx 2
addx 1|noop|addx -10|noop|noop|addx 20|addx 1|addx 2|addx 2|addx -6|addx -11|noop|noop|noop
""".replace(
    "|", "\n"
).strip()

expected_output = """
##..##..##..##..##..##..##..##..##..##..
###...###...###...###...###...###...###.
####....####....####....####....####....
#####.....#####.....#####.....#####.....
######......######......######......####
#######.......#######.......#######.....""".lstrip()


def test():
    import io

    actual = run(io.StringIO(test_input))
    expected = 13140
    assert actual == expected, (actual, expected)
