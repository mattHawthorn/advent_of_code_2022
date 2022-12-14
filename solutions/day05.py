from functools import reduce
from itertools import takewhile, zip_longest
from typing import IO, List, NamedTuple, Optional

from .util import T, compose, nonnull_head

Stacks = List[List[T]]


class Instruction(NamedTuple):
    qty: int
    from_: int
    to: int


def parse_crate_row(line: str) -> List[Optional[str]]:
    chunks = (line[i : i + 4] for i in range(0, len(line) - 1, 4))  # noqa: E203
    return [c[1] if c[0] == "[" else None for c in chunks]


def not_footer(line: str):
    return not all(part.isdigit() for part in line.split())


def parse_crates(input_: IO[str]) -> Stacks[str]:
    lines = map(str.rstrip, input_)
    header = takewhile(not_footer, lines)
    crate_rows = list(map(parse_crate_row, header))
    from_the_bottom = reversed(crate_rows)
    transposed = zip_longest(*from_the_bottom, fillvalue=None)
    crate_columns: Stacks[str] = list(map(compose(nonnull_head, list), transposed))  # type: ignore
    return crate_columns


def parse_instruction(line: str) -> Instruction:
    parts = line.split()
    return Instruction(*map(int, parts[1::2]))


def parse_instructions(input_: IO[str]) -> List[Instruction]:
    return list(map(parse_instruction, filter(bool, map(str.strip, input_))))


def move(stacks: Stacks[T], inst: Instruction, sequential: bool) -> Stacks[T]:
    from_, to_ = inst.from_ - 1, inst.to - 1
    from_stack = stacks[from_]
    to_stack = stacks[to_]
    from_items = [from_stack.pop() for _ in range(inst.qty)]
    to_stack.extend(from_items if sequential else reversed(from_items))
    return stacks


def move_sequential(stacks: Stacks[T], inst: Instruction) -> Stacks[T]:
    return move(stacks, inst, sequential=True)


def move_concurrent(stacks: Stacks[T], inst: Instruction) -> Stacks[T]:
    return move(stacks, inst, sequential=False)


def run(input_: IO[str], part_2: bool = True) -> str:
    stacks = parse_crates(input_)
    instructions = parse_instructions(input_)
    op = move_concurrent if part_2 else move_sequential
    reduce(op, instructions, stacks)
    return "".join(stack[-1] for stack in stacks)


def test():
    import io

    f = io.StringIO(_test_input)
    stacks = parse_crates(f)
    assert stacks == [list("ZN"), list("MCD"), list("P")]
    instrs = parse_instructions(f)
    assert instrs == [
        Instruction(1, 2, 1),
        Instruction(3, 1, 3),
        Instruction(2, 2, 1),
        Instruction(1, 1, 2),
    ]


_test_input = """
    [D]
[N] [C]
[Z] [M] [P]
 1   2   3

move 1 from 2 to 1
move 3 from 1 to 3
move 2 from 2 to 1
move 1 from 1 to 2""".lstrip(
    "\n"
)
