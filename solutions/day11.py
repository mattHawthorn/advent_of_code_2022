import re
from functools import partial, reduce
from itertools import cycle, islice
from operator import add, attrgetter, mul, sub
from typing import IO, Callable, Dict, List, NamedTuple

from .util import lcm, parse_blocks, print_, set_verbose

WorryLevel = int
MonkeyID = int
WorryLevelMod = Callable[[WorryLevel], WorryLevel]
WorryLevelPredicate = Callable[[WorryLevel], bool]

ADD, SUB, MUL = "+", "-", "*"
OP_MAP: Dict[str, Callable[[WorryLevel, WorryLevel], WorryLevel]] = {
    ADD: add,
    MUL: mul,
    SUB: sub,
}


class Monkey(NamedTuple):
    id_: MonkeyID
    items: List[WorryLevel]
    transform: WorryLevelMod
    modulus: int
    if_true: MonkeyID
    if_false: MonkeyID


monkey_re = re.compile(
    r"Monkey (?P<id>\d+):\n\s*"
    r"Starting items:\s*(?P<items>\d+(,\s*\d+)*)\n\s*"
    r"Operation:\s*new\s*=\s*(?P<expr>(?P<l>old|-?\d+)\s*(?P<op>[-+*])\s*(?P<r>old|-?\d+))\n\s*"
    r"Test:\s*divisible by (?P<modulus>\d+)\n\s*"
    r"If true:\s*throw to monkey (?P<if_true>\d+)\n\s*"
    r"If false:\s*throw to monkey (?P<if_false>\d+)"
)


def parse_monkey(s: str) -> Monkey:
    monkey_match = monkey_re.fullmatch(s)
    assert monkey_match is not None
    fields = monkey_match.groupdict()
    id_ = int(fields["id"])
    if_true_id = int(fields["if_true"])
    if_false_id = int(fields["if_false"])
    items = list(map(int, map(str.strip, fields["items"].split(","))))
    transform = parse_transform(fields["l"], fields["op"], fields["r"])
    modulus = int(fields["modulus"])
    return Monkey(
        id_,
        items=items,
        transform=transform,
        modulus=modulus,
        if_true=if_true_id,
        if_false=if_false_id,
    )


def parse_monkeys(input_: IO[str]) -> List[Monkey]:
    parsed_monkeys = parse_blocks(input_, parse_monkey)
    return sorted(parsed_monkeys, key=attrgetter("id_"))


def parse_transform(left: str, op: str, right: str) -> WorryLevelMod:
    l_is_input = left == "old"
    r_is_input = right == "old"
    op_f = OP_MAP[op]
    if l_is_input and r_is_input:
        return partial(self_op, op_f)
    elif op == SUB:
        return partial(sub, int(left)) if r_is_input else partial(rsub, int(right))
    else:
        # other ops are commutative
        return partial(op_f, int(left if r_is_input else right))


def self_op(op: Callable[[WorryLevel, WorryLevel], WorryLevel], value: WorryLevel) -> WorryLevel:
    return op(value, value)


def rsub(operand: WorryLevel, value: WorryLevel):
    return value - operand


def play_keepaway(monkeys: List[Monkey], rounds: int, worry_reduction: int = 1) -> List[int]:
    total_modulus = reduce(lcm, (m.modulus for m in monkeys))
    indexed_monkeys = {m.id_: m for m in monkeys}
    throw_counts = {m.id_: 0 for m in monkeys}
    active_monkeys = islice(cycle(enumerate(monkeys)), len(monkeys) * rounds)
    for ix, monkey in active_monkeys:
        throw_counts[monkey.id_] += len(monkey.items)
        while monkey.items:
            initial_worry_level = monkey.items.pop()
            new_worry_level = monkey.transform(initial_worry_level)
            if worry_reduction != 1:
                new_worry_level = new_worry_level // worry_reduction
            else:
                new_worry_level = new_worry_level % total_modulus
            to_monkey_id = (
                monkey.if_true if new_worry_level % monkey.modulus == 0 else monkey.if_false
            )
            to_monkey = indexed_monkeys[to_monkey_id]
            to_monkey.items.append(new_worry_level)

    return [throw_counts[m.id_] for m in monkeys]


def run(
    input_: IO[str],
    verbose: bool = False,
    rounds: int = 10000,
    worry_reduction: int = 1,
):
    set_verbose(verbose)
    monkeys = parse_monkeys(input_)
    print_(f"Parsed {len(monkeys)} monkeys with ids {[m.id_ for m in monkeys]}")
    throw_counts = play_keepaway(monkeys, rounds, worry_reduction)
    return reduce(mul, sorted(throw_counts, reverse=True)[:2])


test_input = """Monkey 0:
  Starting items: 79, 98
  Operation: new = old * 19
  Test: divisible by 23
    If true: throw to monkey 2
    If false: throw to monkey 3

Monkey 1:
  Starting items: 54, 65, 75, 74
  Operation: new = old + 6
  Test: divisible by 19
    If true: throw to monkey 2
    If false: throw to monkey 0

Monkey 2:
  Starting items: 79, 60, 97
  Operation: new = old * old
  Test: divisible by 13
    If true: throw to monkey 1
    If false: throw to monkey 3

Monkey 3:
  Starting items: 74
  Operation: new = old + 3
  Test: divisible by 17
    If true: throw to monkey 0
    If false: throw to monkey 1"""


def test():
    import io

    monkeys = parse_monkeys(io.StringIO(test_input))
    assert len(monkeys) == 4
    for monkey, (i, items) in zip(
        monkeys, [(0, [79, 98]), (1, [54, 65, 75, 74]), (2, [79, 60, 97]), (3, [74])]
    ):
        assert monkey.id_ == i
        assert monkey.items == items
    counts = play_keepaway(monkeys, 20, worry_reduction=3)
    expected = [101, 95, 7, 105]
    assert counts == expected, (counts, expected)
    counts = play_keepaway(monkeys, 10000, worry_reduction=1)
    most_active = sorted(counts, reverse=True)[:2]
    expected = [52166, 52013]
    assert most_active == expected, (most_active, expected)
