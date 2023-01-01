from dataclasses import dataclass
from functools import partial, singledispatch
from operator import add, floordiv, mul, sub
from typing import IO, Callable, Dict, NamedTuple, Optional, Union

Op = Callable[[int, int], int]
OPS: Dict[str, Op] = {"+": add, "-": sub, "*": mul, "/": floordiv}


class VarMonkey(NamedTuple):
    id_: str

    @property
    def value(self):
        return Var(self.id_)


class NumberMonkey(NamedTuple):
    id_: str
    value: int


class OpMonkey(NamedTuple):
    id_: str
    op: Op
    left: str
    right: str


AnyMonkey = Union[VarMonkey, NumberMonkey, OpMonkey]


class Expr:
    def __add__(self, other: int):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other: int):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __sub__(self, other: int):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __floordiv__(self, other):
        return Div(self, other)

    def __rfloordiv__(self, other):
        return Div(other, self)


@dataclass
class Var(Expr):
    id_: str

    def __str__(self):
        return self.id_


@dataclass
class BinaryOp(Expr):
    left: Union[int, Expr]
    right: Union[int, Expr]

    @property
    def inv_op(self) -> Callable:
        return op_cls_to_inv_op[type(self)]

    @property
    def op(self) -> Callable:
        return op_cls_to_op[type(self)]


class Add(BinaryOp):
    pass


class Mul(BinaryOp):
    pass


class Sub(BinaryOp):
    pass


class Div(BinaryOp):
    pass


op_cls_to_inv_op = {Add: sub, Sub: add, Mul: floordiv, Div: mul}
op_cls_to_op = {Add: add, Sub: sub, Mul: mul, Div: floordiv}


def parse_monkey(s: str, var: Optional[str] = None) -> AnyMonkey:
    id_, expr = s.strip().split(": ", 1)
    if id_ == var:
        return VarMonkey(id_)
    elif expr.isdigit():
        return NumberMonkey(id_, int(expr))
    else:
        l, op, r = expr.strip().split()
        return OpMonkey(id_, OPS[op], l, r)


def parse(input_: IO[str], var: Optional[str] = None) -> Dict[str, AnyMonkey]:
    return {monkey.id_: monkey for monkey in map(partial(parse_monkey, var=var), input_)}


@singledispatch
def eval(monkey: AnyMonkey, monkeys: Dict[str, AnyMonkey], cache: Dict[str, int]):
    raise TypeError(type(monkey))


@eval.register(NumberMonkey)
@eval.register(VarMonkey)
def eval_number(monkey: NumberMonkey, monkeys: Dict[str, AnyMonkey], cache: Dict[str, int]):
    return monkey.value


@eval.register(OpMonkey)
def eval_op(monkey: OpMonkey, monkeys: Dict[str, AnyMonkey], cache: Dict[str, int]):
    value = cache.get(monkey.id_)
    if value is None:
        left = monkeys[monkey.left]
        right = monkeys[monkey.right]
        value = monkey.op(eval(left, monkeys, cache), eval(right, monkeys, cache))
        cache[monkey.id_] = value
    return value


@singledispatch
def solve(expr: Expr, value: int) -> int:
    raise TypeError(type(Expr))


@solve.register(Var)
def solve_var(expr: Var, value: int) -> int:
    return value


@solve.register(Add)
@solve.register(Mul)
def solve_commutative_op(expr: BinaryOp, value: int) -> int:
    l, r = (expr.left, expr.right) if isinstance(expr.left, Expr) else (expr.right, expr.left)
    assert isinstance(l, Expr)
    return solve(l, expr.inv_op(value, r))


@solve.register(Sub)
@solve.register(Div)
def solve_noncommutative_op(expr: BinaryOp, value: int) -> int:
    if isinstance(expr.left, Expr):
        return solve(expr.left, expr.inv_op(value, expr.right))
    else:
        assert isinstance(expr.right, Expr)
        return solve(expr.right, expr.op(expr.left, value))


def run(input_: IO[str], part_2: bool = True):
    monkeys = parse(input_, "humn" if part_2 else None)
    root = monkeys["root"]
    assert isinstance(root, OpMonkey)
    cache: Dict[str, int] = {}
    if part_2:
        l, r = eval(monkeys[root.left], monkeys, cache), eval(monkeys[root.right], monkeys, cache)
        left, right = (l, r) if isinstance(l, Expr) else (r, l)
        assert isinstance(right, int)
        return solve(left, right)
    else:
        return eval(root, monkeys, cache)


test_input = """
root: pppw + sjmn
dbpl: 5
cczh: sllz + lgvd
zczc: 2
ptdq: humn - dvpt
dvpt: 3
lfqf: 4
humn: 5
ljgn: 2
sjmn: drzm * dbpl
sllz: 4
pppw: cczh / lfqf
lgvd: ljgn * ptdq
drzm: hmdt - zczc
hmdt: 32""".strip()


def test():
    import io

    result = run(io.StringIO(test_input), part_2=False)
    expected = 152
    assert result == expected, (result, expected)

    result = run(io.StringIO(test_input), part_2=True)
    expected = 301
    assert result == expected, (result, expected)
