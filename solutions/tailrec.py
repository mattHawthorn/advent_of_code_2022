import dis
from collections import deque
from inspect import Signature, signature
from itertools import chain, count, islice
from operator import itemgetter
from types import CodeType, FunctionType
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from warnings import warn

RETURN_OPS = {name for name in dis.opname if name.startswith("RETURN_")}
SIDE_EFFECT_OPS = RETURN_OPS | {
    name
    for name in dis.opname
    if name.startswith("STORE_")
    or name.startswith("JUMP_")
    or name.startswith("YIELD_")
}
BYTECODE_SIZE = 2
CALL_FUNCTION_OP, CALL_FUNCTION_KW_OP = "CALL_FUNCTION", "CALL_FUNCTION_KW"
CALL_OPS = {CALL_FUNCTION_OP, CALL_FUNCTION_KW_OP}
BUILD_TUPLE_OP = "BUILD_TUPLE"
BUILD_TUPLE_OPCODE = dis.opmap[BUILD_TUPLE_OP]
UNPACK_SEQUENCE_OP = "UNPACK_SEQUENCE"
UNPACK_SEQUENCE_OPCODE = dis.opmap[UNPACK_SEQUENCE_OP]
STORE_FAST_OP = "STORE_FAST"
STORE_FAST_OPCODE = dis.opmap[STORE_FAST_OP]
POP_TOP_OP = "POP_TOP"
POP_TOP_OPCODE = dis.opmap[POP_TOP_OP]
JUMP_ABS_OP = "JUMP_ABSOLUTE"
JUMP_ABS_OPCODE = dis.opmap[JUMP_ABS_OP]
LOAD_CONST_OP = "LOAD_CONST"
LOAD_CONST_OPCODE = dis.opmap[LOAD_CONST_OP]
NO_OP = "NOP"
NO_OPCODE = dis.opmap[NO_OP]


def tailrec(f: Callable) -> FunctionType:
    assert isinstance(f, FunctionType)
    instructions = list(dis.get_instructions(f))
    sig = signature(f)
    code = f.__code__
    new_instructions, new_consts, is_optimized = transform_tail_calls(
        instructions, sig, code
    )
    if not is_optimized:
        warn(
            f"No recursive tail calls found in function {code.co_name}; can't optimize"
        )
        return f

    new_bytecode = assemble(new_instructions)
    new_code = CodeType(
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        new_bytecode,
        new_consts,
        code.co_names,
        code.co_varnames,
        code.co_filename,
        code.co_name,
        code.co_firstlineno,
        code.co_lnotab,
        code.co_freevars,
        code.co_cellvars,
    )
    return FunctionType(
        code=new_code,
        globals=f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )


def recursive_call_span(
    instructions: Sequence[dis.Instruction], func_name: str
) -> Optional[int]:
    # pattern:
    # LOAD_GLOBAL(func_name)
    # [^STORE_*|RETURN_VALUE|YIELD_VALUE|YIELD_FROM_ITER]*
    # CALL_FUNCTION|CALL_FUNCTION_KW
    # RETURN_VALUE
    first = instructions[0]
    if first.opname == "LOAD_GLOBAL" and first.argval == func_name:
        seek_call, seek_return = 0, 1
        state = seek_call
        for j, next_ in enumerate(islice(instructions, 1, None), 1):
            if state == seek_call:
                if next_.opname in CALL_OPS:
                    state = seek_return
                elif next_.opname in SIDE_EFFECT_OPS:
                    return None
            elif state == seek_return:
                if next_.opname in RETURN_OPS:
                    return j
                else:
                    return None
            else:
                return None
        else:
            return None
    else:
        return None


def recursive_call_spans(
    instructions: Iterable[dis.Instruction], func_name: str
) -> Iterator[Tuple[int, int]]:
    i = 0
    instructions = deque(instructions)
    while instructions:
        j = recursive_call_span(instructions, func_name)
        if j is None:
            i += 1
            j = 1
        else:
            j += 1
            yield i, i + j
            i += j
        for _ in range(j):
            instructions.popleft()


def transform_tail_call(
    instructions: List[dis.Instruction],
    sig: Signature,
    code: CodeType,
    consts: Optional[Tuple] = None,
) -> Tuple[Sequence[dis.Instruction], Tuple, Dict[int, int]]:
    # remove function load at beginning of sequence, call and return from end, and append
    call = instructions[-2]
    nargs = call.argval
    if call.opname == CALL_FUNCTION_OP:
        new_instructions = instructions[0:-2]
        old_ix_to_new_ix = dict(
            zip(range(0, len(instructions) - 2), range(len(new_instructions)))
        )
        args = range(nargs)
        kw = {}
    elif call.opname == CALL_FUNCTION_KW_OP:
        new_instructions = instructions[0:-3]
        old_ix_to_new_ix = dict(
            zip(range(0, len(instructions) - 3), range(len(new_instructions)))
        )
        kw_names = instructions[-3].argval
        args = range(nargs - len(kw_names))
        kw = {k: i for i, k in enumerate(kw_names, nargs - len(kw_names))}
    else:
        raise ValueError(
            f"Unknown op where a function call was expected: {call.opname}"
        )

    # Leave this in in place of the function load as a jump target e.g. in case of an if block
    # where the return happens immediately
    new_instructions[0] = dis.Instruction(
        opname=NO_OP,
        opcode=NO_OPCODE,
        arg=0,
        argval=0,
        argrepr="",
        offset=new_instructions[0].offset,
        starts_line=new_instructions[0].starts_line,
        is_jump_target=new_instructions[0].is_jump_target,
    )

    new_consts = list(consts or code.co_consts)
    bound_args = sig.bind(*args, **kw)
    call_args = set(bound_args.arguments)
    arg_names = [k for k, v in sorted(bound_args.arguments.items(), key=itemgetter(1))]
    bound_args.apply_defaults()
    defaults = {k: v for k, v in bound_args.arguments.items() if k not in call_args}

    def next_offset():
        return new_instructions[-1].offset + BYTECODE_SIZE

    new_instructions.append(
        dis.Instruction(
            opname=BUILD_TUPLE_OP,
            opcode=BUILD_TUPLE_OPCODE,
            arg=nargs,
            argval=nargs,
            argrepr="",
            offset=next_offset(),
            starts_line=None,
            is_jump_target=True,
        )
    )
    new_instructions.append(
        dis.Instruction(
            UNPACK_SEQUENCE_OP,
            opcode=UNPACK_SEQUENCE_OPCODE,
            arg=nargs,
            argval=nargs,
            argrepr="",
            offset=next_offset(),
            starts_line=None,
            is_jump_target=True,
        )
    )

    for name in arg_names:
        new_instructions.append(
            dis.Instruction(
                opname=STORE_FAST_OP,
                opcode=STORE_FAST_OPCODE,
                arg=code.co_varnames.index(name),
                argval=name,
                argrepr=name,
                offset=next_offset(),
                starts_line=None,
                is_jump_target=True,
            )
        )

    for name, value in defaults.items():
        if value not in new_consts:
            new_consts.append(value)
        new_instructions.append(
            dis.Instruction(
                opname=LOAD_CONST_OP,
                opcode=LOAD_CONST_OPCODE,
                arg=new_consts.index(value),
                argval=value,
                argrepr=repr(value),
                offset=next_offset(),
                starts_line=None,
                is_jump_target=True,
            )
        )
        new_instructions.append(
            dis.Instruction(
                opname=STORE_FAST_OP,
                opcode=STORE_FAST_OPCODE,
                arg=code.co_varnames.index(name),
                argval=name,
                argrepr=name,
                offset=next_offset(),
                starts_line=None,
                is_jump_target=True,
            )
        )

    new_instructions.append(
        dis.Instruction(
            opname=JUMP_ABS_OP,
            opcode=JUMP_ABS_OPCODE,
            arg=0,
            argval=0,
            argrepr="0",
            offset=next_offset(),
            starts_line=None,
            is_jump_target=True,
        )
    )

    return new_instructions, tuple(new_consts), old_ix_to_new_ix


def _fix_offsets_and_jumps(
    instructions: Iterable[dis.Instruction],
    old_ix_to_new_ix: Dict[int, int],
    start: int = 0,
) -> Iterator[dis.Instruction]:
    new_jump_targets = set()
    new_jump_ixs = set()
    for offset, instr in zip(count(start, BYTECODE_SIZE), instructions):
        arg: Optional[int]
        argval: Optional[int]
        if "JUMP" in instr.opname and instr.arg is not None:
            # fix up jump targets
            old_target = instr.arg
            try:
                new_target = (
                    old_ix_to_new_ix[old_target // BYTECODE_SIZE] * BYTECODE_SIZE
                )
            except KeyError:
                raise ValueError(
                    f"Jump target {old_target} is missing after code reorganization"
                )

            if new_target >= 2 << 8:
                raise ValueError(
                    "Found new jump target exceeding 1-byte size after tail-call optimization; "
                    "this is not currently supported"
                )
            new_jump_ixs.add(new_target)
            arg = argval = new_target
        else:
            arg = instr.arg
            argval = instr.argval

        if instr.is_jump_target:
            new_jump_targets.add(offset)

        i = dis.Instruction(
            opname=instr.opname,
            opcode=instr.opcode,
            arg=arg,
            argval=argval,
            argrepr=instr.argrepr,
            offset=offset,
            starts_line=instr.starts_line,
            is_jump_target=instr.is_jump_target,
        )
        yield i

    if not new_jump_ixs.issubset(new_jump_targets):
        warn(
            "In tail call optimization, new jump offsets were generated that are not explicitly "
            f"marked as jump targets: {new_jump_ixs.difference(new_jump_targets)}"
        )


def transform_tail_calls(
    instructions: List[dis.Instruction], sig: Signature, code: CodeType
) -> Tuple[List[dis.Instruction], Tuple, bool]:
    recursive_spans = list(recursive_call_spans(instructions, code.co_name))
    if not recursive_spans:
        return instructions, code.co_consts, False

    new_code: List[dis.Instruction] = []
    old_ix_to_new_ix: Dict[int, int] = {}
    new_consts = code.co_consts
    prior_stop = 0

    for start, stop in recursive_spans:
        current_len = len(new_code)
        new_code.extend(instructions[prior_stop:start])
        old_ix_to_new_ix.update(
            (prior_stop + i, current_len + i) for i in range(start - prior_stop)
        )

        tail_call = instructions[start:stop]
        jump, new_consts, old_ix_to_new_ix_ = transform_tail_call(
            tail_call, sig, code, new_consts
        )
        current_len = len(new_code)
        new_code.extend(jump)
        old_ix_to_new_ix.update(
            (start + i, current_len + j) for i, j in old_ix_to_new_ix_.items()
        )
        prior_stop = stop

    current_len = len(new_code)
    new_code.extend(instructions[prior_stop:])
    old_ix_to_new_ix.update(
        (prior_stop + i, current_len + i) for i in range(len(instructions) - prior_stop)
    )
    first = new_code[0]
    new_code[0] = dis.Instruction(
        opname=first.opname,
        opcode=first.opcode,
        arg=first.arg,
        argval=first.argval,
        argrepr=first.argrepr,
        offset=first.offset,
        starts_line=first.starts_line,
        is_jump_target=True,
    )
    return list(_fix_offsets_and_jumps(new_code, old_ix_to_new_ix)), new_consts, True


def assemble(instructions: Iterable[dis.Instruction]) -> bytes:
    return bytes(chain.from_iterable((i.opcode, i.arg or 0) for i in instructions))
