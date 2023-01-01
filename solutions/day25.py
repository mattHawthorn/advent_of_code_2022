from operator import mul
from typing import IO, Iterator

DIGIT_TO_VALUE = {"=": -2, "-": -1, "0": 0, "1": 1, "2": 2}
VALUE_TO_DIGIT = {v: k for k, v in DIGIT_TO_VALUE.items()}
MAX_VALUE = max(VALUE_TO_DIGIT.keys())
PLACES = [5**i for i in range(25)]


def parse(input_: IO[str]) -> Iterator[int]:
    return map(decode_snafu, map(str.strip, input_))


def decode_snafu(s: str) -> int:
    digits = map(DIGIT_TO_VALUE.__getitem__, reversed(s))
    return sum(map(mul, digits, PLACES))


def to_base(base: int, max_digit_value: int, n: int) -> Iterator[int]:
    if n > 0:
        quot, rem = divmod(n, base)
        quot_, rem_ = (quot, rem) if rem <= max_digit_value else (quot + 1, rem - base)
        yield rem_
        yield from to_base(base, max_digit_value, quot_)


def encode_snafu(n: int) -> str:
    snafu_digits = to_base(5, MAX_VALUE, n)
    symbols = list(map(VALUE_TO_DIGIT.__getitem__, snafu_digits))
    return "".join(reversed(symbols))


def run(input_: IO[str]) -> str:
    snafus = parse(input_)
    total = sum(snafus)
    snafu = encode_snafu(total)
    return snafu


test_input = """
1=-0-2
12111
2=0=
21
2=01
111
20012
112
1=-1=
1-12
12
1=
122""".strip()


def test():
    import io

    result = run(io.StringIO(test_input))
    expected = "2=-1=0"
    assert result == expected, (expected, result)
