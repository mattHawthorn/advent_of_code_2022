from functools import partial, reduce
from itertools import accumulate, product, starmap, tee
from operator import lt, mul
from typing import IO, Callable, Iterable, Iterator, List, Optional, Tuple, cast

from .util import T

Terrain = List[List[int]]

multiply = cast(Callable[[Iterable[int]], int], partial(reduce, mul))


def parse_row(line: str):
    return list(map(int, line.rstrip()))


def parse_map(input_: IO[str]) -> Tuple[Terrain, int, int]:
    terrain = list(map(parse_row, input_))
    row_len = max(map(len, terrain))
    assert all(len(row) == row_len for row in terrain)
    return terrain, len(terrain), row_len


def visibility(row: Iterable[int]) -> Iterator[bool]:
    row1, row2 = tee(row, 2)
    max_values = accumulate(row1, max, initial=-1)
    return map(lt, max_values, row2)


def viewing_distance(
    terrain: Terrain,
    row_ix: int,
    col_ix: int,
    row_step: int,
    col_step: int,
    acc: int = 0,
    initial: Optional[int] = None,
) -> int:
    if (
        row_ix < 0
        or col_ix < 0
        or row_ix >= len(terrain)
        or col_ix >= len(terrain[row_ix])
    ):
        return acc - 1
    else:
        row = terrain[row_ix]
        value = row[col_ix]
        if value == initial:
            return acc
        elif initial is None or value < initial:
            initial = value if initial is None else initial
            return viewing_distance(
                terrain,
                row_ix + row_step,
                col_ix + col_step,
                row_step,
                col_step,
                acc + 1,
                initial,
            )
        else:  # value > initial:
            return acc - 1


def viewing_distances(
    terrain: Terrain, row_ix: int, col_ix: int
) -> Tuple[int, int, int, int]:
    vd = partial(viewing_distance, terrain, row_ix, col_ix)
    return vd(0, 1), vd(0, -1), vd(1, 0), vd(-1, 0)


def reverse_indices(size: int):
    return range(size - 1, -1, -1)


def enumerate_reversed(it: Iterable[T], size) -> Iterator[Tuple[int, T]]:
    return zip(reverse_indices(size), it)


def run(input_: IO[str], part_2: bool = True) -> int:
    terrain, num_rows, num_cols = parse_map(input_)
    if part_2:
        distances = starmap(
            partial(viewing_distances, terrain),
            product(range(num_rows), range(num_cols)),
        )
        scenic_scores = map(multiply, distances)
        return max(scenic_scores)
    else:
        visibility_from_left = map(visibility, terrain)
        total_visibility = {
            (i, j): v
            for i, row in enumerate(visibility_from_left)
            for j, v in enumerate(row)
        }
        visibility_from_right = map(visibility, map(reversed, terrain))  # type: ignore
        total_visibility = {
            (i, j): total_visibility[i, j] or v
            for i, row in enumerate(visibility_from_right)
            for j, v in enumerate_reversed(row, num_cols)
        }
        visibility_from_above = map(visibility, zip(*terrain))
        total_visibility = {
            (i, j): total_visibility[i, j] or v
            for j, col in enumerate(visibility_from_above)
            for i, v in enumerate(col)
        }
        visibility_from_below = map(visibility, map(reversed, zip(*terrain)))  # type: ignore
        total_visibility = {
            (i, j): total_visibility[i, j] or v
            for j, col in enumerate(visibility_from_below)
            for i, v in enumerate_reversed(col, num_rows)
        }
        return sum(total_visibility.values())


test_input = """
30373
25512
65332
33599
35390
""".strip()


def test():
    import io

    assert list(enumerate_reversed([1, 2, 3], 3)) == [(2, 1), (1, 2), (0, 3)]
    result = run(io.StringIO(test_input), part_2=False)
    expected = 22
    assert result == expected, (result, expected)
    table, _, _ = parse_map(io.StringIO(test_input))
    actual = viewing_distances(table, 1, 1)
    assert actual == (1, 1, 1, 1), actual
    actual = viewing_distances(table, 1, 2)
    assert actual == (2, 1, 2, 1), actual
    result = run(io.StringIO(test_input), part_2=True)
    assert result == 9, result
