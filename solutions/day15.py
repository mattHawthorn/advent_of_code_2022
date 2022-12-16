import re
from functools import partial
from itertools import chain, combinations, starmap
from typing import IO, Iterable, List, Optional, Tuple

from .util import GridCoordinates, reduce_while

Interval = Tuple[int, int]
# slope, y intercept
Line = Tuple[int, int]

line_re = re.compile(
    r"Sensor at x=(?P<sensor_x>-?\d+), y=(?P<sensor_y>-?\d+): "
    r"closest beacon is at x=(?P<beacon_x>-?\d+), y=(?P<beacon_y>-?\d+)"
)


def parse_sensor(line: str) -> Tuple[GridCoordinates, GridCoordinates]:
    match = line_re.match(line)
    assert match is not None
    fields = match.groupdict()
    return (int(fields["sensor_x"]), int(fields["sensor_y"])), (
        int(fields["beacon_x"]),
        int(fields["beacon_y"]),
    )


def parse_sensors(input_: IO[str]) -> List[Tuple[GridCoordinates, GridCoordinates]]:
    return list(map(parse_sensor, input_))


def manhattan_distance(coord1: GridCoordinates, coord2: GridCoordinates) -> int:
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def sensor_range_at_row(
    y: int, sensor_coord: GridCoordinates, beacon_coord: GridCoordinates
) -> Optional[Interval]:
    dist = manhattan_distance(sensor_coord, beacon_coord)
    sensor_x, sensor_y = sensor_coord
    y_diff = abs(y - sensor_y)
    x_max = dist - y_diff
    return sensor_x - x_max, sensor_x + x_max


def is_overlapping(interval1: Interval, interval2: Interval) -> bool:
    start1, end1 = interval1
    start2, end2 = interval2
    return (
        interval_contains(interval1, start2)
        or interval_contains(interval2, start1)
        or interval_contains(interval1, end2)
        or interval_contains(interval2, end1)
    )


def interval_contains(interval: Interval, value: int) -> bool:
    start, end = interval
    return start <= value <= end


def coalesce_intervals(interval1: Interval, interval2: Interval) -> Interval:
    start1, end1 = interval1
    start2, end2 = interval2
    return min(start1, start2), max(end1, end2)


def interval_size(interval: Interval) -> int:
    return interval[1] + 1 - interval[0]


def interval_graph_components(intervals: Iterable[Interval]) -> Iterable[Interval]:
    """Components of an inteval graph with overlap as the edge predicate, as min, max intervals.
    Correctness is ensured via a preliminary sort so this is _not_ a lazy streaming operation"""
    return reduce_while(is_overlapping, coalesce_intervals, sorted(intervals))


def bounding_lines(sensor: GridCoordinates, beacon: GridCoordinates) -> List[Line]:
    """Lines bounding the region within the manhattan distance from `sensor` to `beacon` of `sensor`
    (non-inclusive)"""
    dist = manhattan_distance(sensor, beacon)
    x, y = sensor
    lo = y - dist - 1
    hi = y + dist + 1
    return [(1, hi - x), (1, lo - x), (-1, lo + x), (-1, hi + x)]


def intersection_point(line1: Line, line2: Line) -> Optional[GridCoordinates]:
    slope1, y1 = line1
    slope2, y2 = line2
    if slope1 == slope2:
        return None
    else:
        x = (y2 - y1) // (slope1 - slope2)
        return x, x * slope1 + y1


def outside_of_neighborhood(
    point: GridCoordinates,
    sensor: GridCoordinates,
    beacon: GridCoordinates,
) -> bool:
    dist_to_beacon = manhattan_distance(sensor, beacon)
    dist_to_point = manhattan_distance(sensor, point)
    return dist_to_point > dist_to_beacon


def inside_of_box(xmin: int, xmax: int, ymin: int, ymax: int, point: GridCoordinates) -> bool:
    x, y = point
    return xmin <= x <= xmax and ymin <= y <= ymax


def run(
    input_: IO[str],
    row: int = 2000000,
    xmin: int = 0,
    ymin: int = 0,
    xmax: int = 4000000,
    ymax: int = 4000000,
    part_2: bool = True,
) -> int:
    sensors = parse_sensors(input_)
    beacons = {beacon for _, beacon in sensors}
    if part_2:
        all_lines = chain.from_iterable(starmap(bounding_lines, sensors))
        intersections_ = starmap(intersection_point, combinations(all_lines, 2))
        intersections = (point for point in intersections_ if point is not None)
        candidate_intersections = list(
            filter(partial(inside_of_box, xmin, xmax, ymin, ymax), intersections)
        )
        undetectable_intersections = set(
            point
            for point in candidate_intersections
            if all(starmap(partial(outside_of_neighborhood, point), sensors))
        )
        assert len(undetectable_intersections) == 1
        x, y = next(iter(undetectable_intersections))
        return x * 4000000 + y
    else:
        intervals_ = (sensor_range_at_row(row, sensor, beacon) for sensor, beacon in sensors)
        intervals = (i for i in intervals_ if i is not None)
        components = list(interval_graph_components(intervals))
        sizes = map(interval_size, components)
        beacons_in_row = [
            beacon
            for beacon in beacons
            if beacon[1] == row and any(interval_contains(i, beacon[0]) for i in components)
        ]
        return sum(sizes) - len(beacons_in_row)


test_input = """
Sensor at x=2, y=18: closest beacon is at x=-2, y=15
Sensor at x=9, y=16: closest beacon is at x=10, y=16
Sensor at x=13, y=2: closest beacon is at x=15, y=3
Sensor at x=12, y=14: closest beacon is at x=10, y=16
Sensor at x=10, y=20: closest beacon is at x=10, y=16
Sensor at x=14, y=17: closest beacon is at x=10, y=16
Sensor at x=8, y=7: closest beacon is at x=2, y=10
Sensor at x=2, y=0: closest beacon is at x=2, y=10
Sensor at x=0, y=11: closest beacon is at x=2, y=10
Sensor at x=20, y=14: closest beacon is at x=25, y=17
Sensor at x=17, y=20: closest beacon is at x=21, y=22
Sensor at x=16, y=7: closest beacon is at x=15, y=3
Sensor at x=14, y=3: closest beacon is at x=15, y=3
Sensor at x=20, y=1: closest beacon is at x=15, y=3""".strip()


def test():
    import io

    actual = run(io.StringIO(test_input), row=10, part_2=False)
    expected = 26
    assert actual == expected, (actual, expected)
    actual = run(io.StringIO(test_input), xmax=20, ymax=20, part_2=True)
    expected = 56000011
    assert actual == expected
