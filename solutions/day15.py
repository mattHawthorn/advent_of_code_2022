import re
from typing import IO, Iterable, List, Optional, Tuple

from .util import GridCoordinates, reduce_while

Interval = Tuple[int, int]

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
    """Comonents of an inteval graph with overlap as the edge predicate, as min, max intervals"""
    return reduce_while(is_overlapping, coalesce_intervals, sorted(intervals))


def run(input_: IO[str], row: int = 2000000, part_2: bool = True) -> int:
    sensors = parse_sensors(input_)
    beacons = {beacon for _, beacon in sensors}
    if part_2:
        ...
        return 0
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
