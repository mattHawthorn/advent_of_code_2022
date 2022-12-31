import re
from functools import partial, reduce
from itertools import chain, filterfalse
from operator import add, itemgetter, mul, sub
from typing import IO, Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple

from .tailrec import tailrec
from .util import INF, branch_and_bound, iterate, set_verbose, take_until

Material = str
RobotType = str


class Counts(dict):
    def _op(self, other: "Counts", op: Callable[[int, int], int]) -> "Counts":
        return Counts(
            (k, op(self.get(k, 0), other.get(k, 0)))
            for k in chain(self, filterfalse(self.__contains__, other))
        )

    def __abs__(self):
        return sum(max(v, 0) for v in self.values())

    def __add__(self, other: "Counts") -> "Counts":
        return self._op(other, add)

    def __sub__(self, other: "Counts") -> "Counts":
        return self._op(other, sub)

    def __ge__(self, other: "Counts"):
        return all(self.get(k, 0) >= v for k, v in other.items())

    def __mul__(self, other: int) -> "Counts":
        return MaterialState((k, mul(v, other)) for k, v in self.items())

    def __truediv__(self, other: "Counts") -> int:
        def div(x: int, y: int):
            if y == 0 and x > 0:
                return INF
            else:
                m, n = divmod(x, y)
                return m + (n > 0)

        return max(self._op(other, div).values())


Costs = Counts
RobotState = Counts
MaterialState = Counts
BluePrint = Dict[RobotType, Costs]


class State(NamedTuple):
    budget: int
    blueprint: BluePrint
    materials: MaterialState
    robots: RobotState
    steps: List[Optional[RobotType]]

    @property
    def prior(self) -> Optional["State"]:
        if self.steps:
            last = self.steps[-1]
            if last is None:
                materials = self.materials - self.robots
                robots = self.robots
            else:
                robots = Counts(self.robots)
                robots[last] -= 1
                materials = self.materials - robots + self.blueprint[last]
            return State(self.budget + 1, self.blueprint, materials, robots, self.steps[:-1])
        else:
            return None


# Parsing

ORE, CLAY, OBS, GEO = ("ore", "clay", "obsidian", "geode")
ROBOT_TYPES: List[Material] = [ORE, CLAY, OBS, GEO]
SOURCE_MATERIALS: List[Material] = [ORE, CLAY, OBS]

cost_re = re.compile(r"(?P<qty>\d+) (?P<material>{})".format("|".join(SOURCE_MATERIALS)))
robot_re = re.compile(
    r"Each (?P<robot_type>{}) robot costs (?P<costs>[^.]*).".format("|".join(ROBOT_TYPES))
)
blueprint_re = re.compile(r"Blueprint (?P<id>\d+): (?P<robots>.*)")


def parse_costs(s: str) -> Costs:
    return MaterialState(
        {match.group("material"): int(match.group("qty")) for match in cost_re.finditer(s)}
    )


def parse_robots(s: str) -> BluePrint:
    return MaterialState(
        {
            match.group("robot_type"): parse_costs(match.group("costs"))
            for match in robot_re.finditer(s)
        }
    )


def parse_blueprint(s: str) -> Tuple[int, BluePrint]:
    match = blueprint_re.match(s)
    assert match is not None
    return int(match.group("id")), parse_robots(match.group("robots"))


# State transitions


def can_build(state: State, robot_type: RobotType) -> bool:
    return state.materials >= state.blueprint[robot_type]


def should_build(state: State, robot_type: RobotType) -> bool:
    # we can only build one robot at a time, so never accrue more robots than needed to build
    # one of each    kind in each turn
    return robot_type == GEO or state.robots.get(robot_type, 0) < max(
        bp.get(robot_type, 0) for bp in state.blueprint.values()
    )


def build_options(state: State) -> List[RobotType]:
    robot_types = filter(
        partial(should_build, state), filter(partial(can_build, state), state.blueprint)
    )
    if state.steps and state.steps[-1] is None:
        return list(filterfalse(partial(can_build, state.prior), robot_types))
    else:
        return list(robot_types)


def build(state: State, robot_type: RobotType) -> State:
    new_materials = state.materials - state.blueprint[robot_type] + state.robots
    new_robots = RobotState(state.robots)
    new_robots[robot_type] = state.robots.get(robot_type, 0) + 1
    return State(
        state.budget - 1, state.blueprint, new_materials, new_robots, state.steps + [robot_type]
    )


def harvest(state: State, steps: Optional[int] = None) -> State:
    if steps is None:
        robot_type, steps = min(
            ((rt, (cost - state.materials) / state.robots) for rt, cost in state.blueprint.items()),
            key=itemgetter(1),
        )
        steps_ = max(min(steps, state.budget), 1)
    else:
        steps_ = steps

    new_materials = state.materials + state.robots * steps_
    new_budget = state.budget - steps_
    return State(
        new_budget, state.blueprint, new_materials, state.robots, state.steps + [None] * steps_
    )


def is_final(state: State) -> bool:
    return state.budget == 0


def candidate_solutions(state: State) -> Iterable[State]:
    buildable_bots = list(build_options(state))
    # if every kind of robot can be built, always build one - there is nothing to gain by delaying
    can_build_all = len(buildable_bots) >= len(state.blueprint)
    maybe_harvest = [] if can_build_all else [harvest(state)]
    return chain(maybe_harvest, map(partial(build, state), buildable_bots))


# Objective functions


def score(state: State) -> int:
    return -state.materials.get(GEO, 0)


def score_lower_bound(state: State) -> int:
    steps_remaining = state.budget
    total_production = state.robots.get(GEO, 0) * steps_remaining + state.materials.get(GEO, 0)
    steps_to_next_geode_bot = 0 if can_build(state, GEO) else 1
    max_geode_bots = steps_remaining - steps_to_next_geode_bot
    hypothetical_production = max_geode_bots * (max_geode_bots - 1) // 2
    return -(total_production + hypothetical_production)


# Solutions


@tailrec
def heuristic_optimal_step(
    state: State, target_resource: Material = GEO, verbose: bool = False
) -> State:
    # choose the action (among those available) that will minimize the distance between
    # the robot distribution and the resources required to build a geode-producing robot
    requirements = state.blueprint[target_resource] - state.materials
    build_candidates = [(material, qty) for material, qty in requirements.items() if qty > 0]
    if not build_candidates:  # can build target_resource
        return build(state, target_resource)
    else:
        # harvest or build something else; choose the one with the greatest total cost to have
        # enough to build `target_resource`
        next_target_resource, _ = max(
            build_candidates,
            key=lambda t: t[1] * abs(state.blueprint[t[0]] - state.robots),
        )
        if next_target_resource == target_resource:  # need x to get x; harvest
            return harvest(state)
        else:
            return heuristic_optimal_step(state, next_target_resource, verbose=verbose)


def heuristic_solve(state: State, stop_fn: Callable[[State], bool], verbose: bool = False) -> State:
    states = iterate(partial(heuristic_optimal_step, verbose=verbose), state)
    valid_states = take_until(stop_fn, states)
    return list(valid_states)[-1]


def optimal_solution(state: State):
    heuristic_solution = heuristic_solve(state, is_final, verbose=False)
    return branch_and_bound(
        initial=[state],
        heuristic_solution=heuristic_solution,
        candidate_fn=candidate_solutions,
        stop_fn=is_final,
        objective_fn=score,
        lower_bound_fn=score_lower_bound,
    )


def run(input_: IO[str], part_2: bool = True, verbose: bool = False) -> int:
    set_verbose(verbose)
    blueprints = list(map(parse_blueprint, input_))
    steps = 32 if part_2 else 24
    states = (
        (id_, State(steps, blueprint, MaterialState(), RobotState({ORE: 1}), []))
        for id_, blueprint in blueprints[: 3 if part_2 else len(blueprints)]
    )
    if part_2:
        return reduce(mul, (-score(optimal_solution(state)) for id_, state in states))
    else:
        return sum(id_ * -score(optimal_solution(state)) for id_, state in states)


test_input = (
    """
Blueprint 1:
  Each ore robot costs 4 ore.
  Each clay robot costs 2 ore.
  Each obsidian robot costs 3 ore and 14 clay.
  Each geode robot costs 2 ore and 7 obsidian.

Blueprint 2:
  Each ore robot costs 2 ore.
  Each clay robot costs 3 ore.
  Each obsidian robot costs 3 ore and 8 clay.
  Each geode robot costs 3 ore and 12 obsidian.
""".replace(
        "\n  ", " "
    )
    .replace("\n\n", "\n")
    .strip()
)


def test():
    import io

    state = State(
        budget=5,
        blueprint={GEO: Costs({ORE: 2, OBS: 4}), ORE: Costs(), CLAY: Costs(), OBS: Costs()},
        materials=MaterialState({ORE: 1, CLAY: 2, OBS: 2, GEO: 2}),
        robots=RobotState({ORE: 1, CLAY: 1, GEO: 1, OBS: 1}),
        steps=[],
    )
    test_prior(state)
    test_lower_bound(state)
    test_optimization()
    quality_level = run(io.StringIO(test_input), part_2=False)
    expected_quality_level = 33
    assert quality_level == expected_quality_level, (quality_level, expected_quality_level)

    quality_level = run(io.StringIO(test_input), part_2=True)
    expected_quality_level = 56 * 62
    assert quality_level == expected_quality_level, (quality_level, expected_quality_level)


def test_prior(state: State):
    state1_1 = harvest(state, 1)
    state1_2 = build(state1_1, GEO)
    state2_1 = build(state, GEO)
    state2_2 = harvest(state, 1)
    for prior, next_ in (
        (state, state1_1),
        (state1_1, state1_2),
        (state, state2_1),
        (state, state2_2),
    ):
        assert next_.prior == prior, (next_.prior, prior)


def test_lower_bound(state: State):
    expected_lower_bound = -(5 * 1 + 2 + 4 * (4 - 1) // 2)
    actual_lower_bound = score_lower_bound(state)
    # TODO: put this back
    assert actual_lower_bound == expected_lower_bound, (expected_lower_bound, actual_lower_bound)


def test_optimization():
    steps = [
        None,
        None,
        ORE,
        None,
        ORE,
        CLAY,
        CLAY,
        CLAY,
        CLAY,
        CLAY,
        OBS,
        CLAY,
        OBS,
        OBS,
        OBS,
        CLAY,
        OBS,
        GEO,
        OBS,
        GEO,
        OBS,
        GEO,
        OBS,
        GEO,
    ]
    state_ = State(
        24,
        {
            ORE: Costs({ORE: 2}),
            CLAY: Costs({ORE: 3}),
            OBS: Costs({ORE: 3, CLAY: 8}),
            GEO: Costs({ORE: 3, OBS: 12}),
        },
        MaterialState(),
        RobotState({ORE: 1}),
        [],
    )
    final_score = -12
    assert len(steps) == 24, len(steps)
    for i, to_build in enumerate(steps):
        lb = score_lower_bound(state_)
        assert lb <= final_score, lb
        if to_build is None:
            harvested = harvest(state_)
            assert harvested == next(iter(candidate_solutions(state_))), (
                "harvest",
                i,
                state_.materials,
            )
            state_ = harvest(state_, 1)
        else:
            options = build_options(state_)
            assert to_build in options, (to_build, options, state_.materials, i)
            state_ = build(state_, to_build)
