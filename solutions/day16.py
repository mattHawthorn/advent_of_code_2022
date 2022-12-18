import re
from functools import partial, reduce
from itertools import filterfalse, product, repeat
from operator import itemgetter
from typing import IO, FrozenSet, List, MutableMapping, NamedTuple, Tuple

from .util import WeightedDiGraph, compose, floyd_warshall, induced_subgraph, non_overlapping

line_re = re.compile(
    r"Valve (?P<id>\w+) [\w\s]+ rate=(?P<rate>\d+); [\w\s]+ valves? (?P<neighbors>\w+(?:, \w+)*)"
)

ValveId = str


class Valve(NamedTuple):
    id: ValveId
    rate: int


ValveState = FrozenSet[Valve]


def parse_line(line: str) -> Tuple[Valve, List[ValveId]]:
    match = line_re.match(line)
    assert match is not None
    fields = match.groupdict()
    return Valve(fields["id"], int(fields["rate"])), fields["neighbors"].split(", ")


def parse_graph(input_: IO[str]) -> WeightedDiGraph[Valve]:
    graph_ = dict(map(parse_line, input_))
    id_to_node = {node.id: node for node in graph_}
    return {node: {id_to_node[n]: 1 for n in neighbors} for node, neighbors in graph_.items()}


def total_flow(state: ValveState) -> int:
    return sum(v.rate for v in state)


def optimal_flow(
    graph: WeightedDiGraph[Valve],
    node: Valve,
    budget: int,
    best: MutableMapping[ValveState, int],
    state: ValveState = frozenset(),
    flow: int = 0,
) -> MutableMapping[ValveState, int]:
    if budget > 0:
        best[state] = max(best.get(state, 0), flow)
        is_visited = compose(itemgetter(0), state.__contains__)
        candidate_neighbors = filterfalse(is_visited, graph[node].items())

        def inner(
            graph,
            budget,
            flow,
            state,
            best: MutableMapping[ValveState, int],
            nbr_cost: Tuple[Valve, int],
        ):
            neighbor, cost = nbr_cost
            new_budget = budget - cost - 1
            new_flow = flow + new_budget * neighbor.rate
            new_state = state.union([neighbor])
            return optimal_flow(graph, neighbor, new_budget, best, new_state, new_flow)

        return reduce(partial(inner, graph, budget, flow, state), candidate_neighbors, best)
    else:
        return best


def run(input_: IO[str], start: str = "AA", part_2: bool = True) -> int:
    graph = parse_graph(input_)
    nonzero_flow_nodes = {n for n in graph if n.rate > 0 or n.id == start}
    compressed_graph = induced_subgraph(floyd_warshall(graph), nonzero_flow_nodes)
    budget, n = (26, 2) if part_2 else (30, 1)
    best_states = optimal_flow(compressed_graph, Valve("AA", 0), budget, {})
    candidate_states = list(filter(non_overlapping, product(*repeat(best_states, n))))
    score = compose(partial(map, best_states.__getitem__), sum)  # type: ignore
    return max(map(score, candidate_states))


test_input = """
Valve AA has flow rate=0; tunnels lead to valves DD, II, BB
Valve BB has flow rate=13; tunnels lead to valves CC, AA
Valve CC has flow rate=2; tunnels lead to valves DD, BB
Valve DD has flow rate=20; tunnels lead to valves CC, AA, EE
Valve EE has flow rate=3; tunnels lead to valves FF, DD
Valve FF has flow rate=0; tunnels lead to valves EE, GG
Valve GG has flow rate=0; tunnels lead to valves FF, HH
Valve HH has flow rate=22; tunnel leads to valve GG
Valve II has flow rate=0; tunnels lead to valves AA, JJ
Valve JJ has flow rate=21; tunnel leads to valve II""".strip()


def test():
    import io

    graph = parse_graph(io.StringIO(test_input))
    nonzero_flow_nodes = {n for n in graph if n.rate > 0 or n.id == "AA"}
    compressed_graph = induced_subgraph(floyd_warshall(graph), nonzero_flow_nodes)
    best_states = optimal_flow(compressed_graph, Valve("AA", 0), 30, {})
    max_flow = max(best_states.values())
    expected_max_flow = 1651
    assert max_flow == expected_max_flow, (max_flow, expected_max_flow)
