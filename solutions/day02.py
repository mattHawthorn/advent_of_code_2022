from enum import IntEnum
from itertools import chain, cycle, islice, starmap
from typing import IO, Dict, Tuple


class Outcome(IntEnum):
    loss = 0
    draw = 3
    win = 6


class Play(IntEnum):
    rock = 1
    paper = 2
    scissors = 3


WIN_RELATION: Dict[Play, Play] = dict(zip(Play, islice(chain(Play, Play), 1, 4)))
LOSE_RELATION: Dict[Play, Play] = {q: p for p, q in WIN_RELATION.items()}
PLAY_MAPPING = dict(zip("ABCXYZ", cycle(Play)))
OUTCOME_MAPPING = dict(zip("XYZ", cycle(Outcome)))


def outcome(their_play: Play, our_play: Play) -> Outcome:
    if our_play == their_play:
        return Outcome.draw
    elif WIN_RELATION[their_play] == our_play:
        return Outcome.win
    else:
        return Outcome.loss


# Problem 1


def parse_1(line: str) -> Tuple[Play, Play]:
    their_play, our_play = line.strip().split()
    return PLAY_MAPPING[their_play], PLAY_MAPPING[our_play]


def score_1(their_play: Play, our_play: Play) -> int:
    result = outcome(their_play, our_play)
    return result + our_play


# Problem 2


def parse_2(line: str) -> Tuple[Play, Outcome]:
    their_play, outcome_ = line.strip().split()
    return PLAY_MAPPING[their_play], OUTCOME_MAPPING[outcome_]


def score_2(their_play: Play, outcome_: Outcome):
    if outcome_ == Outcome.draw:
        our_play = their_play
    elif outcome_ == Outcome.win:
        our_play = WIN_RELATION[their_play]
    else:
        our_play = LOSE_RELATION[their_play]
    return outcome_ + our_play


def run(input_: IO[str], encode_play: bool = False):
    if encode_play:
        parse = parse_1
        score = score_1
    else:
        parse = parse_2  # type: ignore
        score = score_2  # type: ignore

    games = map(parse, input_)
    scores = starmap(score, games)
    return sum(scores)


def test():
    ...
