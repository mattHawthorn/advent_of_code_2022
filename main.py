#! /usr/bin/env python
import sys
from importlib import import_module
from pathlib import Path
from typing import IO, Protocol

from bourbaki.application.cli import CommandLineInterface  # type: ignore

INPUT_DIR = Path("inputs")

cli = CommandLineInterface(
    prog="main",
    require_options=False,
    require_subcommand=True,
    implicit_flags=True,
    use_verbose_flag=True,
)


class Problem(Protocol):
    def run(self, input_: IO[str]):
        ...

    def test(self):
        ...


def problem_name(problem: int) -> str:
    assert 1 <= problem <= 25, "problem number must be between 1 and 25, inclusive"
    return f"day{str(problem).zfill(2)}"


def import_problem(problem: int) -> Problem:
    name = problem_name(problem)
    return import_module(f"solutions.{name}")


def get_input(day: int) -> IO[str]:
    name = problem_name(day)
    filename = INPUT_DIR / (name + ".txt")
    return open(filename) if sys.stdin.isatty() else sys.stdin


@cli.definition
class AOC2022:
    """Run and test Matt Hawthorn's solutions to the 2022 Advent of Code problems"""

    def run(self, day: int):
        """Run the solution to a particular day's problem

        :param day: the day number of the problem to solve (1-25)
        """
        problem = import_problem(day)
        input_ = get_input(day)
        problem.run(input_)

    def test(self, day: int):
        """Run unit tests for functions used in the solution to a particular day's problem

        :param day: the day number of the problem to run tests for (1-25)
        """
        problem = import_problem(day)
        problem.test()

    def info(self, day: int):
        """Print the doc string for a particular day's solution, providing some details about methodology

        :param day: the day number of the problem to run tests for (1-25)
        """
        problem = import_problem(day)
        print(problem.__doc__)


if __name__ == "__main__":
    cli.run()
