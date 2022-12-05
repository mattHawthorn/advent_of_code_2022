#! /usr/bin/env python
import json
import sys
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import IO, Protocol, TypeVar, Union

from bourbaki.application.cli import CommandLineInterface, cli_spec  # type: ignore
from bourbaki.application.typed_io.cli_parse import cli_parser  # type: ignore

INPUT_DIR = Path("inputs/")

Solution = TypeVar("Solution", covariant=True)
Param = Union[int, float, bool, str]


class Problem(Protocol[Solution]):
    def run(self, input_: IO[str], **args: Param) -> Solution:
        ...

    def test(self):
        ...


@cli_parser.register(Param, as_const=True, derive_nargs=True)
def parse_param(s: str):
    return json.loads(s)


def print_solution(solution):
    print(solution)


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


cli = CommandLineInterface(
    prog="main",
    require_options=False,
    require_subcommand=True,
    implicit_flags=True,
    use_verbose_flag=True,
)


@cli.definition
class AOC2022:
    """Run and test Matt Hawthorn's solutions to the 2022 Advent of Code problems"""

    @cli_spec.output_handler(print_solution)
    def run(self, day: int, **args: Param):
        """Run the solution to a particular day's problem

        :param day: the day number of the problem to solve (1-25)
        :param args: keyword arguments to pass to the problem solution in case it is parameterized.
          Run the `info` command for the problem in question to see its parameters.
        """
        problem = import_problem(day)
        input_ = get_input(day)
        solution = problem.run(input_, **args)
        return solution

    def test(self, day: int):
        """Run unit tests for functions used in the solution to a particular day's problem

        :param day: the day number of the problem to run tests for (1-25)
        """
        problem = import_problem(day)
        problem.test()
        print(f"Tests pass for day {day}!")

    def info(self, day: int):
        """Print the doc string for a particular day's solution, providing some details about methodology

        :param day: the day number of the problem to run tests for (1-25)
        """
        problem = import_problem(day)
        print(f"Day {day} problem info:")
        if problem.__doc__:
            print(problem.__doc__, end="\n\n")
        print("Signature:")
        print(signature(problem.run))


if __name__ == "__main__":
    cli.run()
