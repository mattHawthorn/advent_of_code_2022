# Advent of Code 2022 daily solutions

In these solutions I strive for elegance, generality, and efficiency, simultaneously if possible.
Time-to-solution was not a metric I measured myself by, preferring aesthetics above all.

## Initialize the environment

From repo root, run:

```shell
pipenv install --dev
```

## Run a solution

```shell
# help on the commands documented below
./main --help

# run solution to day 1 problem
./main run 1

# run tests for day 12 problem
./main test 12

# run day 24 solution with particular input read from stdin as opposed to the default input file
cat my_input.txt | ./main run 24
```
