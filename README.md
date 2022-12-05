# Advent of Code 2022 daily solutions

In these solutions I strive for elegance, generality, and efficiency, simultaneously if possible.
Time-to-solution was not a metric I measured myself by, preferring aesthetics above all.

## Initialize the environment

From repo root, run:

```shell
pipenv install --dev
```

## Run a solution

(activate environment first with `pipenv shell`)

```shell
# help on the commands documented below
./main.py --help

# run solution to day 1 problem
./main.py run 1

# run tests for day 1 problem
./main.py test 1

# run day 1 solution with particular input read from stdin as opposed to the default input file
cat my_input.txt | python main.py run 1
```
