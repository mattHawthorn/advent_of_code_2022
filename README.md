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
cd solutions

# docs
./day01.py --help

# tests
./day01.py test

# solution
./day01.py

# solution with particular input read from stdin - replace the input file with any other
cat day01.txt | python day01.py
```
