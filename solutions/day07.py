from functools import partial
from itertools import chain
from textwrap import indent
from typing import IO, Callable, Iterable, Optional, Tuple, Union, cast

from .util import Leaf, Tree, dfs, identity, print_, set_verbose, tree_acc

DirPath = Tuple[str, ...]
File = Leaf[str, int]
Dir = Tree[str, int]


def file_to_str(file):
    return f"- {file.id} {file.data}"


def dir_to_str(dir_):
    header = f"{dir_.id.rstrip('/')}/ {dir_.data}"
    contents = sorted(dir_, key=lambda f: (isinstance(f, Tree), f.id))  # files first
    return "\n".join(
        chain([header], map(partial(indent, prefix="  "), map(str, contents)))
    )


AnyFile = Union[File, Dir]


# Parsing


def is_command(line: str) -> bool:
    return line.startswith("$ ")


def parse_command(line: str) -> Tuple[str, Optional[str]]:
    tokens = line[2:].strip().split(" ", 1)
    return (tokens[0], None) if len(tokens) == 1 else (tokens[0], tokens[1])


def parse_file_dir(line: str, parent: Optional[Dir] = None) -> Union[File, Dir]:
    size, name = line.strip().split(" ", 1)
    return Dir(name, 0, {}, parent) if size == DIR else File(name, int(size), parent)


# Commands and terminal constants
CD = "cd"
LS = "ls"
DIR = "dir"
PARENT = ".."

# Parsing states
READ_OUTPUT = 1
READ_COMMAND = 2


def parse_terminal(f: IO[str]) -> Dir:
    filesystem = current_tree = Dir("START", 0, {}, None)
    state = READ_COMMAND
    for line in map(str.rstrip, f):
        if state != READ_COMMAND and is_command(line):
            state = READ_COMMAND
        if state == READ_COMMAND:
            cmd, maybe_name = parse_command(line)
            if cmd == LS:
                state = READ_OUTPUT
            elif cmd == CD:
                state = READ_COMMAND
                assert maybe_name is not None
                if maybe_name == PARENT:
                    assert current_tree.parent is not None
                    current_tree = current_tree.parent
                else:
                    sub_tree = current_tree.children.get(maybe_name) or Dir(
                        maybe_name, 0, {}, current_tree
                    )
                    assert isinstance(sub_tree, Tree)
                    current_tree.children[maybe_name] = sub_tree
                    current_tree = sub_tree
        else:
            file_or_dir = parse_file_dir(line, current_tree)
            current_tree.children[file_or_dir.id] = file_or_dir

    assert len(filesystem.children) == 1
    root = next(iter(filesystem))
    assert isinstance(root, Tree)
    return root


def deletion_candidate(
    filesystem: Dir, capacity: int, required: int
) -> Tuple[DirPath, Dir]:
    total_size = filesystem.data
    max_allowable = capacity - required
    must_delete = total_size - max_allowable
    print_(
        f"Capacity {capacity}, {total_size} used, {capacity - total_size} free, "
        f"must delete {must_delete} to free {required}"
    )
    deletion_candidates = (
        (p, d)
        for p, d in dfs(filesystem)
        if isinstance(d, Tree) and d.data >= must_delete
    )
    return min(deletion_candidates, key=lambda t: t[1].data)


def run(
    input_: IO[str],
    max_size: int = 100000,
    capacity: int = 70000000,
    required: int = 30000000,
    part_2: bool = True,
    verbose: bool = True,
) -> str:
    set_verbose(verbose)
    filesystem = parse_terminal(input_)
    sized_filesystem = tree_acc(
        filesystem,
        f=cast(Callable[[int], int], identity),
        acc=cast(Callable[[Iterable[int]], int], sum),
        Tree=Dir,
        Leaf=File,
    )
    print_(sized_filesystem, end="\n\n")
    if part_2:
        delete_path, dir_ = deletion_candidate(sized_filesystem, capacity, required)
        return f"{'/'.join(name.rstrip('/') for name in delete_path)}, {dir_.data}"
    else:
        dir_sizes = (t.data for _, t in dfs(sized_filesystem) if isinstance(t, Tree))
        small_dir_sizes = filter(max_size.__ge__, dir_sizes)
        return str(sum(small_dir_sizes))


test_input = """$ cd /
$ ls
dir a
14848514 b.txt
8504156 c.dat
dir d
$ cd a
$ ls
dir e
29116 f.txt
2557 g.txt
62596 h.lst
$ cd e
$ ls
584 i.txt
$ cd ..
$ cd ..
$ cd d
$ ls
4060174 j.txt
8033020 d.log
5626152 d.ext
7214296 k.txt"""


def test():
    import io
    import re

    result = run(io.StringIO(test_input), part_2=False)
    a_size = 29116 + 2557 + 62596 + 584
    e_size = 584
    expected = a_size + e_size
    assert int(result) == expected, (result, expected)
    total_size = 48381165
    path = run(
        io.StringIO(test_input),
        part_2=True,
        capacity=total_size,
        required=e_size - e_size // 3,
    )
    expected_path = "/a/e"
    assert path == expected_path, (path, expected_path)
    path = run(
        io.StringIO(test_input),
        part_2=True,
        capacity=total_size,
        required=a_size - a_size // 4,
    )
    expected_path = "/a"
    assert path == expected_path, (path, expected_path)
    fs = tree_acc(
        parse_terminal(io.StringIO(test_input)), identity, sum, Tree=Dir, Leaf=File
    )
    expected_size = sum(map(int, re.findall(r"\d+", test_input)))
    file_size = sum(s.data for _, s in dfs(fs) if isinstance(s, Leaf))
    assert file_size == expected_size, (file_size, expected_size)
    assert fs.data == expected_size
