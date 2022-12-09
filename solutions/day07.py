from functools import partial
from itertools import chain
from textwrap import indent
from typing import IO, Callable, NamedTuple, Optional, Tuple, cast

from .util import Tree, dfs, identity, print_, set_verbose, tree_acc


class Stat(NamedTuple):
    is_dir: bool
    size: int = 0


File = Tree[str, Stat]
FilePath = Tuple[str, ...]


def dir_to_str(dir_: File):
    if dir_.data.is_dir:
        header = f"{dir_.id.rstrip('/')}/"
    else:
        header = f"{dir_.id} {dir_.data.size}"
    contents = sorted(dir_, key=lambda f: (f.data.is_dir, f.id))  # files first
    return "\n".join(
        chain([header], map(partial(indent, prefix="  "), map(dir_to_str, contents)))
    )


# Parsing


def is_command(line: str) -> bool:
    return line.startswith("$ ")


def parse_command(line: str) -> Tuple[str, Optional[str]]:
    tokens = line[2:].strip().split(" ", 1)
    return (tokens[0], None) if len(tokens) == 1 else (tokens[0], tokens[1])


def parse_file_dir(line: str, parent: Optional[File] = None) -> File:
    token1, name = line.strip().split(" ", 1)
    is_dir = token1 == DIR
    size = 0 if is_dir else int(token1)
    return File(name, Stat(is_dir, size), {}, parent)


# Commands and terminal constants
CD = "cd"
LS = "ls"
DIR = "dir"
PARENT = ".."

# Parsing states
READ_OUTPUT = 1
READ_COMMAND = 2


def parse_terminal(f: IO[str]) -> File:
    filesystem = current_tree = File("START", Stat(False, 0), {}, None)
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
                    sub_tree = current_tree.children.get(maybe_name) or File(
                        maybe_name, Stat(True, 0), {}, current_tree
                    )
                    assert sub_tree.data.is_dir
                    current_tree.children[maybe_name] = sub_tree
                    current_tree = sub_tree
        else:
            file_or_dir = parse_file_dir(line, current_tree)
            current_tree.children[file_or_dir.id] = file_or_dir

    assert len(filesystem.children) == 1
    root = next(iter(filesystem))
    return root


def deletion_candidate(
    filesystem: File, capacity: int, required: int
) -> Tuple[FilePath, File]:
    total_size = filesystem.data.size
    max_allowable = capacity - required
    must_delete = total_size - max_allowable
    print_(
        f"Capacity {capacity}, {total_size} used, {capacity - total_size} free, "
        f"must delete {must_delete} to free {required}"
    )
    deletion_candidates = (
        (p, d)
        for p, d in dfs(filesystem)
        if d.data.is_dir and d.data.size >= must_delete
    )
    return min(deletion_candidates, key=lambda t: t[1].data.size)


def add_sizes(stats1: Stat, stats2: Stat) -> Stat:
    return Stat(stats1.is_dir or stats2.is_dir, stats1.size + stats2.size)


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
        f=cast(Callable[[Stat], Stat], identity),
        acc=add_sizes,
    )
    if verbose:
        s = dir_to_str(sized_filesystem)
        print_(s, end="\n\n")
    if part_2:
        delete_path, dir_ = deletion_candidate(sized_filesystem, capacity, required)
        return f"{'/'.join(name.rstrip('/') for name in delete_path)} {dir_.data.size}"
    else:
        dir_sizes = (t.data.size for _, t in dfs(sized_filesystem) if t.data.is_dir)
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
    expected_path = "/a/e 584"
    assert path == expected_path, (path, expected_path)
    path = run(
        io.StringIO(test_input),
        part_2=True,
        capacity=total_size,
        required=a_size - a_size // 4,
    )
    expected_path = "/a 94853"
    assert path == expected_path, (path, expected_path)
    fs = tree_acc(
        parse_terminal(io.StringIO(test_input)),
        identity,
        add_sizes,
    )
    expected_size = sum(map(int, re.findall(r"\d+", test_input)))
    file_size = sum(s.data.size for _, s in dfs(fs) if not s.data.is_dir)
    assert file_size == expected_size, (file_size, expected_size)
    assert fs.data.size == expected_size, (fs.data, expected_size)
