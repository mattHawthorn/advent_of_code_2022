from functools import partial
from itertools import chain
from operator import itemgetter
from textwrap import indent
from typing import IO, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

DirPath = Tuple[str, ...]


class File(NamedTuple):
    name: str
    size: int

    def __str__(self):
        return f"- {self.name} {self.size}"


class FileTree(NamedTuple):
    name: str
    contents: Dict[str, Union[File, "FileTree"]]
    parent: Optional["FileTree"] = None

    def __str__(self):
        header = f"{self.name.rstrip('/')}/"
        contents = sorted(
            self.contents.values(), key=lambda f: (isinstance(f, FileTree), f.name)
        )
        return "\n".join(
            chain([header], map(partial(indent, prefix="  "), map(str, contents)))
        )


class FileStat(NamedTuple):
    path: DirPath
    is_dir: bool
    size: int

    @property
    def name(self):
        return "/".join(p if p == "/" else p.rstrip("/") for p in self.path)


def is_comand(line: str) -> bool:
    return line.startswith("$ ")


def parse_command(line: str) -> Tuple[str, Optional[str]]:
    tokens = line[2:].strip().split(" ", 1)
    return (tokens[0], None) if len(tokens) == 1 else (tokens[0], tokens[1])


def parse_file_dir(line: str) -> Union[File, str]:
    size, name = line.strip().split(" ", 1)
    return name if size == DIR else File(name.rstrip(), int(size))


# Commands and terminal constants
CD = "cd"
LS = "ls"
DIR = "dir"
PARENT = ".."

# Parsing states
READ_OUTPUT = 1
READ_COMMAND = 2


def parse_terminal(f: IO[str]) -> FileTree:
    filesystem = current_tree = FileTree("START", {}, None)
    state = READ_COMMAND
    for line in f:
        if state != READ_COMMAND and is_comand(line):
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
                elif maybe_name in current_tree.contents:
                    sub_tree = current_tree.contents[maybe_name]
                    assert isinstance(sub_tree, FileTree)
                    current_tree = sub_tree
                else:
                    sub_tree = FileTree(maybe_name, {}, current_tree)
                    current_tree.contents[maybe_name] = sub_tree
                    current_tree = sub_tree
        else:
            file_or_dir = parse_file_dir(line)
            if isinstance(file_or_dir, File):
                current_tree.contents[file_or_dir.name] = file_or_dir
            elif file_or_dir not in current_tree.contents:
                current_tree.contents[file_or_dir] = FileTree(
                    file_or_dir, {}, current_tree
                )

    assert len(filesystem.contents) == 1
    root = filesystem.contents["/"]
    assert isinstance(root, FileTree)
    return root


def content_sizes(
    filesystem: Union[FileTree, File], path: DirPath = ()
) -> Iterator[FileStat]:
    full_path = (*path, filesystem.name)
    if isinstance(filesystem, File):
        yield FileStat(full_path, False, filesystem.size)
    else:
        dir_size = 0
        for file_or_dir in filesystem.contents.values():
            contents = content_sizes(file_or_dir, full_path)
            for stats in contents:
                yield stats
                if not stats.is_dir:
                    dir_size += stats.size
        yield FileStat(full_path, True, dir_size)


def deletion_candidate(stats: List[FileStat], capacity: int, required: int) -> FileStat:
    total_size = sum(f.size for f in stats if not f.is_dir)
    max_allowable = capacity - required
    must_delete = total_size - max_allowable
    deletion_candidates = (d for d in stats if d.is_dir and d.size >= must_delete)
    return min(deletion_candidates, key=itemgetter(2))


def run(
    input_: IO[str],
    max_size: int = 100000,
    capacity: int = 70000000,
    required: int = 30000000,
    part_2: bool = True,
    verbose: bool = False,
) -> str:
    tree = parse_terminal(input_)
    if verbose:
        print(tree, end="\n\n")
    sizes = content_sizes(tree)
    if part_2:
        stat = deletion_candidate(list(sizes), capacity, required)
        return stat.name
    else:
        dir_sizes = filter(itemgetter(1), sizes)
        small_dir_sizes = filter(max_size.__ge__, map(itemgetter(2), dir_sizes))
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
