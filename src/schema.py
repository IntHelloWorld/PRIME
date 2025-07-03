from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx

from src.interfaces.method_extractor import JMethod


@dataclass
class Tag:
    """Represents a code element in the source file."""

    rel_fname: str
    fname: str
    line: Tuple[int, int]
    name: str
    kind: str  # 'def' or 'ref'
    category: str  # 'class', 'function', 'field', 'interface', or 'enum'
    code: str
    pkg_name: str = None
    parent_class: str = None
    interfaces: Tuple[str] = ()
    is_test: bool = False

    # outer class name is filled for all methods
    # since it can be extrated from the Java file name
    outer_class: str = None

    # for covered methods, these field will be filled after instrumentation
    inner_class: str = None
    is_covered: bool = False

    def __hash__(self):
        return hash(
            (self.rel_fname, self.line, self.name, self.kind, self.category)
        )

    @property
    def method_id(self):
        if self.category != "function":
            return None
        if self.inner_class:
            return f"{self.pkg_name}.{self.outer_class}.{self.inner_class.replace('$', '.')}.{self.name}<{self.line[0]}-{self.line[1]}>"
        return f"{self.pkg_name}.{self.outer_class}.{self.name}<{self.line[0]}-{self.line[1]}>"

    @property
    def method_pos(self):
        if self.category != "function":
            return None
        if self.inner_class:
            return f".{self.inner_class.replace('$', '.')}.{self.name}<{self.line[0]}-{self.line[1]}>"
        return f".{self.name}<{self.line[0]}-{self.line[1]}>"


class MethodID:
    """static class to parse method IDs. A method ID is a string that uniquely identifies a method in the codebase.
    It is constructed as follows:

    package.class_name.method_name<start_line-end_line>
    """

    @staticmethod
    def get_method_name(method_id: str):
        return method_id.split("<")[0].split(".")[-1]

    @staticmethod
    def get_method_full_name(method_id: str):
        return method_id.split("<")[0]

    @staticmethod
    def get_class_name(method_id: str):
        return method_id.split("<")[0].split(".")[-2]

    @staticmethod
    def get_class_full_name(method_id: str):
        return ".".join(method_id.split("<")[0].split(".")[:-1])


@dataclass
class CGMethodNode:
    """Represents a method node in the call graph constructed by Java agent."""

    package: str
    class_name: str
    method_name: str
    start_line: int
    end_line: int

    @property
    def signature(self):
        return f"{self.package}@{self.class_name}:{self.method_name}({self.start_line}-{self.end_line})"

    @property
    def inner_class(self):
        if "$" in self.class_name:
            return "$".join(self.class_name.split("$")[1:])
        return None

    @property
    def outer_class(self):
        return self.class_name.split("$")[0]

    def __hash__(self):
        return hash(self.signature)


@dataclass
class TestCase:
    name: str
    test_method: Optional[JMethod] = None
    test_output: Optional[str] = None
    stack_trace: Optional[str] = None
    test_class_name: Optional[str] = None
    test_method_name: Optional[str] = None

    def __str__(self) -> str:
        return self.name

    def __post_init__(self):
        self.test_class_name, self.test_method_name = self.name.split("::")


@dataclass
class TestClass:
    name: str
    test_cases: List[TestCase]

    def __str__(self) -> str:
        return f"{self.name}: {str(self.test_cases)}"


@dataclass
class TestFailure:
    project: str
    bug_ID: int
    test_classes: List[TestClass]
    buggy_methods: Optional[List[JMethod]] = None


@dataclass
class DebugInput:
    test_name: str
    test_code: str
    error_message: str
    repo_graph: nx.Graph
    loaded_classes: List[str]
    output_path: Path


@dataclass
class SearchInput:
    test_name: str
    test_code: str
    error_message: str
    output_path: Path


@dataclass
class SearchResult:
    debug_report: str
    error_analysis: str
    method_id: str


@dataclass
class VerifyInput:
    test_name: str
    test_code: str
    error_message: str
    method_id: str
    suspected_issue: str
    method_code: str
    output_dir: Path
    process_id: str
    method: Tag


@dataclass
class VerifyResult:
    method_id: str
    category: str
    explanation: str


@dataclass
class RoundResult:
    search_results: List[SearchResult]
    verify_results: List[VerifyResult]
