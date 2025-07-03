import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Set

import chardet
import networkx as nx
from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser

from src.config import BugInfo
from src.interfaces.d4j import get_test_case_dataset_path
from src.schema import CGMethodNode, Tag, TestFailure
from src.utils import Timer


class CodeGraph:
    """Constructs and manages code dependency graphs for Python/Java projects"""

    SUPPORTED_LANGS = {".py": "python", ".java": "java"}

    def __init__(
        self, src_dir: str, test_dir: str, language: str = None, logger=None
    ):
        self.src_dir = src_dir
        self.test_dir = test_dir
        self.language = language
        self.logger = logger
        if language and language not in self.SUPPORTED_LANGS.values():
            raise ValueError(
                f"Unsupported language: {language}. Supported languages are: {list(self.SUPPORTED_LANGS.values())}"
            )

        # Cache for tag locations
        self.location = {}

        # Cache for def tags
        self.def_tags = {}

    def get_code_graph(
        self,
        src_files: List[str],
        test_files: List[str],
        inst_graph_file: Path,
    ) -> nx.MultiDiGraph:
        """Construct code graph from given files"""
        if not src_files or not test_files:
            raise ValueError("No source or test files found")

        src_tags = self.get_file_tags(src_files, False)
        test_tags = self.get_file_tags(test_files, True)
        tags = src_tags + test_tags
        inst_graph = self.load_inst_graph(inst_graph_file)
        repo_graph = self.create_repo_graph(inst_graph, tags)
        return repo_graph

    def load_inst_graph(self, graphml_file: Path) -> nx.DiGraph:
        """
        load the call graph from graphml file.
        Note that a node in this graph is a CGMethodNode object,
        which may not correspond to a method in the source code.
        """

        def get_node_from_str(node_str):
            pattern = r"(.*?)@(.*?):(.*?)\((\d+)-(\d+)\)"
            match = re.match(pattern, node_str)
            if not match:
                raise ValueError(f"Error: Invalid node string: {node_str}")
            package, class_name, method_name, start_line, end_line = (
                match.groups()
            )
            return CGMethodNode(
                package,
                class_name,
                method_name,
                int(start_line),
                int(end_line),
            )

        new_G = nx.DiGraph()
        raw_G = nx.read_gml(graphml_file)
        for u, v in raw_G.edges:
            new_G.add_edge(get_node_from_str(u), get_node_from_str(v))
        return new_G

    def create_dynamic_graph(
        self,
        repo_graph: nx.MultiDiGraph,
        inst_graph: nx.DiGraph,
        tags: List[Tag],
    ):
        """
        Renew the repo graph with the dynamic call graph.
        The dynamic call graph is constructed by runtime method call relationship,
        which is collected by a Java agent.
        """

        def get_all_matched_successors(
            node: CGMethodNode, visited_fathers: Set[CGMethodNode]
        ):
            """Get all matched successors in the repo graph
            Note that their may be self loops in the call graph"""
            visited_fathers.add(node)
            matched_successors = []
            for successor in inst_graph.successors(node):
                if successor in visited_fathers:
                    continue
                if successor in node_to_tag:
                    matched_successors.append(node_to_tag[successor])
                else:
                    matched = get_all_matched_successors(
                        successor, visited_fathers
                    )
                    matched_successors.extend(matched)

            return matched_successors

        def match_tag(node: CGMethodNode):
            founded_tags = []
            if "$" in node.class_name:  # may be a inner class
                last_idx = node.class_name.rfind("$")
                while last_idx != -1:
                    query_key = (
                        f"{node.class_name[:last_idx]}.{node.method_name}"
                    )
                    if query_key in tag_map:
                        for start_line, end_line in tag_map[query_key]:
                            if (
                                node.start_line >= start_line
                                and node.end_line <= end_line
                            ):
                                founded_tags.append(
                                    tag_map[query_key][(start_line, end_line)]
                                )
                    last_idx = node.class_name.rfind("$", 0, last_idx)
            else:
                query_key = f"{node.class_name}.{node.method_name}"
                if query_key in tag_map:
                    for start_line, end_line in tag_map[query_key]:
                        if (
                            node.start_line >= start_line
                            and node.end_line <= end_line
                        ):
                            founded_tags.append(
                                tag_map[query_key][(start_line, end_line)]
                            )

            if not founded_tags:
                return None
            elif len(founded_tags) == 1:
                founded_tag = founded_tags[0]
                node_to_tag[node] = founded_tag
                founded_tag.outer_class = node.outer_class
                founded_tag.inner_class = node.inner_class
                founded_tag.is_covered = True
                return founded_tag
            else:
                # if multiple tags are found, choose the smallest one
                sorted_tags = sorted(
                    founded_tags, key=lambda x: x.line[1] - x.line[0]
                )
                founded_tag = sorted_tags[0]
                node_to_tag[node] = founded_tag
                founded_tag.outer_class = node.outer_class
                founded_tag.inner_class = node.inner_class
                founded_tag.is_covered = True
                return founded_tag

        # group the tags for matching
        tag_map = defaultdict(dict)
        for tag in tags:
            if tag.category == "function":
                file_path = Path(tag.fname)
                outer_class = file_path.stem
                method_name = tag.name
                query_key = f"{outer_class}.{method_name}"
                tag_map[query_key][tag.line] = tag

        # match the dynamic nodes to the static tags
        node_to_tag = {}
        for node in tqdm(inst_graph.nodes, desc="Matching dynamic call graph"):
            tag = match_tag(node)
            if tag:
                node_to_tag[node] = tag

        for u, v in tqdm(inst_graph.edges, desc="Adding dynamic call edges"):
            if u in node_to_tag and v in node_to_tag:
                if not repo_graph.has_edge(
                    node_to_tag[u], node_to_tag[v], key="unique"
                ):
                    repo_graph.add_edge(
                        node_to_tag[u],
                        node_to_tag[v],
                        key="unique",
                        rel="calls",
                    )
            elif u in node_to_tag and v not in node_to_tag:
                # solve the issue caused by Java synthetic methods
                visited_fathers = set()
                successors = get_all_matched_successors(u, visited_fathers)
                if successors:
                    for successor in successors:
                        if not repo_graph.has_edge(
                            node_to_tag[u], successor, key="unique"
                        ):
                            repo_graph.add_edge(
                                node_to_tag[u],
                                successor,
                                key="unique",
                                rel="calls",
                            )
            else:
                pass

    def create_static_graph(
        self, repo_graph: nx.MultiDiGraph, tags: List[Tag]
    ):
        """
        Renew the repo graph with the static call graph.
        The static call graph is constructed by the code dependency relationship,
        which is collected by the tree-sitter parser.

        Note: The static analysis may not be accurate!
        """

        def find_def_tag(ref_tag, categories=[]):
            """Find definition tag for reference"""
            ref_fname = ref_tag.fname
            ref_line = ref_tag.line
            if ref_fname not in self.location:
                return None
            for lines, def_tag in self.location[ref_fname].items():
                if lines[0] <= ref_line[0] and ref_line[1] <= lines[1]:
                    if categories and def_tag.category not in categories:
                        continue
                    return def_tag
            return None

        # Add edges for references
        tags_ref = [tag for tag in tags if tag.kind == "ref"]
        # tags_def = [tag for tag in tags if tag.kind == 'def']
        tags_class = [tag for tag in tags if tag.category == "class"]
        tags_function = [tag for tag in tags if tag.category == "function"]
        tags_field = [tag for tag in tags if tag.category == "field"]
        tags_interface = [tag for tag in tags if tag.category == "interface"]

        # Add nodes
        repo_graph.add_nodes_from(tags)

        # Add edges
        for tag in tqdm(tags_function, desc="Adding function-class edges"):
            tgt_tag = find_def_tag(tag, ["class", "interface", "enum"])
            if tgt_tag:
                repo_graph.add_edge(tgt_tag, tag, rel="defines_function")
            else:
                if self.language == "java":
                    raise ValueError(
                        f"Missing class definition for function {tag.name}"
                    )

        for tag in tqdm(tags_field, desc="Adding field-class edges"):
            tgt_tag = find_def_tag(tag, ["class", "interface", "enum"])
            if tgt_tag:
                repo_graph.add_edge(tgt_tag, tag, rel="defines_field")
            else:
                if self.language == "java":
                    raise ValueError(
                        f"Missing class definition for field {tag.name}"
                    )

        for tag in tqdm(tags_ref, desc="Adding function call edges"):
            src_tag = find_def_tag(tag, ["function", "field"])
            if src_tag:
                tgt_tags = self.def_tags.get(tag.name, [])
                for tgt_tag in tgt_tags:
                    if tgt_tag != src_tag:
                        repo_graph.add_edge(src_tag, tgt_tag, rel="may_calls")

        for tag in tqdm(
            tags_class, desc="Adding inherits and implements edges"
        ):
            if tag.parent_class:
                for parent_tag in tags_class:
                    if parent_tag.name == tag.parent_class:
                        repo_graph.add_edge(tag, parent_tag, rel="inherits")
            if tag.interfaces:
                for interface in tag.interfaces:
                    for interface_tag in tags_interface:
                        if interface_tag.name == interface:
                            repo_graph.add_edge(
                                tag, interface_tag, rel="implements"
                            )

    def create_repo_graph(
        self, G: nx.DiGraph, tags: List[Tag]
    ) -> nx.MultiDiGraph:
        """Create repo graph from static and dynamic graphs"""
        with Timer(self.logger, "Creating repo graph"):
            repo_graph = nx.MultiDiGraph()
            self.create_static_graph(repo_graph, tags)
            self.create_dynamic_graph(repo_graph, G, tags)
            return repo_graph

    def get_file_tags(self, files: List[str], is_test: bool) -> List[Tag]:
        """Extract tags from source files"""
        all_tags = []
        for fname in tqdm(files, desc="Parsing tags"):
            if not Path(fname).is_file():
                continue

            if is_test:
                rel_fname = os.path.relpath(fname, self.test_dir)
            else:
                rel_fname = os.path.relpath(fname, self.src_dir)
            ext = Path(fname).suffix

            lang = self.SUPPORTED_LANGS[ext]
            tags = self.parse_file(fname, rel_fname, lang, is_test)
            all_tags.extend(tags)

        return all_tags

    def parse_file(
        self, fname: str, rel_fname: str, lang: str, is_test: bool
    ) -> List[Tag]:
        """Parse single source file and extract tags"""
        try:
            with open(fname, "r", encoding="utf-8") as f:
                file_content = f.read()
        except UnicodeDecodeError:
            with open(fname, "rb") as f:
                file_content = f.read()
                encoding = chardet.detect(file_content)["encoding"]
                file_content = file_content.decode(encoding)

        parser = get_parser(lang)
        language = get_language(lang)
        tree = parser.parse(bytes(file_content, "utf-8"))

        # Get language-specific query
        query = self.get_query(lang)
        captures = language.query(query).captures(tree.root_node)

        tags = []
        for node, tag_type in captures:
            tag = self.create_tag(
                node, tag_type, fname, rel_fname, file_content, is_test
            )
            if tag:
                tags.append(tag)

        return tags

    def get_node_code(self, node, code: str) -> str:
        """Get code for AST node, keep the indentation"""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        return "\n".join(code.splitlines()[start_line - 1 : end_line])

    def create_tag(
        self,
        node,
        tag_type: str,
        fname: str,
        rel_fname: str,
        file_content: str,
        is_test: bool,
    ) -> Optional[Tag]:
        """Create tag from AST node"""
        pkg_name = ".".join(rel_fname.split(".")[0].split("/")[:-1])
        parent_class = None
        outer_class = rel_fname.split("/")[-1].split(".")[0]
        interfaces = []

        if tag_type == "definition.interface":
            kind = "def"
            name_node = node.child_by_field_name("name")
            code = self.get_node_code(node, file_content)
            category = "interface"
            if name_node:
                name = name_node.text.decode("utf-8")
            else:
                raise ValueError(f"Missing name node in {tag_type}")
        elif tag_type == "definition.enum":
            kind = "def"
            name_node = node.child_by_field_name("name")
            code = self.get_node_code(node, file_content)
            category = "enum"
            if name_node:
                name = name_node.text.decode("utf-8")
            else:
                raise ValueError(f"Missing name node in {tag_type}")
        elif tag_type == "definition.class":
            kind = "def"
            name_node = node.child_by_field_name("name")
            code = self.get_node_code(node, file_content)
            category = "class"
            if name_node:
                name = name_node.text.decode("utf-8")
                if self.language == "python":
                    bases_node = node.child_by_field_name("bases")
                    if bases_node and bases_node.named_children:
                        parent_class = bases_node.named_children[
                            0
                        ].text.decode("utf-8")
                elif self.language == "java":
                    superclass_node = node.child_by_field_name("superclass")
                    if superclass_node and superclass_node.named_children:
                        parent_class = superclass_node.named_children[
                            0
                        ].text.decode("utf-8")
                    interfaces_node = node.child_by_field_name("interfaces")
                    if interfaces_node:
                        interfaces = [
                            child.text.decode("utf-8")
                            for child in interfaces_node.named_children
                        ]
        elif tag_type in ["definition.function", "definition.constructor"]:
            kind = "def"
            category = "function"
            name_node = node.child_by_field_name("name")
            code = self.get_node_code(node, file_content)
            if name_node:
                name = name_node.text.decode("utf-8")
            else:
                raise ValueError(f"Missing name node in {tag_type}")
        elif tag_type == "definition.class.field":
            kind = "def"
            category = "field"
            name_node = node.child_by_field_name(
                "declarator"
            ).child_by_field_name("name")
            code = self.get_node_code(node, file_content)
            if name_node:
                name = name_node.text.decode("utf-8")
            else:
                raise ValueError(f"Missing name node in {tag_type}")
        elif tag_type.startswith("name.reference."):
            kind = "ref"
            name = node.text.decode("utf-8")
            category = None
            code = None
        else:
            return None

        lines = (node.start_point[0] + 1, node.end_point[0] + 1)
        tag = Tag(
            rel_fname,
            fname,
            lines,
            name,
            kind,
            category,
            code,
            pkg_name,
            parent_class,
            tuple(interfaces),
            is_test,
            outer_class,
        )
        try:
            self.location[fname][lines] = tag
        except:
            self.location[fname] = {lines: tag}

        if kind == "def":
            try:
                self.def_tags[name].append(tag)
            except:
                self.def_tags[name] = [tag]
        return tag

    @staticmethod
    def get_query(lang: str) -> str:
        """Get AST query for specified language"""
        queries = {
            "python": """
                (class_definition 
                    name: (identifier) @name.definition.class
                    bases: (argument_list 
                        (identifier)? @name.inherit.class)?) @definition.class
                (function_definition name: (identifier) @name.definition.function) @definition.function
                (class_definition
                    body: (block 
                        (expression_statement
                            (assignment 
                                left: (identifier) @name.definition.field)))) @definition.class.field
                (call function: [(identifier) @name.reference.call (attribute attribute: (identifier) @name.reference.call)]) @reference.call
            """,
            "java": """
                (interface_declaration 
                    name: (identifier) @name.definition.interface) @definition.interface
                    
                (class_declaration 
                    name: (identifier) @name.definition.class) @definition.class
                
                (enum_declaration
                    name: (identifier) @name.definition.enum) @definition.enum

                (method_declaration 
                    name: (identifier) @name.definition.function) @definition.function
                    
                (constructor_declaration 
                    name: (identifier) @name.definition.constructor) @definition.constructor
                    
                (field_declaration 
                    declarator: (variable_declarator 
                        name: (identifier) @name.definition.field)) @definition.class.field
                        
                (method_invocation 
                    name: (identifier) @name.reference.call) @reference.call
                    
                (object_creation_expression 
                    type: (type_identifier) @name.reference.constructor) @reference.call
            """,
        }
        return queries.get(lang, "")

    def logging(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def print_graph_infos(self, G: nx.MultiDiGraph):
        """Print graph statistics"""
        self.logging("-" * 50)
        self.logging(
            f"Generated graph with {len(G.nodes)} nodes and {len(G.edges)} edges"
        )
        self.logging(
            f"covered nodes: {len([n for n in G.nodes(data=True) if n[0].is_covered])}"
        )
        self.logging(
            f"function-class edges: {len([e for e in G.edges(data=True) if e[2]['rel'] == 'defines_function'])}"
        )
        self.logging(
            f"field-class edges: {len([e for e in G.edges(data=True) if e[2]['rel'] == 'defines_field'])}"
        )
        self.logging(
            f"static call edges: {len([e for e in G.edges(data=True) if e[2]['rel'] == 'may_calls'])}"
        )
        self.logging(
            f"dynamic call edges: {len([e for e in G.edges(data=True) if e[2]['rel'] == 'calls'])}"
        )
        self.logging(
            f"inheritance edges: {len([e for e in G.edges(data=True) if e[2]['rel'] == 'inherits'])}"
        )
        self.logging(
            f"interface implementation edges: {len([e for e in G.edges(data=True) if e[2]['rel'] == 'implements'])}"
        )
        self.logging("-" * 50)


def create_repo_graph(
    bug_info: BugInfo, test_failure: TestFailure, language="java"
):
    """Programmatic interface"""

    all_graphs = []
    for test_class in test_failure.test_classes:
        for test_case in test_class.test_cases:
            bug_info.logger.info(
                f"[build repo graph] test case: {bug_info.project}-{bug_info.bug_id} {test_case.name}"
            )

            src_dir = bug_info.buggy_path / bug_info.src_prefix
            test_dir = bug_info.buggy_path / bug_info.test_prefix
            graph = CodeGraph(src_dir, test_dir, language, bug_info.logger)

            test_cache_dir = get_test_case_dataset_path(bug_info, test_case)
            test_cache_dir.mkdir(parents=True, exist_ok=True)
            repo_graph_file = test_cache_dir / "repograph.pkl"
            if repo_graph_file.exists():
                bug_info.logger.info(
                    f"[build repo graph] exist cache {repo_graph_file}"
                )
                with repo_graph_file.open("rb") as f:
                    G = pickle.load(f)
                all_graphs.append(G)
                continue

            ext = [
                ext
                for ext, lang in CodeGraph.SUPPORTED_LANGS.items()
                if lang == language
            ][0]
            src_files = [str(f) for f in Path(src_dir).rglob(f"*{ext}")]
            test_files = [str(f) for f in Path(test_dir).rglob(f"*{ext}")]
            inst_graph_file = test_cache_dir / "callgraph.graphml"

            G = graph.get_code_graph(src_files, test_files, inst_graph_file)
            graph.print_graph_infos(G)
            bug_info.logger.info(f"[build repo graph] OK!")
            all_graphs.append(G)
            with repo_graph_file.open("wb") as f:
                pickle.dump(G, f)

    combined_graph_file = bug_info.bug_path / "combined_graph.pkl"
    if combined_graph_file.exists():
        bug_info.logger.info(
            f"[build combined graph] exist cache {combined_graph_file}"
        )
        return
    combined_graph = nx.compose_all(all_graphs)
    with combined_graph_file.open("wb") as f:
        pickle.dump(combined_graph, f)
    bug_info.logger.info(f"[build combined graph] OK!")
