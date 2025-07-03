import json
from difflib import SequenceMatcher
from typing import Dict, List

import networkx as nx

from src.exceptions import (
    ClassNameNotFoundError,
    ClassNotFoundError,
    MethodIDNotFoundError,
    MethodNotFoundError,
    SnippetNotFoundError,
)
from src.schema import MethodID, Tag


class AutoFLRepoSearcher:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.covered_classes_result = None

        # create a dictionary of methods grouped by class name
        # note that this supports fuzzy matching
        method_dict: Dict[str, List[Tag]] = {}

        # create a map of method ID to method node
        method_id_map: Dict[str, Tag] = {}

        for node in self.graph.nodes(data=True):
            if node[0].category == "function":
                method_node: Tag = node[0]

                method_id_map[method_node.method_id] = method_node
                inner_class_names = []
                if method_node.inner_class:
                    inner_class_names = method_node.inner_class.split("$")
                class_names = [method_node.outer_class] + inner_class_names
                for i in range(1, len(class_names) + 1):
                    class_name = ".".join(class_names[:i])
                    class_full_name = f"{method_node.pkg_name}.{class_name}"
                    try:
                        method_dict[class_full_name].append(method_node)
                    except KeyError:
                        method_dict[class_full_name] = [method_node]

        self.method_dict = method_dict
        self.method_id_map = method_id_map

    def get_method(self, method_id) -> Tag | None:
        return self.method_id_map.get(method_id, None)

    def get_covered_classes(self):
        """Get a dictionary of all covered classes in the graph"""
        if self.covered_classes_result:
            return self.covered_classes_result

        covered_test_classes = {}
        covered_source_classes = {}
        for node in self.graph.nodes(data=True):
            if node[0].category == "function":
                method_node: Tag = node[0]
                if method_node.is_covered:
                    full_class_name = method_node.outer_class
                    if method_node.is_test:
                        try:
                            if (
                                full_class_name
                                not in covered_test_classes[
                                    method_node.pkg_name
                                ]
                            ):
                                covered_test_classes[
                                    method_node.pkg_name
                                ].append(full_class_name)
                        except KeyError:
                            covered_test_classes[method_node.pkg_name] = [
                                method_node.outer_class
                            ]
                    else:
                        try:
                            if (
                                full_class_name
                                not in covered_source_classes[
                                    method_node.pkg_name
                                ]
                            ):
                                covered_source_classes[
                                    method_node.pkg_name
                                ].append(full_class_name)
                        except KeyError:
                            covered_source_classes[method_node.pkg_name] = [
                                method_node.outer_class
                            ]

        template = (
            "Covered classes grouped with package name:\n\n"
            "Covered source classes:\n{covered_source_classes}\n\n"
            "Covered test classes:\n{covered_test_classes}"
        )
        result = template.format(
            covered_source_classes=json.dumps(covered_source_classes),
            covered_test_classes=json.dumps(covered_test_classes),
        )
        self.covered_classes_result = result
        return result

    def get_method_dict(self, methods: List[Tag]) -> str:
        """Get a dictionary of all methods grouped by class name"""
        result = {}
        for method in methods:
            pkg_name = method.pkg_name
            class_name = method.outer_class
            full_class_name = f"{pkg_name}.{class_name}"
            try:
                result[full_class_name].append(method.method_pos)
            except KeyError:
                result[full_class_name] = [method.method_pos]
        template = "Method IDs grouped by class name:\n{methods}"
        return template.format(methods=json.dumps(result))

    def get_covered_method_ids_for_class(self, class_name):
        """Get a list of all covered methods of a class"""
        try:
            if class_name in self.method_dict:
                precise_methods = self.method_dict[class_name]
                covered_methods = [m for m in precise_methods if m.is_covered]
                method_dict = self.get_method_dict(covered_methods)
                return method_dict
            else:
                raise ClassNotFoundError(class_name)
        except Exception as e:
            return e.message

    def get_possible_method_ids(self, false_id):
        false_method_name = MethodID.get_method_name(false_id)
        possible_method_ids = []
        for node in self.graph.nodes(data=True):
            if node[0].category == "function":
                method_node: Tag = node[0]
                if not method_node.is_covered:
                    continue
                if method_node.name == false_method_name:
                    possible_method_ids.append(method_node.method_id)

        possible_method_ids = list(set(possible_method_ids))
        return possible_method_ids

    def get_similar_candidates(
        self,
        false_method_id: str,
        possible_method_ids: List[str],
        num_max_candidates: int = 5,
    ):
        def _compute_similarity(method_id_1, method_id_2):
            method_full_name_1 = MethodID.get_method_full_name(method_id_1)
            method_full_name_2 = MethodID.get_method_full_name(method_id_2)
            return SequenceMatcher(
                None, method_full_name_1, method_full_name_2
            ).ratio()

        similarities = []
        for possible_method_id in possible_method_ids:
            similarity = _compute_similarity(
                false_method_id, possible_method_id
            )
            similarities.append((similarity, possible_method_id))

        candidates = list(
            map(
                lambda t: t[1],
                sorted(similarities, key=lambda t: t[0], reverse=True),
            )
        )
        if num_max_candidates is not None:
            candidates = candidates[:num_max_candidates]
        return candidates

    def get_method_code_for_id(self, method_id):
        """Get the code of a method"""
        method_tag = self.get_method(method_id)
        if not method_tag:
            # find candidate methods
            try:
                possible_method_ids = self.get_possible_method_ids(method_id)
                if not possible_method_ids:
                    raise MethodIDNotFoundError(method_id)
                candidate_methods = self.get_similar_candidates(
                    method_id, possible_method_ids
                )
                result = f"The method ID is not found, maybe you mean one of the following similar method IDs:\n{candidate_methods}"
                return result
            except Exception as e:
                return e.message

        result = (
            f'"Method ID": "{method_id}"\n'
            f'"Method Code":\n```java\n{method_tag.code}\n```\n'
        )
        return result


class RepoSearcher:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.covered_classes_result = None

        # create a dictionary of methods grouped by class name
        # note that this supports fuzzy matching
        method_dict: Dict[str, List[Tag]] = {}
        covered_method_dict: Dict[str, List[Tag]] = {}

        # create a map of method ID to method node
        method_id_map: Dict[str, Tag] = {}

        for node in self.graph.nodes(data=True):
            if node[0].category == "function":
                method_node: Tag = node[0]

                method_id_map[method_node.method_id] = method_node
                inner_class_names = []
                if method_node.inner_class:
                    inner_class_names = method_node.inner_class.split("$")
                class_names = [method_node.outer_class] + inner_class_names
                for i in range(1, len(class_names) + 1):
                    class_name = ".".join(class_names[:i])
                    class_full_name = f"{method_node.pkg_name}.{class_name}"
                    try:
                        method_dict[class_full_name].append(method_node)
                    except KeyError:
                        method_dict[class_full_name] = [method_node]
                    if method_node.is_covered:
                        try:
                            covered_method_dict[class_full_name].append(
                                method_node
                            )
                        except KeyError:
                            covered_method_dict[class_full_name] = [
                                method_node
                            ]
        self.method_dict = method_dict
        self.method_id_map = method_id_map
        self.covered_method_dict = covered_method_dict

    def get_method(self, method_id) -> Tag | None:
        return self.method_id_map.get(method_id, None)

    def get_covered_classes(self):
        """Get a dictionary of all covered classes in the graph"""
        if self.covered_classes_result:
            return self.covered_classes_result

        covered_test_classes = {}
        covered_source_classes = {}
        for node in self.graph.nodes(data=True):
            if node[0].category == "function":
                method_node: Tag = node[0]
                if method_node.is_covered:
                    full_class_name = method_node.outer_class
                    if method_node.is_test:
                        try:
                            if (
                                full_class_name
                                not in covered_test_classes[
                                    method_node.pkg_name
                                ]
                            ):
                                covered_test_classes[
                                    method_node.pkg_name
                                ].append(full_class_name)
                        except KeyError:
                            covered_test_classes[method_node.pkg_name] = [
                                method_node.outer_class
                            ]
                    else:
                        try:
                            if (
                                full_class_name
                                not in covered_source_classes[
                                    method_node.pkg_name
                                ]
                            ):
                                covered_source_classes[
                                    method_node.pkg_name
                                ].append(full_class_name)
                        except KeyError:
                            covered_source_classes[method_node.pkg_name] = [
                                method_node.outer_class
                            ]

        template = (
            "Covered classes grouped with package name:\n\n"
            "Covered source classes:\n{covered_source_classes}\n\n"
            "Covered test classes:\n{covered_test_classes}"
        )
        result = template.format(
            covered_source_classes=json.dumps(covered_source_classes),
            covered_test_classes=json.dumps(covered_test_classes),
        )
        self.covered_classes_result = result
        return result

    def get_method_dict(self, methods: List[Tag]) -> str:
        """Get a dictionary of all methods grouped by class name"""
        result = {}
        for method in methods:
            pkg_name = method.pkg_name
            class_name = method.outer_class
            full_class_name = f"{pkg_name}.{class_name}"
            try:
                result[full_class_name].append(method.method_pos)
            except KeyError:
                result[full_class_name] = [method.method_pos]
        template = "Method IDs grouped by class name:\n{methods}"
        return template.format(methods=json.dumps(result))

    def get_covered_method_ids_for_class(self, class_name):
        """Get a list of all covered methods of a class"""
        try:
            if class_name in self.method_dict:
                precise_methods = self.method_dict[class_name]
                covered_methods = [m for m in precise_methods if m.is_covered]
                method_dict = self.get_method_dict(covered_methods)
                return method_dict
            else:
                raise ClassNotFoundError(class_name)
        except Exception as e:
            return e.message

    def get_possible_method_ids(self, false_id):
        false_method_name = MethodID.get_method_name(false_id)
        possible_method_ids = []
        for node in self.graph.nodes(data=True):
            if node[0].category == "function":
                method_node: Tag = node[0]
                if not method_node.is_covered:
                    continue
                if method_node.name == false_method_name:
                    possible_method_ids.append(method_node.method_id)

        possible_method_ids = list(set(possible_method_ids))
        return possible_method_ids

    def get_similar_candidates(
        self,
        false_method_id: str,
        possible_method_ids: List[str],
        num_max_candidates: int = 5,
    ):
        def _compute_similarity(method_id_1, method_id_2):
            method_full_name_1 = MethodID.get_method_full_name(method_id_1)
            method_full_name_2 = MethodID.get_method_full_name(method_id_2)
            return SequenceMatcher(
                None, method_full_name_1, method_full_name_2
            ).ratio()

        similarities = []
        for possible_method_id in possible_method_ids:
            similarity = _compute_similarity(
                false_method_id, possible_method_id
            )
            similarities.append((similarity, possible_method_id))

        candidates = list(
            map(
                lambda t: t[1],
                sorted(similarities, key=lambda t: t[0], reverse=True),
            )
        )
        if num_max_candidates is not None:
            candidates = candidates[:num_max_candidates]
        return candidates

    def get_method_code_for_id(self, method_id):
        """Get the code of a method"""
        method_tag = self.get_method(method_id)
        if not method_tag:
            # find candidate methods
            try:
                possible_method_ids = self.get_possible_method_ids(method_id)
                if not possible_method_ids:
                    raise MethodIDNotFoundError(method_id)
                candidate_methods = self.get_similar_candidates(
                    method_id, possible_method_ids
                )
                result = f"The method ID is not found, maybe you mean one of the following similar method IDs:\n{candidate_methods}"
                return result
            except Exception as e:
                return e.message

        result = (
            f'"Method ID": "{method_id}"\n'
            f'"Method Code":\n```java\n{method_tag.code}\n```\n'
        )
        return result

    def search_covered_class_full_name(self, class_name: str):
        """Get a list of class full names that match the given class name"""

        def split_class_name(class_name):
            items = class_name.split(".")
            for idx in range(len(items)):
                if items[idx][0].isupper():
                    break
            pkg_name = ".".join(items[:idx])
            class_name = ".".join(items[idx:])
            return pkg_name, class_name

        try:
            class_full_names = []
            for class_full_name in self.covered_method_dict.keys():
                if class_full_name.endswith(class_name):
                    class_full_names.append(class_full_name)

            if not class_full_names:
                raise ClassNameNotFoundError(class_name)
            class_name_dict = {}
            for class_full_name in class_full_names:
                pkg_name, class_name = split_class_name(class_full_name)
                try:
                    class_name_dict[pkg_name].append(class_name)
                except KeyError:
                    class_name_dict[pkg_name] = [class_name]

            result = f"The following covered classes match the provided class name:\n{class_name_dict}"
            return result
        except Exception as e:
            return e.message

    def search_covered_method_id(
        self, method_name: str, class_name: str = None
    ):
        """Get a list of method IDs that match the method name and the class name (optional)"""
        try:
            possible_methods = []
            if class_name:
                possible_classes = [
                    c for c in self.method_dict if c.endswith(class_name)
                ]
                if not possible_classes:
                    raise ClassNameNotFoundError(class_name)

                for c in possible_classes:
                    class_methods = self.method_dict[c]
                    for m in class_methods:
                        if m.name != method_name:
                            continue
                        if m not in possible_methods:
                            possible_methods.append(m)
            else:
                for _, method_node in self.method_id_map.items():
                    if method_node.name != method_name:
                        continue
                    if method_node not in possible_methods:
                        possible_methods.append(method_node)

            if not possible_methods:
                raise MethodNotFoundError(method_name)
            possible_methods = [m for m in possible_methods if m.is_covered]
            method_dict = self.get_method_dict(possible_methods)

            result = (
                f"We found the following covered method IDs:\n{method_dict}"
            )
            return result

        except Exception as e:
            return e.message


class NewRepoSearcher:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.covered_classes_result = None

        # create a dictionary of methods grouped by class name
        # note that this supports fuzzy matching
        method_dict: Dict[str, List[Tag]] = {}

        # create a map of method ID to method node
        method_id_map: Dict[str, Tag] = {}

        for node in self.graph.nodes(data=True):
            if node[0].category == "function":
                method_node: Tag = node[0]

                method_id_map[method_node.method_id] = method_node
                inner_class_names = []
                if method_node.inner_class:
                    inner_class_names = method_node.inner_class.split("$")
                class_names = [method_node.outer_class] + inner_class_names
                for i in range(1, len(class_names) + 1):
                    class_name = ".".join(class_names[:i])
                    class_full_name = f"{method_node.pkg_name}.{class_name}"
                    try:
                        method_dict[class_full_name].append(method_node)
                    except KeyError:
                        method_dict[class_full_name] = [method_node]
        self.method_dict = method_dict
        self.method_id_map = method_id_map

    def get_method(self, method_id) -> Tag | None:
        return self.method_id_map.get(method_id, None)

    def get_predecessors(self, node, edge_type):
        predecessors = [
            u
            for u in self.graph.pred[node]
            for key in self.graph[u][node]
            if self.graph[u][node][key]["rel"] == edge_type
        ]
        return predecessors

    def get_successors(self, node, edge_type):
        successors = [
            v
            for v in self.graph.succ[node]
            for key in self.graph[node][v]
            if self.graph[node][v][key]["rel"] == edge_type
        ]
        return successors

    def get_covered_classes(self):
        """Get a dictionary of all covered classes in the graph"""
        if self.covered_classes_result:
            return self.covered_classes_result

        covered_test_classes = {}
        covered_source_classes = {}
        for node in self.graph.nodes(data=True):
            if node[0].category == "function":
                method_node: Tag = node[0]
                if method_node.is_covered:
                    full_class_name = method_node.outer_class
                    if method_node.is_test:
                        try:
                            if (
                                full_class_name
                                not in covered_test_classes[
                                    method_node.pkg_name
                                ]
                            ):
                                covered_test_classes[
                                    method_node.pkg_name
                                ].append(full_class_name)
                        except KeyError:
                            covered_test_classes[method_node.pkg_name] = [
                                method_node.outer_class
                            ]
                    else:
                        try:
                            if (
                                full_class_name
                                not in covered_source_classes[
                                    method_node.pkg_name
                                ]
                            ):
                                covered_source_classes[
                                    method_node.pkg_name
                                ].append(full_class_name)
                        except KeyError:
                            covered_source_classes[method_node.pkg_name] = [
                                method_node.outer_class
                            ]

        template = (
            "Covered classes grouped with package name:\n\n"
            "Covered source classes:\n{covered_source_classes}\n\n"
            "Covered test classes:\n{covered_test_classes}"
        )
        result = template.format(
            covered_source_classes=json.dumps(covered_source_classes),
            covered_test_classes=json.dumps(covered_test_classes),
        )
        self.covered_classes_result = result
        return result

    def get_possible_methods(
        self,
        class_name: str,
        method_name: str = None,
        covered_only: bool = False,
    ) -> List[Tag]:
        possible_classes = [
            c for c in self.method_dict if c.endswith(class_name)
        ]
        if not possible_classes:
            return None

        possible_methods: List[Tag] = []
        for c in possible_classes:
            class_methods = self.method_dict[c]
            for m in class_methods:
                if covered_only and not m.is_covered:
                    continue
                if method_name:
                    if m.name != method_name:
                        continue
                if m not in possible_methods:
                    possible_methods.append(m)
        return possible_methods

    def get_method_dict(self, methods: List[Tag]) -> Dict[str, List[str]]:
        """Get a dictionary of all methods grouped by class name"""
        result = {}
        for method in methods:
            pkg_name = method.pkg_name
            class_name = method.outer_class
            full_class_name = f"{pkg_name}.{class_name}"
            try:
                result[full_class_name].append(method.method_pos)
            except KeyError:
                result[full_class_name] = [method.method_pos]
        return result

    def get_covered_method_ids_for_class(self, class_name):
        """Get a list of all covered methods of a class"""
        try:
            # precise match
            if class_name in self.method_dict:
                precise_methods = self.method_dict[class_name]
                covered_methods = [m for m in precise_methods if m.is_covered]
                method_dict = self.get_method_dict(covered_methods)
                return json.dumps(method_dict)
            # fuzzy match
            else:
                possible_methods = self.get_possible_methods(
                    class_name, covered_only=True
                )
                if not possible_methods:
                    raise ClassNotFoundError(class_name)
                method_dict = self.get_method_dict(possible_methods)
                return json.dumps(method_dict)
        except Exception as e:
            return e.message

    def get_possible_method_ids(self, false_ids, covered_only=False):
        all_possible_methods = []
        for false_id in false_ids:
            class_name = MethodID.get_class_name(false_id)
            method_name = MethodID.get_method_name(false_id)
            possible_methods = self.get_possible_methods(
                class_name, method_name, covered_only
            )
            if possible_methods:
                all_possible_methods.extend(possible_methods)

        possible_ids = [m.method_id for m in all_possible_methods]
        possible_ids = list(set(possible_ids))
        return possible_ids

    def get_similar_candidates(
        self,
        false_method_id: str,
        possible_method_ids: List[str],
        num_max_candidates: int = 5,
    ):
        def _compute_similarity(method_id_1, method_id_2):
            method_full_name_1 = MethodID.get_method_full_name(method_id_1)
            method_full_name_2 = MethodID.get_method_full_name(method_id_2)
            return SequenceMatcher(
                None, method_full_name_1, method_full_name_2
            ).ratio()

        similarities = []
        for possible_method_id in possible_method_ids:
            similarity = _compute_similarity(
                false_method_id, possible_method_id
            )
            similarities.append((similarity, possible_method_id))
        print(similarities)

        candidates = list(
            map(
                lambda t: t[1],
                sorted(similarities, key=lambda t: t[0], reverse=True),
            )
        )
        if num_max_candidates is not None:
            candidates = candidates[:num_max_candidates]
        return candidates

    def get_method_code_for_id(self, method_id):
        """Get the code of a method"""
        method_tag = self.get_method(method_id)
        if not method_tag:
            # find candidate methods
            possible_method_ids = self.get_possible_method_ids(
                [method_id], covered_only=True
            )
            if not possible_method_ids:
                raise MethodIDNotFoundError(method_id)
            candidate_methods = self.get_similar_candidates(
                method_id, possible_method_ids
            )
            result = f"The method ID is not found, maybe you mean one of the following similar method IDs:\n{candidate_methods}"
            return result

        result = (
            f'"Method ID": "{method_id}"\n'
            f'"Is Covered": "{method_tag.is_covered}"\n'
            f'"Method Code":\n```java\n{method_tag.code}\n```\n'
        )
        return result

    def get_method_ids_contain_string(self, string_content, max_count=50):
        """
        Get a list of all production methods containing a specific string content.
        This tool is particularly useful when the test output contains some special strings.",
        """
        try:
            methods = []
            for method in self.method_id_map.values():
                if method.is_test:  # skip test methods
                    continue
                if string_content in method.code:
                    methods.append(method)
            if not methods:
                raise SnippetNotFoundError(string_content)
        except Exception as e:
            return e.message

        methods = list(set(methods))
        suffix = ""
        if len(methods) > max_count:
            methods = methods[:max_count]
            suffix = f"\n... and {len(methods) - max_count} more methods."
        template = '{idx}. "{method_id}"'
        prefix = "The following methods contain the provided string content:\n"
        result = prefix + "\n".join(
            [
                template.format(
                    idx=i + 1,
                    method_id=m.method_id,
                )
                for i, m in enumerate(methods)
            ]
        )
        return result + suffix
