import argparse
import itertools
import json
import multiprocessing
import pickle
import random
import re
import sys
from argparse import Namespace
from copy import deepcopy
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from pprint import pprint
from typing import List

import networkx as nx
import numpy as np
import pandas as pd

from src.interfaces.method_extractor import JMethod
from src.schema import MethodID, Tag, TestFailure

random.seed(123)

root = Path(__file__).resolve().parent
sys.path.append(str(root))

from src.config import BugInfo
from src.interfaces.d4j import get_failed_tests, get_properties


def get_node_distance(
    graph: nx.MultiDiGraph, node1: Tag, node2: Tag, simple=False
):
    # if the two nodes are the same, return 0
    if node1 == node2:
        return 0

    if simple:
        return 0.1

    dynamic_graph = nx.Graph()
    static_graph = nx.Graph()
    for edge in graph.edges(data=True):
        static_graph.add_edge(edge[0], edge[1])
        if edge[2]["rel"] == "calls":
            dynamic_graph.add_edge(edge[0], edge[1])

    # first try to find the shortest path in dynamic graph
    try:
        shortest_path = nx.shortest_path(dynamic_graph, node1, node2)
        return len(shortest_path)
    except Exception:
        # if not found, try to find the shortest path in static graph
        try:
            shortest_path = nx.shortest_path(static_graph, node1, node2)
            return len(shortest_path)
        except Exception:
            # if still not found, return -1
            return -1


def get_relative_distance(
    call_graph: nx.MultiDiGraph,
    modified_methods: List[JMethod],
    method_id: str,
):
    """
    Get the relative distance between buggy methods and the predicted method.

    For example, if the number of buggy methods is 3,
    the output will be a list contains 3 distances such as [2, 3, 4].
    """

    def fuzzy_match(method_id_1: str, method_id_2: str):
        if method_id_1 == method_id_2:
            return True
        if method_id_1.split(".")[-1] == method_id_2.split(".")[-1]:
            return True
        return False

    buggy_nodes = []
    predict_node = None
    should_find_methods = deepcopy(modified_methods)
    for nodes in call_graph.nodes(data=True):
        if nodes[0].category == "function":
            method_node: Tag = nodes[0]
            for m in should_find_methods:
                if method_node.outer_class in m.class_name:
                    if m.name == method_node.name:
                        if (
                            m.loc[0][0] + 1 <= method_node.line[0]
                            and m.loc[1][0] + 1 >= method_node.line[1]
                        ):
                            buggy_nodes.append(method_node)
                            should_find_methods.remove(m)
                            break
            if fuzzy_match(method_node.method_id, method_id):
                predict_node = method_node

    assert (
        len(should_find_methods) == 0
    ), f"Buggy methods not found in graph: {[m.get_signature() for m in should_find_methods]}"

    if predict_node is None:
        print(f"Predict method not found: {method_id}")
        return []

    distances = []
    for buggy_node in buggy_nodes:
        distance = get_node_distance(
            call_graph,
            buggy_node,
            predict_node,
            simple=True,
        )
        if distance != -1:
            distances.append(distance)
    return distances


def get_distance(
    test_failure_obj: TestFailure,
    ranked_methods: dict,
    call_graph: nx.MultiDiGraph,
):
    evaluate_result = {}
    modified_methods = test_failure_obj.buggy_methods
    for centrality_type in ranked_methods:
        evaluate_result[centrality_type] = []
        methods = [Tag(**m) for m in ranked_methods[centrality_type]]
        method_ids = [m.method_id for m in methods]
        for method_id in method_ids:
            distances = get_relative_distance(
                call_graph, modified_methods, method_id
            )
            if distances:
                rd = max([1 / (d + 1) for d in distances])
                evaluate_result[centrality_type].append(rd)
            else:
                # if the method is not found in the graph
                evaluate_result[centrality_type].append(0)
    return evaluate_result


def get_most_similar_candidate(
    false_method_id: str, possible_method_nodes: List[Tag]
):
    def _compute_similarity(false_method_id, method_node_2):
        method_full_name_1 = MethodID.get_class_full_name(false_method_id)
        method_full_name_2 = MethodID.get_class_full_name(
            method_node_2.method_id
        )
        return SequenceMatcher(
            None, method_full_name_1, method_full_name_2
        ).ratio()

    similarities = []
    for possible_method_node in possible_method_nodes:
        similarity = _compute_similarity(false_method_id, possible_method_node)
        similarities.append((similarity, possible_method_node))

    candidates = list(
        map(
            lambda t: t[1],
            sorted(similarities, key=lambda t: t[0], reverse=True),
        )
    )
    return candidates[0]


def get_possible_method_nodes(graph, false_id):
    false_method_name = MethodID.get_method_name(false_id)
    possible_method_nodes = []
    for node in graph.nodes(data=True):
        if node[0].category == "function":
            method_node: Tag = node[0]
            if not method_node.is_covered:
                continue
            if method_node.name == false_method_name:
                possible_method_nodes.append(method_node)

    possible_method_nodes = list(set(possible_method_nodes))
    return possible_method_nodes


def calculate_node_importance(mdi_graph: nx.MultiDiGraph) -> dict:
    """
    Calculate the importance of each node in the MDI graph using various centrality measures.
    """
    importance_scores = {
        "in_degree_centrality": {},
        "betweenness_centrality": {},
        "eigenvector_centrality": {},
        "pagerank": {},
    }
    graph = nx.DiGraph(mdi_graph)
    if graph.number_of_nodes() == 0:
        print(
            "Warning: The graph is empty. Returning empty importance scores."
        )
        return importance_scores

    # 1. In-Degree
    importance_scores["in_degree_centrality"] = {
        node: graph.in_degree(node, weight="weight") for node in graph.nodes()
    }

    # 2. Betweenness Centrality
    importance_scores["betweenness_centrality"] = nx.betweenness_centrality(
        graph, weight="weight", normalized=False, seed=123
    )

    # 3. Eigenvector Centrality
    try:
        importance_scores["eigenvector_centrality"] = (
            nx.eigenvector_centrality(graph, weight="weight", max_iter=1000)
        )
    except nx.PowerIterationFailedConvergence:
        print(
            "Warning: Eigenvector centrality did not converge. "
            "Consider increasing the max_iter parameter."
        )
        importance_scores["eigenvector_centrality"] = {}

    # 4. PageRank
    importance_scores["pagerank"] = nx.pagerank(graph, weight="weight")

    return importance_scores


def parse_debug_result(
    debug_result_file: Path, call_graph: nx.MultiDiGraph, id_dict: dict
):
    debug_result = json.loads(debug_result_file.read_text())
    all_pred_method_nodes = []
    node_frequency = {}
    n_test = len(debug_result)
    for test_name in debug_result:
        n_process = len(debug_result[test_name])
        for process_id in debug_result[test_name]:
            pred_lines = debug_result[test_name][process_id]["prediction"]
            pred_method_nodes = []
            for line in pred_lines.split("\n"):
                if line:
                    line = line.strip()
                    if line in id_dict:
                        pred_method_nodes.append(id_dict[line])
                    else:
                        possible_method_nodes = get_possible_method_nodes(
                            call_graph, line
                        )
                        if not possible_method_nodes:
                            print(f"Method ID {line} not found in graph")
                        else:
                            pred_method_nodes.append(
                                get_most_similar_candidate(
                                    line, possible_method_nodes
                                )
                            )

            n_pred = len(pred_method_nodes)
            for pred_node in pred_method_nodes:
                try:
                    node_frequency[pred_node] += 1 / (
                        n_pred * n_process * n_test
                    )
                except KeyError:
                    node_frequency[pred_node] = 1 / (
                        n_pred * n_process * n_test
                    )
            all_pred_method_nodes.extend(pred_method_nodes)
    all_pred_method_nodes = list(set(all_pred_method_nodes))
    return all_pred_method_nodes, node_frequency


def parse_debug_result_new(
    debug_result_file: Path, call_graph: nx.MultiDiGraph, id_dict: dict
):
    debug_result = json.loads(debug_result_file.read_text())
    all_pred_method_nodes = []
    node_frequency = {}
    n_test = len(debug_result)
    for test_name in debug_result:
        n_process = len(debug_result[test_name])
        for process_id in debug_result[test_name]:
            pred_lines = debug_result[test_name][process_id]["prediction"]
            pattern = r"(?:\w+\.)*\w+<\d+-\d+>"
            method_ids = list(set(re.findall(pattern, pred_lines)))
            pred_method_nodes = []
            for method_id in method_ids:
                if method_id in id_dict:
                    pred_method_nodes.append(id_dict[method_id])
                else:
                    possible_method_nodes = get_possible_method_nodes(
                        call_graph, method_id
                    )
                    if not possible_method_nodes:
                        print(f"Method ID {method_id} not found in graph")
                    else:
                        pred_method_nodes.append(
                            get_most_similar_candidate(
                                method_id, possible_method_nodes
                            )
                        )

            n_pred = len(pred_method_nodes)
            for pred_node in pred_method_nodes:
                try:
                    node_frequency[pred_node] += 1 / (
                        n_pred * n_process * n_test
                    )
                except KeyError:
                    node_frequency[pred_node] = 1 / (
                        n_pred * n_process * n_test
                    )
            all_pred_method_nodes.extend(pred_method_nodes)
    all_pred_method_nodes = list(set(all_pred_method_nodes))
    return all_pred_method_nodes, node_frequency


def get_ranked(
    bug_info: BugInfo, combined_graph: nx.MultiDiGraph, alpha: float = 0.5
):
    """
    Get the ranked methods from the combined graph.
    """

    # solve the combined_graph, only keep the 'calls', 'may_calls' edges and the 'function' nodes
    call_graph = nx.MultiDiGraph()
    id_dict = {}
    for node in combined_graph.nodes(data=True):
        if node[0].category == "function":
            id_dict[node[0].method_id] = node[0]
            call_graph.add_node(node[0])

    for edge in combined_graph.edges(data=True):
        if edge[2]["rel"] in ["calls", "may_calls"]:
            if (
                edge[0].category == "function"
                and edge[1].category == "function"
            ):
                call_graph.add_edge(edge[0], edge[1], rel=edge[2]["rel"])
    print_graph_info(call_graph, "Call Graph")

    ranked_result_file = bug_info.res_path / "method_rank_list.json"
    # TODO: uncomment this to use cached results
    if ranked_result_file.exists():
        with ranked_result_file.open("r") as f:
            ranked_methods = json.load(f)
            return ranked_methods, call_graph

    debug_result_file = bug_info.res_path / "debug_result.json"
    all_pred_method_nodes, node_frequency = parse_debug_result(
        debug_result_file, call_graph, id_dict
    )

    # normalize the frequency using min-max scaling
    node_frequency = normalize(node_frequency)

    # get subgraph
    subgraph = get_connected_subgraph(call_graph, all_pred_method_nodes)

    # calculate importance
    importance_scores = calculate_node_importance(subgraph)

    # filter out nodes that are not in all_pred_method_nodes
    for centrality_type in importance_scores:
        importance_scores[centrality_type] = {
            node: score
            for node, score in importance_scores[centrality_type].items()
            if node in all_pred_method_nodes
        }

    # normalize the importance scores
    for centrality_type in importance_scores:
        importance_scores[centrality_type] = normalize(
            importance_scores[centrality_type]
        )

    # combine centrality scores with frequency scores
    result = {}
    for centrality_type in importance_scores:
        if centrality_type not in result:
            result[centrality_type] = {}
        for node, centrality_score in importance_scores[
            centrality_type
        ].items():
            frequency_score = node_frequency[node]
            final_score = (alpha * centrality_score) + (
                (1 - alpha) * frequency_score
            )
            result[centrality_type][node] = final_score

    # do not compute the node importance scores if the subgraph is too small
    if subgraph.number_of_nodes() < 20:
        for centrality_type in importance_scores:
            result[centrality_type] = node_frequency

    # add the frequency score to the result
    result["frequency"] = node_frequency

    ranked_methods = {}
    for centrality_type in result:
        ranked_methods[centrality_type] = []
        suspicious_method_list = []
        for node in result[centrality_type]:
            suspicious_method_list.append(
                (node, result[centrality_type][node])
            )
        if not suspicious_method_list:
            continue
        suspicious_method_list = sorted(
            suspicious_method_list, key=lambda x: x[1], reverse=True
        )
        method_list = list(zip(*suspicious_method_list))[0]
        method_list = [asdict(m) for m in method_list]
        ranked_methods[centrality_type] = method_list

    with ranked_result_file.open("w") as f:
        json.dump(ranked_methods, f, indent=4)
    return ranked_methods, call_graph


def get_ranked_raw(bug_info: BugInfo, combined_graph: nx.MultiDiGraph):
    """
    Get the ranked methods from the combined graph.
    """

    # solve the combined_graph, only keep the 'calls', 'may_calls' edges and the 'function' nodes
    call_graph = nx.MultiDiGraph()
    id_dict = {}
    for node in combined_graph.nodes(data=True):
        if node[0].category == "function":
            id_dict[node[0].method_id] = node[0]
            call_graph.add_node(node[0])

    for edge in combined_graph.edges(data=True):
        if edge[2]["rel"] in ["calls", "may_calls"]:
            if (
                edge[0].category == "function"
                and edge[1].category == "function"
            ):
                call_graph.add_edge(edge[0], edge[1], rel=edge[2]["rel"])
    print_graph_info(call_graph, "Call Graph")

    ranked_result_file = bug_info.res_path / "method_rank_list_raw.json"
    # TODO: uncomment this to use cached results
    if ranked_result_file.exists():
        with ranked_result_file.open("r") as f:
            ranked_methods = json.load(f)
            return ranked_methods, call_graph

    debug_result_file = bug_info.res_path / "debug_result.json"
    all_pred_method_nodes, node_frequency = parse_debug_result(
        debug_result_file, call_graph, id_dict
    )

    method_list = [asdict(m) for m in all_pred_method_nodes]
    ranked_methods = {
        "frequency": method_list,
    }

    with ranked_result_file.open("w") as f:
        json.dump(ranked_methods, f, indent=4)
    return ranked_methods, call_graph


def min_max_normalize(node_dict: dict) -> dict:
    """Normalize the node dictionary using min-max scaling."""
    min_value = min(node_dict.values())
    max_value = max(node_dict.values())
    normalized_dict = {
        node: (value - min_value) / (max_value - min_value)
        for node, value in node_dict.items()
    }
    return normalized_dict


def normalize(node_dict: dict) -> dict:
    """Normalize the node dictionary using Z-Score + Sigmoid scaling."""
    if len(node_dict) == 0:
        return {}
    mean = sum(node_dict.values()) / len(node_dict)
    std_dev = (
        sum((x - mean) ** 2 for x in node_dict.values()) / len(node_dict)
    ) ** 0.5

    normalized_dict = {}
    for node, value in node_dict.items():
        z_score = (value - mean) / std_dev if std_dev != 0 else 0
        normalized_value = 1 / (1 + np.exp(-z_score))  # Sigmoid function
        normalized_dict[node] = normalized_value

    return normalized_dict


def get_connected_subgraph(graph: nx.MultiDiGraph, nodes: List[Tag]):
    """Get the connected subgraph from the combined graph based on the given nodes."""
    path_nodes = set(nodes)
    for u, v in itertools.combinations(nodes, 2):
        if graph.has_edge(u, v) or graph.has_edge(v, u):
            path_nodes.add(u)
            path_nodes.add(v)
            continue
        try:
            path = nx.shortest_path(graph, u, v)
            path_nodes.update(path)
        except nx.NetworkXNoPath:
            pass

    subgraph = graph.subgraph(path_nodes)
    sorted_nodes = sorted(
        subgraph.nodes(data=True), key=lambda x: x[0].method_id
    )
    sorted_edges = sorted(
        subgraph.edges(data=True),
        key=lambda x: (x[0].method_id, x[1].method_id, x[2]["rel"]),
    )
    stable_subgraph = nx.MultiDiGraph()
    stable_subgraph.add_nodes_from(sorted_nodes)
    stable_subgraph.add_edges_from(sorted_edges)
    print_graph_info(stable_subgraph, "Connected Subgraph")
    return stable_subgraph


def print_graph_info(graph: nx.MultiDiGraph, desc: str = ""):
    """Print the information of the combined graph."""
    if desc:
        print(f"Graph Info: {desc}")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")


def evaluate(project, bugID, config, use_raw):
    args = Namespace(project=project, bugID=bugID, config=config)
    bug_info = BugInfo(args, eval=True)
    result_path: Path = bug_info.evaluation_path / Path(config).stem
    if not result_path.exists():
        result_path.mkdir(parents=True, exist_ok=True)
    if use_raw:
        result_file = result_path / f"{project}-{bugID}_raw.json"
    else:
        result_file = result_path / f"{project}-{bugID}.json"
    # TODO: uncomment this to use cached results
    if result_file.exists():
        return

    print(f"Evaluating {project}-{bugID}")
    # collect basic bug information from cache
    # For preprocessing please run `preprocess.py`
    get_properties(bug_info)
    test_failure_obj = get_failed_tests(bug_info)

    graph_file = bug_info.bug_path / "combined_graph.pkl"
    with graph_file.open("rb") as f:
        combined_graph = pickle.load(f)

    # combine the result for all test cases to get the ranked methods
    if use_raw:
        ranked_methods, call_graph = get_ranked_raw(bug_info, combined_graph)
    else:
        ranked_methods, call_graph = get_ranked(bug_info, combined_graph)

    # get the distance between the ranked methods and the buggy methods
    distances = get_distance(test_failure_obj, ranked_methods, call_graph)
    with result_file.open("w") as f:
        json.dump(distances, f, indent=4)


def print_result(bug_names, config_file, use_raw):
    root_path = Path(__file__).resolve().parent
    config_name = Path(config_file).stem
    output = {}
    top_5_bugs = []
    for bug_name in bug_names:
        proj, bug_id = bug_name.split("_")
        if use_raw:
            distance_file = (
                root_path
                / "EvaluationResult"
                / config_name
                / f"{proj}-{bug_id}_raw.json"
            )
        else:
            distance_file = (
                root_path
                / "EvaluationResult"
                / config_name
                / f"{proj}-{bug_id}.json"
            )
        if not distance_file.exists():
            raise FileNotFoundError(f"{distance_file} not found, please check")
        with distance_file.open("r") as f:
            distance = json.load(f)

        for score_type in distance:
            if score_type not in output:
                output[score_type] = {}
                output[score_type]["all"] = {
                    "Top-1": 0,
                    "Top-3": 0,
                    "Top-5": 0,
                    "Top-10": 0,
                    "RD@1": [],
                    "RD@3": [],
                    "RD@5": [],
                }

            if proj not in output[score_type]:
                output[score_type][proj] = {
                    "Top-1": 0,
                    "Top-3": 0,
                    "Top-5": 0,
                    "Top-10": 0,
                    "RD@1": [],
                    "RD@3": [],
                    "RD@5": [],
                }
            for idx, d in enumerate(distance[score_type]):
                if d == 1.0:
                    if idx == 0:
                        output[score_type][proj]["Top-1"] += 1
                        output[score_type]["all"]["Top-1"] += 1
                    if idx < 3:
                        output[score_type][proj]["Top-3"] += 1
                        output[score_type]["all"]["Top-3"] += 1
                    if idx < 5:
                        output[score_type][proj]["Top-5"] += 1
                        output[score_type]["all"]["Top-5"] += 1
                        top_5_bugs.append(f"{proj}-{bug_id}")
                    if idx < 10:
                        output[score_type][proj]["Top-10"] += 1
                        output[score_type]["all"]["Top-10"] += 1
                    break
            for i in [1, 3, 5]:
                if distance[score_type][:i]:
                    output[score_type][proj][f"RD@{i}"].append(
                        max(distance[score_type][:i])
                    )
                else:
                    print(f"Warning: {proj}-{bug_id} no results!")

    for score_type in output:
        for proj in output[score_type]:
            for i in [1, 3, 5]:
                output[score_type][proj][f"RD@{i}"] = sum(
                    output[score_type][proj][f"RD@{i}"]
                )

    top_5_file = root_path / "utils" / f"{config_name}_top_5_bugs.txt"
    with open(top_5_file, "w") as f:
        f.write("\n".join(top_5_bugs))

    pprint(output)


def main(dataset_file, config_file, processes, use_raw):
    df = pd.read_csv(dataset_file, header=None)
    bug_names = df.iloc[:, 0].tolist()

    # TODO: evaluation: control bug for test
    # bug_names = ["Mockito_12"]

    if processes > 1:
        with multiprocessing.Pool(processes=processes) as pool:
            async_results = []
            for bug_name in bug_names:
                proj, bug_id = bug_name.split("_")
                async_result = pool.apply_async(
                    evaluate, (proj, bug_id, config_file, use_raw)
                )
                async_results.append(async_result)

            for i, async_result in enumerate(async_results):
                try:
                    async_result.get()
                except Exception as e:
                    print(f"{bug_names[i]} error: {str(e)}")
                    return
    else:
        for bug_name in bug_names:
            proj, bug_id = bug_name.split("_")
            evaluate(proj, bug_id, config_file, use_raw)

    print_result(bug_names, config_file, use_raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all bugs")
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset file",
        # default="dataset/JacksonDatabind.csv",
        # default="dataset/Mockito.csv",
        # default="dataset/Time.csv",
        default="dataset/all_bugs.csv",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="config file",
        # default="config/default.yml",
        default="config/default_path6.yml",
        # default="config/default_path_select.yml",
        # default="config/default_path_select_path6.yml",
    )
    parser.add_argument(
        "--processes",
        type=int,
        help="processes",
        default=10,
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="use raw ranked methods",
    )
    args = parser.parse_args()
    main(args.dataset, args.config, args.processes, args.raw)
