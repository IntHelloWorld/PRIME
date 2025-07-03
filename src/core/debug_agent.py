import json
import pickle
import time
from dataclasses import dataclass
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Any, Dict, List

from src.config import BugInfo
from src.core.llm_backend import AnthropicBackend, OpenAIBackend
from src.core.memory import Memory
from src.core.search_agent import SearchAgent
from src.core.utils import extract_json_block
from src.interfaces.d4j import (
    get_test_case_dataset_path,
    get_test_case_output_path,
)
from src.repograph.graph_searcher import RepoSearcher
from src.schema import DebugInput, SearchInput, TestFailure
from src.utils import Timer


@dataclass
class DebugState:
    """Manage debug status and handle communication between agents"""

    bug_name: str
    input: DebugInput

    def prepare_search_input(self) -> SearchInput:
        """Prepare the input for the search agent"""
        return SearchInput(
            test_name=self.input.test_name,
            test_code=self.input.test_code,
            error_message=self.input.error_message,
            output_path=self.input.output_path,
        )


def debug_process(bug_info: BugInfo, debug_state: DebugState):
    """Debug process for a single test case"""
    searcher = RepoSearcher(debug_state.input.repo_graph)
    search_agent = SearchAgent(bug_info=bug_info, searcher=searcher)

    # perform search
    search_input = debug_state.prepare_search_input()
    with Timer(
        bug_info.logger,
        f"{search_input.test_name} - search",
    ):
        search_file = search_input.output_path / "search.json"
        if search_file.exists():
            print(
                f"Search result file {search_file} already exists, skipping search."
            )
            return
        search_agent.run(search_input)


class DebugAgent:
    def __init__(self, bug_info: BugInfo):
        self.bug_info = bug_info
        self.debug_states: Dict[int, DebugState] = {}
        self.num_process = self.bug_info.config.hyper.debug_workers

        self.org = bug_info.config.search_model.org
        assert self.org in ["openai", "anthropic"]
        if self.org == "openai":
            self.llm_backend = OpenAIBackend
        else:
            self.llm_backend = AnthropicBackend

    def run_multiprocess(self):
        # Create a pool of processes
        with Pool(processes=self.num_process) as pool:
            # Start the debug processes
            async_results: List[AsyncResult] = []
            for i, debug_state in self.debug_states.items():
                result = pool.apply_async(
                    debug_process,
                    args=(self.bug_info, debug_state),
                )
                async_results.append(result)

            for r in async_results:
                try:
                    r.get()
                except Exception as e:
                    self.bug_info.logger.error(f"Error in subprocess: {e}")

    def run_singleprocess(self):
        for _, debug_state in self.debug_states.items():
            debug_process(self.bug_info, debug_state)

    def prepare_debug_inputs(
        self, test_failure: TestFailure
    ) -> List[DebugInput]:
        debug_inputs = []
        test_cases = []
        for test_class in test_failure.test_classes:
            test_cases.append(test_class.test_cases)

        # Limit the number of test cases
        # We try to keep test cases from different test classes
        num_test_cases = sum(len(cs) for cs in test_cases)
        new_test_cases = []
        max_test_cases = self.bug_info.config.hyper.max_test_cases
        if num_test_cases > max_test_cases:
            while len(new_test_cases) < max_test_cases:
                for cs in test_cases:
                    if cs:
                        new_test_cases.append(cs.pop())
                        if len(new_test_cases) == max_test_cases:
                            break
        else:
            new_test_cases = [tc for cs in test_cases for tc in cs]

        for test_case in new_test_cases:
            # TODO: Control single test case for test
            # if test_case.test_method_name != "testIssue787":
            #     continue
            dataset_path = get_test_case_dataset_path(self.bug_info, test_case)
            repo_graph_file = dataset_path / "repograph.pkl"
            with repo_graph_file.open("rb") as f:
                repo_graph = pickle.load(f)
            loaded_classes_file = dataset_path / "loaded_classes.txt"
            loaded_classes = loaded_classes_file.read_text().split("\n")

            stack_trace_file = dataset_path / "stack_trace.txt"
            test_output_file = dataset_path / "test_output.txt"
            test_name = test_case.name
            test_code = test_case.test_method.text
            if self.bug_info.config.ablation.test_output:
                error_message = stack_trace_file.read_text()
            elif self.bug_info.config.ablation.stack_trace:
                error_message = test_output_file.read_text()
            else:
                error_message = (
                    stack_trace_file.read_text()
                    + "\n\n"
                    + test_output_file.read_text()
                )
            output_path = get_test_case_output_path(self.bug_info, test_case)
            if not output_path.exists():
                output_path.mkdir(parents=True)

            debug_inputs.append(
                DebugInput(
                    test_name=test_name,
                    test_code=test_code,
                    error_message=error_message,
                    repo_graph=repo_graph,
                    loaded_classes=loaded_classes,
                    output_path=output_path,
                )
            )
        return debug_inputs

    # def combine_test_case_results(self, inputs: List[DebugInput]):
    #     """Combine the results for all test cases to generate a final suspicious methods rank list"""
    #     # check if the debug report for all test cases are generated
    #     for input in inputs:
    #         search_file = input.output_path / "search.json"
    #         if not search_file.exists():
    #             raise Exception(
    #                 f"Search result file {search_file} is not found."
    #             )
    #         if not json.loads(search_file.read_text())["debug_report"]:
    #             raise Exception(
    #                 f"Search result file {search_file} does not contain debug report."
    #             )

    #     # check if already generated the final report
    #     result_file: Path = self.bug_info.res_path / "debug_result.json"
    #     if result_file.exists():
    #         return

    #     llm = self.llm_backend(
    #         api_key=self.bug_info.config.search_model.api_key,
    #         base_url=self.bug_info.config.search_model.base_url,
    #     )
    #     memory = Memory(
    #         system_prompt=SUMMARIZE_SYSTEM_PROMPT,
    #         model_name=self.bug_info.config.search_model.model,
    #     )
    #     memory.add_message(
    #         {
    #             "role": "user",
    #             "content": self.get_result_report(inputs),
    #         }
    #     )

    #     result = None
    #     combined_graph_file = self.bug_info.bug_path / "combined_graph.pkl"
    #     with combined_graph_file.open("rb") as f:
    #         combined_graph = pickle.load(f)
    #     combined_searcher = RepoSearcher(combined_graph)
    #     while True:
    #         response = llm.call(
    #             messages=memory.get_messages(),
    #             model=self.bug_info.config.search_model.model,
    #             **self.bug_info.config.search_model.llm_args.asdict(),
    #         )

    #         message = self.llm_backend.get_msg(response)
    #         message_text = self.llm_backend.get_msg_text(message)

    #         memory.add_message({"role": "assistant", "content": message_text})
    #         result = extract_json_block(message_text)
    #         if result is None:
    #             error_message = f"Reponse format error, please return a JSON format verification result wrapped with ```json...``` block."
    #             memory.add_message(
    #                 {"role": "user", "content": error_message}, "retry"
    #             )
    #             continue

    #         suspicious_methods = json.loads(result)
    #         method_ids = [m["method_id"] for m in suspicious_methods]
    #         false_ids = []
    #         for method_id in method_ids:
    #             if combined_searcher.get_method(method_id) is None:
    #                 false_ids.append(method_id)
    #         if false_ids:
    #             possible_ids = combined_searcher.get_possible_method_ids(
    #                 false_ids
    #             )
    #             if not possible_ids:
    #                 possible_ids = "No possible method IDs found."
    #             error_message = (
    #                 "The following method IDs in your result are not found:\n"
    #                 + json.dumps(false_ids)
    #                 + "\n\nHere are some possible method IDs:\n"
    #                 + json.dumps(possible_ids)
    #                 + "\n\nPlease correct the method IDs or delete the non-existent method IDs and regenerate the JSON format verification result."
    #             )
    #             memory.add_message(
    #                 {"role": "user", "content": error_message}, "retry"
    #             )
    #             continue

    #         input_tokens, output_tokens = self.llm_backend.get_tokens(response)
    #         memory.add_cost(output_tokens, input_tokens)
    #         break

    #     result_dict = {
    #         "results": suspicious_methods,
    #         "memory": memory.serialize(),
    #     }
    #     result_file.write_text(json.dumps(result_dict, indent=4))

    def get_debug_result(self):
        debug_result = {}
        search_result_files = self.bug_info.res_path.rglob("search.json")
        for search_result_file in search_result_files:
            test_method_name = search_result_file.parent.name
            test_class_name = search_result_file.parent.parent.name.replace(
                "_", "."
            )
            test_name = f"{test_class_name}::{test_method_name}"
            if test_name not in debug_result:
                debug_result[test_name] = {}
            search_result = json.loads(search_result_file.read_text())
            for process_id in search_result:
                debug_result[test_name][process_id] = {
                    "prediction": search_result[process_id]["memory"][
                        "messages"
                    ][-1]["content"],
                    "debug_report": search_result[process_id]["debug_report"],
                }

        debug_result_file = self.bug_info.res_path / "debug_result.json"
        debug_result_file.write_text(json.dumps(debug_result, indent=2))

    def run(self, test_failure: TestFailure) -> List[Dict[str, Any]]:
        # Prepare the debug inputs
        inputs = self.prepare_debug_inputs(test_failure)

        # Create debug states and run the debug process
        for i, debug_input in enumerate(inputs):
            debug_state = DebugState(
                bug_name=self.bug_info.bug_name,
                input=debug_input,
            )
            self.debug_states[i] = debug_state

        if self.num_process > 1:
            self.run_multiprocess()
        else:
            self.run_singleprocess()

        # Get the final debugging result
        self.get_debug_result()
        # with Timer(self.bug_info.logger, "Generate final report"):
        #     self.combine_test_case_results(inputs)
