import copy
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from anthropic.types import ToolUseBlock
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)

from src.config import BugInfo
from src.core.llm_backend import AnthropicBackend, LLMBackend, OpenAIBackend
from src.core.memory import Memory
from src.core.path_selector import PathSelector
from src.core.prompt import (
    DEBUGGING_PROMPT,
    FAULT_LOCALIZATION_PROMPT,
    PRUNE_PROMPT,
    PRUNE_USER_PROMPT,
    REACH_MAX_TOOL_CALLS,
    SEARCH_AGENT_TOOLS_ANTHROPIC,
    SUMMARIZATION_PROMPT,
    TOOLS,
    USER_PROMPT,
)
from src.repograph.graph_searcher import RepoSearcher
from src.schema import SearchInput

DEFAULT_FUNCTION = {
    "content": "First, let's look at all the classes covered by the failing test to understand the debugging scope.",
    "refusal": None,
    "role": "assistant",
    "audio": None,
    "function_call": None,
    "tool_calls": [
        {
            "id": "call_default",
            "function": {
                "arguments": "{}",
                "name": "get_covered_classes",
            },
            "type": "function",
        }
    ],
}

DEFAULT_FUNCTION_NO_THOUGHT = {
    "content": None,
    "refusal": None,
    "role": "assistant",
    "audio": None,
    "function_call": None,
    "tool_calls": [
        {
            "id": "call_default",
            "function": {
                "arguments": "{}",
                "name": "get_covered_classes",
            },
            "type": "function",
        }
    ],
}


@dataclass
class ThreadState:
    input: SearchInput
    llm: LLMBackend
    memory: Memory
    id: str
    function_calls: List[str] = field(default_factory=list)
    verify_rounds: int = 0

    @property
    def num_function_calls(self):
        return len(self.function_calls)


class SearchAgent:

    def __init__(self, bug_info: BugInfo, searcher: RepoSearcher):
        self.bug_info = bug_info
        self.searcher = searcher

        # search settings
        self.cur_paths = 1
        self.max_paths = self.bug_info.config.hyper.max_search_paths
        self.max_tool_calls = bug_info.config.hyper.max_tool_calls

        # branch settings
        self.use_select_path = bug_info.config.hyper.use_select_path
        self.branch_count = self.bug_info.config.hyper.branch_count
        self.num_branch_sampling = (
            self.bug_info.config.hyper.num_branch_sampling
        )

        # prune settings
        self.use_prune = self.bug_info.config.hyper.use_prune
        self.prune_step = self.bug_info.config.hyper.prune_step
        self.prune_threshold = self.bug_info.config.hyper.prune_threshold
        self.num_prune_sampling = self.bug_info.config.hyper.num_prune_sampling

        self.debug_prompt = DEBUGGING_PROMPT.format(
            max_tool_calls=self.max_tool_calls,
        )
        self.default_function = DEFAULT_FUNCTION

        self.functions = {
            "get_covered_classes": self.searcher.get_covered_classes,
            "get_covered_method_ids_for_class": self.searcher.get_covered_method_ids_for_class,
            "get_method_code_for_id": self.searcher.get_method_code_for_id,
            "search_covered_class_full_name": self.searcher.search_covered_class_full_name,
            "search_covered_method_id": self.searcher.search_covered_method_id,
        }

        self.org = bug_info.config.search_model.org
        assert self.org in ["openai", "anthropic"]
        if self.org == "openai":
            self.llm_backend = OpenAIBackend
            self.tool_set = TOOLS
        else:
            self.llm_backend = AnthropicBackend
            self.tool_set = SEARCH_AGENT_TOOLS_ANTHROPIC

        self.path_pool = []
        if self.use_select_path:
            self.path_selector = PathSelector(
                **bug_info.config.embedding_model.asdict()
            )

        self.threads: Dict[int, ThreadState] = {}
        self.futures: List[Future] = []
        self.thread_counter = 0
        self.thread_lock = threading.Lock()
        self.search_workers = bug_info.config.hyper.search_workers
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.search_workers,
        )
        self.futures = []

    def create_thread(self, input: SearchInput, parent_id=None) -> ThreadState:
        with self.thread_lock:
            thread_id = self.thread_counter
            if parent_id is not None:
                parent_thread = self.threads[parent_id]
                self.threads[thread_id] = ThreadState(
                    input=input,
                    llm=self.llm_backend(
                        api_key=self.bug_info.config.search_model.api_key,
                        base_url=self.bug_info.config.search_model.base_url,
                    ),
                    memory=copy.deepcopy(parent_thread.memory),
                    id=f"{parent_thread.id}-{thread_id}",
                    function_calls=copy.deepcopy(parent_thread.function_calls),
                )
            else:
                self.threads[thread_id] = ThreadState(
                    input=input,
                    llm=self.llm_backend(
                        api_key=self.bug_info.config.search_model.api_key,
                        base_url=self.bug_info.config.search_model.base_url,
                    ),
                    memory=Memory(
                        self.debug_prompt,
                        self.bug_info.config.search_model.model,
                    ),
                    id=str(thread_id),
                )
            self.thread_counter += 1
            return thread_id

    def save_memory(self):
        memory_cache = {}
        for thread in self.threads.values():
            memory_cache[thread.id] = {
                "memory": thread.memory.serialize(),
                "debug_report": thread.memory.get_debug_report(),
            }
        # save the memory cache to a file
        search_file = thread.input.output_path / "search.json"
        search_file.write_text(json.dumps(memory_cache, indent=2))
        self.bug_info.logger.info(f"Save search memory cache to {search_file}")

    def load_memory(self, thread: ThreadState):
        # load the memory cache from a file
        search_file = thread.input.output_path / "search.json"
        memory_cache = json.loads(search_file.read_text())
        thread_id = list(memory_cache.keys())[0]

        cached_messages = []
        function_calls = []
        for message in memory_cache[thread_id]["memory"]["messages"]:
            if "tool_calls" in message:
                function_name = message["tool_calls"][0]["function"]["name"]
                if function_name == "nominate_suspicious_method":
                    break
                function_calls.append(function_name)
            cached_messages.append(message)

        thread.id = thread_id
        thread.function_calls = function_calls
        for cached_message in cached_messages:
            thread.memory.add_message(
                self.llm_backend.recover_msg(cached_message)
            )

        self.bug_info.logger.info(
            f"Load search memory cache from {search_file}"
        )

    def init_memory(self, input: SearchInput, thread_id: str) -> None:
        thread = self.threads[thread_id]

        if self.bug_info.config.hyper.use_ablation:
            # load the memory cache from a file
            self.load_memory(thread)
            return

        default_messages = [
            {
                "role": "user",
                "content": USER_PROMPT.format(**asdict(input)),
            },
            self.llm_backend.recover_msg(self.default_function),
            {
                "role": "tool",
                "tool_call_id": "call_default",
                "content": self.functions["get_covered_classes"](),
            },
        ]
        for message in default_messages:
            thread.memory.add_message(message)

        thread.function_calls.append("get_covered_classes")

    def run(self, input: SearchInput):
        entry_thread_id = self.create_thread(input)
        self.init_memory(input, entry_thread_id)

        entry_future = self.thread_pool.submit(
            self.run_thread, entry_thread_id
        )
        entry_future.thread_id = entry_thread_id
        self.futures.append(entry_future)

        # wait for all futures to finish
        while not all(future.done() for future in self.futures):
            time.sleep(1)

        # check for exceptions in the futures
        has_exception = False
        for future in self.futures:
            try:
                result = future.result()
            except Exception as e:
                self.bug_info.logger.error(
                    (
                        f"<{self.threads[future.thread_id].input.test_name}> "
                        f"- encountered an exception: {e}"
                    ),
                    exc_info=True,
                )
                has_exception = True

        if has_exception:
            raise Exception(
                "Search agent encountered exceptions. "
                "Please check the logs for details."
            )
        self.save_memory()

    def execute_function(
        self,
        tool_call: ChatCompletionMessageToolCall | ToolUseBlock,
    ):
        function_args = self.llm_backend.get_tool_args(tool_call)
        function_name = self.llm_backend.get_tool_name(tool_call)
        function_to_call = self.functions[function_name]
        function_response = function_to_call(**function_args)
        return function_response

    def remove_duplicate_msg(
        self,
        messages: List[ChatCompletionMessage],
    ):
        """
        Remove duplicate tool calls from the messages,
        and keep a single end message without tool call.
        """
        seen_tool_calls = set()
        has_end_msg = False  # We only need one message without a tool call
        tool_msgs = []
        end_msg = None
        for message in messages:
            tool_call = self.llm_backend.get_tool_call(message)
            if tool_call:
                function = tool_call.function
                function_str = str(function)
                if function_str in seen_tool_calls:
                    # Skip duplicate tool calls
                    continue
                seen_tool_calls.add(function_str)
                tool_msgs.append(message)
            else:
                if not has_end_msg:
                    # Keep messages without tool calls
                    end_msg = message
                    has_end_msg = True
        return tool_msgs, end_msg

    def extend_nodes(
        self,
        thread_id: int,
        messages: List[ChatCompletionMessage],
    ) -> None:
        """Extend the search paths based on the messages."""
        tool_messages, end_message = self.remove_duplicate_msg(messages)

        # check how many paths we can extend
        with self.thread_lock:
            max_parallel = self.branch_count
            if self.cur_paths >= self.max_paths:
                # do not create new threads
                max_parallel = 1
            else:
                max_new_paths = min(len(tool_messages), self.branch_count) - 1
                max_parallel = (
                    min(max_new_paths, self.max_paths - self.cur_paths) + 1
                )
            if max_parallel > 1:
                self.cur_paths += max_parallel - 1

        if self.use_select_path:
            # we only perform path selection for tool messages
            # note that we prefer end message by default
            top_n = max_parallel - 1 if end_message else max_parallel
            if 0 < top_n < len(tool_messages):
                selected_indices, selected_embeddings = (
                    self.path_selector.select_paths(
                        [str(msg) for msg in tool_messages],
                        self.path_pool,
                        top_n=top_n,
                    )
                )
                selected_tool_messages = [
                    tool_messages[i] for i in selected_indices
                ]
                self.path_pool.extend(
                    list(zip(selected_tool_messages, selected_embeddings))
                )
            elif top_n == 0:
                selected_tool_messages = []
            else:
                selected_tool_messages = tool_messages
                self.path_pool.extend(
                    self.path_selector.embed_paths(
                        [str(msg) for msg in tool_messages]
                    )
                )
        else:
            # if we do not use path selection, we just use all tool messages
            selected_tool_messages = tool_messages[:max_parallel]

        # extend the search paths
        for message in selected_tool_messages:
            tool_call = self.llm_backend.get_tool_call(message)

            if tool_call:
                # create a new thread for each message
                new_thread_id = self.create_thread(
                    input=copy.deepcopy(self.threads[thread_id].input),
                    parent_id=thread_id,
                )
                future = self.thread_pool.submit(
                    self.run_thread,
                    new_thread_id,
                    message,
                )
                future.thread_id = new_thread_id
                self.futures.append(future)

        # terminate the search path
        all_need_extend = True
        if end_message:
            all_need_extend = False
            self.run_fault_localization(thread_id, end_message)

        # if all paths need to be extended,
        # remove the parent thread and its futures
        if all_need_extend:
            self.remove_thread(thread_id)

    def remove_thread(self, thread_id: str) -> None:
        """
        Remove a thread and its associated futures.
        """
        with self.thread_lock:
            # remove the parent thread and its futures
            self.threads.pop(thread_id)
            for future in self.futures:
                if future.thread_id == thread_id:
                    self.futures.remove(future)

    def run_function_calls(
        self, thread_id: str, message: ChatCompletionMessage
    ) -> None:
        thread = self.threads[thread_id]
        tool_call = self.llm_backend.get_tool_call(message)
        tool_call_name = self.llm_backend.get_tool_name(tool_call)
        self.bug_info.logger.info(
            f"{self.bug_info.bug_name} "
            f"- <{thread.input.test_name}> "
            f"- thread {thread.id} "
            f"- call function {tool_call_name}"
        )

        try:
            tool_call_result = self.execute_function(tool_call)
            thread.function_calls.append(
                self.llm_backend.get_tool_name(tool_call)
            )
        except Exception as e:
            self.bug_info.logger.error(
                f"{self.bug_info.bug_name} "
                f"- <{thread.input.test_name}> "
                f"- thread {thread.id} "
                f"- Error when executing function {tool_call}: {str(e)}"
            )
            tool_call_result = (
                "Function cannot be called with the given arguments. "
                "Please try something else."
            )
            thread.function_calls.append("retry")
        thread.memory.add_message(message)
        thread.memory.add_message(
            self.llm_backend.get_tool_result_msg(tool_call, tool_call_result)
        )

    def run_fault_localization(
        self, thread_id: str, message: ChatCompletionMessage
    ) -> None:
        thread = self.threads[thread_id]
        self.bug_info.logger.info(
            f"{self.bug_info.bug_name} "
            f"- <{thread.input.test_name}> "
            f"- thread {thread.id} - start fault localization"
        )

        message_text = self.llm_backend.get_msg_text(message)
        terminate_message = {
            "role": "assistant",
            "content": message_text,
        }
        thread.memory.add_message(terminate_message)

        # summarize the debugging report
        summarization_message = {
            "role": "user",
            "content": SUMMARIZATION_PROMPT,
        }
        thread.memory.add_message(summarization_message)
        summary_response = thread.llm.call(
            messages=thread.memory.get_messages(),
            model=self.bug_info.config.search_model.model,
            **self.bug_info.config.search_model.llm_args.asdict(),
        )
        summary_message = self.llm_backend.get_msg(summary_response)
        summary_text = self.llm_backend.get_msg_text(summary_message)
        input_tokens, output_tokens = self.llm_backend.get_tokens(
            summary_response
        )
        thread.memory.add_cost(output_tokens, input_tokens)
        thread.memory.add_message(
            {
                "role": "assistant",
                "content": summary_text,
            }
        )

        # get the bug localization result
        fault_localization_message = {
            "role": "user",
            "content": FAULT_LOCALIZATION_PROMPT,
        }
        thread.memory.add_message(fault_localization_message)
        fl_response = thread.llm.call(
            messages=thread.memory.get_messages(),
            model=self.bug_info.config.search_model.model,
            **self.bug_info.config.search_model.llm_args.asdict(),
        )
        fl_message = self.llm_backend.get_msg(fl_response)
        fl_result_text = self.llm_backend.get_msg_text(fl_message)
        input_tokens, output_tokens = self.llm_backend.get_tokens(fl_response)
        thread.memory.add_cost(output_tokens, input_tokens)
        thread.memory.add_message(
            {
                "role": "assistant",
                "content": fl_result_text,
            }
        )

    def prune_search_paths(self, thread_id: str):
        """
        Prune the search paths based on the current thread's memory.
        """
        thread = self.threads[thread_id]
        self.bug_info.logger.debug(
            f"{self.bug_info.bug_name} "
            f"- <{thread.input.test_name}> "
            f"- Process {thread.id} - start pruning search paths"
        )

        debugging_process = thread.memory.to_debug_process()
        input_messages = [
            {"role": "system", "content": PRUNE_PROMPT},
            {
                "role": "user",
                "content": PRUNE_USER_PROMPT.format(
                    **asdict(thread.input),
                    debugging_process=debugging_process,
                ),
            },
        ]

        response = thread.llm.call(
            messages=input_messages,
            model=self.bug_info.config.search_model.model,
            n=self.num_prune_sampling,
            **self.bug_info.config.search_model.llm_args.asdict(),
        )
        input_tokens, output_tokens = self.llm_backend.get_tokens(response)
        thread.memory.add_cost(output_tokens, input_tokens)

        messages = self.llm_backend.get_msgs(response)
        prune_scores = []
        prune_reason = []
        for message in messages:
            msg_text = self.llm_backend.get_msg_text(message)
            prune_reason.append(msg_text)
            if "YES" in msg_text:
                prune_scores.append(1)
            elif "NO" in msg_text:
                prune_scores.append(0)
        prune_score = sum(prune_scores) / len(prune_scores)
        if_prune = True if prune_score < self.prune_threshold else False
        thread.memory.prune_info = {
            "pruned": if_prune,
            "prune_score": prune_score,
            "prune_reason": prune_reason,
        }

        if if_prune:
            self.bug_info.logger.debug(
                f"{self.bug_info.bug_name} "
                f"- <{thread.input.test_name}> "
                f"- Process {thread.id} - search path pruned"
            )

        return if_prune

    def run_thread(
        self, thread_id: str, message: ChatCompletionMessage = None
    ) -> None:
        thread = self.threads[thread_id]

        if message:
            self.run_function_calls(thread_id, message)

        if self.use_prune:
            if (
                thread.num_function_calls > 0
                and thread.num_function_calls % self.prune_step == 0
            ):
                # try to prune the search path
                if self.prune_search_paths(thread_id):
                    return

        if thread.num_function_calls >= self.max_tool_calls:
            # if the thread has reached the maximum number of tool calls,
            # return a handmade message
            self.bug_info.logger.debug(
                f"{self.bug_info.bug_name} "
                f"- <{thread.input.test_name}> "
                f"- Process {thread.id} - reached max tool calls"
            )
            end_msg = ChatCompletionMessage(
                role="assistant",
                content=REACH_MAX_TOOL_CALLS,
            )
            messages = [end_msg]
        else:
            # if the thread has not reached the maximum number of tool calls,
            # continue the search
            response = thread.llm.call(
                messages=thread.memory.get_messages(),
                tools=self.tool_set,
                model=self.bug_info.config.search_model.model,
                n=self.num_branch_sampling,
                parallel_tool_calls=False,
                **self.bug_info.config.search_model.llm_args.asdict(),
            )
            messages = self.llm_backend.get_msgs(response)
            input_tokens, output_tokens = self.llm_backend.get_tokens(response)
            thread.memory.add_cost(output_tokens, input_tokens)

        self.extend_nodes(thread_id, messages)
