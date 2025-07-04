import copy
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import anthropic
import openai
from anthropic.types import Message, ToolUseBlock
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from tenacity import retry, stop_after_attempt, wait_exponential


def before_sleep_output(retry_state):
    print(f"\nRetry attempt {retry_state.attempt_number} failed.")
    print(f"Error: {retry_state.outcome.exception()}")
    print(
        f"Waiting {retry_state.next_action.sleep} seconds before next attempt...\n"
    )


class LLMBackend(ABC):
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def call(self, **kwargs) -> ChatCompletion | Message:
        pass

    @staticmethod
    @abstractmethod
    def get_msg(msg: ChatCompletion | Message) -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_msg_text(msg: ChatCompletion | Message) -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_single_tool_call_msg(
        msg: ChatCompletion | Message, index: int
    ) -> ChatCompletion | Message:
        pass

    @staticmethod
    @abstractmethod
    def get_tool_result_msg(
        tool_call: ChatCompletionMessageToolCall | ToolUseBlock,
        tool_result: str,
    ):
        pass

    @staticmethod
    @abstractmethod
    def get_tokens(msg: ChatCompletion | Message) -> Tuple[int, int]:
        pass

    @staticmethod
    @abstractmethod
    def recover_msg(obj) -> ChatCompletion | Message:
        pass

    @staticmethod
    @abstractmethod
    def get_tool_name(
        tool_call: ToolUseBlock | ChatCompletionMessageToolCall,
    ) -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_tool_args(
        tool_call: ToolUseBlock | ChatCompletionMessageToolCall,
    ) -> Dict[str, str]:
        pass


class OpenAIBackend(LLMBackend):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_output,
    )
    def call(self, **kwargs) -> ChatCompletion:
        response = self.client.chat.completions.create(**kwargs)
        return response

    @staticmethod
    def get_msgs(response: ChatCompletion):
        return [c.message for c in response.choices]

    @staticmethod
    def get_msg(response: ChatCompletion):
        return response.choices[0].message

    @staticmethod
    def get_msg_text(msg: ChatCompletionMessage) -> str:
        text = msg.content
        if text is None:
            text = ""
        return text

    @staticmethod
    def get_tool_calls(
        msg: ChatCompletionMessage,
    ) -> List[ChatCompletionMessageToolCall]:
        try:
            return msg.tool_calls
        except Exception:
            return None

    @staticmethod
    def get_tool_call(
        msg: ChatCompletionMessage,
    ) -> ChatCompletionMessageToolCall:
        try:
            return msg.tool_calls[0]
        except Exception:
            return None

    @staticmethod
    def get_single_tool_call_msg(
        msg: ChatCompletionMessage, index: int
    ) -> ChatCompletionMessage:
        new_msg = copy.deepcopy(msg)
        new_msg.tool_calls = [msg.tool_calls[index]]
        return new_msg

    @staticmethod
    def get_tool_result_msg(
        tool_call: ChatCompletionMessageToolCall, tool_result: str
    ) -> ChatCompletionMessage:
        result = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result,
        }
        return result

    @staticmethod
    def get_tokens(response: ChatCompletion) -> Tuple[int, int]:
        return response.usage.prompt_tokens, response.usage.completion_tokens

    @staticmethod
    def recover_msg(obj) -> ChatCompletion:
        if "tool_calls" in obj:
            return ChatCompletionMessage.model_validate(obj)
        else:
            return obj

    @staticmethod
    def get_tool_name(tool_call: ChatCompletionMessageToolCall) -> str:
        return tool_call.function.name

    @staticmethod
    def get_tool_args(
        tool_call: ChatCompletionMessageToolCall,
    ) -> Dict[str, str]:
        return json.loads(tool_call.function.arguments)


class AnthropicBackend(LLMBackend):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=60,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_output,
    )
    def call(self, **kwargs) -> ChatCompletion | Message:
        response = self.client.messages.create(**kwargs)
        return response

    @staticmethod
    def get_msg(response: Message):
        return response

    @staticmethod
    def get_msg_text(msg: Message) -> str:
        return msg.content[0].text

    @staticmethod
    def get_tool_calls(msg: Message) -> List[ToolUseBlock]:
        return [b for b in msg.content if isinstance(b, ToolUseBlock)]

    @staticmethod
    def get_tool_name(tool_call: ToolUseBlock) -> str:
        return tool_call.name

    @staticmethod
    def get_single_tool_call_msg(msg: Message, index: int) -> Message:
        new_msg = copy.deepcopy(msg)
        tool_calls = AnthropicBackend.get_tool_calls(msg)
        new_msg.content = [tool_calls[index]]
        return new_msg

    @staticmethod
    def get_tool_result_msg(
        tool_call: ToolUseBlock, tool_result: str
    ) -> Message:
        result = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": tool_result,
                }
            ],
        }
        return result

    @staticmethod
    def get_tokens(response: Message) -> Tuple[int, int]:
        return response.usage.input_tokens, response.usage.output_tokens

    @staticmethod
    def recover_msg(obj) -> Message:
        if obj["type"] == "tool_use":
            return Message.model_validate(obj)
        else:
            return obj

    @staticmethod
    def get_tool_args(
        tool_call: ToolUseBlock,
    ) -> Dict[str, str]:
        return tool_call.input
