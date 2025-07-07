import json
from typing import Dict, List

from anthropic.types import Message
from openai.types.chat import ChatCompletionMessage

PRICE_PER_MILLION_TOKENS = {
    "gpt-3.5-turbo": (0.5, 1.5),
    "gpt-4": (30, 60),
    "gpt-4o-2024-11-20": (2.5, 10),
    "gpt-4-turbo-2024-04-09": (10, 30),
    "gpt-4-0125-preview": (10, 30),
    "claude-3-5-sonnet-20241022": (3, 15),
    "deepseek-chat": (0.2857, 1.1429),
    "deepseek-v3-250324": (0.2857, 1.1429),
    "gpt-4.1-mini-2025-04-14": (0.4, 1.6),
}


class MyMessage:
    def __init__(self, llm_msg, type):
        self.llm_msg = llm_msg
        self.type = type

    def dump(self):
        if isinstance(self.llm_msg, ChatCompletionMessage):
            return json.loads(self.llm_msg.model_dump_json())
        elif isinstance(self.llm_msg["content"], List):
            return {
                "role": self.llm_msg["role"],
                "content": [
                    block.model_dump_json()
                    for block in self.llm_msg["content"]
                ],
            }
        else:
            return self.llm_msg

    def __hash__(self):
        return self.llm_msg.__hash__()


class Memory:
    def __init__(
        self, system_prompt: str, model_name: str = "gpt-4o-2024-11-20"
    ):
        self.messages: List[MyMessage] = []
        self.add_message({"role": "system", "content": system_prompt})
        self.retry_msgs = []
        self.model_name = model_name
        self.costs = []
        self.purne_info = {
            "purned": False,
            "purne_score": 1,
            "purne_reason": [],
        }

    def add_message(self, message, type="normal"):
        my_msg = MyMessage(message, type)
        self.messages.append(my_msg)
        if type == "retry":
            self.retry_msgs.append(my_msg)

    def add_cost(self, completion_tokens: int, prompt_tokens: int):
        cost = self.in_tokens_cost(prompt_tokens) + self.out_tokens_cost(
            completion_tokens
        )
        self.costs.append(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
            }
        )

    def get_messages(self) -> List[Dict]:
        """
        Get all messages. To save cost, we remove previous retry messages
        except the last two retry messages (including question and the error message).
        """
        messages = []
        for msg in self.messages:
            if len(self.retry_msgs) > 2 and msg in self.retry_msgs[:-2]:
                continue
            messages.append(msg.llm_msg)
        return messages

    def get_debug_report(self) -> str:
        debug_report = ""
        for msg in self.messages:
            if msg.type == "debug_report":
                debug_report = msg.llm_msg["content"]
                break
        return debug_report

    def clear(self):
        self.messages = []

    def serialize(self) -> str:
        messages = [m.dump() for m in self.messages]
        return {
            "messages": messages,
            "all_in_tokens": sum(c["prompt_tokens"] for c in self.costs),
            "all_out_tokens": sum(c["completion_tokens"] for c in self.costs),
            "money": sum([c["cost"] for c in self.costs]),
            "purne_info": self.purne_info,
        }

    def to_debug_process(self) -> str:
        debug_process = []
        messages = [m.dump() for m in self.messages]
        for message in messages[2:]:
            if "tool_calls" in message:
                debug_process.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                        "tool_calls": message["tool_calls"],
                    }
                )
            elif message["role"] == "tool":
                debug_process.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                    }
                )
            else:
                debug_process.append(message)
        return json.dumps(debug_process, indent=2)

    def in_tokens_cost(self, tokens: int):
        return tokens / 1000000 * PRICE_PER_MILLION_TOKENS[self.model_name][0]

    def out_tokens_cost(self, tokens: int):
        return tokens / 1000000 * PRICE_PER_MILLION_TOKENS[self.model_name][1]
