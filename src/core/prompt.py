# TOOLS = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_covered_method_ids_for_class",
#             "description": "This function returns the IDs of all covered methods in a class.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "class_name": {
#                         "type": "string",
#                         "description": "The full class name such as 'com.example.MyClass'.",
#                     },
#                 },
#                 "required": ["class_name"],
#                 "additionalProperties": False,
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "get_method_code_for_id",
#             "description": "This function returns the source code of the method with the specified method ID.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "method_id": {
#                         "type": "string",
#                         "description": "The complete method id to search for its code, e.g., 'com.example.MyClass.InnerClass.methodName<20-30>'.",
#                     },
#                 },
#                 "required": ["method_id"],
#                 "additionalProperties": False,
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "search_covered_class_full_name",
#             "description": "This function returns the possible full class name for a given class name.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "class_name": {
#                         "type": "string",
#                         "description": "The short class name to search for, e.g., 'MyClass'.",
#                     },
#                 },
#                 "required": ["class_name"],
#                 "additionalProperties": False,
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "search_covered_method_id",
#             "description": "This function returns the possible method IDs for the given method(constructor) name and class name (optional).",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "method_name": {
#                         "type": "string",
#                         "description": "The method(constructor) name to search for, e.g., 'myMethod'.",
#                     },
#                     "class_name": {
#                         "type": "string",
#                         "description": "The short class name to search for, e.g., 'MyClass'.",
#                     },
#                 },
#                 "required": ["method_name"],
#                 "additionalProperties": False,
#             },
#         },
#     },
# ]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_covered_method_ids_for_class",
            "description": "This function returns the IDs of all covered methods in a class.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {
                        "type": "string",
                        "description": "The full class name such as 'com.example.MyClass'.",
                    },
                },
                "required": ["class_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_method_code_for_id",
            "description": "This function returns the source code of the method with the specified method ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method_id": {
                        "type": "string",
                        "description": "The complete method id to search for its code, e.g., 'com.example.MyClass.InnerClass.methodName<20-30>'.",
                    },
                },
                "required": ["method_id"],
                "additionalProperties": False,
            },
        },
    },
]

SEARCH_AGENT_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "get_covered_method_ids_for_class",
            "description": "This function returns the IDs of all covered methods in a class. It supports precise and fuzzy matches for class name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your analysis and the reason for initiate the function call.",
                    },
                    "class_name": {
                        "type": "string",
                        "description": "The class name. For precise matches, input the full class name such as 'com.example.MyClass'. For fuzzy matches, input the class name such as 'MyClass'.",
                    },
                },
                "required": ["thought", "class_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_method_code_for_id",
            "description": "This function returns the source code of the method with the specified method ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your analysis and the reason for initiate the function call.",
                    },
                    "method_id": {
                        "type": "string",
                        "description": "The complete method id to search for its code, e.g., 'com.example.MyClass.InnerClass.methodName<20-30>'.",
                    },
                },
                "required": ["thought", "method_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_method_ids_contain_string",
            "description": "This function returns the IDs of all methods containing a specific string content. It can be used to search for methods responsible for printing specific string or to statically find caller/callee methods by method name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your analysis and the reason for initiate the function call.",
                    },
                    "string_content": {
                        "type": "string",
                        "description": "The string content to search for, requires proper indentation.",
                    },
                },
                "required": ["thought", "string_content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_caller_method_ids",
            "description": "This function returns the IDs of all methods that have called the specified method during runtime.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your analysis and the reason for initiate the function call.",
                    },
                    "method_id": {
                        "type": "string",
                        "description": "The method id to search for its callers.",
                    },
                },
                "required": ["thought", "method_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_callee_method_ids",
            "description": "This function returns the IDs of all methods that have been called by the specified method during runtime.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your analysis and the reason for initiate the function call.",
                    },
                    "method_id": {
                        "type": "string",
                        "description": "The method id to search for its callees.",
                    },
                },
                "required": ["thought", "method_id"],
                "additionalProperties": False,
            },
        },
    },
]

SEARCH_AGENT_TOOLS_ANTHROPIC = [
    {
        "name": "get_covered_method_ids_for_class",
        "description": "Get a list of all covered methods of a class. Support precise and fuzzy matches for class name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "A text to describe your analysis and the reason for the tool call.",
                },
                "class_name": {
                    "type": "string",
                    "description": "The class name. For precise matches, input the full class name such as 'com.example.MyClass'. For fuzzy matches, input a partial class name such as 'MyClass'.",
                },
            },
            "required": ["thought", "class_name"],
        },
    },
    {
        "name": "get_method_code_for_id",
        "description": "Get the code of the method(s) from a specified class. Support precise and fuzzy matches for class name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "A text to describe your analysis and the reason for the tool call.",
                },
                "class_name": {
                    "type": "string",
                    "description": "The class name. For precise matches, input the full class name such as 'com.example.MyClass'. For fuzzy matches, input a partial class name such as 'MyClass'.",
                },
                "method_name": {
                    "type": "string",
                    "description": "The method name. Directly input the method name such as 'myMethod'.",
                },
            },
            "required": ["thought", "class_name", "method_name"],
        },
    },
    {
        "name": "get_method_ids_contain_string",
        "description": "Get a list of all production methods containing a specific string content. This tool can be used to search for methods responsible for printing test output strings or to statically find caller/callee methods by method name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "A text to describe your analysis and the reason for the tool call.",
                },
                "string_content": {
                    "type": "string",
                    "description": "The string content to search for, requires proper indentation.",
                },
            },
            "required": ["thought", "string_content"],
        },
    },
    {
        "name": "get_caller_method_ids",
        "description": "Get a list of covered method IDs that have called the specified method during runtime.",
        "input_schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "A text to describe your analysis and the reason for the tool call.",
                },
                "method_id": {
                    "type": "string",
                    "description": "The method id to search for its callers.",
                },
            },
            "required": ["thought", "method_id"],
        },
    },
    {
        "name": "get_callee_method_ids",
        "description": "Get a list of covered method IDs that have been called by the specified method during runtime.",
        "input_schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "A text to describe your analysis and the reason for the tool call.",
                },
                "method_id": {
                    "type": "string",
                    "description": "The method id to search for its callees.",
                },
            },
            "required": ["thought", "method_id"],
        },
    },
]

SUMMARIZATION_PROMPT = """Based on the available information, provide a step-by-step explanation of how the bug occurred."""

FAULT_LOCALIZATION_PROMPT = """Based on the available information, provide the method IDs of the most likely culprit Java methods for the bug. Your answer will be processed automatically, so make sure to only answer with the accurate IDs of all likely culprits (e.g., 'com.example.MyClass.InnerClass.methodName<20-30>'), without commentary (one per line)."""

NEW_FAULT_LOCALIZATION_PROMPT = """Based on the available information, please list all the culprit Java methods for the bug. Your answer will be automatically processed, so please make sure your answer includes the accurate ID of each culprit method (e.g., 'com.example.MyClass.InnerClass.methodName<20-30>')."""

REACH_MAX_TOOL_CALLS = """The maximum number of tool calls has been reached, we have to stop the debugging session."""


USER_PROMPT = """# Test Failure Information

The test `{test_name}` failed.

The source code of the failing test method is:

```java
{test_code}
```

It failed with the following error message and call stack:

```text
{error_message}
```
"""

DEBUGGING_PROMPT = """You are a Software Debugging Assistant. You will be provided with the test failure information and a set of callable functions to help you debug the issue. Your task is to understand the root cause of the bug step-by-step using the callable functions.

NOTE:
- Explain your analysis and thoughts before each function call you initiate.
- You have up to {max_tool_calls} chances to call the functions.
- If you have understood the root cause, please terminate the debugging session by providing a response without any function call.
"""

PRUNE_PROMPT = """You are an expert software engineer specializing in debugging and code analysis. Your task is to evaluate the reasoning and actions of a Large Language Model (LLM) agent designed for automated fault localization.
You will be given the details of a failed test case and the step-by-step debugging process the agent has taken so far. This process includes the agent's reasoning, the tools it called, and the outputs from those tools.
Your goal is to determine if the agent is on a promising path to finding the root cause of the bug. You are not judging the final outcome, as the process is incomplete. Instead, you are evaluating the quality and direction of the debugging process itself.

## Evaluation Criteria

Carefully analyze the agent's debugging process based on the following criteria:

* "YES" (On the right track to success): The agent's actions are logical and directed. The agent is making tangible progress towards identifying the root cause of the bug, or the agent has found clues related to the test failure.

* "NO" (Not on the right track to success): The agent's actions are irrelevant, or inefficient. The agent goes in circles without making tangible progress, or the agent has not found any clues related to the test failure.

## Output Format

You must format your response into two lines as shown below:

Thoughts: <Your brief thoughts and reasoning process for the decision>
On the right track to success: "YES" or "NO"
"""

PRUNE_USER_PROMPT = """## Test Failure Information

The test `{test_name}` failed.

The source code of the failing test method is:

```java
{test_code}
```

It failed with the following error message and call stack:

```text
{error_message}
```

## Agent's Debugging Process So Far

The following is the sequence of thoughts, tool calls, and tool outputs from the agent so far:

```json
{debugging_process}
```

Based on the information above, please evaluate the fault localization agent's performance.
"""
