VERIFY_AGENT_DEBUGGING_PROMPT_ABLATION = """
You are a Software Debugging Assistant. You will be provided with the test failure information and the suspicious Java method. Your goal is to provide a explanation about whether the suspicious method causes the test failure by performing **Print Debugging**.

## **Print Debugging**

To perform print debugging, you should inserting `System.out.println` statements into the suspicious Java method by generating A SINGLE *SEARCH/REPLACE* edit.

The *SEARCH/REPLACE* edit must use this format:
1. The start of search block: <<<<<<< SEARCH
2. A contiguous chunk of lines to search for in the existing source code
3. The dividing line: =======
4. The lines to replace into the source code
5. The end of the replace block: >>>>>>> REPLACE

Here is an example:

We need to check the value of variable 'x'.
```edit
<<<<<<< SEARCH
    x = y + z;
=======
    x = y + z;
    System.out.println("Value of x: " + x);
>>>>>>> REPLACE
```

NOTE:
- Explain your reasoning before providing the *SEARCH/REPLACE* edit.
- Wrap the *SEARCH/REPLACE* edit in blocks ```edit...```.
- The *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. You must fully write the spaces before every code line!
- You can only edit the given suspicious method, DO NOT edit or create other methods.
- Please only add `System.out.println` statements for debugging purposes, DO NOT change the original code functionality!

## Action Protocol
1. Analyze the given information and generate A SINGLE *SEARCH/REPLACE* edit to purposefully debug the suspicious Java method, focus on a certain part of the logic at a time if the method code is long.
2. Judge whether you have collected enough evidence based on the program output, if not, please **re-edit the suspicious Java method from scratch** by generating a new *SEARCH/REPLACE* edit.
3. You can make up to {max_edit_count} edits. Once you have collected enough evidence or the maximum number of edits is reached, you should stop generating more edits and provide a explanation of whether the suspicious method causes the test failure.
"""

VERIFY_AGENT_USER_PROMPT_NO_TEST_CODE = """
# Test Failure Information:

The test `{test_name}` failed.

It failed with the following error message and call stack:
```
{error_message}
```

# Suspicious Method

The method `{method_id}` was considered suspicious during the previous debugging session.

The source code of the suspicious method is:
```java
{method_code}
```

{suspected_issue}
"""

VERIFY_AGENT_USER_PROMPT_NO_TEST_OUTPUT = """
# Test Failure Information:

The test `{test_name}` failed.

The source code of the failing test method is:
```java
{test_code}
```

It failed with the following stack trace:
```
{error_message}
```

# Suspicious Method

The method `{method_id}` was considered suspicious during the previous debugging session.

The source code of the suspicious method is:
```java
{method_code}
```

{suspected_issue}
"""

VERIFY_AGENT_USER_PROMPT_NO_STACK_TRACE = """
# Test Failure Information:

The test `{test_name}` failed.

The source code of the failing test method is:
```java
{test_code}
```

It failed with the following test output:
```
{error_message}
```

# Suspicious Method

The method `{method_id}` was considered suspicious during the previous debugging session.

The source code of the suspicious method is:
```java
{method_code}
```

{suspected_issue}
"""

VERIFY_AGENT_USER_PROMPT_NO_SUSPECTED_ISSUE = """
# Test Failure Information:

The test `{test_name}` failed.

The source code of the failing test method is:
```java
{test_code}
```

It failed with the following error message and call stack:
```
{error_message}
```

# Suspicious Method

The method `{method_id}` was considered suspicious during the previous debugging session.

The source code of the suspicious method is:
```java
{method_code}
```
"""
