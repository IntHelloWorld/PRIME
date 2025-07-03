import re
from typing import List, Tuple


def extract_search_replace_block(text):
    pattern = r"<+ SEARCH\n(.*?)\n=+\n(.*?)\n>+ REPLACE"
    matche = re.search(pattern, text, re.DOTALL)
    if matche:
        return (matche.group(1), matche.group(2))
    return None


def extract_edit_block(text):
    pattern = r"// \.\.\. existing code \.\.\.\n(.*?)\n// \.\.\. existing code \.\.\."
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def extract_diff_block(text):
    pattern = r"```diff\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def extract_json_block(text):
    pattern = r"```json\n(.*?)\n```"
    matches = re.search(pattern, text, re.DOTALL)
    if matches:
        return matches.group(1)
    return None


def extract_java_block(text):
    pattern = r"```java\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def extract_edit_block(text):
    pattern = r"```edit\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def extract_print_blocks(text) -> List[Tuple[str, str]]:
    pattern = r"(System\.out\.println\((.*?)\);)"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def remove_whitespace(lines: List[str]):
    new_text_lines = []
    for line in lines:
        new_text_lines.append(line.strip())
    return "\n".join(new_text_lines)


def extract_replace_lines(text: str):
    prefix_pattern = r"// PREFIX_START\n(.*?)\n// PREFIX_END"
    suffix_pattern = r"// SUFFIX_START\n(.*?)\n// SUFFIX_END"
    replace_pattern = r"// PREFIX_END\n(.*?)\n// SUFFIX_START"
    prefix_match = re.search(prefix_pattern, text, re.DOTALL)
    suffix_match = re.search(suffix_pattern, text, re.DOTALL)
    replace_match = re.search(replace_pattern, text, re.DOTALL)
    if prefix_match and suffix_match and replace_match:
        return (
            prefix_match.group(1),
            suffix_match.group(1),
            replace_match.group(1),
        )
    return None, None, None


def get_original_lines(diff_lines: List[str]):
    original_lines = []
    for line in diff_lines:
        if line.startswith("-"):
            original_lines.append(line[1:].strip())
        if line.startswith("+"):
            continue
        else:
            original_lines.append(line.strip())
    return original_lines


def get_replace_lines(diff_lines: List[str]):
    replace_lines = []
    for line in diff_lines:
        if line.startswith("+"):
            replace_lines.append(line[1:])
        elif line.startswith("-"):
            continue
        else:
            replace_lines.append(line)
    return replace_lines


def add_line_numbers(code: str, start_line: int, end_line: int):
    code_lines = code.splitlines()
    assert len(code_lines) == end_line - start_line + 1

    numbered_lines = [
        f"{i+start_line:4d} {line}" for i, line in enumerate(code_lines)
    ]
    return "\n".join(numbered_lines)


class ContextMatcher:
    def __init__(self, context_lines: List[str], replace_lines: List[str]):
        context_lines = [line.strip() for line in context_lines]
        replace_lines = [line.strip() for line in replace_lines]
        self.ctxt_lines, self.ctxt_idx_map = self.remove_empty_lines(
            context_lines
        )
        self.rep_lines, self.rep_idx_map = self.remove_empty_lines(
            replace_lines
        )
        self.n_tolerance = 1

    def remove_empty_lines(self, lines: List[str]):
        line_index_map = {}  # new line index -> old line index
        new_lines = []
        for i, line in enumerate(lines):
            if line != "":
                line_index_map[len(new_lines)] = i
                new_lines.append(line)
        return new_lines, line_index_map

    def find_context(self):
        matches = []

        for start, ctxt_line in enumerate(self.ctxt_lines):
            if ctxt_line == self.rep_lines[0]:
                cur_rep = 0
                end = start
                for i in range(start, len(self.ctxt_lines)):
                    for j in range(cur_rep, len(self.rep_lines)):
                        if self.ctxt_lines[i] == self.rep_lines[j]:
                            cur_rep = j
                            end = i
                            break

                matches.append((start, end))

        if len(matches) == 0:
            raise Exception("No match found!")
        matches = [
            (self.ctxt_idx_map[start], self.ctxt_idx_map[end])
            for start, end in matches
        ]
        start_line, end_line = max(matches, key=lambda x: x[1] - x[0])
        return start_line, end_line
