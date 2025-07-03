import ast
import os

import tree_sitter_languages


def create_structure_java(directory_path):
    """Create the structure of the repository directory by parsing Java files.
    :param directory_path: Path to the repository directory.
    :return: A dictionary representing the structure.
    """
    structure = {}
    # tree-sitter parser
    parser = tree_sitter_languages.get_parser("java")

    for root, _, files in os.walk(directory_path):
        repo_name = os.path.basename(directory_path)
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            relative_root = repo_name
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]
        for file_name in files:
            if file_name.endswith(".java"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_java_file(file_path, parser)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure


def parse_java_file(file_path, parser, file_content=None):
    """Parse a Java file to extract class and method definitions with their line numbers.
    :param file_path: Path to the Java file
    :param file_content: Optional pre-loaded file content
    :return: Class info, method info, and file contents as tuple
    """

    if file_content is None:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                file_content = file.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return [], [], ""

    try:
        tree = parser.parse(bytes(file_content, 'utf-8'))
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return [], [], ""

    class_info = []
    function_names = []
    file_lines = file_content.splitlines()

    def get_node_text(node):
        return file_content[node.start_byte:node.end_byte]

    def get_node_lines(node):
        return file_lines[node.start_point[0]:node.end_point[0] + 1]

    def process_methods(class_node):
        methods = []
        for child in class_node.child_by_field_name('body').children:
            if child.type == 'method_declaration':
                # get method name
                name_node = child.child_by_field_name('name')
                if name_node:
                    methods.append({
                        'name': name_node.text.decode('utf-8'),
                        'start_line': child.start_point[0] + 1,
                        'end_line': child.end_point[0] + 1,
                        'text': get_node_lines(child)
                    })
        return methods

    def traverse_tree(node):
        if node.type == 'class_declaration':
            # get class name
            name_node = node.child_by_field_name('name')
            if name_node:
                class_name = name_node.text.decode('utf-8')
                methods = process_methods(node)
                class_info.append({
                    'name': class_name,
                    'start_line': node.start_point[0] + 1,
                    'end_line': node.end_point[0] + 1,
                    'text': get_node_lines(node),
                    'methods': methods
                })

        for child in node.children:
            traverse_tree(child)

    traverse_tree(tree.root_node)
    return class_info, function_names, file_lines


def create_structure(directory_path):
    """Create the structure of the repository directory by parsing Python files.
    :param directory_path: Path to the repository directory.
    :return: A dictionary representing the structure.
    """
    structure = {}

    for root, _, files in os.walk(directory_path):
        repo_name = os.path.basename(directory_path)
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            relative_root = repo_name
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_python_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure

def parse_python_file(file_path, file_content=None):
    """Parse a Python file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Python file.
    :return: Class names, function names, and file contents
    """
    if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[
                                n.lineno - 1 : n.end_lineno
                            ],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[
                        node.lineno - 1 : node.end_lineno
                    ],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) and not isinstance(
            node, ast.AsyncFunctionDef
        ):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ],
                    }
                )

    return class_info, function_names, file_content.splitlines()