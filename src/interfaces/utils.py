import os
import re
import subprocess as sp

import chardet


def run_cmd(cmd: str, debug=False):
    if debug:
        print("-" * 50)
        print(f"run command: {cmd}")
    p = sp.Popen(cmd.split(" "), stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    output, err = p.communicate()
    out = output.decode("utf-8")
    err = err.decode("utf-8")
    if debug:
        print(err)
        print(out)
        print("-" * 50)
    return out, err


def git_clean(git_dir):
    cwd = os.getcwd()
    os.chdir(git_dir)
    run_cmd("git clean -df")
    os.chdir(cwd)


def clean_doc(doc: str) -> str:
    """
    Turn multi-line doc into one line.
    """
    new_doc_lines = []
    doc_lines = doc.split("\n")
    for doc_line in doc_lines:
        doc_str = re.match(r"^([/\s\*]*)(.*)", doc_line)
        if doc_str is not None:
            line = doc_str.group(2)
            if not line.startswith("@author"):
                new_doc_lines.append(line)
    return " ".join(new_doc_lines)


def auto_read(file):
    with open(file, "rb") as f:
        content = f.read()
    detected_encoding = chardet.detect(content)["encoding"]
    text = content.decode(detected_encoding, errors="ignore")
    return text


def filter_compile_error(log: str) -> str:
    javac_lines = re.findall(r"\[javac\] .*\n", log)
    javac_warning = r"\[javac\] .*warning: .*\n"
    javac_error = r"(\[javac\] ).*(error: .*)\n"
    javac_note = r"\[javac\] .*Note: .*\n"
    inserted_prefix = (
        "try { java.nio.file.Path filePath = java.nio.file.Paths.get("
    )
    error_lines = []
    flag = False
    for line in javac_lines:
        if re.match(javac_warning, line):
            flag = False
        elif re.match(javac_note, line):
            flag = False
        elif inserted_prefix in line:
            flag = False
        elif line == "\n":
            flag = False
        elif re.match(javac_error, line):
            flag = True
            match = re.match(javac_error, line)
            line = match.group(1) + match.group(2)
        else:
            line = line.strip()

        if flag:
            error_lines.append(line)

    return "\n".join(error_lines)


class WorkDir:
    def __init__(self, path):
        self.work_dir = path
        self.cwd = os.getcwd()
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def __enter__(self):
        os.chdir(self.work_dir)
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.cwd)
        return True


if __name__ == "__main__":
    a, b = run_cmd("defects4j compile -w /home/qyh/projects/FixFL/Closure-4")
    print(a)
    print("---------------")
    print(b)
