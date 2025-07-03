import copy
import json
import os
import pickle
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces.method_extractor import JavaMethodExtractor, JMethod
from interfaces.utils import (
    WorkDir,
    auto_read,
    filter_compile_error,
    git_clean,
    run_cmd,
)
from src.config import BugInfo
from src.exceptions import CompileError
from src.schema import TestCase, TestClass, TestFailure
from src.utils import Timer

filepath = Path(__file__).parent
root = filepath.parent


class JavaMethod:
    def __init__(self, class_name: str, inst_sig: str, inner: bool):
        self.class_name = class_name
        self.inst_sig = inst_sig
        self.inst_id = class_name + "::" + inst_sig
        self._inner = inner  # if the method is in an inner class
        self._covered = False

        self.src_sig = None
        self.src_id = None
        self.doc = ""
        self.enhanced_doc = ""
        self.code = None

    def set_covered(self):
        self._covered = True

    def __hash__(self) -> int:
        return hash(self.inst_id)

    def __eq__(self, other: object) -> bool:
        return self.inst_id == other.inst_id


class JavaClass:
    """
    Only for outer class, for inner class, we save its methods as the JavaMethod which name in
    format InnerClass1$InnerClass2.method_sig
    """

    def __init__(self, class_name):
        self.class_name = class_name
        self.methods = (
            {}
        )  # first extracted from instrument file, then enriched by source code
        self.doc = ""

    def add_methods(self, method: JavaMethod):
        self.methods[method.class_name + "::" + method.inst_sig] = method

    def statistic(self):
        n_covered_methods = 0
        for method in self.methods.values():
            if method._covered:
                n_covered_methods += 1
        self.n_covered_methods = n_covered_methods
        self.n_all_methods = len(self.methods)
        self.porpotion = self.n_covered_methods / self.n_all_methods * 100

    def __hash__(self) -> int:
        return hash(self.class_name)

    def __eq__(self, other: object) -> bool:
        return self.class_name == other.class_name


def check_out(bugInfo: BugInfo):
    with WorkDir(bugInfo.proj_tmp_path):
        if not bugInfo.buggy_path.exists():
            run_cmd(
                f"{bugInfo.bug_exec} checkout -p {bugInfo.project} -v {bugInfo.bug_id}b -w buggy"
            )
        if not bugInfo.fixed_path.exists():
            run_cmd(
                f"{bugInfo.bug_exec} checkout -p {bugInfo.project} -v {bugInfo.bug_id}f -w fixed"
            )


def check_out_playground(bugInfo: BugInfo, playground_path: Path):
    parent_path = playground_path.parent
    dirname = playground_path.name
    with WorkDir(parent_path):
        if not playground_path.exists():
            run_cmd(
                f"{bugInfo.bug_exec} checkout -p {bugInfo.project} -v {bugInfo.bug_id}b -w {dirname}"
            )


def run_single_test_playground(
    bugInfo: BugInfo, playground_path: Path, test_name: str, max_lines=50
):
    """Run a single test case in the playground and return the result adaptively"""
    assert playground_path.exists()
    console_out, console_err = run_cmd(
        f"{bugInfo.bug_exec} test -t {test_name} -w {playground_path}"
    )

    cmd_text = None
    print_text = "empty"

    # check if the compilation is successful
    if console_err.splitlines()[0].endswith("FAIL"):
        cmd_text = filter_compile_error(console_err)
    else:
        # collect the print text
        print_file = playground_path / "output.txt"
        if print_file.exists():
            print_text = print_file.read_text()
            print_file.unlink()

    if cmd_text:
        raise CompileError(cmd_text)
    else:
        print_lines = print_text.splitlines()
        if len(print_lines) > max_lines:
            print_text = "\n".join(print_lines[:max_lines])
            print_text += f"\n... omitting the rest {len(print_lines) - max_lines} lines ..."
        return f"The log printed by the edited code is as follows:\n```\n{print_text}\n```"


def get_test_case_output_path(bugInfo: BugInfo, test_case: TestCase) -> Path:
    output_path = (
        bugInfo.res_path
        / test_case.test_class_name.replace(".", "_")
        / test_case.test_method_name
    )
    return output_path


def get_test_case_dataset_path(bugInfo: BugInfo, test_case: TestCase) -> Path:
    dataset_path = (
        bugInfo.bug_path
        / test_case.test_class_name.replace(".", "_")
        / test_case.test_method_name
    )
    return dataset_path


def run_single_test(test_case: TestCase, bugInfo: BugInfo):
    test_output_dir = (
        bugInfo.cache_path / test_case.test_class_name / test_case.name
    )
    test_output_dir.mkdir(parents=True, exist_ok=True)
    test_output_file = test_output_dir / "test_output.txt"
    stack_trace_file = test_output_dir / "stack_trace.txt"

    if test_output_file.exists() and stack_trace_file.exists():
        test_output = test_output_file.read_text().splitlines()
        stack_trace = stack_trace_file.read_text().splitlines()
        return test_output, stack_trace

    git_clean(bugInfo.buggy_path)
    out, err = run_cmd(f"{bugInfo.bug_exec} compile -w {bugInfo.buggy_path}")
    out, err = run_cmd(
        f"timeout 90 {bugInfo.bug_exec} test -n -t {test_case.name} -w {bugInfo.buggy_path}"
    )
    with open(f"{bugInfo.buggy_path}/failing_tests", "r") as f:
        test_res = f.readlines()
    test_output, stack_trace = parse_test_report(test_res, bugInfo)
    with open(test_output_file, "w") as f:
        f.writelines(test_output)
    with open(stack_trace_file, "w") as f:
        f.writelines(stack_trace)
    return test_output, stack_trace


def run_test_with_instrument_old(test_case: TestCase, bugInfo: BugInfo):
    test_cache_dir = get_test_case_dataset_path(bugInfo, test_case)
    run_methods_file = test_cache_dir / "run.log"
    test_output_file = test_cache_dir / "test_output.txt"
    stack_trace_file = test_cache_dir / "stack_trace.txt"
    all_files = [run_methods_file, test_output_file, stack_trace_file]
    src_class_path = bugInfo.buggy_path / bugInfo.src_class_prefix
    test_class_path = bugInfo.buggy_path / bugInfo.test_class_prefix

    if all(f.exists() for f in all_files):
        bugInfo.logger.info(
            "[run all tests]     instrumentation already done, skip!"
        )
    else:
        shutil.rmtree(test_cache_dir, ignore_errors=True)
        test_cache_dir.mkdir(parents=True, exist_ok=True)
        git_clean(bugInfo.buggy_path)
        cmd = (
            f"{bugInfo.bug_exec} test -n -w {bugInfo.buggy_path} "
            f"-t {test_case.name} "
            f"-a -Djvmargs=-javaagent:{bugInfo.java_agent_lib}="
            f"outputDir={test_cache_dir},"
            f"srcClassPath={src_class_path},"
            f"testClassPath={test_class_path}"
        )
        run_cmd(cmd)
        test_report_file = bugInfo.buggy_path / "failing_tests"
        test_report = test_report_file.read_text().splitlines()
        test_output, stack_trace = parse_test_report(test_report, bugInfo)
        test_output_file.write_text("\n".join(test_output))
        stack_trace_file.write_text("\n".join(stack_trace))
        assert all(f.exists() for f in all_files)

    test_case.test_output = test_output_file.read_text()
    test_case.stack_trace = stack_trace_file.read_text()


def run_test_with_instrument(test_case: TestCase, bugInfo: BugInfo):
    test_cache_dir = get_test_case_dataset_path(bugInfo, test_case)
    loaded_classes_file = test_cache_dir / "loaded_classes.txt"
    calltrace_file = test_cache_dir / "callgraph.graphml"
    test_output_file = test_cache_dir / "test_output.txt"
    stack_trace_file = test_cache_dir / "stack_trace.txt"
    all_files = [
        loaded_classes_file,
        calltrace_file,
        test_output_file,
        stack_trace_file,
    ]
    src_class_path = bugInfo.buggy_path / bugInfo.src_class_prefix
    test_class_path = bugInfo.buggy_path / bugInfo.test_class_prefix

    if all(os.path.exists(f) for f in all_files):
        bugInfo.logger.info("instrumentation already done, skip!")
    else:
        shutil.rmtree(test_cache_dir, ignore_errors=True)
        test_cache_dir.mkdir(parents=True, exist_ok=True)
        git_clean(bugInfo.buggy_path)
        cmd = (
            f"{bugInfo.bug_exec} test -n "
            f"-t {test_case.name} "
            f"-a -Djvmargs=-javaagent:{bugInfo.java_agent_lib}="
            f"srcClassPath={src_class_path},"
            f"testClassPath={test_class_path}"
        )
        with WorkDir(bugInfo.buggy_path):
            run_cmd(cmd)
            shutil.copy(
                f"{bugInfo.buggy_path}/callgraph.graphml", test_cache_dir
            )
            shutil.copy(
                f"{bugInfo.buggy_path}/loaded_classes.txt", test_cache_dir
            )
        test_report_file = bugInfo.buggy_path / "failing_tests"
        test_report = test_report_file.read_text().splitlines()
        test_output, stack_trace = parse_test_report(test_report, bugInfo)
        test_output_file.write_text("\n".join(test_output))
        stack_trace_file.write_text("\n".join(stack_trace))
        assert all(f.exists() for f in all_files)

    test_case.test_output = test_output_file.read_text()
    test_case.stack_trace = stack_trace_file.read_text()


def get_test_method(
    bugInfo: BugInfo, test_class_name: str, test_method_name: str
):
    buggy_path = bugInfo.buggy_path
    test_path = bugInfo.test_prefix
    test_file = (
        buggy_path
        / test_path
        / Path(test_class_name.replace(".", "/") + ".java")
    )

    if not test_file.exists():
        raise FileNotFoundError(f"Error: {test_file} not exists.")

    code = test_file.read_text()

    function_extractor = JavaMethodExtractor()
    methods = function_extractor.get_java_methods(code)
    assert len(methods) > 0, f"Error: No method found in {test_file}."
    for method in methods:
        if method.name == test_method_name:
            return method
    else:
        # the test method may be in the father class
        try:
            dot_idx = test_class_name.rfind(".")
            pkg_name = test_class_name[:dot_idx]
            short_name = test_class_name[dot_idx + 1 :]
            match_cls = re.search(rf"{short_name}\s+extends\s+(\w+)", code)
            f_class_name = match_cls.group(1)
            match_pkg = re.search(rf"import\s+([\w.]+).{f_class_name};", code)
            f_pkg_name = match_pkg.group(1) if match_pkg else pkg_name
            f_class_full_name = f_pkg_name + "." + f_class_name

            return get_test_method(
                bugInfo, f_class_full_name, test_method_name
            )
        except Exception:
            raise ValueError(
                f"Error: No method named {test_method_name} in {test_file}."
            )


def get_modified_methods(bugInfo: BugInfo):
    buggy_path = bugInfo.buggy_path
    fixed_path = bugInfo.fixed_path
    src_path = bugInfo.src_prefix
    modified_classes = bugInfo.modified_classes
    buggy_methods = []

    for class_name in modified_classes:

        # fix errors in GrowingBugs
        if bugInfo.project == "IO":
            extra_prefix = src_path.replace("/", ".") + "."
            class_name = class_name.replace(extra_prefix, "")
        elif bugInfo.project == "Dagger_core":
            extra_prefix = "core."
            class_name = class_name.replace(extra_prefix, "")

        buggy_file = (
            buggy_path
            / src_path
            / Path(class_name.replace(".", "/") + ".java")
        )

        fixed_file = (
            fixed_path
            / src_path
            / Path(class_name.replace(".", "/") + ".java")
        )

        if not (fixed_file.exists() and buggy_file.exists()):
            raise FileNotFoundError(
                f"Warning: {fixed_file} or {buggy_file} not exists."
            )

        buggy_code = auto_read(buggy_file)
        fixed_code = auto_read(fixed_file)

        function_extractor = JavaMethodExtractor()
        methods = function_extractor.get_buggy_methods(buggy_code, fixed_code)
        for method in methods:
            method.class_full_name = class_name
            method.text = method.text.replace("\r", "")
        buggy_methods.extend(methods)
    return buggy_methods


def get_properties(bugInfo: BugInfo):
    """
    Retrieves properties related to the project.
    """
    if (bugInfo.bug_path / "properties.json").exists():
        with open(bugInfo.bug_path / "properties.json", "r") as f:
            properties = json.load(f)
    else:
        properties = {}

        # for some project such as Pool we have to compile first
        cmd = f"{bugInfo.bug_exec} compile -w {bugInfo.buggy_path}"
        out, err = run_cmd(cmd)

        cmd = f"{bugInfo.bug_exec} export -p tests.trigger -w {bugInfo.buggy_path}"
        out, err = run_cmd(cmd)
        properties["failed_test_names"] = out.split("\n")

        cmd = f"{bugInfo.bug_exec} export -p dir.bin.classes -w {bugInfo.buggy_path}"
        out, err = run_cmd(cmd)
        properties["src_class_prefix"] = out

        cmd = f"{bugInfo.bug_exec} export -p dir.bin.tests -w {bugInfo.buggy_path}"
        out, err = run_cmd(cmd)
        properties["test_class_prefix"] = out

        cmd = f"{bugInfo.bug_exec} export -p dir.src.classes -w {bugInfo.buggy_path}"
        out, err = run_cmd(cmd)
        properties["src_prefix"] = out

        cmd = f"{bugInfo.bug_exec} export -p dir.src.tests -w {bugInfo.buggy_path}"
        out, err = run_cmd(cmd)
        properties["test_prefix"] = out

        cmd = f"{bugInfo.bug_exec} export -p classes.modified -w {bugInfo.buggy_path}"
        out, err = run_cmd(cmd)
        properties["modified_classes"] = out.split("\n")

        with open(bugInfo.bug_path / "properties.json", "w") as f:
            json.dump(properties, f, indent=4)

    bugInfo.failed_test_names = properties["failed_test_names"]
    bugInfo.src_class_prefix = properties["src_class_prefix"]
    bugInfo.test_class_prefix = properties["test_class_prefix"]
    bugInfo.src_prefix = properties["src_prefix"]
    bugInfo.test_prefix = properties["test_prefix"]
    bugInfo.modified_classes = properties["modified_classes"]


def get_failed_tests(bugInfo: BugInfo) -> TestFailure:
    """Get the TestFailure object for a defect4j bug."""

    try:
        with open(bugInfo.test_failure_file, "rb") as f:
            test_failure = pickle.load(f)
            return test_failure
    except FileNotFoundError:
        pass

    # initialize test failure
    test_classes = {}
    for test_name in bugInfo.failed_test_names:
        test_class_name, test_method_name = test_name.split("::")
        test_case = TestCase(test_name)
        test_case.test_method = get_test_method(
            bugInfo,
            test_class_name,
            test_case.test_method_name,
        )
        if test_class_name not in test_classes:
            test_classes[test_class_name] = TestClass(
                test_class_name, [test_case]
            )
        else:
            test_classes[test_class_name].test_cases.append(test_case)

    # get modified methods as the buggy methods for evaluation
    bugInfo.logger.info(
        "[get test failure object] get modified methods as the buggy methods for evaluation..."
    )
    buggy_methods = get_modified_methods(bugInfo)

    bugInfo.logger.info(
        "[get test failure object] construct the TestFailure object..."
    )
    test_failure = TestFailure(
        bugInfo.project,
        bugInfo.bug_id,
        list(test_classes.values()),
        buggy_methods,
    )

    with open(bugInfo.test_failure_file, "wb") as f:
        pickle.dump(test_failure, f)
        bugInfo.logger.info(
            f"[get test failure object] Save failed tests to {bugInfo.test_failure_file}"
        )

    return test_failure


def merge_classes(
    class_name: str, covered_classes: List[Dict[str, JavaClass]]
) -> JavaClass:
    merged_class = JavaClass(class_name)
    all_covered_methods = [
        [m for m in c[class_name].methods.values() if m._covered]
        for c in covered_classes
    ]
    spc_methods = {}
    for covered_methods in all_covered_methods:
        for method in covered_methods:
            if method.inst_id not in spc_methods:
                spc_methods[method.inst_id] = method
    if (
        len(spc_methods) == 0
    ):  # no suspicious methods, which means nether of the methods in the class can be buggy
        return None
    merged_class.methods = spc_methods
    return merged_class


def filter_classes_Ochiai(
    project, bugID, extracted_classes: List[JavaClass]
) -> List[JavaClass]:
    """
    Filter the classes according to the top 20 result of Ochiai (https://github.com/Instein98/D4jOchiai).
    """

    def parse_ochiai(path):
        """
        Parse the Ochiai result from line level to method level.
        """
        res = []
        with open(path, "r") as f:
            line = f.readline()
            line = f.readline()
            while line:
                name, _ = line.split(";")
                name = name.split(":")[0]
                if res == []:
                    res.append(name)
                else:
                    if name != res[-1]:
                        res.append(name)
                if len(res) == 20:
                    break
                line = f.readline()
        return res

    ochiai_res_path = (
        Path("functions/OchiaiResult")
        / project
        / str(bugID)
        / "ochiai.ranking.csv"
    )
    if not ochiai_res_path.exists():
        print(f"Warning: No Ochiai result for {project}-{bugID}")
        return []
    ochiai_res = parse_ochiai(ochiai_res_path)
    filtered_classes = []
    bug_result_dict = {}
    for m in ochiai_res:
        class_name = m.split("#")[0].replace("$", ".")
        method_name = m.split("#")[1].split("(")[0]
        if class_name not in bug_result_dict:
            bug_result_dict[class_name] = [method_name]
        else:
            if method_name not in bug_result_dict[class_name]:
                bug_result_dict[class_name].append(method_name)

    # filter out useless classes and methods
    for javaclass in extracted_classes:
        if javaclass.class_name in bug_result_dict:
            new_javaclass = copy.deepcopy(javaclass)
            for inst_id in javaclass.methods:
                inst_method_name = inst_id.split("::")[1].split("(")[0]
                if (
                    inst_method_name
                    not in bug_result_dict[javaclass.class_name]
                ):
                    new_javaclass.methods.pop(inst_id)
            filtered_classes.append(new_javaclass)
    return filtered_classes


def filter_classes_Grace(
    project, bugID, extracted_classes: List[JavaClass]
) -> List[JavaClass]:
    """
    Filter the classes according to the top 10 result of Grace (https://github.com/yilinglou/Grace/tree/master).
    """
    filtered_classes = []
    with open("functions/grace_result_dict.json", "r") as f:
        grace_result = json.load(f)
    if str(bugID) not in grace_result[project]:
        print(f"Warning: No Grace result for {project}-{bugID}")
        return filtered_classes
    bug_result = grace_result[project][str(bugID)]["top10_result"]
    bug_result_dict = {}
    for m in bug_result:
        class_name = m.split(":")[0].split("$")[0]
        method_name = m.split(":")[1].split("(")[0]
        if class_name not in bug_result_dict:
            bug_result_dict[class_name] = [method_name]
        else:
            if method_name not in bug_result_dict[class_name]:
                bug_result_dict[class_name].append(method_name)

    # filter out useless classes and methods
    for javaclass in extracted_classes:
        if javaclass.class_name in bug_result_dict:
            new_javaclass = copy.deepcopy(javaclass)
            for inst_id in javaclass.methods:
                inst_method_name = inst_id.split("::")[1].split("(")[0]
                if (
                    inst_method_name
                    not in bug_result_dict[javaclass.class_name]
                ):
                    new_javaclass.methods.pop(inst_id)
            filtered_classes.append(new_javaclass)
    return filtered_classes


def run_all_tests(bugInfo: BugInfo, test_failure: TestFailure):
    """
    Run all tests to collect bug information and coverage information.
    """

    for test_class in test_failure.test_classes:
        bugInfo.logger.info(
            f"[run all tests] test class: {bugInfo.project}-{bugInfo.bug_id} {test_class.name}"
        )
        for test_case in test_class.test_cases:
            bugInfo.logger.info(
                f"[run all tests]   \u14aa test case: {bugInfo.project}-{bugInfo.bug_id} {test_case.name}"
            )
            with Timer(bugInfo.logger, "instrumentation"):
                run_test_with_instrument(test_case, bugInfo)


def get_class_name_from_msg(tmp_path, test_class):
    """
    Some buggy classes may have low method level coverage proportion rank because of the crash,
    so we add these classes according to the error messages.
    """

    def get_target_classes(match):
        target_classes = []
        class_name = match.split(".")[-1]
        class_names = re.findall(r"[A-Z][a-zA-Z0-9]*", class_name)
        for class_name in class_names:
            if "Tests" in class_name:
                target_classes.append(class_name.replace("Tests", ""))
            elif "Test" in class_name:
                target_classes.append(class_name.replace("Test", ""))
            else:
                target_classes.append(class_name)
        return target_classes

    extra_class_names = set()
    for test_case in test_class.test_cases:
        test_name = test_case.name
        test_tmp_dir = tmp_path / test_class.name / test_name
        stack_trace_file = test_tmp_dir / "stack_trace.txt"
        with open(stack_trace_file, "r") as f:
            stack_trace = f.read()
        matches = re.findall(r"\b(?:\w*\.)+[A-Z]\w*", stack_trace)
        matches = list(set(matches))
        candidates = []
        for match in matches:
            candidates.extend(get_target_classes(match))
        for candidate in candidates:
            extra_class_names.add(candidate)
    return list(extra_class_names)


def parse_test_report(lines: List[str], bug_info: BugInfo, max_lines=50):
    """Seperate the raw test information into output and report."""

    def is_in_project(line: str) -> bool:
        """Check if the stack trace line is in the project."""
        # get class full name
        pattern = re.compile(r"(\w+\.)+\w+")
        method_name = pattern.search(line).group()
        class_name = ".".join(method_name.split(".")[:-1])
        if bug_info.get_class_file(class_name):
            return True
        return False

    def compress_trace(trace: str, test_method_name: str) -> List[str]:
        compressed_trace = []
        repeat_times = 1
        last_line = ""
        for i, line in enumerate(trace):
            if i > max_lines:  # too many lines
                break
            if line.startswith("\tat"):
                if f".{test_method_name}(" in line:  # reach the test method
                    compressed_trace.append(line)
                    break
                elif not is_in_project(
                    line
                ):  # ignore the lines not in the project
                    continue
                else:  # fold the repeated lines
                    if line == last_line:
                        repeat_times += 1
                    else:
                        if repeat_times > 1:
                            compressed_trace[-1] = (
                                f"\t{compressed_trace[-1].strip()} (repeated {repeat_times} times)"
                            )
                        compressed_trace.append(line)
                        repeat_times = 1
                    last_line = line
            else:
                compressed_trace.append(line)
        if repeat_times > 1:
            compressed_trace[-1] = (
                f"\t{compressed_trace[-1].strip()} (repeated {repeat_times} times)"
            )
        return compressed_trace

    output = []
    trace = []
    test_method = lines[0].strip("\n").split("::")[1]
    for i, line in enumerate(lines):
        if i < 2:
            trace.append(line)
        elif line.startswith("\tat"):
            trace.append(line)
        else:
            output.append(line)

    if len(output) > max_lines:
        output = output[:max_lines]
        output.append("// omitting the rest ...")
    compressed_trace = compress_trace(trace, test_method)
    return output, compressed_trace


def parse_stack_trace(lines):
    """parse the bug error stack trace, find the location of the error assert statement, e.g.:

    --- com.google.javascript.jscomp.TypeCheckTest::testBadInterfaceExtendsNonExistentInterfaces
    java.lang.NullPointerException
        at com.google.javascript.jscomp.TypeCheck.checkInterfaceConflictProperties(TypeCheck.java:1574)
        at com.google.javascript.jscomp.TypeCheck.visitFunction(TypeCheck.java:1664)
        at com.google.javascript.jscomp.TypeCheck.visit(TypeCheck.java:778)
        at com.google.javascript.jscomp.NodeTraversal.traverseBranch(NodeTraversal.java:505)
        at com.google.javascript.jscomp.NodeTraversal.traverseBranch(NodeTraversal.java:498)
        at com.google.javascript.jscomp.NodeTraversal.traverseWithScope(NodeTraversal.java:343)
        at com.google.javascript.jscomp.TypeCheck.check(TypeCheck.java:404)
        at com.google.javascript.jscomp.TypeCheck.process(TypeCheck.java:375)
        at com.google.javascript.jscomp.TypeCheck.processForTesting(TypeCheck.java:393)
        at com.google.javascript.jscomp.TypeCheckTest.testTypes(TypeCheckTest.java:11530)
        at com.google.javascript.jscomp.TypeCheckTest.testBadInterfaceExtendsNonExistentInterfaces(TypeCheckTest.java:3780)

    return 3780
    """
    test_classs = []
    test_method_name = ""
    location = -1
    for line in lines:
        if line.startswith("---"):
            _, clazz, test_method_name = re.split(r" |::", line.strip("\n"))
            test_classs.append(clazz)
        elif line.startswith("\tat"):
            method_full_name = re.search(r"at (.*)\(", line).group(1)
            method_name = method_full_name.split(".")[-1]
            clazz = ".".join(method_full_name.split(".")[:-1])
            if method_name == test_method_name:
                location = int(re.search(r":(\d+)", line).group(1))
                break
    if location == -1:
        print("Warning: No assert statement found in stack trace!")
    return location


def parse_inst_method_sig(inst_method):
    """
    To match the code, parse the method signature in instrument file to expected format, e.g.:
    testTypes(java.lang.String,java.lang.String[]) -> testTypes(String,String[])

    We only focus on the coverage of source code methods, so if the method not exists in source code, return None, e.g.:
    access$100(com.google.javascript.rhino.Node) -> None
    """
    if re.match(r"access\$.*", inst_method):
        return None

    match = re.search(r"(.*)\((.*)\)", inst_method)
    method_name = match.group(1)
    params = match.group(2).split(",")

    new_params = []
    for param in params:
        new_param = param.replace("$", ".")  # ignore the inner class name
        new_param = new_param.split(".")[-1]  # ignore the package name
        if new_param == "LatticeElement":
            new_param = "L"
        new_params.append(new_param)
    simple_params = ",".join(new_params)
    method_sig = f"{method_name}({simple_params})"
    return method_sig


def parse_test_run_log(lines):
    """parse the method run log of test, find all runed methods, e.g.:

    com.google.javascript.jscomp.TypeCheckTest testBadInterfaceExtendsNonExistentInterfaces() void
    com.google.javascript.jscomp.TypeCheckTest testTypes(java.lang.String,java.lang.String[]) void
    com.google.javascript.jscomp.TypeCheckTest makeTypeCheck() com.google.javascript.jscomp.TypeCheck
    """
    methods = {}
    for line in lines:
        if line == "\n":
            continue
        clazz, method, _ = line.split(" ")
        method = parse_inst_method_sig(method)
        try:
            methods[clazz].append(method)
        except KeyError:
            methods[clazz] = [method]
    return methods


def parse_coverage(bugInfo: BugInfo, run_file: Path):
    """Parse the coverage information from the instrument file, e.g.:

    com.google.javascript.jscomp@PrepareAst:PrepareAst(43-46)
    com.google.javascript.jscomp.graph@LinkedDirectedGraph$LinkedDirectedGraphEdge:LinkedDirectedGraph$LinkedDirectedGraphEdge(472-476)

    ->

    {
        "com/google/javascript/jscomp/PrepareAst.java": [
            (43, 46, "com.google.javascript.jscomp", "PrepareAst", "PrepareAst#43-46")
        ],
        "com/google/javascript/jscomp/graph/LinkedDirectedGraph.java": [
            (472, 476, "com.google.javascript.jscomp.graph", "LinkedDirectedGraph", "LinkedDirectedGraphEdge.LinkedDirectedGraphEdge#472-476")
        ]
    }
    """
    assert run_file.exists(), "Error: No instrument file:\n"
    text = run_file.read_text()

    pattern = re.compile(
        r"^(?P<package>[a-zA-Z0-9_.]+)@(?P<class>[a-zA-Z0-9_$]+):(?P<method>[a-zA-Z0-9_]+)\((?P<start>\d+)-(?P<end>\d+)\)$"
    )
    result = {}

    for line in text.strip().split("\n"):
        match = pattern.match(line)
        if match:
            package = match.group("package")
            class_name = match.group("class")
            method_name = match.group("method")
            start_line = int(match.group("start"))
            end_line = int(match.group("end"))
            outer_class_name = class_name
            inner_class_name = None

            # solve the inner class name
            if "$" in class_name:
                outer_class_name = class_name.split("$")[0]
                inner_class_name = "$".join(class_name.split("$")[1:])

            rel_file_path = (
                f"{package.replace('.', '/')}/{outer_class_name}.java"
            )
            src_file_path = (
                bugInfo.buggy_path / bugInfo.src_prefix / rel_file_path
            )
            test_file_path = (
                bugInfo.buggy_path / bugInfo.test_prefix / rel_file_path
            )
            assert (
                src_file_path.exists() or test_file_path.exists()
            ), f"Error: No source file {src_file_path} or test file {test_file_path}"

            if rel_file_path in result:
                result[rel_file_path].append(
                    (
                        start_line,
                        end_line,
                        package,
                        outer_class_name,
                        inner_class_name,
                    )
                )
            else:
                result[rel_file_path] = [
                    (
                        start_line,
                        end_line,
                        package,
                        outer_class_name,
                        inner_class_name,
                    )
                ]
    return result


example_output = """--- com.google.javascript.jscomp.TypeCheckTest::testConversionFromInterfaceToRecursiveConstructor
java.lang.StackOverflowError
	at com.google.javascript.rhino.jstype.ObjectType.isUnknownType(ObjectType.java:569)
	at com.google.javascript.rhino.jstype.JSType.checkEquivalenceHelper(JSType.java:659)
	at com.google.javascript.rhino.jstype.JSType.isEquivalentTo(JSType.java:626)
	at com.google.javascript.rhino.jstype.JSType.isSubtypeHelper(JSType.java:1350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:319)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
	at com.google.javascript.rhino.jstype.PrototypeObjectType.isSubtype(PrototypeObjectType.java:350)
"""


def test():
    a, b = parse_test_report(example_output.splitlines())
    print(b)


if __name__ == "__main__":
    test()
