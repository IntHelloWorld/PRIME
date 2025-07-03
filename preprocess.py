import argparse
import multiprocessing as mp
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent
sys.path.append(str(root))

from src.config import BugInfo
from src.interfaces.d4j import (
    check_out,
    get_failed_tests,
    get_properties,
    run_all_tests,
)
from src.repograph.construct_graph import create_repo_graph


def worker(bug_name, config_file):
    proj, bug_id = bug_name.split("_")
    run_one(proj, bug_id, config_file)


def check_exists(bug_name):
    proj, bug_id = bug_name.split("_")
    combined_graph_file = (
        root
        / "dataset"
        / "bugInfo"
        / proj
        / f"{proj}@{bug_id}"
        / "combined_graph.pkl"
    )
    if combined_graph_file.exists():
        print(f"Skip {bug_name}")
        return True
    return False


def main(dataset_file, config_file, num_processes):
    df = pd.read_csv(dataset_file, header=None)
    bug_names = df.iloc[:, 0].tolist()

    # TODO: Control bug for preprocess
    bug_names = ["Mockito_5"]

    if num_processes > 1:
        preprocess_failed = []
        with mp.Pool(num_processes) as pool:
            async_results = []
            for bug_name in bug_names:
                if check_exists(bug_name):
                    continue
                async_result = pool.apply_async(
                    worker, (bug_name, config_file)
                )
                async_results.append(async_result)

            for i, async_result in enumerate(async_results):
                try:
                    async_result.get()
                except Exception as e:
                    preprocess_failed.append(f"{bug_names[i]} {str(e)}")

        preprocess_failed_file = root / "preprocess_failed.txt"
        if preprocess_failed:
            with open(preprocess_failed_file, "w") as f:
                f.write("\n".join(preprocess_failed))
    else:
        for bug_name in bug_names:
            if check_exists(bug_name):
                continue
            worker(bug_name, config_file)

    # clean up
    checkout_tmp_path = (
        root / "DebugResult" / BugInfo.get_config_hash(config_file)
    )
    shutil.rmtree(checkout_tmp_path)


def run_one(proj, bug_id, config_file):
    args = Namespace(project=proj, bugID=bug_id, config=config_file)
    bug_info = BugInfo(args)

    check_out(bug_info)

    # collect basic bug information
    get_properties(bug_info)
    test_failure_obj = get_failed_tests(bug_info)
    run_all_tests(bug_info, test_failure_obj)

    # create the raw repository graph
    create_repo_graph(bug_info, test_failure_obj)

    # clean up
    shutil.rmtree(bug_info.proj_tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess all bugs")
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset file",
        default="dataset/Mockito.csv",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="config file",
        default="config/preprocess.yml",
    )
    parser.add_argument(
        "--processes",
        type=int,
        help="processes",
        default=1,
    )
    args = parser.parse_args()
    main(args.dataset, args.config, args.processes)
