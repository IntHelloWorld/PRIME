import argparse
import multiprocessing
import sys
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent
sys.path.append(str(root))

import subprocess


def run_script(project, bugID, config):
    cmd = (
        f"python run.py --project {project} --bugID {bugID} --config {config}"
    )
    subprocess.run(cmd.split())


def main(dataset_file, config_file, processes):
    df = pd.read_csv(dataset_file, header=None)
    bug_names = df.iloc[:, 0].tolist()
    root_path = Path(__file__).resolve().parent

    # TODO: Control bug for test
    # bug_names = [
    #     b
    #     for b in bug_names
    #     if not b.startswith("Closure") and not b.startswith("Chart")
    # ]
    # bug_names = [
    #     b for b in bug_names if b.startswith("Cli") or b.startswith("Csv")
    # ]
    # bug_names = ["Mockito_5"]

    run_failed = []

    if processes > 1:
        with multiprocessing.Pool(processes=processes) as pool:
            async_results = []

            for bug_name in bug_names:
                proj, bug_id = bug_name.split("_")
                debug_result_file = (
                    root_path
                    / "DebugResult"
                    / Path(config_file).stem
                    / proj
                    / f"{proj}-{bug_id}"
                    / "debug_result.json"
                )
                if not debug_result_file.exists():
                    async_result = pool.apply_async(
                        run_script, (proj, bug_id, config_file)
                    )
                    async_results.append(async_result)

            for i, async_result in enumerate(async_results):
                try:
                    async_result.get()
                except Exception as e:
                    print(e)
                    run_failed.append(bug_names[i])
    else:
        for bug_name in bug_names:
            proj, bug_id = bug_name.split("_")
            debug_result_file = (
                root_path
                / "DebugResult"
                / Path(config_file).stem
                / proj
                / f"{proj}-{bug_id}"
                / "debug_result.json"
            )
            if not debug_result_file.exists():
                run_script(proj, bug_id, config_file)

    if run_failed:
        run_failed_file = root / "run_all_failed.txt"
        with open(run_failed_file, "w") as f:
            f.write("\n".join(run_failed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all bugs")
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset file",
        # default="dataset/Mockito.csv",
        # default="dataset/Time.csv",
        default="dataset/all_bugs.csv",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="config file",
        # default="config/default.yml",
        # default="config/default_path_select_path2.yml",
        # default="config/default_path_select_path6.yml",
        # default="config/default_path_select_path8.yml",
        # default="config/default_path_select_branch3.yml",
        # default="config/default_path_select_branch4.yml",
        default="config/default_path_select_branch5.yml",
    )
    parser.add_argument(
        "--processes",
        type=int,
        help="processes",
        default=4,
    )
    args = parser.parse_args()
    main(args.dataset, args.config, args.processes)
