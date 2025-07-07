import argparse
import sys
from argparse import Namespace
from pathlib import Path

from src.core.debug_agent import DebugAgent

root = Path(__file__).resolve().parent
sys.path.append(str(root))

from src.config import BugInfo
from src.interfaces.d4j import get_failed_tests, get_properties


def main(proj, bug_id, config_file):
    args = Namespace(project=proj, bugID=bug_id, config=config_file)
    bug_info = BugInfo(args)

    # collect basic bug information from cache
    # for preprocessing please run `preprocess.py`
    get_properties(bug_info)
    test_failure_obj = get_failed_tests(bug_info)

    # run the debug agent
    debug_agent = DebugAgent(bug_info)
    debug_agent.run(test_failure_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the debug agent")
    parser.add_argument(
        "--project", type=str, help="project name", default="Mockito"
    )
    parser.add_argument("--bugID", type=str, help="bug id", default="5")
    parser.add_argument(
        "--config",
        type=str,
        help="config file",
        default="config/default_new_purne.yml",
    )
    args = parser.parse_args()
    main(args.project, args.bugID, args.config)
