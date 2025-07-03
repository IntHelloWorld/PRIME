import hashlib
import logging
import logging.config
import os
import sys
from pathlib import Path
from time import time
from typing import Optional

import yaml

log_config = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "%(levelname)s - %(asctime)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "simple",
            "level": "DEBUG",
            "filename": "",
            "mode": "w",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "default": {"handlers": ["file", "console"], "level": "DEBUG"},
    },
}


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def asdict(self):
        return {
            k: v.asdict() if isinstance(v, Config) else v
            for k, v in self.__dict__.items()
        }


class BugInfo:
    def __init__(self, args, eval=False):
        self.root_path = Path(__file__).resolve().parents[1]

        # bug info
        self.project = args.project
        self.bug_id = args.bugID
        self.bug_name = f"{args.project}@{args.bugID}"
        config_file = Path(args.config)
        self.config_name = config_file.stem
        self.config_hash = BugInfo.get_config_hash(config_file)

        # global paths/files
        self.output_path = self.root_path / "DebugResult"
        self.evaluation_path = self.root_path / "EvaluationResult"
        self.dataset_path = self.root_path / "dataset"
        self.res_path = (
            self.output_path
            / self.config_name
            / args.project
            / f"{args.project}-{args.bugID}"
        )
        self.res_file = self.res_path / "result.json"
        self.bug_path = (
            self.dataset_path / "bugInfo" / args.project / self.bug_name
        )
        self.test_failure_file = self.bug_path / "test_failure.pkl"
        self.checkout_tmp_path = self.output_path / self.config_hash
        self.proj_tmp_path = self.checkout_tmp_path / self.bug_name
        self.buggy_path = self.proj_tmp_path / "buggy"
        self.fixed_path = self.proj_tmp_path / "fixed"

        # temp paths for each test case
        self.failed_test_names = []
        self.modified_classes = []
        self.src_prefix = None
        self.test_prefix = None
        self.src_class_prefix = None
        self.test_class_prefix = None

        if not eval:
            # Create directories if they don't exist
            for path in [self.res_path, self.bug_path]:
                path.mkdir(parents=True, exist_ok=True)

        if Path(args.config).exists():
            # read config file
            with Path(args.config).open("r") as f:
                self.config = Config(yaml.safe_load(f))

            # dependencies
            self.java_agent_lib = Path(self.config.dependencies.java_agent_lib)
            self.bug_exec = self.config.dependencies.bug_exec

        # init logger with time
        if eval:
            if "file" in log_config["handlers"]:
                log_config["handlers"].pop("file")
        else:
            log_config["handlers"]["file"]["filename"] = str(
                self.res_path / f"{int(time())}.log"
            )
            logging.config.dictConfig(log_config)
            self.logger = logging.getLogger("default")

    def get_class_file(self, class_name) -> Optional[Path]:
        class_file = (
            self.buggy_path
            / self.src_prefix
            / Path(class_name.split("$")[0].replace(".", "/") + ".java")
        )

        if not class_file.exists():
            class_file = (
                self.buggy_path
                / self.test_prefix
                / Path(class_name.split("$")[0].replace(".", "/") + ".java")
            )

        if not class_file.exists():
            return None

        return class_file

    @staticmethod
    def get_config_hash(config_file: Path) -> str:
        md5_hash = hashlib.md5()
        md5_hash.update(str(config_file).encode("utf-8"))
        return md5_hash.hexdigest()
