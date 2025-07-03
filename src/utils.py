import logging
import time
from contextlib import ContextDecorator
from typing import Optional


class Timer(ContextDecorator):
    def __init__(self, logger: logging.Logger, desc: str):
        self.logger = logger
        self.desc = desc
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self.logger.debug(f"{self.desc} - time:{elapsed_time:.2f}s")
        return False
