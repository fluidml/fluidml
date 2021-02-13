from abc import abstractmethod
import logging
import multiprocessing
from multiprocessing import Process, Queue, Lock
import sys
# import traceback
from typing import Dict

from rich.console import Console

from fluidml.common.logging import QueueHandler, StdoutHandler, StderrHandler

logger = logging.getLogger(__name__)
console = Console()


class Whale(Process):
    def __init__(self,
                 exception: Dict[str, Exception],
                 exit_on_error: bool,
                 logging_queue: Queue,
                 lock: Lock):
        super().__init__(target=self.work,
                         args=())
        self.exception = exception
        self.exit_on_error = exit_on_error
        self.logging_queue = logging_queue
        self.lock = lock

    def _configure_logging(self):
        h = QueueHandler(self.logging_queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)

    def _redirect_stdout_stderr(self):
        sys.stdout = StdoutHandler(self.logging_queue)
        sys.stderr = StderrHandler(self.logging_queue)

    @abstractmethod
    def _work(self):
        raise NotImplementedError

    def work(self):
        self._configure_logging()
        self._redirect_stdout_stderr()
        try:
            self._work()
        except Exception as e:
            if self.exit_on_error:
                # exc_type, exc_value, exc_traceback = sys.exc_info()
                # traceback.print_exception(exc_type, exc_value, exc_traceback)
                with self.lock:
                    logger.exception(e)
                    console.print_exception(extra_lines=2)
                    self.exception['value'] = e
            raise
