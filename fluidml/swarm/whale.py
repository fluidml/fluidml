from abc import abstractmethod
import logging
from multiprocessing import Process, Queue
import sys
from typing import Dict

from fluidml.common.logging import QueueHandler, StdoutHandler, StderrHandler


class Whale(Process):
    def __init__(self,
                 exception: Dict[str, Exception],
                 exit_on_error: bool,
                 logging_queue: Queue):
        super().__init__(target=self.work,
                         args=())
        self.exception = exception
        self.exit_on_error = exit_on_error
        self.logging_queue = logging_queue

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
        try:
            self._configure_logging()
            self._redirect_stdout_stderr()
            self._work()
        except Exception as e:
            if self.exit_on_error:
                self.exception['message'] = e
            raise
