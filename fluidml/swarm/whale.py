from abc import abstractmethod
import logging
import multiprocessing
from multiprocessing import Process, Queue, Lock
import os
import sys
from typing import Dict

from fluidml.common.logging import QueueHandler, StdoutHandler, StderrHandler

logger = logging.getLogger(__name__)


class Whale(Process):
    def __init__(self,
                 exception: Dict[str, Exception],
                 exit_on_error: bool,
                 logging_queue: Queue,
                 lock: Lock):
        super().__init__(target=self.work,
                         args=())
        self.exception = exception
        self._exit_on_error = exit_on_error
        self._logging_queue = logging_queue
        self._lock = lock

    def _configure_logging(self):
        h = QueueHandler(self._logging_queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)

    def _redirect_stdout_stderr(self):
        sys.stdout = StdoutHandler(self._logging_queue)
        sys.stderr = StderrHandler(self._logging_queue)

    @abstractmethod
    def _work(self):
        raise NotImplementedError

    def work(self):
        try:
            self._configure_logging()
            self._redirect_stdout_stderr()
            self._work()
        except Exception as e:
            if self._exit_on_error:
                with self._lock:
                    # worker_name = multiprocessing.current_process().name
                    # logger.exception(f'{worker_name:13}{e}')
                    logger.exception(e)
                    self.exception['value'] = e
            raise
        finally:
            # sys.stdout = sys.__stdout__
            # sys.stderr = sys.__stderr__
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
