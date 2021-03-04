from abc import abstractmethod
import logging
from multiprocessing import Process, Queue, Lock, Event
import os
import sys

from fluidml.common.logging import QueueHandler, StdoutHandler, StderrHandler

logger = logging.getLogger(__name__)


class Whale(Process):
    def __init__(self,
                 exit_event: Event,
                 exit_on_error: bool,
                 logging_queue: Queue,
                 lock: Lock):
        super().__init__(target=self.work,
                         args=())
        self.exit_event = exit_event
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
                    logger.exception(e)
                    self.exit_event.set()
            raise
        finally:
            # sys.stdout = sys.__stdout__
            # sys.stderr = sys.__stderr__
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
