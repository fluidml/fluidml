from abc import abstractmethod
import logging
import logging.handlers
from multiprocessing import Process, Queue
from typing import Dict


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
        h = logging.handlers.QueueHandler(self.logging_queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)

    @abstractmethod
    def _work(self):
        raise NotImplementedError

    def work(self):
        try:
            self._configure_logging()
            self._work()
        except Exception as e:
            if self.exit_on_error:
                self.exception['message'] = e
            raise
