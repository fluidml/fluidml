import logging
import sys
from abc import abstractmethod
from multiprocessing import Process, Queue, RLock, Event

from fluidml.common.logging import QueueHandler, StdoutHandler, StderrHandler

logger = logging.getLogger(__name__)


class Whale(Process):
    def __init__(
        self,
        exit_event: Event,
        exit_on_error: bool,
        logging_queue: Queue,
        error_queue: Queue,
        logging_lvl: int,
        lock: RLock,
        use_multiprocessing: bool = True,
    ):
        self.use_multiprocessing = use_multiprocessing
        # only init the Process parent class if multiprocessing is configured.
        if self.use_multiprocessing:
            super().__init__(target=self.work, args=())
            self._logging_queue = logging_queue
            self._logging_lvl = logging_lvl

        self.exit_event = exit_event
        self._exit_on_error = exit_on_error
        self._error_queue = error_queue
        self._lock = lock

    def _configure_logging(self):
        h = QueueHandler(self._logging_queue, self.name)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(self._logging_lvl)

    def _redirect_stdout_stderr(self):
        sys.stdout = StdoutHandler(self._logging_queue, self.name)
        sys.stderr = StderrHandler(self._logging_queue, self.name)

    @abstractmethod
    def _work(self):
        raise NotImplementedError

    def work(self):
        try:
            if self.use_multiprocessing:
                self._redirect_stdout_stderr()
                self._configure_logging()
            self._work()
        except Exception as e:
            logger.exception(e)
            self._error_queue.put(e)
            self.exit_event.set()
