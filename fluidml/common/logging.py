import logging
import logging.handlers
from multiprocessing import Queue
from threading import Thread
import threading

from rich.logging import RichHandler


class LoggingListener(Thread):
    """ Listens to and handles child process log messages
    This class, when instantiated, listens to the logging queue to receive log messages from child processes
    and handles these messages using the configured root logger in the main process.
    """

    def __init__(self, logging_queue: Queue):
        super().__init__(target=self.work,
                         args=())
        self._logging_queue = logging_queue
        self._stop_event = threading.Event()

    def work(self):
        while True:
            record = self._logging_queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)

    def stop(self):
        self._stop_event.set()


def configure_logging():
    root = logging.getLogger()
    rich_formatter = logging.Formatter('%(processName)-10s\n%(message)s')
    rich_handler = RichHandler(rich_tracebacks=True, markup=True)
    rich_handler.setLevel(logging.DEBUG)
    rich_handler.setFormatter(rich_formatter)
    root.addHandler(rich_handler)
    root.setLevel(logging.DEBUG)
