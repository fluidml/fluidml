import logging
import logging.handlers
from multiprocessing import Queue
from threading import Thread
import threading


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
    try:
        from rich.logging import RichHandler as StreamHandler
    except ImportError:
        from logging import StreamHandler
    root = logging.getLogger()
    formatter = logging.Formatter('%(processName)-10s\n%(message)s')
    stream_handler = StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    root.setLevel(logging.DEBUG)
