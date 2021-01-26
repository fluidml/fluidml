import logging
import logging.handlers
from multiprocessing import Queue
from threading import Thread


class LoggingListener(Thread):
    """ Listens to and handles child process log messages
    This class, when instantiated, listens to the logging queue to receive log messages from child processes
    and handles these messages using the configured root logger in the main process.
    """
    def __init__(self, logging_queue: Queue):
        super().__init__(target=self.work,
                         args=())
        self.logging_queue = logging_queue

    def work(self):
        while True:
            record = self.logging_queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
