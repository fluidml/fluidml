import copy
import logging
from logging import LogRecord
from multiprocessing import Queue
from queue import Empty
import sys
from threading import Thread
import threading
from typing import List, Union

try:
    from rich.console import Console
    from rich.logging import RichHandler
    console = Console(stderr=True)
    logging.lastResort = RichHandler(console=console,
                                     level='WARNING',
                                     rich_tracebacks=True,
                                     tracebacks_extra_lines=2,
                                     show_path=False)
    rich_logging = True
except ImportError:
    from logging import StreamHandler
    rich_logging = False

try:
    from tblib import pickling_support
    pickling_support.install()
    tb_lib = True
except ImportError:
    tb_lib = False


class QueueHandler(logging.Handler):
    """
    This handler sends events to a queue. Typically, it would be used together
    with a multiprocessing Queue to centralise logging to file in one process
    (in a multi-process application), so as to avoid file write contention
    between processes.
    This code is new in Python 3.2, but this class can be copy pasted into
    user code for use with earlier Python versions.
    """

    def __init__(self, queue: Queue):
        """
        Initialise an instance, using the passed queue.
        """
        logging.Handler.__init__(self)
        self.queue = queue

    def enqueue(self, record: List[LogRecord]):
        """
        Enqueue a record.
        The base implementation uses put_nowait. You may want to override
        this method if you want to use blocking, timeouts or custom queue
        implementations.
        """
        self.queue.put_nowait(record)

    def prepare(self, record: LogRecord):
        """
        Prepares a record for queuing. The object returned by this method is
        enqueued.
        The base implementation formats the record to merge the message
        and arguments, and removes unpickleable items from the record
        in-place.
        You might want to override this method if you want to convert
        the record to a dict or JSON string, or send a modified copy
        of the record while leaving the original intact.
        """
        # The format operation gets traceback text into record.exc_text
        # (if there's exception data), and also returns the formatted
        # message. We can then use this to replace the original
        # msg + args, as these might be unpickleable. We also zap the
        # exc_info and exc_text attributes, as they are no longer
        # needed and, if not None, will typically not be pickleable.
        if not tb_lib:
            msg = self.format(record)
            # bpo-35726: make copy of record to avoid affecting other handlers in the chain.
            record = copy.copy(record)
            record.message = msg
            record.msg = msg
            record.args = None
            record.exc_info = None
            record.exc_text = None
        return ['log_msg', record]

    def emit(self, record: LogRecord):
        """
        Emit a record.
        Writes the LogRecord to the queue, preparing it for pickling first.
        """
        try:
            self.enqueue(self.prepare(record))
        except Exception:
            self.handleError(record)


class StdoutHandler:
    def __init__(self, queue: Queue):
        self.queue = queue

    def write(self, msg):
        self.queue.put(['stdout_msg', msg])

    @staticmethod
    def flush():
        sys.__stdout__.flush()


class StderrHandler:
    def __init__(self, queue: Queue):
        self.queue = queue

    def write(self, msg):
        self.queue.put(['stderr_msg', msg])

    @staticmethod
    def flush():
        sys.__stderr__.flush()


class LoggingListener(Thread):
    """ Listens to and handles child process log messages
    This class, when instantiated, listens to the logging queue to receive log messages from child processes
    and handles these messages using the configured root logger in the main process.
    """

    def __init__(self,
                 logging_queue: Queue
                 ):
        super().__init__(target=self.work,
                         args=())
        self._logging_queue = logging_queue
        self._stop_event = threading.Event()

        self.record_type_to_handle_fn = {'log_msg': LoggingListener._handle_log_msg,
                                         'stdout_msg': LoggingListener._handle_stdout_msg,
                                         'stderr_msg': LoggingListener._handle_stderr_msg}

    @staticmethod
    def _handle_log_msg(record: LogRecord):
        logger = logging.getLogger(record.name)
        logger.handle(record)

    @staticmethod
    def _handle_stdout_msg(record: str):
        sys.stdout.write(record)
        sys.stdout.flush()

    @staticmethod
    def _handle_stderr_msg(record: str):
        sys.stderr.write(record)
        sys.stderr.flush()

    def work(self):
        while True:
            # Using queue.get(block=False) is necessary for python 3.6. queue.get() sometimes
            # leads to weird deadlocks when waiting for logging messages from child processes.
            try:
                record = self._logging_queue.get(block=False, timeout=0.01)
            except Empty:
                continue

            if record is None:
                break
            record_type, record = record

            handle_record = self.record_type_to_handle_fn[record_type]
            handle_record(record)

    def stop(self):
        self._stop_event.set()


def configure_logging(level: Union[str, int] = 'INFO'):
    assert level in ['DEBUG', 'INFO', 'WARNING', 'WARN', 'ERROR', 'FATAL', 'CRITICAL',
                     10, 20, 30, 40, 50]
    logger = logging.getLogger()
    if rich_logging:
        formatter = logging.Formatter('%(processName)-13s%(message)s')
        stream_handler = RichHandler(
            rich_tracebacks=True,
            tracebacks_extra_lines=2,
            show_path=False
        )
    else:
        formatter = logging.Formatter('%(asctime)s %(levelname)-10s %(processName)-10s %(message)s')
        stream_handler = StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(level)
