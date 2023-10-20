import copy
import logging
import os
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from logging import LogRecord
from multiprocessing import Event, Lock, Queue
from queue import Empty
from threading import Thread
from typing import IO, Dict, List, Optional, Tuple, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as rich_install
from tblib import pickling_support

from fluidml.exception import TmuxError

pickling_support.install()
logger_ = logging.getLogger(__name__)


def configure_logging(
    level: Union[str, int] = "INFO",
    rich_logging: bool = True,
    rich_traceback: bool = True,
):
    """A Convenience function to initialize and configure logging in the application.

    Args:
        level: Logging level to use, e.g. "DEBUG", "INFO", etc.
        rich_logging: Whether to use the `rich` library to prettify logging.
        rich_traceback: Whether to use the `rich` library to prettify error tracebacks.
    """

    assert level in [
        "DEBUG",
        "INFO",
        "WARNING",
        "WARN",
        "ERROR",
        "FATAL",
        "CRITICAL",
        10,
        20,
        30,
        40,
        50,
    ]
    if rich_traceback:
        rich_install(extra_lines=2)
    logger = logging.getLogger()
    stream_handler = create_stream_handler(rich_logging=rich_logging)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)
    logger.setLevel(level)


def create_stream_handler(stream: Optional[IO] = None, rich_logging: bool = True) -> logging.Handler:
    if rich_logging:
        console_ = Console(file=stream, color_system="standard") if stream else None
        formatter = logging.Formatter("%(processName)-13s%(message)s")
        stream_handler = RichHandler(
            console=console_,
            rich_tracebacks=True,
            tracebacks_extra_lines=2,
            show_path=False,
            omit_repeated_times=False,
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(processName)-13s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler(stream) if stream else logging.StreamHandler()

    stream_handler.setFormatter(formatter)
    return stream_handler


class QueueHandler(logging.Handler):
    """
    This handler sends events to a queue. Typically, it would be used together
    with a multiprocessing Queue to centralise logging to file in one process
    (in a multi-process application), so as to avoid file write contention
    between processes.
    This code is new in Python 3.2, but this class can be copy pasted into
    user code for use with earlier Python versions.
    """

    def __init__(self, queue: Queue, worker_name: str):
        """
        Initialise an instance, using the passed queue.
        """
        logging.Handler.__init__(self)
        self.queue = queue
        self.worker_name = worker_name

    def enqueue(self, record: Tuple[str, str, LogRecord]):
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

        # Not needed, since we use tblib
        # msg = self.format(record)
        # # bpo-35726: make copy of record to avoid affecting other handlers in the chain.
        # record = copy.copy(record)
        # record.message = msg
        # record.msg = msg
        # record.args = None
        # record.exc_info = None
        # record.exc_text = None
        return "log_msg", self.worker_name, record

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
    def __init__(self, queue: Queue, worker_name: str):
        self.queue = queue
        self.worker_name = worker_name

    def write(self, msg):
        self.queue.put(("stdout_msg", self.worker_name, msg))

    @staticmethod
    def flush():
        sys.__stdout__.flush()


class StderrHandler:
    def __init__(self, queue: Queue, worker_name: str):
        self.queue = queue
        self.worker_name = worker_name

    def write(self, msg):
        self.queue.put(("stderr_msg", self.worker_name, msg))

    @staticmethod
    def flush():
        sys.__stderr__.flush()


def create_tmux_handler(stream: IO) -> logging.Handler:
    """Creates a stream handler to handle tmux logging.

    First, we try to copy an existing console stream handler from the root logger and use that for tmux logging.
    If no handler exists we create a new one.

    Args:
        stream: The input stream for the handler. In tmux logging this is a fifo.

    Returns:
        A tmux logging handler.
    """

    # retrieve root logger
    root_logger = logging.getLogger()

    # iterate all existing handlers and retrieve a stream handler logging to the console (stdout, stderr)
    # if a handler is found, copy the handler and alter its stream input to use it for tmux logging
    tmux_handler = None
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream.name in [
            "<stderr>",
            "<stdout>",
        ]:
            tmux_handler = copy.copy(handler)
            tmux_handler.setStream(stream)
            break
        elif isinstance(handler, RichHandler) and handler.console.file.name in [
            "<stderr>",
            "<stdout>",
        ]:
            tmux_handler = copy.copy(handler)
            console_ = Console(file=stream, color_system="standard")
            tmux_handler.console = console_
            break

    # if no console handler is attached to root logger, we create a new default tmux handler
    if tmux_handler is None:
        tmux_handler = create_stream_handler(stream=stream, rich_logging=False)
    return tmux_handler


class TmuxManager:
    def __init__(
        self,
        worker_names: List[str],
        session_name: Optional[str] = "fluidml",
        max_panes_per_window: int = 4,
    ):
        self.worker_names = worker_names
        self.session_name = session_name
        self.session_created = False
        self.max_panes_per_window = max_panes_per_window
        self._current_tmux_window = 0

        self.pipes = self._setup_tmux_logging()

    @staticmethod
    def is_tmux_installed() -> bool:
        try:
            subprocess.check_output("tmux -V", shell=True, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            logger_.warning("No tmux logging available since tmux is not installed.")
            return False

    def _setup_tmux_logging(self) -> Dict[str, Dict[str, str]]:
        tmux_pipes = {}
        for i, worker_name in enumerate(self.worker_names):
            # create named pipes (fifos) for stdout and stderr messages
            stdout_pipe, stderr_pipe = TmuxManager._create_stdout_stderr_pipes(worker_name)

            # command enables tmux session to read from pipes to show stdout and stderr from child process
            # read_from_pipe_cmd = f"cat {stdout_pipe} & cat {stderr_pipe}"
            read_from_pipe_cmd = f"cat {stdout_pipe} & cat {stderr_pipe}"

            # logic to create tmux command to start a session, add a new pane to the active window
            # or create a new window when the screen is split in `max_panes_per_window` panes
            tmux_cmd = self._create_tmux_cmd(read_from_pipe_cmd, pane_counter=i)

            # Execute created command and handle potential tmux errors
            TmuxManager._execute_tmux_cmd(tmux_cmd)

            # assign pipes to each worker
            tmux_pipes[worker_name] = {"stdout": stdout_pipe, "stderr": stderr_pipe}
        return tmux_pipes

    @staticmethod
    def init_handlers(pipes: Dict[str, Dict[str, IO]]) -> Dict[str, logging.Handler]:
        tmux_handlers = {}
        for worker_name, pipe in pipes.items():
            stream = pipe["stderr"]
            handler = create_tmux_handler(stream)
            tmux_handlers[worker_name] = handler
        return tmux_handlers

    @staticmethod
    def _create_stdout_stderr_pipes(worker_name: str) -> Tuple[str, str]:
        process_dir = tempfile.mkdtemp(worker_name)
        stdout_pipe = os.path.join(process_dir, "stdout")
        stderr_pipe = os.path.join(process_dir, "stderr")
        os.mkfifo(stdout_pipe)
        os.mkfifo(stderr_pipe)
        return stdout_pipe, stderr_pipe

    def _create_tmux_cmd(self, read_from_pipe_cmd: str, pane_counter: int) -> str:
        if not self.session_created:
            # create new tmux session with session name 'self.session_name', window name 'self._current_tmux_window'
            #  and with the command to continuously read from provided pipes
            #  also remain-on-exit keeps the session open after the fluidml main process is finished or exited
            cmd = (
                f"tmux new-session -d -s {self.session_name} -n {self._current_tmux_window} "
                f"&& tmux send-keys -t {self.session_name}:{self._current_tmux_window} '{read_from_pipe_cmd}' Enter"
            )
            self.session_created = True
        else:
            if pane_counter % self.max_panes_per_window == 0:
                self._current_tmux_window += 1
                # create a new window in the created session if the previous one holds `self.max_panes_per_window` panes
                cmd = (
                    f"tmux new-window -n {self._current_tmux_window} -t {self.session_name} "
                    f"&& tmux send-keys -t {self.session_name}:{self._current_tmux_window} '{read_from_pipe_cmd}' Enter "
                    f"&& tmux select-layout -t {self.session_name}: even-vertical"
                )
            else:
                # split the active window vertically to create a new pane
                cmd = (
                    f"tmux split-window -t {self.session_name}:{self._current_tmux_window} "
                    f"&& tmux send-keys -t {self.session_name}:{self._current_tmux_window} '{read_from_pipe_cmd}' Enter "
                    f"&& tmux select-layout -t {self.session_name}: even-vertical"
                )
        return cmd

    @staticmethod
    def _execute_tmux_cmd(cmd: str):
        try:
            subprocess.check_output(cmd, shell=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise TmuxError(f"Please resolve the following tmux error: '{e.stderr.decode('utf-8').strip()}'.") from None

    # TODO 1: Finalize tmux logging -> Call when Swarm exits context manager -> close pipes and delete tmp directories
    #  make tmux_manager in Swarm a member attribute to call .finalize() during exit()
    # def finalize
    # TODO 2: Check queue putting and getting performance with new changes


class LoggingListener(Thread):
    """Listens to and handles child process log messages
    This class, when instantiated, listens to the logging queue to receive log messages from child processes
    and handles these messages using the configured root logger in the main process.
    """

    def __init__(
        self,
        logging_queue: Queue,
        error_queue: Queue,
        lock: Lock,
        exit_event: Event,
        exit_on_error: bool,
        tmux_manager: Optional[TmuxManager] = None,
    ):
        super().__init__(target=self.work, args=())
        self._logging_queue = logging_queue
        self.tmux_manager = tmux_manager

        self.exit_event = exit_event
        self._exit_on_error = exit_on_error
        self._error_queue = error_queue
        self._lock = lock

    @staticmethod
    def _handle_log_msg(record: LogRecord, tmux_handler: Optional[logging.Handler] = None):
        logger = logging.getLogger(record.name)
        logger.handle(record)

        if tmux_handler and record.levelno >= logger.getEffectiveLevel():
            tmux_handler.handle(record)

    @staticmethod
    def _handle_stdout_msg(record: str, tmux_pipe: Optional[IO] = None):
        sys.stdout.write(record)
        sys.stdout.flush()
        if tmux_pipe:
            tmux_pipe.write(record)
            tmux_pipe.flush()

    @staticmethod
    def _handle_stderr_msg(record: str, tmux_pipe: Optional[IO] = None):
        sys.stderr.write(record)
        sys.stderr.flush()
        if tmux_pipe:
            tmux_pipe.write(record)
            tmux_pipe.flush()

    def _work(self):
        with ExitStack() as stack:
            pipes, handlers = None, None
            if self.tmux_manager:
                pipes = {
                    worker_name: {
                        "stdout": stack.enter_context(open(pipe["stdout"], "w")),
                        "stderr": stack.enter_context(open(pipe["stderr"], "w")),
                    }
                    for worker_name, pipe in self.tmux_manager.pipes.items()
                }
                handlers = self.tmux_manager.init_handlers(pipes)

            while True:
                # Using queue.get(block=False) is necessary for python 3.6. queue.get() sometimes
                # leads to weird deadlocks when waiting for logging messages from child processes.
                try:
                    record = self._logging_queue.get(block=True, timeout=0.05)
                except Empty:
                    continue

                if record is None:
                    break
                record_type, worker_name, record = record

                tmux_handler = handlers[worker_name] if handlers else None
                tmux_stdout_pipe = pipes[worker_name]["stdout"] if pipes else None
                tmux_stderr_pipe = pipes[worker_name]["stderr"] if pipes else None

                if record_type == "log_msg":
                    self._handle_log_msg(record, tmux_handler=tmux_handler)
                elif record_type == "stdout_msg":
                    self._handle_stdout_msg(record, tmux_pipe=tmux_stdout_pipe)
                elif record_type == "stderr_msg":
                    self._handle_stderr_msg(record, tmux_pipe=tmux_stderr_pipe)

    def work(self):
        try:
            self._work()
        except Exception as e:
            logger_.exception(e)
            self._error_queue.put(e)
            if self._exit_on_error:
                with self._lock:
                    self.exit_event.set()
