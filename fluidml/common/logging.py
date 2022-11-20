import logging
import os
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from logging import LogRecord
from multiprocessing import Queue, Lock, Event
from queue import Empty
from threading import Thread
from typing import Tuple, Union, List, Optional, Dict, IO, Callable

from rich.console import Console
from rich.logging import RichHandler
from tblib import pickling_support

from fluidml.common.exception import TmuxError


pickling_support.install()
logger_ = logging.getLogger(__name__)


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
    """Default function to create a stream handler used for tmux logging"""

    console_ = Console(file=stream, color_system="standard")
    formatter = logging.Formatter("%(processName)-13s%(message)s")
    stream_handler = RichHandler(
        console=console_, rich_tracebacks=True, tracebacks_extra_lines=2, show_path=False, omit_repeated_times=False
    )
    stream_handler.setFormatter(formatter)
    return stream_handler


class TmuxManager:
    def __init__(
        self,
        worker_names: List[str],
        session_name: Optional[str] = "fluidml",
        max_panes_per_window: int = 4,
        create_tmux_handler_fn: Optional[Callable] = None,
    ):
        self.worker_names = worker_names
        self.session_name = session_name
        self.session_created = False
        self.max_panes_per_window = max_panes_per_window
        self._current_tmux_window = 0
        self._create_tmux_handler_fn = create_tmux_handler_fn if create_tmux_handler_fn else create_tmux_handler

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

    def init_handlers(self, pipes: Dict[str, Dict[str, IO]]) -> Dict[str, logging.Handler]:
        tmux_handlers = {}
        for worker_name, pipe in pipes.items():
            stream = pipe["stderr"]
            handler = self._create_tmux_handler_fn(stream)
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
                    record = self._logging_queue.get(block=False, timeout=0.01)
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
            if self._exit_on_error:
                with self._lock:
                    logger_.exception(e)
                    self._error_queue.put(e)
                    self.exit_event.set()


def configure_logging(level: Union[str, int] = "INFO"):
    assert level in ["DEBUG", "INFO", "WARNING", "WARN", "ERROR", "FATAL", "CRITICAL", 10, 20, 30, 40, 50]
    logger = logging.getLogger()
    # formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(processName)-13s %(message)s',
    #                               datefmt='%Y-%m-%d %H:%M:%S')
    # stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(processName)-13s%(message)s")
    stream_handler = RichHandler(
        rich_tracebacks=True, tracebacks_extra_lines=2, show_path=False, omit_repeated_times=False
    )
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(level)
