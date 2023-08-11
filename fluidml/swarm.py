import contextlib
import logging.handlers
import multiprocessing
import random
import signal
from multiprocessing import Event as MPEvent
from multiprocessing import Queue as MPQueue
from multiprocessing import RLock, set_start_method
from multiprocessing.managers import SyncManager
from queue import Queue
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

from fluidml.dolphin import Dolphin
from fluidml.logging import LoggingListener, TmuxManager
from fluidml.storage import InMemoryStore, ResultsStore
from fluidml.storage.controller import pack_pipeline_results
from fluidml.task import Task, TaskResults, TaskState
from fluidml.utils import create_unique_hash_from_config

logger = logging.getLogger(__name__)


def _manager_init():
    """An initialization function passed to the multiprocessing manager.

    It causes the manager object to ignore KeyboardInterrupt signals in the child process.
    This is necessary to let the child processes exit gracefully. Otherwise, an error would be thrown, since
    the child processes tries to access the manager object during graceful exit, but it has been terminated already.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class Event:
    """A non-multiprocessing Event Class.

    Imitates the API and functionality of multiprocessing.Event().
    """

    def __init__(self):
        self._flag = False

    def is_set(self):
        if self._flag:
            return True
        return False

    def set(self):
        self._flag = True

    def clear(self):
        self._flag = False


class Swarm:
    """Configure workers, resources, results_store which are used to run the tasks.

    Args:
        n_dolphins: Number of parallel workers. Defaults to None.
        resources: A list of resources that are assigned to workers.
            If len(resources) < n_dolphins, resources are assigned randomly to workers.
        start_method: Start method for multiprocessing. Defaults to 'spawn'.
        exit_on_error: When an error happens all workers finish their current tasks and exit gracefully.
            Defaults to False.
        log_to_tmux: Log to tmux session if True. Defaults to True.
        max_panes_per_window: Max number of panes per tmux window.
    """

    def __init__(
        self,
        n_dolphins: Optional[int] = None,
        resources: Optional[List[Any]] = None,
        start_method: str = "spawn",
        exit_on_error: bool = False,
        log_to_tmux: bool = False,
        max_panes_per_window: int = 4,
    ):
        self.n_dolphins = n_dolphins if n_dolphins else multiprocessing.cpu_count()

        # set in the beginning of self.work()
        self.tasks: Dict[str, Task] = {}
        self.exit_on_error = exit_on_error
        self.resources = Swarm._allocate_resources(self.n_dolphins, resources)
        # get effective logging lvl
        logging_lvl = logger.getEffectiveLevel()

        # Initialize all relevant attributes for sequential Task Graph execution
        # Use a normal Queue and Event file
        self.task_states = {}
        self.lock = contextlib.suppress()  # dummy contextmanager (has no effect)
        self.scheduled_queue = Queue()
        self.error_queue = Queue()
        self.exit_event = Event()
        self.logging_queue = None
        self.manager = None

        # If more than 1 worker is configures -> overwrite attributes for multiprocessing setting
        if self.n_dolphins > 1:
            set_start_method(start_method, force=True)
            self.manager = SyncManager()
            self.manager.start(_manager_init)
            self.task_states = self.manager.dict()
            self.lock = RLock()
            self.scheduled_queue = MPQueue()
            self.logging_queue = MPQueue()
            self.error_queue = MPQueue()
            self.exit_event = MPEvent()

            # tmux logging args (only relevant for multiprocessing)
            self.log_to_tmux = log_to_tmux
            self.max_panes_per_window = max_panes_per_window

        # dolphin workers for task execution
        self.dolphins = [
            Dolphin(
                resource=self.resources[i],
                scheduled_queue=self.scheduled_queue,
                task_states=self.task_states,
                logging_queue=self.logging_queue,
                error_queue=self.error_queue,
                lock=self.lock,
                tasks=self.tasks,
                exit_event=self.exit_event,
                exit_on_error=exit_on_error,
                logging_lvl=logging_lvl,
                use_multiprocessing=True if self.n_dolphins > 1 else False,
            )
            for i in range(self.n_dolphins)
        ]

        # rename workers
        for i, dolphin in enumerate(self.dolphins, 1):
            dolphin.name = f"{dolphin.__class__.__name__}-{i}"

    def _init_mp_logging(self, project_name: Optional[str] = None, run_name: Optional[str] = None) -> LoggingListener:
        """Initialize logging in the case of multiprocessing."""
        tmux_manager = None
        if self.log_to_tmux and TmuxManager.is_tmux_installed():
            session_name = project_name if run_name is None else f"{project_name}--{run_name}"
            tmux_manager = TmuxManager(
                worker_names=[dolphin.name for dolphin in self.dolphins],
                session_name=session_name,
                max_panes_per_window=self.max_panes_per_window,
            )

        # listener thread to handle stdout, stderr and logging from child processes
        logging_listener = LoggingListener(
            logging_queue=self.logging_queue,
            tmux_manager=tmux_manager,
            exit_on_error=self.exit_on_error,
            lock=self.lock,
            error_queue=self.error_queue,
            exit_event=self.exit_event,
        )
        return logging_listener

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ):
        if self.n_dolphins > 1:
            self.close()

    @staticmethod
    def _allocate_resources(n_dolphins: int, resources: List[Any]) -> List[Any]:
        if not resources:
            resources = [None] * n_dolphins
        elif len(resources) != n_dolphins:
            # we assign resources to dolphins uniformly (from uniform distribution)
            resources = random.choices(resources, k=n_dolphins)
        return resources

    @staticmethod
    def _get_entry_point_tasks(tasks: List[Task]) -> List[str]:
        """Gets tasks that are run first (tasks with no predecessors)."""
        entry_task_ids = []
        for task in tasks:
            if len(task.predecessors) == 0:
                entry_task_ids.append(task.unique_name)
        return entry_task_ids

    def _join(self, logging_listener: LoggingListener):
        # wait for all workers to finish
        for dolphin in self.dolphins:
            try:
                # Only call join() if child processes have been started.
                # If they haven't been started, an AssertionError is thrown and we pass.
                dolphin.join()
            except AssertionError:
                pass

        self.logging_queue.put(None)
        logging_listener.join()

    def _run_parallel(self, project_name: str, run_name: str):
        """Execute the Task graph in parallel using multiprocessing."""
        # start the listener thread to receive log messages from child processes
        logging_listener = self._init_mp_logging(project_name, run_name)
        logging_listener.daemon = True
        logging_listener.start()

        try:
            # start the workers
            for i, dolphin in enumerate(self.dolphins, 1):
                # set workers as non-daemonic to allow for nested multiprocessing
                # see https://superfastpython.com/daemon-process-in-python/#Need_for_Daemon_Processes for further
                #  explanations
                dolphin.daemon = False
                dolphin.start()

            # join processes and logging thread
            self._join(logging_listener)

        except KeyboardInterrupt as e:
            # handle KeyboardInterrupt
            self.exit_event.set()
            self.error_queue.put(e)

            # join processes and logging thread
            self._join(logging_listener)

    def _run_sequential(self):
        """Execute the Task graph sequentially."""
        dolphin = self.dolphins[0]  # only 1 Worker (no Subprocess) exists
        dolphin.work()

    def work(
        self,
        tasks: List[Task],
        run_name: str,
        project_name: str = "uncategorized",
        results_store: Optional[ResultsStore] = None,
        return_results: Optional[str] = None,
    ) -> Optional[Dict[str, List[TaskResults]]]:
        """Handles the sequential or parallel (multiprocessing) execution of a directed Task Graph.

        Schedules the entry point tasks to the Task Queue and executes the Graph.

        Args:
            tasks: A list of expanded Task objects.
            run_name: Name of run.
            project_name: Name of project. Defaults to ``"uncategorized"``.
            results_store: An instance of results store for results management.
                If nothing is provided, a non-persistent InMemoryStore store is used.
            return_results: Return results-dictionary after ``run()``. Defaults to ``all``.
                Choices: "all", "latest", None

        Returns:
            A Dict with the tasks results and configs.
        """
        # setup results store
        results_store = results_store if results_store is not None else InMemoryStore(self.manager)

        # get entry point task ids
        entry_point_tasks: List[str] = self._get_entry_point_tasks(tasks)

        # also update the current tasks and assign results store
        for task in tasks:
            task.run_name = run_name
            task.project_name = project_name
            task.unique_config_hash = create_unique_hash_from_config(task.unique_config)
            task.results_store = results_store

            # set task state and assign tasks to tasks dict object
            self.task_states[task.unique_name] = TaskState.PENDING
            self.tasks[task.unique_name] = task

        # schedule entry point tasks
        for task_unique_name in entry_point_tasks:
            logger.debug(f'Scheduling task "{task_unique_name}"')
            self.scheduled_queue.put(task_unique_name)
            self.task_states[task_unique_name] = TaskState.SCHEDULED

        if self.n_dolphins > 1:
            logger.info(f'Execute run "{run_name}" using multiprocessing with {self.n_dolphins} workers')
            self._run_parallel(project_name, run_name)
        else:
            logger.info(f'Execute run "{run_name}" sequentially (no multiprocessing)')
            self._run_sequential()

        # if an exception was raised by a child process, exit the parent process.
        if self.exit_event.is_set():
            err = self.error_queue.get()
            raise err

        # return all results
        results: Optional[Dict[str, List[TaskResults]]] = pack_pipeline_results(
            all_tasks=list(self.tasks.values()), return_results=return_results
        )
        return results

    def close(self):
        """Join and finish all running sub-processes and clearing all queues."""
        self.tasks = {}
        for dolphin in self.dolphins:
            try:
                # Only call join() if child processes have been started.
                # If they haven't been started, an AssertionError is thrown and we pass.
                dolphin.join()
            except AssertionError:
                pass
            # for python 3.6 dolphin.terminate() is the backup where .close() is not available.
            try:
                dolphin.close()
            except AttributeError:
                dolphin.terminate()
        self.manager.shutdown()
