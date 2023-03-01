import contextlib
import logging.handlers
import multiprocessing
import random
import signal
import time
from multiprocessing import Manager, set_start_method, Queue as MPQueue, Lock, Event as MPEvent
from multiprocessing.managers import SyncManager
from queue import Queue
from types import TracebackType
from typing import Optional, Type, List, Dict, Union, Any

from fluidml.common.task import Task, TaskState
from fluidml.common.logging import LoggingListener, TmuxManager
from fluidml.common.utils import create_unique_run_id_from_config
from fluidml.storage import ResultsStore, InMemoryStore
from fluidml.storage.controller import pack_pipeline_results
from fluidml.swarm import Dolphin

logger = logging.getLogger(__name__)


def init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def handler(signalname):
    """
    Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
    Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
    """

    def f(signal_received, frame):
        time.sleep(10)
        raise KeyboardInterrupt(f"{signalname} received")

    return f


class Event:
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
    def __init__(
        self,
        n_dolphins: Optional[int] = None,
        resources: Optional[List[Any]] = None,
        start_method: str = "spawn",
        exit_on_error: bool = True,
        log_to_tmux: bool = False,
        max_panes_per_window: int = 4,
    ):
        """Configure workers, resources, results_store which are used to run the tasks.

        Args:
            n_dolphins: Number of parallel workers. Defaults to None.
            resources: A list of resources that are assigned to workers.
                If len(resources) < n_dolphins, resources are assigned randomly to workers.
            start_method: Start method for multiprocessing. Defaults to 'spawn'.
            exit_on_error: When an error happens all workers finish their current tasks and exit gracefully.
                Defaults to True.
            log_to_tmux: Log to tmux session if True. Defaults to True.
            max_panes_per_window: Max number of panes per tmux window.
        """
        self.n_dolphins = n_dolphins if n_dolphins else multiprocessing.cpu_count()

        # set in the beginning of self.work()
        self.tasks: Dict[str, Task] = {}
        self.exit_on_error = exit_on_error
        self.resources = Swarm._allocate_resources(self.n_dolphins, resources)
        # get effective logging lvl
        logging_lvl = logger.getEffectiveLevel()

        self.task_states = {}
        self.lock = contextlib.suppress()  # null_context (has no effect)
        self.scheduled_queue = Queue()
        self.error_queue = Queue()
        self.exit_event = Event()
        self.logging_queue = None
        self.manager = None

        if self.n_dolphins > 1:
            set_start_method(start_method, force=True)
            self.manager = SyncManager()
            self.manager.start(init)
            self.task_states = self.manager.dict()
            self.lock = Lock()
            self.scheduled_queue = MPQueue()
            self.logging_queue = MPQueue()
            self.error_queue = MPQueue()
            self.exit_event = MPEvent()

            # tmux logging args
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

    def _init_mp_logging(self, project_name: Optional[str] = None, run_name: Optional[str] = None) -> LoggingListener:
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
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
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

    def _run_parallel(self, project_name: str, run_name: str):
        # This will be inherited by the child process if it is forked (not spawned)
        # signal.signal(signal.SIGINT, handler("SIGINT"))
        # signal.signal(signal.SIGTERM, handler("SIGTERM"))
        # signal.signal(signal.SIGINT, signal.SIG_IGN)

        # start the listener thread to receive log messages from child processes
        logging_listener = self._init_mp_logging(project_name, run_name)
        logging_listener.daemon = True
        logging_listener.start()

        # start the workers
        for i, dolphin in enumerate(self.dolphins, 1):
            dolphin.name = f"{dolphin.__class__.__name__}-{i}"
            dolphin.daemon = True
            dolphin.start()

        # wait for them to finish
        for dolphin in self.dolphins:
            dolphin.join()

        # try:
        #     # start the workers
        #     for dolphin in self.dolphins:
        #         dolphin.daemon = True
        #         dolphin.start()
        #
        #     # wait for them to finish
        #     for dolphin in self.dolphins:
        #         dolphin.join()
        # except KeyboardInterrupt:
        #     time.sleep(10)
        #     self.exit_event.set()
        #     for dolphin in self.dolphins:
        #         dolphin.join()
        #     raise
        # while [p for p in self.dolphins if p.is_alive()]:
        #     # if time() > t + grace_period:
        #     #     for p in alive_procs:
        #     #         os.kill(p.pid, signal.SIGINT)
        #     #         logger.warning("Sending SIGINT to %s", p)
        #     # elif time() > t + kill_period:
        #     #     for p in alive_procs:
        #     #         logger.warning("Sending SIGKILL to %s", p)
        #     #         # Queues and other inter-process communication primitives can break when
        #     #         # process is killed, but we don't care here
        #     #         p.kill()
        #     time.sleep(0.01)
        # for dolphin in self.dolphins:
        #     dolphin.join()
        #
        # raise

        # join the listener thread with the main thread
        self.logging_queue.put(None)
        logging_listener.join()

    def _run_sequential(self):
        dolphin = self.dolphins[0]
        dolphin.work()

    def work(
        self,
        tasks: List[Task],
        run_name: str,
        project_name: str = "uncategorized",
        results_store: Optional[ResultsStore] = None,
        return_results: Optional[str] = None,
    ) -> Dict[str, Union[List[Dict], Dict]]:
        """Handles the parallel execution of a list of tasks.

        Starts and finishes all workers, starts and finishes the logging thread.
        Schedules the entry point tasks to the Task Queue,

        Args:
            tasks: A list of expanded Task objects.
            run_name: Name of run.
            project_name: Name of project. Defaults to ``"uncategorized"``.
            results_store: An instance of results store for results management.
                If nothing is provided, a non-persistent InMemoryStore store is used.
            return_results: Return results-dictionary after ``run()``. Defaults to ``all``.
                Choices: "all", "latest", None

        Returns:
            A Dict if tasks containing their respective configs and their published results
                (if ``return_results is True``)

        """
        # setup results store
        results_store = results_store if results_store is not None else InMemoryStore(self.manager)

        # get entry point task ids
        entry_point_tasks: List[str] = self._get_entry_point_tasks(tasks)

        # also update the current tasks and assign results store
        for task in tasks:
            task.run_name = run_name
            task.project_name = project_name
            task.unique_config_hash = create_unique_run_id_from_config(task.unique_config)
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
        results: Dict[str, Any] = pack_pipeline_results(
            all_tasks=list(self.tasks.values()), return_results=return_results
        )
        return results

    def close(self):
        """Join and finish all running sub-processes and clearing all queues."""
        self.tasks = {}
        # self.task_states = self.manager.dict()
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


# class Swarm:
#     def __init__(
#         self,
#         n_dolphins: Optional[int] = None,
#         resources: Optional[List[Any]] = None,
#         start_method: str = "spawn",
#         exit_on_error: bool = True,
#         log_to_tmux: bool = False,
#         max_panes_per_window: int = 4,
#     ):
#         """Configure workers, resources, results_store which are used to run the tasks.
#
#         Args:
#             n_dolphins: Number of parallel workers. Defaults to None.
#             resources: A list of resources that are assigned to workers.
#                 If len(resources) < n_dolphins, resources are assigned randomly to workers.
#             start_method: Start method for multiprocessing. Defaults to 'spawn'.
#             exit_on_error: When an error happens all workers finish their current tasks and exit gracefully.
#                 Defaults to True.
#             log_to_tmux: Log to tmux session if True. Defaults to True.
#             max_panes_per_window: Max number of panes per tmux window.
#         """
#         set_start_method(start_method, force=True)
#
#         self.manager = Manager()
#
#         self.task_states = self.manager.dict()
#
#         self.n_dolphins = n_dolphins if n_dolphins else multiprocessing.cpu_count()
#         self.resources = Swarm._allocate_resources(self.n_dolphins, resources)
#         self.lock = Lock()
#         self.scheduled_queue = Queue()
#         self.logging_queue = Queue()
#         self.error_queue = Queue()
#         self.exit_event = Event()
#         self.exit_on_error = exit_on_error
#
#         # set in the beginning of self.work()
#         self.tasks: Dict[str, Task] = {}
#
#         # get effective logging lvl
#         logging_lvl = logger.getEffectiveLevel()
#
#         # dolphin workers for task execution
#         self.dolphins = [
#             Dolphin(
#                 resource=self.resources[i],
#                 scheduled_queue=self.scheduled_queue,
#                 task_states=self.task_states,
#                 logging_queue=self.logging_queue,
#                 error_queue=self.error_queue,
#                 lock=self.lock,
#                 tasks=self.tasks,
#                 exit_event=self.exit_event,
#                 exit_on_error=exit_on_error,
#                 logging_lvl=logging_lvl,
#             )
#             for i in range(self.n_dolphins)
#         ]
#
#         # logging args
#         self.log_to_tmux = log_to_tmux
#         self.max_panes_per_window = max_panes_per_window
#
#     def _init_logging(self, project_name: Optional[str] = None, run_name: Optional[str] = None) -> LoggingListener:
#         tmux_manager = None
#         if self.log_to_tmux and TmuxManager.is_tmux_installed():
#             session_name = project_name if run_name is None else f"{project_name}--{run_name}"
#             tmux_manager = TmuxManager(
#                 worker_names=[dolphin.name for dolphin in self.dolphins],
#                 session_name=session_name,
#                 max_panes_per_window=self.max_panes_per_window,
#             )
#
#         # listener thread to handle stdout, stderr and logging from child processes
#         logging_listener = LoggingListener(
#             logging_queue=self.logging_queue,
#             tmux_manager=tmux_manager,
#             exit_on_error=self.exit_on_error,
#             lock=self.lock,
#             error_queue=self.error_queue,
#             exit_event=self.exit_event,
#         )
#         return logging_listener
#
#     def __enter__(self):
#         return self
#
#     def __exit__(
#         self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
#     ):
#         self.close()
#
#     @staticmethod
#     def _allocate_resources(n_dolphins: int, resources: List[Any]) -> List[Any]:
#         if not resources:
#             resources = [None] * n_dolphins
#         elif len(resources) != n_dolphins:
#             # we assign resources to dolphins uniformly (from uniform distribution)
#             resources = random.choices(resources, k=n_dolphins)
#         return resources
#
#     @staticmethod
#     def _get_entry_point_tasks(tasks: List[Task]) -> List[str]:
#         """Gets tasks that are run first (tasks with no predecessors)."""
#         entry_task_ids = []
#         for task in tasks:
#             if len(task.predecessors) == 0:
#                 entry_task_ids.append(task.unique_name)
#         return entry_task_ids
#
#     def work(
#         self,
#         tasks: List[Task],
#         results_store: Optional[ResultsStore] = None,
#         return_results: Optional[str] = None,
#     ) -> Dict[str, Union[List[Dict], Dict]]:
#         """Handles the parallel execution of a list of tasks.
#
#         Starts and finishes all workers, starts and finishes the logging thread.
#         Schedules the entry point tasks to the Task Queue,
#
#         Args:
#             tasks: A list of expanded Task objects.
#             results_store: An instance of results store for results management.
#                 If nothing is provided, a non-persistent InMemoryStore store is used.
#             return_results: Return results-dictionary after ``run()``. Defaults to ``all``.
#                 Choices: "all", "latest", None
#
#         Returns:
#             A Dict if tasks containing their respective configs and their published results
#                 (if ``return_results is True``)
#
#         """
#         # setup results store
#         results_store = results_store if results_store is not None else InMemoryStore(self.manager)
#
#         # setup logging
#         project_name = tasks[0].info.project_name
#         run_name = tasks[0].info.run_name
#         logging_listener = self._init_logging(project_name, run_name)
#
#         # get entry point task ids
#         entry_point_tasks: List[str] = self._get_entry_point_tasks(tasks)
#
#         # also update the current tasks and assign results store
#         for task in tasks:
#             task.results_store = results_store
#             self.task_states[task.unique_name] = TaskState.PENDING
#             # task.state = TaskState.PENDING
#             self.tasks[task.unique_name] = task
#
#         # start the listener thread to receive log messages from child processes
#         logging_listener.daemon = True
#         logging_listener.start()
#
#         # schedule entry point tasks
#         for task_unique_name in entry_point_tasks:
#             logger.debug(f'Scheduling task "{task_unique_name}"')
#             self.scheduled_queue.put(task_unique_name)
#             self.task_states[task_unique_name] = TaskState.SCHEDULED
#             # self.tasks[task_unique_name].state = TaskState.SCHEDULED
#
#         # start the workers
#         for dolphin in self.dolphins:
#             dolphin.start()
#
#         # wait for them to finish
#         for dolphin in self.dolphins:
#             dolphin.join()
#
#         # join the listener thread with the main thread
#         self.logging_queue.put(None)
#         logging_listener.join()
#
#         # if an exception was raised by a child process, exit the parent process.
#         if self.exit_event.is_set():
#             err = self.error_queue.get()
#             raise err
#
#         # return all results
#         results: Dict[str, Any] = pack_pipeline_results(
#             all_tasks=list(self.tasks.values()), return_results=return_results
#         )
#         return results
#
#     def close(self):
#         """Join and finish all running sub-processes and clearing all queues."""
#         self.tasks = {}
#         self.task_states = self.manager.dict()
#         for dolphin in self.dolphins:
#             try:
#                 # Only call join() if child processes have been started.
#                 # If they haven't been started, an AssertionError is thrown and we pass.
#                 dolphin.join()
#             except AssertionError:
#                 pass
#             # for python 3.6 dolphin.terminate() is the backup where .close() is not available.
#             try:
#                 dolphin.close()
#             except AttributeError:
#                 dolphin.terminate()
#         self.manager.shutdown()
