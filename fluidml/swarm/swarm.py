import logging.handlers
import multiprocessing
import random
from multiprocessing import Manager, set_start_method, Queue, Lock, Event
from types import TracebackType
from typing import Optional, Type, List, Dict, Union, Any

# from fluidml.common.task import TaskState
from fluidml.common.logging import LoggingListener, TmuxManager
from fluidml.flow.task_spec import TaskSpec
from fluidml.storage import ResultsStore, InMemoryStore
from fluidml.storage.controller import pack_pipeline_results
from fluidml.swarm import Dolphin

logger = logging.getLogger(__name__)


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
        set_start_method(start_method, force=True)

        self.manager = Manager()

        # self.task_states = self.manager.dict()

        self.n_dolphins = n_dolphins if n_dolphins else multiprocessing.cpu_count()
        self.resources = Swarm._allocate_resources(self.n_dolphins, resources)
        self.lock = Lock()
        self.scheduled_queue = Queue()
        self.running_queue = self.manager.list()
        self.done_queue = self.manager.list()
        self.failed_queue = self.manager.list()
        self.logging_queue = Queue()
        self.error_queue = Queue()
        self.exit_event = Event()
        self.exit_on_error = exit_on_error

        # set in the beginning of self.work()
        self.tasks: Dict[str, TaskSpec] = {}  # self.manager.dict()

        # get effective logging lvl
        logging_lvl = logger.getEffectiveLevel()

        # dolphin workers for task execution
        self.dolphins = [
            Dolphin(
                resource=self.resources[i],
                scheduled_queue=self.scheduled_queue,
                # task_states=self.task_states,
                running_queue=self.running_queue,
                done_queue=self.done_queue,
                failed_queue=self.failed_queue,
                logging_queue=self.logging_queue,
                error_queue=self.error_queue,
                lock=self.lock,
                tasks=self.tasks,
                exit_event=self.exit_event,
                exit_on_error=exit_on_error,
                logging_lvl=logging_lvl,
            )
            for i in range(self.n_dolphins)
        ]

        # logging args
        self.log_to_tmux = log_to_tmux
        self.max_panes_per_window = max_panes_per_window

    def _init_logging(self, project_name: Optional[str] = None, run_name: Optional[str] = None) -> LoggingListener:
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
    def _get_entry_point_tasks(tasks: List[TaskSpec]) -> List[str]:
        """Gets tasks that are run first (tasks with no predecessors)."""
        entry_task_ids = []
        for task in tasks:
            if len(task.predecessors) == 0:
                entry_task_ids.append(task.unique_name)
        return entry_task_ids

    def work(
        self,
        tasks: List[TaskSpec],
        results_store: Optional[ResultsStore] = None,
        return_results: Optional[str] = None,
    ) -> Dict[str, Union[List[Dict], Dict]]:
        """Handles the parallel execution of a list of tasks.

        Starts and finishes all workers, starts and finishes the logging thread.
        Schedules the entry point tasks to the Task Queue,

        Args:
            tasks: A list of expanded TaskSpec objects.
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

        # setup logging
        project_name = tasks[0].info.project_name
        run_name = tasks[0].info.run_name
        logging_listener = self._init_logging(project_name, run_name)

        # get entry point task ids
        entry_point_tasks: List[str] = self._get_entry_point_tasks(tasks)

        # also update the current tasks and assign results store
        for task in tasks:
            task.results_store = results_store
            # task.status = TaskState.CUED
            self.tasks[task.unique_name] = task
            # self.task_states[task.unique_name] = TaskState.CUED

        # start the listener thread to receive log messages from child processes
        logging_listener.daemon = True
        logging_listener.start()

        # schedule entry point tasks
        for task_unique_name in entry_point_tasks:
            logger.debug(f'Scheduling task "{task_unique_name}"')
            self.scheduled_queue.put(task_unique_name)
            self.running_queue.append(task_unique_name)
            # self.task_states[task_unique_name] = Status.SCHEDULED
            # self.tasks[task_unique_name].state = Status.SCHEDULED

        # start the workers
        for dolphin in self.dolphins:
            dolphin.start()

        # wait for them to finish
        for dolphin in self.dolphins:
            dolphin.join()

        # join the listener thread with the main thread
        self.logging_queue.put(None)
        logging_listener.join()

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
        self.done_queue = self.manager.list()
        self.running_queue = self.manager.list()
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
