import logging.handlers
import multiprocessing
import random
from multiprocessing import Manager, set_start_method, Queue, Lock, Event
from types import TracebackType
from typing import Optional, Type, List, Dict, Union, Any, Callable

from fluidml.common import Resource
from fluidml.common.logging import LoggingListener, TmuxManager
from fluidml.flow.task_spec import TaskSpec
from fluidml.storage import ResultsStore, InMemoryStore
from fluidml.storage.controller import pack_pipeline_results
from fluidml.swarm import Dolphin

logger = logging.getLogger(__name__)


class Swarm:
    def __init__(self,
                 n_dolphins: Optional[int] = None,
                 resources: Optional[List[Resource]] = None,
                 results_store: Optional[ResultsStore] = None,
                 start_method: str = 'spawn',
                 exit_on_error: bool = True,
                 return_results: bool = False,
                 log_to_tmux: bool = True,
                 create_tmux_handler_fn: Optional[Callable] = None,
                 max_panes_per_window: Optional[int] = None,
                 project_name: Optional[str] = None,
                 run_name: Optional[str] = None):
        """Configure workers, resources, results_store which are used to run the tasks

        Args:
            n_dolphins (Optional[int], optional): number of parallel workers. Defaults to None.
            resources (Optional[List[Resource]], optional): a list of resources that are assigned to workers.
                    If len(resources) < n_dolphins, resources are assigned randomly to workers.
            results_store (Optional[ResultsStore], optional): an instance of results store for results management.
                    If nothing is provided, a non-persistent InMemoryStore store is used.
            start_method (str, optional): start method for multiprocessing. Defaults to 'spawn'.
            exit_on_error (bool, optional): when an error happens all workers finish their current tasks
                    and exit gracefully. Defaults to True.
            return_results (bool, optional): return results-dictionary after run(). Defaults to False.
            log_to_tmux (bool, optional): log to tmux session if True. Defaults to True
            create_tmux_handler_fn
            max_panes_per_window (Optional[int], optional): max number of panes per tmux window
            project_name (Optional[str], optional): Name of project.
            run_name (Optional[str], optional): Name of run.
        """

        set_start_method(start_method, force=True)

        self.project_name = project_name if project_name is not None else 'uncategorized'
        self.run_name = run_name

        self.n_dolphins = n_dolphins if n_dolphins else multiprocessing.cpu_count()
        self.resources = Swarm._allocate_resources(self.n_dolphins, resources)
        self.manager = Manager()
        self.lock = Lock()

        self.scheduled_queue = Queue()
        self.running_queue = self.manager.list()
        self.done_queue = self.manager.list()
        self.logging_queue = Queue()
        self.error_queue = Queue()

        self.results_store = results_store if results_store is not None else InMemoryStore(self.manager)

        self.exit_event = Event()
        self.return_results = True if isinstance(self.results_store, InMemoryStore) else return_results
        self.tasks: Dict[int, TaskSpec] = {}

        # get effective logging lvl
        logging_lvl = logger.getEffectiveLevel()

        # dolphin workers for task execution
        self.dolphins = [Dolphin(resource=self.resources[i],
                                 scheduled_queue=self.scheduled_queue,
                                 running_queue=self.running_queue,
                                 done_queue=self.done_queue,
                                 logging_queue=self.logging_queue,
                                 error_queue=self.error_queue,
                                 lock=self.lock,
                                 tasks=self.tasks,
                                 exit_event=self.exit_event,
                                 exit_on_error=exit_on_error,
                                 results_store=self.results_store,
                                 logging_lvl=logging_lvl)
                         for i in range(self.n_dolphins)]

        tmux_manager = None
        if log_to_tmux and TmuxManager.is_tmux_installed():
            session_name = self.project_name if self.run_name is None else f'{self.project_name}--{self.run_name}'
            tmux_manager = TmuxManager(worker_names=[dolphin.name for dolphin in self.dolphins],
                                       session_name=session_name,
                                       max_panes_per_window=max_panes_per_window if max_panes_per_window else 4,
                                       create_tmux_handler_fn=create_tmux_handler_fn)

        # listener thread to handle stdout, stderr and logging from child processes
        self.logging_listener = LoggingListener(logging_queue=self.logging_queue,
                                                tmux_manager=tmux_manager,
                                                exit_on_error=exit_on_error,
                                                lock=self.lock,
                                                error_queue=self.error_queue,
                                                exit_event=self.exit_event)

    def __enter__(self):
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]):
        self.close()

    @staticmethod
    def _allocate_resources(n_dolphins: int, resources: List[Resource]) -> List[Resource]:
        if not resources:
            resources = [None] * n_dolphins
        elif len(resources) != n_dolphins:
            # we assign resources to dolphins uniformly (from uniform distribution)
            resources = random.choices(resources, k=n_dolphins)
        return resources

    @staticmethod
    def _get_entry_point_tasks(tasks: List[TaskSpec]) -> Dict[int, str]:
        """
        Gets tasks that are run first (tasks with no predecessors)
        """
        entry_task_ids = {}
        for task in tasks:
            if len(task.predecessors) == 0:
                entry_task_ids[task.id_] = task.unique_name
        return entry_task_ids

    def _collect_results(self) -> Dict[str, Any]:
        results = pack_pipeline_results(all_tasks=list(self.tasks.values()),
                                        results_store=self.results_store,
                                        return_results=self.return_results)
        return results

    def work(self, tasks: List[TaskSpec]) -> Optional[Dict[str, Union[List[Dict], Dict]]]:

        # get entry point task ids
        entry_point_tasks: Dict[int, str] = self._get_entry_point_tasks(tasks)

        # also update the current tasks and assign project- and run name
        for task in tasks:
            task.project_name = self.project_name
            task.run_name = self.run_name
            self.tasks[task.id_] = task

        # start the listener thread to receive log messages from child processes
        self.logging_listener.daemon = True
        self.logging_listener.start()

        # schedule entry point tasks
        for task_id, task_name_unique in entry_point_tasks.items():
            logger.debug(f'Swarm scheduling task {task_name_unique}.')
            self.scheduled_queue.put(task_id)
            self.running_queue.append(task_id)

        # todo: enable for debug mode to turn off multiprocessing
        # self.dolphins[0]._work()

        # start the workers
        for dolphin in self.dolphins:
            dolphin.start()

        # wait for them to finish
        for dolphin in self.dolphins:
            dolphin.join()

        # join the listener thread with the main thread
        self.logging_queue.put(None)
        self.logging_listener.join()

        # if an exception was raised by a child process, exit the parent process.
        if self.exit_event.is_set():
            err = self.error_queue.get()
            raise err

        # return all results
        return self._collect_results()

    def close(self):
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
