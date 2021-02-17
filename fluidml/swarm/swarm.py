import logging.handlers
import multiprocessing
from multiprocessing import Manager, set_start_method, Queue, Lock
import random
from types import TracebackType
from typing import Optional, Type, List, Dict, Union, Any

from fluidml.common.logging import LoggingListener
from fluidml.common import Task, Resource
from fluidml.swarm import Dolphin
from fluidml.storage import ResultsStore, InMemoryStore
from fluidml.storage.utils import pack_results

try:
    from rich.traceback import install
    install(extra_lines=2)
except ImportError:
    pass


logger = logging.getLogger(__name__)


class Swarm:
    def __init__(self,
                 n_dolphins: Optional[int] = None,
                 resources: Optional[List[Resource]] = None,
                 results_store: Optional[ResultsStore] = None,
                 start_method: str = 'spawn',
                 exit_on_error: bool = True,
                 return_results: bool = False):
        """

        Configure workers, resources, results_store which are used to run the tasks

        Args:
            n_dolphins (Optional[int], optional): number of parallel workers. Defaults to None.
            resources (Optional[List[Resource]], optional): a list of resources that are assigned to workers.
                    If len(resources) < n_dolphins, resources are assigned randomly to workers
            results_store (Optional[ResultsStore], optional): an instance of results store for results management
                    If nothing is provided, a non-persistent InMemoryStore store is used
            start_method (str, optional): start method for multiprocessing. Defaults to 'spawn'.
            exit_on_error (bool, optional): whether to exit when an error happens. Defaults to True.
            return_results (bool, optional): return results dictionary after run(). Defaults to False.
        """
        set_start_method(start_method, force=True)

        self.n_dolphins = n_dolphins if n_dolphins else multiprocessing.cpu_count()
        self.resources = Swarm._allocate_resources(self.n_dolphins, resources)
        self.manager = Manager()
        self.lock = Lock()

        self.scheduled_queue = Queue()
        self.running_queue = self.manager.list()
        self.done_queue = self.manager.list()
        self.logging_queue = Queue()

        self.results_store = results_store if results_store is not None else InMemoryStore(self.manager)
        self.exception = self.manager.dict()
        self.return_results = True if isinstance(
            self.results_store, InMemoryStore) else return_results
        self.tasks: Dict[int, Task] = {}

        self.logging_listener = LoggingListener(logging_queue=self.logging_queue)

        # dolphin workers for task execution
        self.dolphins = [Dolphin(resource=self.resources[i],
                                 scheduled_queue=self.scheduled_queue,
                                 running_queue=self.running_queue,
                                 done_queue=self.done_queue,
                                 logging_queue=self.logging_queue,
                                 lock=self.lock,
                                 tasks=self.tasks,
                                 exception=self.exception,
                                 exit_on_error=exit_on_error,
                                 results_store=self.results_store) for i in range(self.n_dolphins)]

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
    def _get_entry_point_tasks(tasks: List[Task]) -> Dict[int, str]:
        """
        Gets tasks that are run first (tasks with no predecessors)
        """
        entry_task_ids = {}
        for task in tasks:
            if len(task.predecessors) == 0:
                entry_task_ids[task.id_] = task.name
        return entry_task_ids

    def _collect_results(self) -> Dict[str, Any]:
        results = pack_results(all_tasks=list(self.tasks.values()),
                               results_store=self.results_store,
                               return_results=self.return_results)
        return results

    def work(self, tasks: List[Task]) -> Optional[Dict[str, Union[List[Dict], Dict]]]:

        # get entry point task ids
        entry_point_tasks: Dict[int, str] = self._get_entry_point_tasks(tasks)

        # also update the current tasks
        for task in tasks:
            self.tasks[task.id_] = task

        # start the listener thread to receive log messages from child processes
        self.logging_listener.start()

        # schedule entry point tasks
        for task_id, task_name in entry_point_tasks.items():
            logger.debug(f'Swarm scheduling task {task_name}-{task_id}.')
            self.scheduled_queue.put(task_id)

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
        if self.exception:
            raise ChildProcessError

        # return all results
        return self._collect_results()

    def close(self):
        self.tasks = {}
        self.done_queue = self.manager.list()
        self.running_queue = self.manager.list()
        for dolphin in self.dolphins:
            # dolphin.terminate() is the backup for python 3.6 where .close() is not available
            try:
                dolphin.close()
            except AttributeError:
                dolphin.terminate()
        self.logging_listener.stop()
